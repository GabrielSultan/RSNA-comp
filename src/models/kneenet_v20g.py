# -*- coding: utf-8 -*-
"""KneeNet v20g — v20b + prior (label, slot) T1/FS, pas de gate globale.

Base v20b inchangee : ResNet-50 RadImageNet, stem A+B, CAM 14x14 dilated,
BiLSTM residuel init 0, BN gelee, 6 slots protocole (FS / T1 par plan).
Pas de MHA (c'est v20a). Pas de masque dur T1 (c'est v18).

Levier : bias additif log(prior[t, slot]) sur attn_score. Table proche de 1
partout -> epoch 0 ≈ v20b. Un residu `slot_bias` init 0 peut s'en ecarter.
Ce n'est pas un filtre dur : un slot malus reste dans le softmax.

  MCL / OA lateral / ACL -> bonus SAG_T1 / COR_T1
  effusion / synovite / Baker -> bonus FS, malus T1
  menisques -> bonus SAG_FS / COR_FS

Entree : (B, 6, 24, H, W) grayscale [0, 1]. Ordre des slots = v20b.
Juger vs subset_v20b_crop130_frac025 (val ET gold, Δ > 0.01).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50

N_TARGETS = 12
DEFAULT_WEIGHTS = "weights/ResNet50.pt"
MAP_HW = 14
LSTM_HIDDEN = 256
N_PROTOCOL_SLOTS = 6

# Ordre TARGETS : ACL, MCL, Med Meniscus, Lat Meniscus, Med OA, Lat OA,
# PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.
# Colonnes = 6 slots v20b. Valeurs dans (0, 1+] proches de 1 : log ~ 0.
_HI = 1.12
_LO = 0.88
_ONE = 1.00
SLOT_PRIOR = (
    (_ONE, _ONE, _ONE, _HI, _HI, _ONE),  # ACL (osseux / T1)
    (_ONE, _ONE, _ONE, _HI, _HI, _ONE),  # MCL
    (_HI, _HI, _ONE, _ONE, _ONE, _ONE),  # Medial Meniscus
    (_HI, _HI, _ONE, _ONE, _ONE, _ONE),  # Lateral Meniscus
    (_ONE, _ONE, _ONE, _ONE, _ONE, _ONE),  # Medial OA
    (_ONE, _ONE, _ONE, _HI, _HI, _ONE),  # Lateral OA
    (_ONE, _ONE, _ONE, _ONE, _ONE, _ONE),  # PF OA
    (_HI, _HI, _HI, _LO, _LO, _LO),  # Effusion
    (_HI, _HI, _HI, _LO, _LO, _LO),  # Synovitis
    (_HI, _HI, _HI, _LO, _LO, _LO),  # Baker's
    (_ONE, _ONE, _ONE, _ONE, _ONE, _ONE),  # Contusion
    (_ONE, _ONE, _ONE, _ONE, _ONE, _ONE),  # Fracture
)


def _dilate_layer4(layer4: nn.Sequential, dilate: int = 2) -> None:
    """stride 32 -> 16, in-place. Poids 3x3 / 1x1 inchanges."""
    for i, block in enumerate(layer4):
        if i == 0:
            block.conv2.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        block.conv2.dilation = (dilate, dilate)
        block.conv2.padding = (dilate, dilate)


class _ResNet50Encoder(nn.Module):
    """ResNet-50 jusqu'a layer4 dilated. Cles `backbone.N.*` = v14/v16/v17."""

    def __init__(self) -> None:
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        _dilate_layer4(self.backbone[7])
        self.num_features = 2048
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _AttnScoreSlotPrior(nn.Linear):
    """Linear(C -> T) + log(prior[t, slot]) broadcast sur D.

    Signature (B, S, D, C) -> (B, S, D, T) comme nn.Linear sur le dernier
    axe. Un tenseur 2D (hooks CAM) n'a pas d'axe slot : prior ignore.
    """

    def __init__(self, feat_dim: int, n_targets: int, n_series: int) -> None:
        super().__init__(feat_dim, n_targets)
        self.n_targets = n_targets
        self.n_series = n_series
        table = torch.ones(n_targets, n_series)
        n_col = min(n_series, len(SLOT_PRIOR[0]))
        n_row = min(n_targets, len(SLOT_PRIOR))
        table[:n_row, :n_col] = torch.tensor(
            [row[:n_col] for row in SLOT_PRIOR[:n_row]], dtype=torch.float32,
        )
        self.register_buffer("slot_prior_p", table, persistent=True)
        self.slot_bias = nn.Parameter(torch.zeros(n_targets, n_series))

    def slot_prior_logits(self, s: int, dtype: torch.dtype) -> torch.Tensor:
        """(1, S, 1, T) : log(table) + residu appris (init 0)."""
        p = self.slot_prior_p[:, :s].clamp(min=1e-4)
        bias = self.slot_bias[:, :s]
        log_p = (p.log() + bias).to(dtype=dtype)
        return log_p.transpose(0, 1).view(1, s, 1, self.n_targets)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        scores = super().forward(feats)
        if feats.ndim == 4:
            s = feats.size(1)
            scores = scores + self.slot_prior_logits(s, scores.dtype)
        return scores


class KneeNet(nn.Module):
    arch_id = "kneenet_v20g"

    def __init__(self, backbone: str = "resnet50", n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 16, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 lstm_hidden: int = LSTM_HIDDEN) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.map_hw = MAP_HW
        self.lstm_hidden = int(lstm_hidden)

        self.encoder = _ResNet50Encoder()
        if pretrained:
            self._load_radimagenet(backbone_weights)
        feat_dim = self.encoder.num_features

        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        self.attn_score = _AttnScoreSlotPrior(feat_dim, n_targets, n_series)
        self.spatial_score = nn.Conv2d(feat_dim, n_targets, kernel_size=1)
        nn.init.zeros_(self.spatial_score.weight)
        nn.init.zeros_(self.spatial_score.bias)

        self.slice_lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Linear(2 * self.lstm_hidden, feat_dim)
        nn.init.zeros_(self.lstm_proj.weight)
        nn.init.zeros_(self.lstm_proj.bias)

        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.stem_mix = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self._init_stem_mix()
        self.register_buffer(
            "img_mean", torch.zeros(1, 3, 1, 1), persistent=True,
        )
        self.register_buffer(
            "img_std", torch.ones(1, 3, 1, 1), persistent=True,
        )

    def extra_log(self) -> dict[str, float]:
        """Amplitude du prior et du residu (history.json)."""
        attn = self.attn_score
        p = attn.slot_prior_p.detach()
        logp = p.clamp(min=1e-4).log() + attn.slot_bias.detach()
        return {
            "slot_prior_p_min": float(p.min()),
            "slot_prior_p_max": float(p.max()),
            "slot_prior_log_min": float(logp.min()),
            "slot_prior_log_max": float(logp.max()),
            "slot_prior_bias_abs": float(attn.slot_bias.detach().abs().mean()),
        }

    def _init_stem_mix(self) -> None:
        """out_c = in_G pour les 3 canaux -> R=G=B=k au depart."""
        with torch.no_grad():
            self.stem_mix.weight.zero_()
            self.stem_mix.weight[:, 1, 0, 0] = 1.0

    def _load_radimagenet(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"poids RadImageNet introuvables : {p.resolve()}"
            )
        sd = torch.load(p, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(sd, strict=True)
        print(
            f"RadImageNet ResNet-50 charge : {p} | CAM {MAP_HW}x{MAP_HW} "
            f"| BiLSTM hidden {self.lstm_hidden} | slots {self.n_series} "
            f"| prior (label, slot) p in [{_LO}, {_HI}]",
            flush=True,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def encoder_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("encoder."):
                yield p

    def _encode_chunks(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb : (N, 3, H, W) dans [0, 1] -> (N, C, 14, 14)."""
        n = rgb.size(0)
        c = self.encoder.num_features
        if n == 0:
            return rgb.new_zeros(0, c, MAP_HW, MAP_HW)
        outs = []
        chunk = max(1, self.encode_chunk)
        for i in range(0, n, chunk):
            part = rgb[i:i + chunk]
            if self.training:
                feats = checkpoint(self.encoder, part, use_reentrant=False)
            else:
                feats = self.encoder(part)
            outs.append(feats)
        return torch.cat(outs, dim=0)

    def _spatial_pool(self, maps: torch.Tensor, return_attn: bool = False):
        """maps (N, C, H, W) -> gap (N, C), local (N, T, C) [, attn (N, T, H, W)]."""
        n, c, h, w = maps.shape
        gap = maps.mean(dim=(-2, -1))
        logits = self.spatial_score(maps)
        attn = torch.softmax(
            logits.float().reshape(n, self.n_targets, h * w), dim=-1,
        ).to(maps.dtype)
        flat = maps.reshape(n, c, h * w)
        local = torch.einsum("nth,nch->ntc", attn, flat)
        if return_attn:
            return gap, local, attn.reshape(n, self.n_targets, h, w)
        return gap, local

    def _stem_rgb(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Triplet A+B pour les tokens valides. x (B,S,D,H,W), idx plat."""
        _b, s, d, _h, _w = x.shape
        bs = idx // (s * d)
        rest = idx % (s * d)
        ss = rest // d
        dd = rest % d
        d_m = (dd - 1).clamp(0, d - 1)
        d_p = (dd + 1).clamp(0, d - 1)
        center = x[bs, ss, dd]
        rgb = torch.stack(
            [
                0.5 + 0.5 * (center - x[bs, ss, d_m]),
                center,
                0.5 + 0.5 * (x[bs, ss, d_p] - center),
            ],
            dim=1,
        )
        rgb = self.stem_mix(rgb)
        return (rgb - self.img_mean) / self.img_std

    def _temporal_delta(self, gaps: torch.Tensor, b: int, s: int,
                        d: int) -> torch.Tensor:
        """gaps (BSD, C) -> residu LSTM (B, S, D, C). Epoch 0 : 0."""
        c = gaps.size(-1)
        seq = gaps.reshape(b * s, d, c)
        with torch.autocast("cuda", enabled=False):
            h, _ = self.slice_lstm(seq.float())
            delta = self.lstm_proj(h)
        return delta.to(dtype=gaps.dtype).reshape(b, s, d, c)

    def _slice_pool(self, x: torch.Tensor, valid: torch.Tensor,
                    gaps: torch.Tensor, local: torch.Tensor):
        """gaps (BSD,C), local (BSD,T,C) -> logits (B,T), slice_attn (B,S,D,T)."""
        b, s, d = valid.shape
        delta = self._temporal_delta(gaps, b, s, d)
        pe = self.plane_embed[:s].view(1, s, 1, -1)
        se = self.slice_embed[:d].view(1, 1, d, -1)
        feats = gaps.reshape(b, s, d, -1) + delta + pe + se
        scores = self.attn_score(feats)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        scores = scores.reshape(b, s * d, self.n_targets)
        attn = torch.softmax(scores.float(), dim=1).to(feats.dtype)
        tokens = local.reshape(b, s, d, self.n_targets, -1)
        tokens = tokens + delta.unsqueeze(3) + pe.unsqueeze(3) + se.unsqueeze(3)
        tokens = tokens.reshape(b, s * d, self.n_targets, -1)
        pooled = torch.einsum("bnt,bntc->btc", attn, tokens)
        logits = self.head(self.norm(pooled)).squeeze(-1)
        return logits, attn.reshape(b, s, d, self.n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, S, D, H, W) dans [0, 1] -> logits (B, 12)."""
        b, s, d, _h, _w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        if d > self.slice_embed.size(0):
            raise ValueError(
                f"D={d} > max_depth={self.slice_embed.size(0)} : "
                "augmenter max_depth"
            )

        valid = x.amax(dim=(-2, -1)) > 1e-6
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        gaps = x.new_zeros(n_tok, feat_dim)
        local = x.new_zeros(n_tok, self.n_targets, feat_dim)
        if idx.numel() > 0:
            maps = self._encode_chunks(self._stem_rgb(x, idx))
            if gaps.dtype != maps.dtype:
                gaps = gaps.to(dtype=maps.dtype)
                local = local.to(dtype=maps.dtype)
            gap, loc = self._spatial_pool(maps)
            gaps[idx] = gap
            local[idx] = loc
        logits, _attn = self._slice_pool(x, valid, gaps, local)
        return logits

    @torch.no_grad()
    def interpret(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward eval + cartes. spatial_attn : (B, S, D, T, 14, 14)."""
        self.eval()
        b, s, d, _h, _w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        valid = x.amax(dim=(-2, -1)) > 1e-6
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        hw = MAP_HW
        gaps = x.new_zeros(n_tok, feat_dim)
        local = x.new_zeros(n_tok, self.n_targets, feat_dim)
        spatial = x.new_zeros(n_tok, self.n_targets, hw, hw)
        if idx.numel() > 0:
            maps = self._encode_chunks(self._stem_rgb(x, idx))
            if gaps.dtype != maps.dtype:
                gaps = gaps.to(dtype=maps.dtype)
                local = local.to(dtype=maps.dtype)
                spatial = spatial.to(dtype=maps.dtype)
            gap, loc, spat = self._spatial_pool(maps, return_attn=True)
            gaps[idx] = gap
            local[idx] = loc
            spatial[idx] = spat
        logits, slice_attn = self._slice_pool(x, valid, gaps, local)
        delta = self._temporal_delta(gaps, b, s, d)
        pe = self.plane_embed[:s].view(1, s, 1, 1, -1)
        se = self.slice_embed[:d].view(1, 1, d, 1, -1)
        cam_tokens = (
            local.reshape(b, s, d, self.n_targets, -1)
            + delta.unsqueeze(3) + pe + se
        )
        return {
            "logits": logits,
            "slice_attn": slice_attn,
            "spatial_attn": spatial.reshape(
                b, s, d, self.n_targets, hw, hw,
            ),
            "cam_tokens": cam_tokens,
            "valid": valid,
        }
