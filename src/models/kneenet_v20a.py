# -*- coding: utf-8 -*-
"""KneeNet v20a — v17 + queries MHA + prior (label, slot).

Base v17 inchangee : ResNet-50 RadImageNet, stem A+B, CAM 14x14 dilated,
BiLSTM residuel init 0, n_series=3, BN gelee, dataset crop130.

v17 score chaque coupe par un Linear(C -> 12) sur le GAP+LSTM.
v20a : 12 queries apprises (une par cible) font une attention multi-tetes
sur les  S*D  tokens. Un bias additif (label, slot) initialise par un
prior anatomique (ACL->sag, MCL->cor, PF OA->ax, ...) ; un residu appris
peut s'en ecarter. Ce n'est pas un filtre dur.

`attn_score` reste un nn.Module appele dans le forward (hooks d'inspect).
Les valeurs du pool restent les tokens CAM, comme v17.

Juger vs subset_v17_crop130_frac025 (val ET gold, Δ > 0.01).
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
ATTN_HEADS = 8

# Ordre TARGETS : ACL, MCL, Med Meniscus, Lat Meniscus, Med OA, Lat OA,
# PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.
# Colonnes = slots v17 : Sagittal, Coronal, Axial (fluid-prioritaire).
# Valeurs dans (0, 1], transformees en log : bias additif sur les scores.
SLOT_PRIOR = (
    (1.00, 0.65, 0.40),  # ACL
    (0.65, 1.00, 0.45),  # MCL
    (0.95, 0.95, 0.60),  # Medial Meniscus
    (1.00, 0.90, 0.65),  # Lateral Meniscus
    (0.85, 1.00, 0.40),  # Medial OA
    (0.90, 1.00, 0.40),  # Lateral OA
    (0.90, 0.55, 1.00),  # PF OA
    (0.90, 0.85, 0.90),  # Effusion
    (0.90, 0.85, 0.90),  # Synovitis
    (0.95, 0.70, 1.00),  # Baker's
    (1.00, 0.90, 0.80),  # Contusion
    (1.00, 0.95, 0.80),  # Fracture
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


class _LabelQueryMHA(nn.Module):
    """Scores (B, S, D, T) : queries par label × cles GAP, + prior (T, S).

    Signature compatible avec nn.Linear(C, T) sur le dernier axe, pour
    garder le hook `attn_score` de inspect_attention.
    """

    def __init__(self, feat_dim: int, n_targets: int, n_series: int,
                 n_heads: int = ATTN_HEADS, prior_scale: float = 1.0) -> None:
        super().__init__()
        if feat_dim % n_heads != 0:
            raise ValueError(f"feat_dim={feat_dim} non divisible par n_heads={n_heads}")
        self.n_targets = n_targets
        self.n_series = n_series
        self.n_heads = int(n_heads)
        self.head_dim = feat_dim // self.n_heads
        self.scale = self.head_dim ** -0.5

        self.label_query = nn.Parameter(torch.zeros(n_targets, feat_dim))
        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        nn.init.eye_(self.q_proj.weight)
        nn.init.eye_(self.k_proj.weight)

        table = torch.full((n_targets, n_series), 0.5)
        n_col = min(n_series, len(SLOT_PRIOR[0]))
        n_row = min(n_targets, len(SLOT_PRIOR))
        table[:n_row, :n_col] = torch.tensor(
            [row[:n_col] for row in SLOT_PRIOR[:n_row]], dtype=torch.float32,
        )
        self.register_buffer("slot_prior_p", table, persistent=True)
        self.slot_bias = nn.Parameter(torch.zeros(n_targets, n_series))
        self.prior_scale = nn.Parameter(torch.tensor(float(prior_scale)))

    def _prior_logits(self, s: int, dtype: torch.dtype) -> torch.Tensor:
        """(T, S) : log(table) * scale + bias appris."""
        p = self.slot_prior_p[:, :s].clamp(min=1e-4)
        bias = self.slot_bias[:, :s]
        return (p.log() * self.prior_scale + bias).to(dtype=dtype)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (B, S, D, C) -> scores (B, S, D, T)."""
        b, s, d, c = feats.shape
        n = s * d
        x = feats.reshape(b, n, c)
        q = self.q_proj(self.label_query).view(
            self.n_targets, self.n_heads, self.head_dim,
        )
        k = self.k_proj(x).view(b, n, self.n_heads, self.head_dim)
        scores = torch.einsum("thd,bnhd->bthn", q, k) * self.scale
        scores = scores.mean(dim=2)
        prior = self._prior_logits(s, scores.dtype)
        prior = prior.view(1, self.n_targets, s, 1).expand(b, -1, s, d)
        scores = scores.reshape(b, self.n_targets, s, d) + prior
        return scores.permute(0, 2, 3, 1).contiguous()


class KneeNet(nn.Module):
    arch_id = "kneenet_v20a"

    def __init__(self, backbone: str = "resnet50", n_series: int = 3,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 16, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 lstm_hidden: int = LSTM_HIDDEN,
                 attn_heads: int = ATTN_HEADS,
                 prior_scale: float = 1.0) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.map_hw = MAP_HW
        self.lstm_hidden = int(lstm_hidden)
        self.attn_heads = int(attn_heads)

        self.encoder = _ResNet50Encoder()
        if pretrained:
            self._load_radimagenet(backbone_weights)
        feat_dim = self.encoder.num_features

        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        self.attn_score = _LabelQueryMHA(
            feat_dim, n_targets, n_series,
            n_heads=self.attn_heads, prior_scale=prior_scale,
        )
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
            f"| BiLSTM hidden {self.lstm_hidden} | MHA {self.attn_heads}h "
            f"+ prior (label, slot)",
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
