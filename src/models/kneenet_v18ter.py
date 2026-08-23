# -*- coding: utf-8 -*-
"""KneeNet v18ter — T1 en residu hors softmax + stem T1 centre x3.

v18/v18bis mettent le T1 dans le meme softmax que les 72 tokens fluides,
avec une gate logsigmoid. Le T1 n'ouvre pas : g reste ~0.10, macro = v17.

Ici le softmax fluide ignore le slot 3. On pool le T1 a part (softmax
sur ses 24 coupes) et on ajoute : pooled = fluide + sigmoid(g) * T1.
Init g ≈ 0.10. Le T1 n'a plus a battre le fluide pour contribuer.

Stem fluide = A+B (comme v17). Stem T1 = coupe centrale x3 : le contraste
T1 n'est pas un voisinage fluide, A+B melangeait du bruit.

LSTM toujours partage sur les 4 sequences. BN gelee.
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
T1_SLOT = 3
# sigmoid(-2.2) ≈ 0.10
T1_GATE_INIT = -2.2


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


class KneeNet(nn.Module):
    arch_id = "kneenet_v18ter"

    def __init__(self, backbone: str = "resnet50", n_series: int = 4,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 16, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 lstm_hidden: int = LSTM_HIDDEN,
                 t1_gate_init: float = T1_GATE_INIT,
                 t1_drop_p: float = 0.0) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.map_hw = MAP_HW
        self.lstm_hidden = int(lstm_hidden)
        self.t1_drop_p = float(t1_drop_p)
        # inspect_attention : ne pas ajouter log_g dans le softmax fluide.
        self.t1_fusion = "residual"
        # scalaire : echelle du residu T1, init g ≈ 0.10.
        self.t1_gate_logit = nn.Parameter(torch.tensor(float(t1_gate_init)))

        self.encoder = _ResNet50Encoder()
        if pretrained:
            self._load_radimagenet(backbone_weights)
        feat_dim = self.encoder.num_features

        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        self.attn_score = nn.Linear(feat_dim, n_targets)
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
        g0 = float(torch.sigmoid(self.t1_gate_logit.detach()).item())
        print(
            f"RadImageNet ResNet-50 charge : {p} | CAM {MAP_HW}x{MAP_HW} "
            f"| BiLSTM hidden {self.lstm_hidden} "
            f"| T1 residu g0={g0:.3f} "
            f"(logit {float(self.t1_gate_logit.detach()):.2f}) "
            f"| stem T1=centre x3",
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
        """A+B fluide, centre x3 sur le slot T1. x (B,S,D,H,W), idx plat."""
        _b, s, d, _h, _w = x.shape
        bs = idx // (s * d)
        rest = idx % (s * d)
        ss = rest // d
        dd = rest % d
        d_m = (dd - 1).clamp(0, d - 1)
        d_p = (dd + 1).clamp(0, d - 1)
        center = x[bs, ss, dd]
        rgb_ab = torch.stack(
            [
                0.5 + 0.5 * (center - x[bs, ss, d_m]),
                center,
                0.5 + 0.5 * (x[bs, ss, d_p] - center),
            ],
            dim=1,
        )
        rgb_c = center.unsqueeze(1).expand(-1, 3, -1, -1)
        is_t1 = (ss == T1_SLOT).view(-1, 1, 1, 1)
        rgb = torch.where(is_t1, rgb_c, rgb_ab)
        rgb = self.stem_mix(rgb)
        return (rgb - self.img_mean) / self.img_std

    def _maybe_drop_t1(self, valid: torch.Tensor) -> torch.Tensor:
        """Masque le slot T1 (train seulement). valid (B, S, D)."""
        b, s, _d = valid.shape
        if not (self.training and s > T1_SLOT and self.t1_drop_p > 0):
            return valid
        valid = valid.clone()
        drop = torch.rand(b, device=valid.device) < self.t1_drop_p
        valid[drop, T1_SLOT] = False
        return valid

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
        tokens = local.reshape(b, s, d, self.n_targets, -1)
        tokens = tokens + delta.unsqueeze(3) + pe.unsqueeze(3) + se.unsqueeze(3)

        # Softmax fluide : le slot T1 est masque, il n'entre pas dans attn.
        scores_f = scores.clone()
        if s > T1_SLOT:
            scores_f[:, T1_SLOT] = -1.0e4
        scores_f = scores_f.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        attn_f = torch.softmax(
            scores_f.float().reshape(b, s * d, self.n_targets), dim=1,
        ).to(feats.dtype)
        pooled = torch.einsum(
            "bnt,bntc->btc",
            attn_f,
            tokens.reshape(b, s * d, self.n_targets, -1),
        )

        if s > T1_SLOT:
            scores_t1 = scores[:, T1_SLOT]
            valid_t1 = valid[:, T1_SLOT]
            scores_t1 = scores_t1.masked_fill(~valid_t1.unsqueeze(-1), -1.0e4)
            attn_t1 = torch.softmax(scores_t1.float(), dim=1).to(feats.dtype)
            pooled_t1 = torch.einsum(
                "bdt,bdtc->btc", attn_t1, tokens[:, T1_SLOT],
            )
            # (B,1,1) pour broadcast sur pooled (B, T, C) — pas (B,1)
            # qui alignait T=12 contre B=4.
            has_t1 = valid_t1.any(dim=1).to(dtype=pooled.dtype).view(b, 1, 1)
            g = torch.sigmoid(self.t1_gate_logit).to(dtype=pooled.dtype)
            pooled = pooled + g * pooled_t1 * has_t1

        logits = self.head(self.norm(pooled)).squeeze(-1)
        return logits, attn_f.reshape(b, s, d, self.n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, S, D, H, W) dans [0, 1] -> logits (B, 12)."""
        b, s, d, _h, _w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        if d > self.slice_embed.size(0):
            raise ValueError(
                f"D={d} > max_depth={self.slice_embed.size(0)} : "
                "augmenter max_depth"
            )

        valid = self._maybe_drop_t1(x.amax(dim=(-2, -1)) > 1e-6)
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
