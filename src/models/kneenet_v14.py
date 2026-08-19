# -*- coding: utf-8 -*-
"""KneeNet v14 — v11 + pooling spatial 7x7 (CAM appris).

Meme stem A+B que v11 (diffs + stem_mix). L'encodeur ResNet-50 s'arrete
apres layer4 : carte (2048, 7, 7) au lieu du GAP.

Deux softmax emboites, 12 tetes chacun :
  1. spatial_score : « ou dans la coupe » (softmax sur 49 positions)
  2. attn_score    : « quelle coupe » (softmax sur 72 tokens, comme v11)

`spatial_score` est init a 0 -> softmax uniforme 1/49 = GAP. A l'epoch 0
le pooling spatial recopie v11 ; le modele peut ensuite piquer par cible
(ACL vs Baker sur la meme sagittale).

Entree : (B, S, D, H, W) grayscale [0, 1] — dataset crop130 inchange.
BN de l'encodeur : running stats gelees (comme v11).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50

N_TARGETS = 12
DEFAULT_WEIGHTS = "weights/ResNet50.pt"
MAP_HW = 7


class _ResNet50Encoder(nn.Module):
    """ResNet-50 jusqu'a layer4. Meme cles `backbone.N.*` que v11 (avgpool
    n'a pas de parametres) : le checkpoint RadImageNet charge en strict."""

    def __init__(self) -> None:
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.num_features = 2048
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class KneeNet(nn.Module):
    arch_id = "kneenet_v14"

    def __init__(self, backbone: str = "resnet50", n_series: int = 3,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 16, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)

        self.encoder = _ResNet50Encoder()
        if pretrained:
            self._load_radimagenet(backbone_weights)
        feat_dim = self.encoder.num_features

        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        self.attn_score = nn.Linear(feat_dim, n_targets)
        # CAM appris : 1x1, zero -> uniforme = GAP (identite v11)
        self.spatial_score = nn.Conv2d(feat_dim, n_targets, kernel_size=1)
        nn.init.zeros_(self.spatial_score.weight)
        nn.init.zeros_(self.spatial_score.bias)
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
        print(f"RadImageNet ResNet-50 charge : {p}", flush=True)

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
        """rgb : (N, 3, H, W) dans [0, 1] -> (N, C, 7, 7)."""
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

    def _slice_pool(self, x: torch.Tensor, valid: torch.Tensor,
                    gaps: torch.Tensor, local: torch.Tensor):
        """gaps (BSD,C), local (BSD,T,C) -> logits (B,T), slice_attn (B,S,D,T)."""
        b, s, d = valid.shape
        pe = self.plane_embed[:s].view(1, s, 1, -1)
        se = self.slice_embed[:d].view(1, 1, d, -1)
        feats = gaps.reshape(b, s, d, -1) + pe + se
        scores = self.attn_score(feats)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        scores = scores.reshape(b, s * d, self.n_targets)
        attn = torch.softmax(scores.float(), dim=1).to(feats.dtype)
        tokens = local.reshape(b, s, d, self.n_targets, -1)
        tokens = tokens + pe.unsqueeze(3) + se.unsqueeze(3)
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
        """Forward eval + cartes. spatial_attn : (B, S, D, T, 7, 7)."""
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
        return {
            "logits": logits,
            "slice_attn": slice_attn,
            "spatial_attn": spatial.reshape(
                b, s, d, self.n_targets, hw, hw,
            ),
            "valid": valid,
        }
