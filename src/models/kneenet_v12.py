# -*- coding: utf-8 -*-
"""KneeNet v12 — A+B+D (ne pas modifier v11).

v10 ResNet-50 RadImageNet + attention 12 tetes, avec :
  D : entree (B, S, D, 3, H, W) deja R=k-1, G=k, B=k+1 **DICOM**
      (NPZ crop130_rgb25d). Pas de re-stack sur le linspace.
  B : canaux = diffs a travers le plan (G=centre, R/B = gradients).
  A : Conv2d 1x1, init = tout depuis G (epoch 0 == v10 / centre seul).

BN gelée comme v10/v11. Dataset / aug doivent passer rgb_neighbors=True.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50

N_TARGETS = 12
DEFAULT_WEIGHTS = "weights/ResNet50.pt"


class _ResNet50Encoder(nn.Module):
    """ResNet-50 sans tete, cles = `backbone.N.*` (checkpoint RadImageNet)."""

    def __init__(self) -> None:
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.num_features = 2048
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


class KneeNet(nn.Module):
    arch_id = "kneenet_v12"
    rgb_neighbors = True

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

        self.channel_mix = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.channel_mix.weight.zero_()
            self.channel_mix.weight[:, 1, 0, 0] = 1.0

        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        self.attn_score = nn.Linear(feat_dim, n_targets)
        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.register_buffer(
            "img_mean", torch.zeros(1, 3, 1, 1), persistent=True,
        )
        self.register_buffer(
            "img_std", torch.ones(1, 3, 1, 1), persistent=True,
        )

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

    @staticmethod
    def _diff_channels(rgb: torch.Tensor) -> torch.Tensor:
        """rgb (N, 3, H, W) R=k-1 G=k B=k+1 -> G=k, R/B = diffs [0, 1]."""
        prev, center, nxt = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        return torch.stack((
            (0.5 + 0.5 * (center - prev)).clamp(0, 1),
            center,
            (0.5 + 0.5 * (nxt - center)).clamp(0, 1),
        ), dim=1)

    def _encode_chunks(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb : (N, 3, H, W) dans [0, 1] -> (N, F)."""
        n = rgb.size(0)
        if n == 0:
            return rgb.new_zeros(0, self.encoder.num_features)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, S, D, 3, H, W) dans [0, 1] -> logits (B, 12)."""
        if x.ndim != 6:
            raise ValueError(
                f"v12 attend (B, S, D, 3, H, W) rgb DICOM, recu {tuple(x.shape)}"
            )
        b, s, d, c, h, w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        if c != 3:
            raise ValueError(f"attendu 3 canaux RGB, recu C={c}")
        if d > self.slice_embed.size(0):
            raise ValueError(
                f"D={d} > max_depth={self.slice_embed.size(0)} : "
                "augmenter max_depth"
            )

        valid = x[:, :, :, 1].amax(dim=(-2, -1)) > 1e-6
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        feats = x.new_zeros(n_tok, feat_dim)
        if idx.numel() > 0:
            rgb = x.reshape(n_tok, 3, h, w).index_select(0, idx)
            rgb = self._diff_channels(rgb)
            rgb = self.channel_mix(rgb)
            rgb = (rgb - self.img_mean) / self.img_std
            enc = self._encode_chunks(rgb)
            if feats.dtype != enc.dtype:
                feats = feats.to(dtype=enc.dtype)
            feats[idx] = enc
        feats = feats.reshape(b, s, d, -1)
        feats = feats + self.plane_embed[:s].view(1, s, 1, -1)
        feats = feats + self.slice_embed[:d].view(1, 1, d, -1)

        scores = self.attn_score(feats)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        scores = scores.reshape(b, s * d, self.n_targets)
        attn = torch.softmax(scores.float(), dim=1).to(feats.dtype)

        tokens = feats.reshape(b, s * d, -1)
        pooled = torch.einsum("bnt,bnf->btf", attn, tokens)
        return self.head(self.norm(pooled)).squeeze(-1)
