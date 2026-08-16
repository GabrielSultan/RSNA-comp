# -*- coding: utf-8 -*-
"""KneeNet v9 — FIGEE. Ne pas modifier.

= v8 + 4e slot T1 (non-fluide). L'attention masque le slot s'il est a zero
(serie absente). plane_embed[3] apprend l'identite « T1 / anatomie ».

Entree : (B, 4, 24, 224, 224)
  slot 0-2 : sag / cor / ax fluid-prioritaire (comme v8)
  slot 3   : non-fluide, sagittal d'abord, jamais une serie deja dans 0-2

~96 forwards ConvNeXt / etude (vs 72 en v8). Meme tete d'attention 12 cibles.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

N_TARGETS = 12
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class KneeNet(nn.Module):
    arch_id = "kneenet_v9"

    def __init__(self, backbone: str = "convnext_tiny", n_series: int = 4,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 16, max_depth: int = 64) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)

        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, in_chans=3, num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.encoder.num_features  # 768 pour convnext_tiny

        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        # 12 tetes : un score d'attention par cible, partage sur tous les tokens
        self.attn_score = nn.Linear(feat_dim, n_targets)
        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.register_buffer(
            "img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=True,
        )

    def encoder_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("encoder."):
                yield p

    def _encode_chunks(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb : (N, 3, H, W) deja normalise ImageNet -> (N, F).

        Checkpoint des l'entrainement : les coupes du loader n'ont pas
        requires_grad, donc on ne doit PAS conditionner dessus (sinon toutes
        les activations ConvNeXt restent en VRAM).
        """
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
        """x : (B, S, D, H, W) dans [0, 1] -> logits (B, 12)."""
        b, s, d, h, w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        if d > self.slice_embed.size(0):
            raise ValueError(
                f"D={d} > max_depth={self.slice_embed.size(0)} : "
                "augmenter max_depth"
            )

        valid = x.amax(dim=(-2, -1)) > 1e-6          # (B, S, D)
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        feats = x.new_zeros(n_tok, feat_dim)
        if idx.numel() > 0:
            gray = x.reshape(n_tok, h, w).index_select(0, idx)
            rgb = gray.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
            rgb = (rgb - self.img_mean) / self.img_std
            feats[idx] = self._encode_chunks(rgb)
        feats = feats.reshape(b, s, d, -1)
        feats = feats + self.plane_embed[:s].view(1, s, 1, -1)
        feats = feats + self.slice_embed[:d].view(1, 1, d, -1)

        # scores (B, S, D, T) -> softmax sur les S*D tokens, par cible
        scores = self.attn_score(feats)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        scores = scores.reshape(b, s * d, self.n_targets)
        attn = torch.softmax(scores.float(), dim=1).to(feats.dtype)  # (B, N, T)

        tokens = feats.reshape(b, s * d, -1)                         # (B, N, F)
        pooled = torch.einsum("bnt,bnf->btf", attn, tokens)          # (B, T, F)
        return self.head(self.norm(pooled)).squeeze(-1)              # (B, T)
