# -*- coding: utf-8 -*-
"""KneeNet : 2.5D multi-series -> 12 logits par etude.

Entree  : (B, S, D, H, W) = (batch, 3 plans, 24 slices, 224, 224), valeurs
          normalisees (x - NORM_MEAN) / NORM_STD.
Backbone partage : timm convnext_tiny avec in_chans=24 (slices = canaux).
La stem 3-canaux ImageNet est adaptee : chaque canal recoit la moyenne des
poids RGB (la LayerNorm juste apres la stem reabsorbe l'echelle).
Fusion  : embeddings GAP des S plans concat -> LayerNorm -> MLP -> 12 logits.
Les plans manquants arrivent comme volumes de zeros : la LayerNorm de fusion
rend ca inoffensif, et le series dropout du Dataset entraine ce cas.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

N_TARGETS = 12


def _adapt_stem_24ch(model: nn.Module) -> None:
    """Replique la moyenne des poids RGB de la stem sur les 24 canaux."""
    stem = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            stem = m
            break
    assert stem is not None and stem.weight.shape[1] == 3
    with torch.no_grad():
        w = stem.weight  # (out, 3, k, k)
        w_new = w.mean(dim=1, keepdim=True).repeat(1, 24, 1, 1)
        stem.weight = nn.Parameter(w_new.contiguous())


class KneeNet(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", n_series: int = 3,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True) -> None:
        super().__init__()
        self.n_series = n_series
        if pretrained:
            # creation 3 canaux pour charger ImageNet, puis adaptation 24
            self.encoder = timm.create_model(
                backbone, pretrained=True, in_chans=3, num_classes=0,
                global_pool="avg",
            )
            _adapt_stem_24ch(self.encoder)
        else:
            self.encoder = timm.create_model(
                backbone, pretrained=False, in_chans=24, num_classes=0,
                global_pool="avg",
            )

        feat_dim = self.encoder.num_features  # 768 pour convnext_tiny
        self.fusion = nn.Sequential(
            nn.LayerNorm(n_series * feat_dim),
            nn.Linear(n_series * feat_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d, h, w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        xf = x.reshape(b * s, d, h, w)
        feats = self.encoder(xf)                 # (B*S, F)
        feats = feats.reshape(b, s * feats.shape[-1])
        return self.fusion(feats)                # (B, 12)


def count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{n/1e6:.1f}M"
