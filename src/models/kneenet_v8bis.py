# -*- coding: utf-8 -*-
"""KneeNet v8bis — v8 + ConvNeXt-Small.

Meme stem R=G=B / attn 12 tetes que v8. Small a besoin d'un depart plus
doux que Tiny (sinon la BCE ecrase les logits vers une constante) :

  - bias_init : biais de sortie par cible = logit des priors (train.py)
  - freeze_encoder_epochs (train.py) : encodeur gele 2 epochs, puis degel
    avec re-warmup + lr_backbone 2e-5
  - gamma LayerScale geles (valeurs ImageNet)

Run subset de reference : subset_v8bis_lr2e5_frac025.
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
    arch_id = "kneenet_v8bis"

    def __init__(self, backbone: str = "convnext_small", n_series: int = 3,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 16, max_depth: int = 64,
                 bias_init: list[float] | None = None) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.bias_init = bias_init

        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, in_chans=3, num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.encoder.num_features
        self._freeze_layer_scale()

        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        self.attn_score = nn.Linear(feat_dim, n_targets)
        self.norm = nn.LayerNorm(feat_dim)
        self.out_bias = nn.Parameter(torch.zeros(n_targets))
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
        self._init_head()

    def _init_head(self) -> None:
        if self.bias_init is None:
            return
        b = torch.tensor(self.bias_init, dtype=self.out_bias.dtype)
        if b.numel() != self.out_bias.numel():
            raise ValueError(
                f"bias_init de taille {b.numel()}, "
                f"attendu {self.out_bias.numel()}"
            )
        with torch.no_grad():
            self.out_bias.copy_(b)

    def _freeze_layer_scale(self) -> None:
        n = 0
        for name, p in self.encoder.named_parameters():
            if name.endswith("gamma"):
                p.requires_grad = False
                n += 1
        print(f"LayerScale gele : {n} tenseurs gamma", flush=True)

    def set_encoder_frozen(self, flag: bool) -> None:
        """Gele/degel de l'encodeur. Les gamma LayerScale restent geles."""
        for name, p in self.encoder.named_parameters():
            p.requires_grad = (not flag) and not name.endswith("gamma")

    def encoder_parameters(self):
        for p in self.encoder.parameters():
            if p.requires_grad:
                yield p

    def head_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("encoder."):
                yield p

    def _encode_chunks(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb : (N, 3, H, W) deja normalise ImageNet -> (N, F)."""
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

        valid = x.amax(dim=(-2, -1)) > 1e-6
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

        scores = self.attn_score(feats)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        scores = scores.reshape(b, s * d, self.n_targets)
        attn = torch.softmax(scores.float(), dim=1).to(feats.dtype)

        tokens = feats.reshape(b, s * d, -1)
        pooled = torch.einsum("bnt,bnf->btf", attn, tokens)
        return self.head(self.norm(pooled)).squeeze(-1) + self.out_bias
