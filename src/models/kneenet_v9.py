# -*- coding: utf-8 -*-
"""KneeNet v9 — 4e slot T1 (non-fluide) + gate.

Entree : (B, 4, 24, 224, 224)
  slot 0-2 : sag / cor / ax fluid-prioritaire (comme v8)
  slot 3   : non-fluide, sagittal d'abord, jamais une serie deja dans 0-2

Encode = v8/v9 naif : UN SEUL passage dans ConvNeXt (fluide + T1 ensemble),
gradient checkpointing. Les essais avec detach / eval() / deux buffers
collapseaient (proba std -> 0) : le checkpoint + bf16 ne supporte pas ces
manips inplace.

Le slot T1 ne dilue pas le softmax :
  gate g = sigmoid(t1_gate_logit), init g ≈ 0 : les scores d'attention
  du slot 3 recoivent +log(g). Le modele peut ignorer le T1 au depart
  puis l'ouvrir s'il aide. Le gradient vers ConvNeXt passe par tous les
  tokens (comme v9 naif) ; la gate n'empeche que le POOLING, pas l'encode.

Attention aux probas constantes (proba std -> 0, AUC ~0.5) : la sortie
depend des pixels via UN SEUL chemin (Linear attn_score de 768). BCE sur
des logits ~0 -> gradient lineaire ~0 -> seuls les biais du head bougent.
t1_drop_p / embeddings / gate ne changent rien. FIX : attn_temp plus
petit (init 0.2) pour amplifier les scores d'attention, et head_init_std
pour etaler les biais de depart.

t1_drop_p (optionnel) : pendant l'entrainement, masque tout le slot 3
avec cette proba (en plus du series_drop dataset).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

N_TARGETS = 12
T1_SLOT = 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class KneeNet(nn.Module):
    arch_id = "kneenet_v9"

    def __init__(self, backbone: str = "convnext_tiny", n_series: int = 4,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 16, max_depth: int = 64,
                 t1_gate_init: float = -4.0, t1_drop_p: float = 0.0,
                 attn_temp: float = 0.2, head_init_std: float = 0.02) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.t1_drop_p = float(t1_drop_p)
        self.attn_temp = float(attn_temp)
        self.head_init_std = float(head_init_std)

        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, in_chans=3, num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.encoder.num_features  # 768 pour convnext_tiny

        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        # log(g) ajoute aux scores du slot T1 ; sigmoid(-4) ≈ 0.018
        self.t1_gate_logit = nn.Parameter(torch.tensor(float(t1_gate_init)))
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
        self._init_head()

    def _init_head(self) -> None:
        """Etale les biais du head (init) : logits de depart disperses, le
        BCE voit tout de suite un signal par etude au lieu de ~prevalence."""
        std = self.head_init_std
        if std <= 0:
            return
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.bias, mean=0.0, std=std)

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
        has_t1 = s > T1_SLOT
        if has_t1 and self.training and self.t1_drop_p > 0:
            valid = valid.clone()
            drop = torch.rand(b, device=x.device) < self.t1_drop_p
            valid[drop, T1_SLOT] = False

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
        # attn_temp < 1 : scores divises par temp -> differences amplifiees
        scores = self.attn_score(feats) / self.attn_temp
        if has_t1:
            # log(sigmoid(logit)) : g≈0 au depart, g→1 si le T1 aide
            log_g = F.logsigmoid(self.t1_gate_logit).to(dtype=scores.dtype)
            scores = scores.clone()
            scores[:, T1_SLOT] = scores[:, T1_SLOT] + log_g
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        scores = scores.reshape(b, s * d, self.n_targets)
        attn = torch.softmax(scores.float(), dim=1).to(feats.dtype)  # (B, N, T)

        tokens = feats.reshape(b, s * d, -1)                         # (B, N, F)
        pooled = torch.einsum("bnt,bnf->btf", attn, tokens)          # (B, T, F)
        return self.head(self.norm(pooled)).squeeze(-1)              # (B, T)
