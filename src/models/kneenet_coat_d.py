# -*- coding: utf-8 -*-
"""KneeNet-CoAt-d — CoAt-a + CAM du stage 2 (14x14), residu init 0.

CoAt-a ne lit que la carte finale 7x7 (stage 3, stride 32). Le stage 2
est deja une carte 14x14 x 384 (7 blocs Transformer, resolution de la
CAM v20b) et le premier bloc du stage 3 la jette dans un AvgPool 2x2
avant l'attention 7x7. Forcer le stage 3 a rester en 14x14 casserait
le biais de position relatif, appris en 7x7.

Ici la CAM 7x7, le GAP et le BiLSTM restent ceux de CoAt-a. Une seconde
CAM sur le stage 2 produit un vecteur par cible, projete 384 -> 768 par
un Linear init 0. Epoch 0 = CoAt-a. `extra_log` suit `mid_proj_l1` :
s'il reste a 0, la branche n'a pas ete utilisee.
"""
from __future__ import annotations

import inspect

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.kneenet_coat_a import KneeNet as _CoatA

# forward_intermediates : 0 = stem, 1 = stage0, 2 = stage1, 3 = stage2.
_MID_INDEX = 3


def _stage2_channels(encoder: nn.Module) -> int:
    norm = encoder.backbone.stages[2].blocks[-1].norm2
    shape = norm.normalized_shape
    if isinstance(shape, int):
        return shape
    return int(shape[0])


class KneeNet(_CoatA):
    arch_id = "kneenet_coat_d"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        mid_ch = _stage2_channels(self.encoder)
        self.spatial_score_mid = nn.Conv2d(mid_ch, self.n_targets, kernel_size=1)
        nn.init.zeros_(self.spatial_score_mid.weight)
        nn.init.zeros_(self.spatial_score_mid.bias)
        self.mid_proj = nn.Linear(mid_ch, self.encoder.num_features)
        nn.init.zeros_(self.mid_proj.weight)
        nn.init.zeros_(self.mid_proj.bias)
        print(
            f"KneeNet-CoAt-d | CAM stage2 {mid_ch}ch 14x14 "
            f"residu init 0 sur la CAM 7x7",
            flush=True,
        )

    @torch.no_grad()
    def extra_log(self) -> dict[str, float]:
        out = super().extra_log()
        out["mid_proj_l1"] = float(self.mid_proj.weight.detach().abs().mean())
        return out

    def _forward_pair(self, rgb: torch.Tensor):
        final, inter = self.encoder.backbone.forward_intermediates(
            rgb, indices=(_MID_INDEX,),
        )
        return final, inter[0]

    def _encode_pair_chunks(self, rgb: torch.Tensor):
        n = rgb.size(0)
        feat_dim = self.encoder.num_features
        mid_ch = self.mid_proj.in_features
        if n == 0:
            h7, w7 = rgb.size(-2) // 32, rgb.size(-1) // 32
            h14, w14 = rgb.size(-2) // 16, rgb.size(-1) // 16
            return (
                rgb.new_zeros(0, feat_dim, h7, w7),
                rgb.new_zeros(0, mid_ch, h14, w14),
            )
        finals, mids = [], []
        chunk = max(1, self.encode_chunk)
        for i in range(0, n, chunk):
            part = rgb[i:i + chunk]
            if self.training:
                final, mid = checkpoint(
                    self._forward_pair, part, use_reentrant=False,
                )
            else:
                final, mid = self._forward_pair(part)
            finals.append(final)
            mids.append(mid)
        return torch.cat(finals, dim=0), torch.cat(mids, dim=0)

    def _fuse_mid(self, local: torch.Tensor, maps_mid: torch.Tensor) -> torch.Tensor:
        """local (N, T, C) + CAM 14x14 projetee. Linear a 0 -> identite."""
        n, _c, h, w = maps_mid.shape
        logits = self.spatial_score_mid(maps_mid)
        attn = torch.softmax(
            logits.float().reshape(n, self.n_targets, h * w), dim=-1,
        ).to(maps_mid.dtype)
        flat = maps_mid.reshape(n, maps_mid.size(1), h * w)
        local_mid = torch.einsum("nth,nch->ntc", attn, flat)
        return local + self.mid_proj(local_mid)

    def _encode_valid(self, x: torch.Tensor, valid: torch.Tensor,
                      return_attn: bool = False):
        b, s, d, img_h, img_w = x.shape
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        gaps = x.new_zeros(n_tok, feat_dim)
        local = x.new_zeros(n_tok, self.n_targets, feat_dim)
        spatial = None
        if idx.numel() > 0:
            maps, maps_mid = self._encode_pair_chunks(self._stem_rgb(x, idx))
            pooled = self._spatial_pool(maps, return_attn=return_attn)
            gap, loc = pooled[0], pooled[1]
            loc = self._fuse_mid(loc, maps_mid)
            gaps = gaps.to(dtype=gap.dtype)
            local = local.to(dtype=loc.dtype)
            gaps[idx] = gap
            local[idx] = loc
            if return_attn:
                spat = pooled[2]
                spatial = spat.new_zeros(n_tok, self.n_targets, *spat.shape[-2:])
                spatial[idx] = spat
        if return_attn and spatial is None:
            spatial = x.new_zeros(n_tok, self.n_targets, img_h // 32, img_w // 32)
        return gaps, local, spatial


KneeNet.__init__.__signature__ = inspect.signature(_CoatA.__init__)
