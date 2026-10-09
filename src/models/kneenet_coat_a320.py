# -*- coding: utf-8 -*-
"""KneeNet-CoAt-a320 — CoAt-a sur crop 130 mm **320x320**.

Meme graphe que ``kneenet_coat_a`` : stem R=k-1 / G=k / B=k+1, CAM sur
la carte finale, BiLSTM, 6 slots, BN gelé. Le YAML LLRD 0.8 / no_decay
ne change pas le module.

CoAtNet-0 (timm ``coatnet_0_rw_224``) a un biais relatif fixe, une
fenetre par stage Transformer. A 224 : stage 2 = 14 (carte 14x14),
stage 3 = 7 (carte 7x7). A 320 : stage 2 = 20, stage 3 = 10 (CAM 10x10).
``create_model(..., img_size=320)`` passe par ``checkpoint_filter_fn``,
qui interpole ``relative_position_bias_table`` (14 -> 20, 7 -> 10).
Les deux cartes sont divisibles par leur fenetre.

Entree : (B, 6, 24, 320, 320) grayscale [0, 1], slots v20b.
Volumes : ``data/crop130_320`` (gris ; le stem reconstruit le triplet).
Ne pas utiliser coat_d : sa CAM intermediaire suppose 14x14 / 7x7.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from src.models.kneenet_coat_a import (
    DEFAULT_WEIGHTS,
    LSTM_HIDDEN,
    N_PROTOCOL_SLOTS,
    N_TARGETS,
    TIMM_NAME,
    KneeNet as KneeNetA,
)

IMG_SIZE = 320
# stride cumule jusqu'a la carte finale (stem 2 + 4 stages a stride 2).
_DOWNSAMPLE = 32


class KneeNet(KneeNetA):
    arch_id = "kneenet_coat_a320"

    # Signature explicite : build_model ne transmet que les cles YAML
    # nommees ici. Avec **kwargs, encode_chunk / dropout du YAML etaient
    # ignores.
    def __init__(self, img_size: int = IMG_SIZE, backbone: str = TIMM_NAME,
                 n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 8, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 lstm_hidden: int = LSTM_HIDDEN) -> None:
        super().__init__(
            backbone=backbone, n_series=n_series, n_targets=n_targets,
            head_dim=head_dim, dropout=dropout, pretrained=pretrained,
            encode_chunk=encode_chunk, max_depth=max_depth,
            backbone_weights=backbone_weights, lstm_hidden=lstm_hidden,
            img_size=img_size,
        )
        if self.img_size % _DOWNSAMPLE:
            raise ValueError(
                f"img_size {self.img_size} non divisible par {_DOWNSAMPLE} "
                "(fenetre CoAt = img_size/32)"
            )
        grid = self.img_size // _DOWNSAMPLE
        print(
            f"KneeNet-CoAt-a320 | entree {self.img_size} | "
            f"CAM {grid}x{grid} | rel-pos stage2 {grid * 2} "
            f"(14 -> {grid * 2}) stage3 {grid} (7 -> {grid}) | "
            f"encode_chunk {self.encode_chunk} | dropout {dropout:g}",
            flush=True,
        )

    def _check_size(self, x: torch.Tensor) -> None:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if h != self.img_size or w != self.img_size:
            raise ValueError(
                f"coat_a320 attend {self.img_size}x{self.img_size}, recu {h}x{w} "
                "(zip_path=data/crop130_320, img_size=320, pas de rgb_neighbors)"
            )

    def _encode_indexed(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Stem + encodeur par paquets.

        Le stem 2.5D materialise R/G/B en fp32. Sur un batch 4 plein
        (576 coupes, 320) ca fait un seul bloc de 675 Mo, que le
        backward n'arrive pas a allouer sur 8 Go Windows (cudaMalloc
        echoue, expandable_segments indisponible). Decoupe : memes
        gradients, plus de gros bloc contigu.
        """
        chunk = max(1, int(self.encode_chunk))
        outs = []
        n = int(idx.numel())
        for i in range(0, n, chunk):
            rgb = self._stem_rgb(x, idx[i:i + chunk])
            if self.training:
                feats = checkpoint(self.encoder, rgb, use_reentrant=False)
            else:
                feats = self.encoder(rgb)
            outs.append(feats)
        return torch.cat(outs, dim=0)

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
            maps = self._encode_indexed(x, idx)
            pooled = self._spatial_pool(maps, return_attn=return_attn)
            gap, loc = pooled[0], pooled[1]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_size(x)
        return super().forward(x)
