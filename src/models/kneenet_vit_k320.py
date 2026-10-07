# -*- coding: utf-8 -*-
"""KneeNet-ViT-k320 — ViT-k (graphe e) sur crop 130 mm **320x320**.

Meme graphe que ``kneenet_vit_e`` : stem R=k-1 / G=k / B=k+1, CAM patch,
BiLSTM init 0, LayerScale gele. Le YAML k (LLRD 0.8, no_decay) ne change
pas le module.

DINOv3 ViT-B/16 n'a pas de pos embed absolu. Le RoPE (``rope_type='dinov3'``)
renormalise les coordonnees dans [-1, 1] par la taille de la grille
(``make_coords_dinov3``, normalize ``separate``). Passer de 224 (14x14)
a 320 (20x20) n'exige donc aucune reinterpolation : ``get_embed(shape=...)``
est deja le chemin utilise a 224. 320 est divisible par le patch 16.

Entree : (B, 6, 24, 320, 320) grayscale [0, 1], slots v20b.
Volumes : ``data/crop130_320`` (pas crop130_rgb25d : ces NPZ sont en
gris, le stem reconstruit le triplet).
"""
from __future__ import annotations

import torch

from src.models.kneenet_vit_e import PATCH_SIZE, KneeNet as KneeNetE

IMG_SIZE = 320


class KneeNet(KneeNetE):
    arch_id = "kneenet_vit_k320"

    def __init__(self, img_size: int = IMG_SIZE, **kwargs) -> None:
        super().__init__(**kwargs)
        self.img_size = int(img_size)
        if self.img_size % PATCH_SIZE:
            raise ValueError(
                f"img_size {self.img_size} non divisible par le patch {PATCH_SIZE}"
            )
        grid = self.img_size // PATCH_SIZE
        print(
            f"KneeNet-ViT-k320 | entree {self.img_size} | "
            f"CAM {grid}x{grid} (RoPE DINOv3 renormalise, pas de pos embed)",
            flush=True,
        )

    def _check_size(self, x: torch.Tensor) -> None:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if h != self.img_size or w != self.img_size:
            raise ValueError(
                f"vit_k320 attend {self.img_size}x{self.img_size}, recu {h}x{w} "
                "(zip_path=data/crop130_320, img_size=320, pas de rgb_neighbors)"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_size(x)
        return super().forward(x)

    def interpret(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._check_size(x)
        return super().interpret(x)
