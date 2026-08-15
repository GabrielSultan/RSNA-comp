# -*- coding: utf-8 -*-
"""Augmentations de volumes — version batch GPU.

Entree : (B, S, D, H, W) float dans [0, 1] (S = nb de series/plans).
Memes parametres pour les D slices d'un volume ; parametres differents
par volume du batch. Un seul affine_grid sur (B*S*D) images -> quasi gratuit
sur GPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def augment_batch(
    x: torch.Tensor,
    max_rotate_deg: float = 12.0,
    max_translate: float = 0.08,
    scale_range: tuple[float, float] = (0.92, 1.08),
    brightness: float = 0.15,
    contrast: float = 0.15,
    gamma_range: tuple[float, float] = (0.85, 1.15),
) -> torch.Tensor:
    """x : (B, S, D, H, W) sur device GPU. Retourne meme shape.

    Pas de flip horizontal : il echange medial/lateral (menisques, OA).
    """
    b, s, d, h, w = x.shape
    device = x.device
    n_vol = b * s

    xf = x.reshape(n_vol * d, 1, h, w)

    # affine : une matrice par volume, repetee sur les D slices
    max_rot = max_rotate_deg * np.pi / 180.0
    angle = (torch.rand(n_vol, device=device) * 2 - 1) * max_rot
    scale = torch.empty(n_vol, device=device).uniform_(*scale_range)
    tx = (torch.rand(n_vol, device=device) * 2 - 1) * max_translate
    ty = (torch.rand(n_vol, device=device) * 2 - 1) * max_translate

    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(n_vol, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty
    theta = theta.repeat_interleave(d, dim=0)  # (B*S*D, 2, 3)

    grid = F.affine_grid(theta, xf.shape, align_corners=False)
    xf = F.grid_sample(xf, grid, align_corners=False, padding_mode="zeros")

    # photometrique : un scalaire par volume
    bshift = (torch.rand(n_vol, 1, 1, 1, 1, device=device) * 2 - 1) * brightness
    cscale = torch.empty(n_vol, 1, 1, 1, 1, device=device).uniform_(
        1 - contrast, 1 + contrast)
    gamma = torch.empty(n_vol, 1, 1, 1, 1, device=device).uniform_(*gamma_range)

    xv = xf.reshape(n_vol, d, 1, h, w)
    xv = ((xv + bshift).clamp(0, 1) - 0.5) * cscale + 0.5
    xv = xv.clamp(1e-4, 1).pow(gamma)

    return xv.reshape(b, s, d, h, w)


# TTA inference : identity + petites rotations / zooms. Pas de hflip
# (echangerait medial/lateral : menisques, OA).
TTA_VARIANTS: tuple[tuple[float, float], ...] = (
    (0.0, 1.00),
    (5.0, 1.00),
    (-5.0, 1.00),
    (0.0, 0.95),
    (0.0, 1.05),
)


def apply_affine(
    x: torch.Tensor,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> torch.Tensor:
    """x : (B, S, D, H, W) dans [0, 1]. Meme affine pour tous les plans.

    Convention identique a `augment_batch` (cos/sin divises par scale).
    """
    if angle_deg == 0.0 and scale == 1.0 and tx == 0.0 and ty == 0.0:
        return x
    b, s, d, h, w = x.shape
    device = x.device
    xf = x.reshape(b * s * d, 1, h, w)
    angle = x.new_tensor(angle_deg * np.pi / 180.0)
    sc = x.new_tensor(scale)
    cos, sin = torch.cos(angle) / sc, torch.sin(angle) / sc
    theta = xf.new_zeros(b, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty
    theta = theta.repeat_interleave(s * d, dim=0)
    grid = F.affine_grid(theta, xf.shape, align_corners=False)
    xf = F.grid_sample(xf, grid, align_corners=False, padding_mode="zeros")
    return xf.reshape(b, s, d, h, w)
