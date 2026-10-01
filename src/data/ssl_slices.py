# -*- coding: utf-8 -*-
"""Series crop130 pour le pretrain SSL, sans labels.

Une entree = un volume (D, H, W) dans [0, 1]. Le tirage des coupes
et les deux vues SimCLR sont faits dans la boucle, sur GPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.dataset_npz import (
    _dataloader_worker_init,
    load_volume,
    resolve_volume_source,
)


def list_ssl_series(series_csv: str, zip_path: str) -> list[tuple[str, str]]:
    """(SeriesInstanceUID, zip) presentes dans train_series ET dans les volumes."""
    volume_index = resolve_volume_source(zip_path)
    sids = pd.read_csv(series_csv, usecols=["SeriesInstanceUID"])
    uniq = sorted(set(sids.SeriesInstanceUID.astype(str)))
    items: list[tuple[str, str]] = []
    skipped = 0
    for sid in uniq:
        if volume_index is None:
            items.append((sid, zip_path))
        elif sid in volume_index:
            items.append((sid, volume_index[sid]))
        else:
            skipped += 1
    print(
        f"SSL series : {len(items)} | absentes des volumes : {skipped}",
        flush=True,
    )
    if not items:
        raise FileNotFoundError(
            f"aucune serie SSL dans {series_csv} x {zip_path}"
        )
    return items


class SslSeriesDataset(Dataset):
    """Volume grayscale (D, H, W) float32 [0, 1]. Pas de label."""

    def __init__(
        self,
        items: list[tuple[str, str]],
        img_size: int = 224,
    ) -> None:
        self.items = items
        self.img_size = int(img_size)
        if self.img_size < 32:
            raise ValueError(f"img_size={self.img_size} : trop petit")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> torch.Tensor:
        sid, src = self.items[i]
        vol = load_volume(src, sid)
        if vol.ndim != 3:
            raise ValueError(f"{sid}: attendu (D, H, W), recu {vol.shape}")
        t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))
        t = t / 255.0
        if t.shape[-2] != self.img_size or t.shape[-1] != self.img_size:
            t = F.interpolate(
                t.unsqueeze(1),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return t


def make_ssl_loader(
    series_csv: str,
    zip_path: str,
    series_batch: int,
    num_workers: int,
    img_size: int = 224,
    seed: int = 42,
):
    from torch.utils.data import DataLoader

    items = list_ssl_series(series_csv, zip_path)
    ds = SslSeriesDataset(items, img_size=img_size)
    loader = DataLoader(
        ds,
        batch_size=int(series_batch),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        generator=torch.Generator().manual_seed(int(seed)),
        worker_init_fn=_dataloader_worker_init if num_workers > 0 else None,
    )
    return loader
