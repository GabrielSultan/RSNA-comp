# -*- coding: utf-8 -*-
"""Dataset etude-level : 3 plans (sag/cor/ax, fluid prioritaire) -> tensor
(3, 24, 224, 224), 12 labels avec masque (-1 = masque).

Lecture NPZ en streaming depuis volumes.zip (un ZipFile par worker).
Series dropout : avec proba `series_drop_p`, un plan est mis a zero
(robustesse aux series manquantes au test + augmentation).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PLANES = ["Sagittal", "Coronal", "Axial"]
N_SLICES = 24
NORM_MEAN = 0.1648
NORM_STD = 0.1941

# ZipFile partage par processus worker (pas picklable -> lazy par pid)
_ZIPS: dict[int, zipfile.ZipFile] = {}


def _zip_for(path: str) -> zipfile.ZipFile:
    import os

    pid = os.getpid()
    if pid not in _ZIPS:
        _ZIPS[pid] = zipfile.ZipFile(path)
    return _ZIPS[pid]


def load_volume(zip_path: str, series_uid: str) -> np.ndarray:
    z = _zip_for(zip_path)
    # .copy() : detache du buffer/mmap pour limiter l'empreinte RAM des workers
    return np.load(io.BytesIO(z.read(series_uid + ".npz")))["data"].copy()


class StudySeriesIndex:
    """Index etude -> candidats par plan (fluid d'abord)."""

    def __init__(self, series_csv: str | Path) -> None:
        df = pd.read_csv(series_csv)
        df = df.sort_values("SeriesInstanceUID")
        self.by_study: dict[str, dict[str, dict[int, list[str]]]] = {}
        for uid, g in df.groupby("StudyInstanceUID"):
            plans: dict[str, dict[int, list[str]]] = {}
            for plane in PLANES:
                gp = g[g.Anatomical_Plane == plane]
                plans[plane] = {
                    1: gp[gp.Fluid_Sensitive == 1].SeriesInstanceUID.tolist(),
                    0: gp[gp.Fluid_Sensitive == 0].SeriesInstanceUID.tolist(),
                }
            self.by_study[uid] = plans

    def pick(self, study_uid: str, plane: str, rng: np.random.Generator | None):
        """Serie fluid si possible, sinon non-fluid du meme plan, sinon None."""
        plans = self.by_study.get(study_uid)
        if plans is None:
            return None
        cands = plans[plane][1] or plans[plane][0]
        if not cands:
            return None
        if rng is None:
            return cands[0]
        return cands[int(rng.integers(len(cands)))]


class StudyDataset(Dataset):
    """items : liste de StudyInstanceUID. labels_df indexe par StudyInstanceUID,
    colonnes = 12 cibles (NaN = masque)."""

    """Les volumes sont renvoyes dans [0, 1] SANS normalisation ni augmentation :
    l'aug GPU est dans la boucle d'entrainement, ImageNet dans KneeNet.
    Les NPZ crop130 sont deja restreints a la bande profondeur 20-80 %."""

    def __init__(
        self,
        items: list[str],
        labels_df: pd.DataFrame | None,
        index: StudySeriesIndex,
        zip_path: str = "data/volumes.zip",
        train: bool = True,
        series_drop_p: float = 0.15,
    ) -> None:
        self.items = items
        self.labels = (
            labels_df.set_index("StudyInstanceUID") if labels_df is not None else None
        )
        self.index = index
        self.zip_path = zip_path
        self.train = train
        self.series_drop_p = series_drop_p

    def __len__(self) -> int:
        return len(self.items)

    def _load_plane(self, study_uid: str, plane: str,
                    rng: np.random.Generator | None) -> torch.Tensor:
        sid = self.index.pick(study_uid, plane, rng if self.train else None)
        if sid is None:
            return torch.zeros(N_SLICES, 224, 224)
        vol = load_volume(self.zip_path, sid).astype(np.float32) / 255.0
        return torch.from_numpy(vol)

    def __getitem__(self, i: int):
        uid = self.items[i]
        rng = np.random.default_rng(
            (hash(uid) & 0xFFFFFFFF) ^ (np.random.randint(0, 2**31) if self.train else 0)
        )
        vols = []
        for plane in PLANES:
            if self.train and torch.rand(()).item() < self.series_drop_p:
                vols.append(torch.zeros(N_SLICES, 224, 224))
            else:
                vols.append(self._load_plane(uid, plane, rng))
        x = torch.stack(vols)  # (3, 24, 224, 224)

        if self.labels is None:
            return x, uid
        y = self.labels.loc[uid].values.astype(np.float32)
        y = np.nan_to_num(y, nan=-1.0)
        return x, torch.from_numpy(y), uid


def _exclude_gold(df: pd.DataFrame, labels_csv: str) -> pd.DataFrame:
    """Les 58 gold restent hors train/val (reserve sanity-check)."""
    train_csv = Path(labels_csv).resolve().parent.parent / "csv" / "train.csv"
    if not train_csv.exists():
        return df
    from src.nlp.lexicon import TARGETS
    train_df = pd.read_csv(train_csv, usecols=["StudyInstanceUID"] + TARGETS)
    gold_uids = set(train_df[train_df[TARGETS].notna().all(axis=1)]["StudyInstanceUID"])
    n = int(df["StudyInstanceUID"].isin(gold_uids).sum())
    if n:
        print(f"gold exclus du train/val : {n} etudes", flush=True)
        df = df[~df["StudyInstanceUID"].isin(gold_uids)]
    return df


def _ensure_split(df: pd.DataFrame, targets: list[str],
                  val_frac: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Split 85/15 si le CSV n'a pas de colonne split (ex. blend candidat)."""
    if "split" in df.columns:
        return df
    rng = np.random.default_rng(seed)
    n_pos = (df[targets] >= 0.5).sum(axis=1)
    pos_bin = pd.cut(n_pos, bins=[-1, 0, 2, 4, 20], labels=["0", "1-2", "3-4", "5+"])
    split = pd.Series("train", index=df.index)
    for _, group in df.groupby(pos_bin, observed=False):
        n_val = max(1, int(round(len(group) * val_frac))) if len(group) >= 4 else 0
        if n_val > 0:
            chosen = rng.choice(group.index.to_numpy(),
                                size=min(n_val, len(group)), replace=False)
            split.loc[chosen] = "val"
    out = df.copy()
    out["split"] = split.values
    print(f"split auto 85/15 : train {(out.split == 'train').sum()} | "
          f"val {(out.split == 'val').sum()}", flush=True)
    return out


def make_loaders(labels_csv: str = "data/labels/train_val_labels.csv",
                 series_csv: str = "data/csv/train_series.csv",
                 zip_path: str = "data/volumes.zip",
                 batch_size: int = 8, num_workers: int = 4):
    from torch.utils.data import DataLoader

    df = pd.read_csv(labels_csv)
    targets = [c for c in df.columns if c not in ("StudyInstanceUID", "lang", "split")]
    df = df.dropna(subset=targets, how="all")  # etudes sans aucun label : exclues
    df = _exclude_gold(df, labels_csv)

    index = StudySeriesIndex(series_csv)
    # ne garder que les etudes ayant au moins une serie indexee
    df = df[df.StudyInstanceUID.isin(index.by_study.keys())]
    df = _ensure_split(df, targets)

    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]

    train_ds = StudyDataset(
        train_df.StudyInstanceUID.tolist(), train_df[["StudyInstanceUID"] + targets],
        index, zip_path, train=True, series_drop_p=0.15,
    )
    val_ds = StudyDataset(
        val_df.StudyInstanceUID.tolist(), val_df[["StudyInstanceUID"] + targets],
        index, zip_path, train=False, series_drop_p=0.0,
    )

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)
    return train_ld, val_ld, targets
