# -*- coding: utf-8 -*-
"""Dataset etude-level : 3 plans fluid, slot T1 optionnel, ou 6 slots proto.

Sortie grayscale (S, 24, H, W) ou RGB DICOM (S, 24, 3, H, W)
si `rgb_neighbors=True` (NPZ crop130_rgb25d). H=W=`img_size` (224 ou 320).
S=3 (v8/v17), S=4 (v9/v18 T1 extra), S=6 (v20b protocole FS/T1 par plan).
12 labels avec masque (-1 = masque). Lecture NPZ depuis un zip unique
(`data/volumes.zip`) ou un dossier de shards (`data/crop130/*.zip`).
Series dropout : un slot mis a zero avec proba `series_drop_p`.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PLANES = ["Sagittal", "Coronal", "Axial"]
# v20b : contraste fixe par slot, pas de repli fluide -> T1.
PROTOCOL_SLOTS: list[tuple[str, int]] = [
    ("Sagittal", 1),
    ("Coronal", 1),
    ("Axial", 1),
    ("Sagittal", 0),
    ("Coronal", 0),
    ("Axial", 0),
]
PROTOCOL_SLOT_NAMES = [
    "SAG_FS", "COR_FS", "AX_FS", "SAG_T1", "COR_T1", "AX_T1",
]
N_SLICES = 24
NORM_MEAN = 0.1648
NORM_STD = 0.1941

# ZipFile partage par processus worker (pas picklable -> lazy par pid + chemin)
_ZIPS: dict[tuple[int, str], zipfile.ZipFile] = {}
_VOLUME_SOURCE_CACHE: dict[str, dict[str, str] | None] = {}


def _zip_for(path: str) -> zipfile.ZipFile:
    key = (os.getpid(), path)
    if key not in _ZIPS:
        _ZIPS[key] = zipfile.ZipFile(path)
    return _ZIPS[key]


def load_volume(zip_file: str, series_uid: str) -> np.ndarray:
    z = _zip_for(zip_file)
    # .copy() : detache du buffer/mmap pour limiter l'empreinte RAM des workers
    return np.load(io.BytesIO(z.read(series_uid + ".npz")))["data"].copy()


def _index_from_manifests(dir_path: Path) -> dict[str, str]:
    """manifest_partN.csv + *partN.zip -> SeriesInstanceUID -> chemin zip."""
    mapping: dict[str, str] = {}
    for man in sorted(dir_path.glob("manifest_part*.csv")):
        part = man.stem.replace("manifest_", "")
        zips = sorted(dir_path.glob(f"*{part}.zip"))
        if not zips:
            continue
        zp = str(zips[0])
        df = pd.read_csv(man, usecols=["SeriesInstanceUID", "ok"])
        for sid, ok in zip(df.SeriesInstanceUID, df.ok):
            if int(ok) == 1:
                mapping[str(sid)] = zp
    return mapping


def _index_from_zips(dir_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for zp in sorted(dir_path.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.endswith(".npz"):
                    mapping[Path(name).stem] = str(zp)
    return mapping


def resolve_volume_source(zip_path: str) -> dict[str, str] | None:
    """None = un seul zip (lecture directe). dict = sid -> zip (dossier shards)."""
    p = Path(zip_path)
    cache_key = str(p.resolve()) if p.exists() else zip_path
    if cache_key in _VOLUME_SOURCE_CACHE:
        return _VOLUME_SOURCE_CACHE[cache_key]
    if p.is_file():
        _VOLUME_SOURCE_CACHE[cache_key] = None
        return None
    if not p.is_dir():
        raise FileNotFoundError(f"zip_path introuvable : {zip_path}")
    mapping = _index_from_manifests(p)
    for sid, zp in _index_from_zips(p).items():
        mapping.setdefault(sid, zp)
    if not mapping:
        raise FileNotFoundError(f"aucune serie NPZ dans {p}")
    n_z = len(set(mapping.values()))
    print(f"volumes {p} : {len(mapping)} series dans {n_z} zip(s)", flush=True)
    _VOLUME_SOURCE_CACHE[cache_key] = mapping
    return mapping


def make_series_index(
    series_csv: str | Path, zip_path: str,
) -> tuple["StudySeriesIndex", dict[str, str] | None]:
    volume_index = resolve_volume_source(zip_path)
    keep = None if volume_index is None else set(volume_index)
    return StudySeriesIndex(series_csv, keep_sids=keep), volume_index


class StudySeriesIndex:
    """Index etude -> candidats par plan x contraste (FS vs T1)."""

    def __init__(
        self,
        series_csv: str | Path,
        keep_sids: set[str] | None = None,
    ) -> None:
        df = pd.read_csv(series_csv)
        if keep_sids is not None:
            n0 = len(df)
            df = df[df.SeriesInstanceUID.isin(keep_sids)]
            print(f"series presentes dans les volumes : {len(df)} / {n0}",
                  flush=True)
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

    def pick_nonfluid(self, study_uid: str, exclude: set[str],
                      rng: np.random.Generator | None):
        """T1 / non-fluide pas deja utilisee. Sagittal d'abord, puis cor, ax."""
        plans = self.by_study.get(study_uid)
        if plans is None:
            return None
        for plane in PLANES:
            cands = [s for s in plans[plane][0] if s not in exclude]
            if not cands:
                continue
            if rng is None:
                return cands[0]
            return cands[int(rng.integers(len(cands)))]
        return None

    def pick_protocol(
        self, study_uid: str, plane: str, fluid: int,
        rng: np.random.Generator | None,
    ):
        """Un contraste exact (FS ou T1) sur un plan. Pas de repli."""
        plans = self.by_study.get(study_uid)
        if plans is None:
            return None
        cands = plans[plane][int(fluid)]
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
        n_series: int = 3,
        volume_index: dict[str, str] | None = None,
        rgb_neighbors: bool = False,
        img_size: int = 224,
    ) -> None:
        self.items = items
        self.labels = (
            labels_df.set_index("StudyInstanceUID") if labels_df is not None else None
        )
        self.index = index
        self.zip_path = zip_path
        self.volume_index = volume_index
        self.train = train
        self.series_drop_p = series_drop_p
        self.n_series = int(n_series)
        self.rgb_neighbors = bool(rgb_neighbors)
        self.img_size = int(img_size)
        if self.n_series not in (3, 4, 6):
            raise ValueError(f"n_series={self.n_series} : attendu 3, 4 ou 6")
        if self.img_size < 32:
            raise ValueError(f"img_size={self.img_size} : trop petit")

    def _empty(self) -> torch.Tensor:
        s = self.img_size
        if self.rgb_neighbors:
            return torch.zeros(N_SLICES, 3, s, s)
        return torch.zeros(N_SLICES, s, s)

    def __len__(self) -> int:
        return len(self.items)

    def _load_sid(self, sid: str | None) -> torch.Tensor:
        if sid is None:
            return self._empty()
        src = self.zip_path if self.volume_index is None else self.volume_index.get(sid)
        if src is None:
            return self._empty()
        try:
            vol = load_volume(src, sid)
        except Exception:
            return self._empty()
        if self.rgb_neighbors:
            if vol.ndim != 4 or vol.shape[1] != 3:
                raise ValueError(
                    f"{sid}: rgb_neighbors attend (D, 3, H, W), recu {vol.shape}"
                )
        elif vol.ndim != 3:
            raise ValueError(
                f"{sid}: grayscale attend (D, H, W), recu {vol.shape} "
                "(activer rgb_neighbors pour les NPZ crop130_rgb25d)"
            )
        return torch.from_numpy(vol.astype(np.float32) / 255.0)

    def _maybe_drop(self) -> bool:
        return self.train and torch.rand(()).item() < self.series_drop_p

    def __getitem__(self, i: int):
        uid = self.items[i]
        rng = np.random.default_rng(
            (hash(uid) & 0xFFFFFFFF) ^ (np.random.randint(0, 2**31) if self.train else 0)
        )
        used: set[str] = set()
        vols = []
        pick_rng = rng if self.train else None
        if self.n_series == 6:
            for plane, fluid in PROTOCOL_SLOTS:
                if self._maybe_drop():
                    vols.append(self._empty())
                    continue
                sid = self.index.pick_protocol(uid, plane, fluid, pick_rng)
                vols.append(self._load_sid(sid))
        else:
            for plane in PLANES:
                if self._maybe_drop():
                    vols.append(self._empty())
                    continue
                sid = self.index.pick(uid, plane, pick_rng)
                if sid is not None:
                    used.add(sid)
                vols.append(self._load_sid(sid))
            if self.n_series >= 4:
                if self._maybe_drop():
                    vols.append(self._empty())
                else:
                    vols.append(self._load_sid(
                        self.index.pick_nonfluid(uid, used, pick_rng)))
        x = torch.stack(vols)  # (3|4|6, 24, [3,] H, W)

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


def _subsample_train(train_df: pd.DataFrame, targets: list[str],
                     frac: float, seed: int) -> pd.DataFrame:
    """Sous-echantillon stratifie du train seulement (val inchangee)."""
    if frac >= 1.0:
        return train_df
    if frac <= 0.0:
        raise ValueError(f"train_frac={frac} : doit etre dans (0, 1]")
    rng = np.random.default_rng(seed)
    n_pos = (train_df[targets] >= 0.5).sum(axis=1)
    pos_bin = pd.cut(n_pos, bins=[-1, 0, 2, 4, 20], labels=["0", "1-2", "3-4", "5+"])
    keep = []
    for _, group in train_df.groupby(pos_bin, observed=False):
        if len(group) == 0:
            continue
        n = max(1, int(round(len(group) * frac)))
        n = min(n, len(group))
        chosen = rng.choice(group.index.to_numpy(), size=n, replace=False)
        keep.extend(chosen.tolist())
    out = train_df.loc[keep]
    print(f"train_frac={frac:g} seed={seed} : {len(out)} / {len(train_df)} etudes",
          flush=True)
    return out


def make_loaders(labels_csv: str = "data/labels/train_val_labels.csv",
                 series_csv: str = "data/csv/train_series.csv",
                 zip_path: str = "data/volumes.zip",
                 batch_size: int = 8, num_workers: int = 4,
                 n_series: int = 3, train_frac: float = 1.0,
                 train_frac_seed: int = 42, rgb_neighbors: bool = False,
                 img_size: int = 224):
    from torch.utils.data import DataLoader

    df = pd.read_csv(labels_csv)
    targets = [c for c in df.columns if c not in ("StudyInstanceUID", "lang", "split")]
    df = df.dropna(subset=targets, how="all")  # etudes sans aucun label : exclues
    df = _exclude_gold(df, labels_csv)

    index, volume_index = make_series_index(series_csv, zip_path)
    # ne garder que les etudes ayant au moins une serie indexee
    df = df[df.StudyInstanceUID.isin(index.by_study.keys())]
    df = _ensure_split(df, targets)

    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]
    train_df = _subsample_train(train_df, targets, train_frac, train_frac_seed)

    if rgb_neighbors:
        print("rgb_neighbors : volumes (D, 3, H, W) DICOM k-1/k/k+1",
              flush=True)
    print(f"img_size={img_size}", flush=True)

    if n_series == 6:
        n_train = len(train_df)
        for name, (plane, fluid) in zip(PROTOCOL_SLOT_NAMES, PROTOCOL_SLOTS):
            n_ok = sum(
                1 for uid in train_df.StudyInstanceUID
                if index.pick_protocol(uid, plane, fluid, None) is not None
            )
            print(f"slot {name}: {n_ok}/{n_train} etudes train", flush=True)
    elif n_series >= 4:
        n_t1 = sum(
            1 for uid in train_df.StudyInstanceUID
            if index.pick_nonfluid(uid, set(), None) is not None
        )
        print(f"slot T1 (non-fluide) : {n_t1}/{len(train_df)} etudes train",
              flush=True)

    train_ds = StudyDataset(
        train_df.StudyInstanceUID.tolist(), train_df[["StudyInstanceUID"] + targets],
        index, zip_path, train=True, series_drop_p=0.15, n_series=n_series,
        volume_index=volume_index, rgb_neighbors=rgb_neighbors,
        img_size=img_size,
    )
    val_ds = StudyDataset(
        val_df.StudyInstanceUID.tolist(), val_df[["StudyInstanceUID"] + targets],
        index, zip_path, train=False, series_drop_p=0.0, n_series=n_series,
        volume_index=volume_index, rgb_neighbors=rgb_neighbors,
        img_size=img_size,
    )

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)
    return train_ld, val_ld, targets
