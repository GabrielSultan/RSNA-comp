# -*- coding: utf-8 -*-
"""Pretraitement DICOM -> volume (24, 224, 224) uint8, reproduisant les NPZ.

Pipeline par serie :
1. Lecture des .dcm (pydicom), tri par InstanceNumber (fallback : position
   le long de la normale ImageOrientationPatient).
2. pixel_array float32, RescaleSlope/Intercept si presents, inversion si
   MONOCHROME1.
3. Resize de chaque coupe a 224x224 (INTER_AREA).
4. Normalisation robuste : clip percentiles 1-99 -> [0, 255] uint8.
5. Resample a 24 coupes par indices linspace.

NOTE : la recette exacte des NPZ communautaires est inconnue ; ce pipeline
est la meilleure approximation. A valider au premier run Kaggle en comparant
les distributions (moments) aux NPZ train (voir notebook 06).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

N_SLICES = 24
IMG_SIZE = 224


def _slice_sort_key(ds) -> float:
    """Cle de tri : InstanceNumber, sinon projection sur la normale."""
    inst = getattr(ds, "InstanceNumber", None)
    if inst is not None:
        try:
            return float(inst)
        except (TypeError, ValueError):
            pass
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if ipp is not None and iop is not None:
        row = np.array(iop[:3], dtype=np.float64)
        col = np.array(iop[3:], dtype=np.float64)
        normal = np.cross(row, col)
        return float(np.dot(np.array(ipp, dtype=np.float64), normal))
    if ipp is not None:
        return float(ipp[2])
    return 0.0


def load_series_dicom(series_dir: str | Path) -> np.ndarray:
    """Un dossier de .dcm -> volume (n_slices, 224, 224) uint8."""
    import pydicom

    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"pas de .dcm dans {series_dir}")

    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f))
            datasets.append(ds)
        except Exception:
            continue
    if not datasets:
        raise RuntimeError(f"aucun DICOM lisible dans {series_dir}")

    datasets.sort(key=_slice_sort_key)

    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr
        arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        slices.append(arr)

    vol = np.stack(slices)  # (n, 224, 224) float32

    # normalisation robuste par percentiles -> uint8
    lo, hi = np.percentile(vol, 1.0), np.percentile(vol, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)

    # resample a 24 coupes
    n = vol.shape[0]
    idx = np.linspace(0, n - 1, N_SLICES).round().astype(int)
    vol = vol[idx]

    return (vol * 255.0).round().astype(np.uint8)


def study_to_tensor(study_dir: str | Path, series_rows,
                    n_series_planes=("Sagittal", "Coronal", "Axial")) -> np.ndarray:
    """Une etude -> (3, 24, 224, 224) float32 dans [0, 1].

    series_rows : DataFrame des series de l'etude (colonnes SeriesInstanceUID,
    Anatomical_Plane, Fluid_Sensitive, Fat_Suppression).
    Selection : fluid-sensitive d'abord (Fluid_Sensitive ; si absent au test,
    repli sur Fat_Suppression — les deux colonnes restent distinctes).
    """
    study_dir = Path(study_dir)
    out = np.zeros((len(n_series_planes), N_SLICES, IMG_SIZE, IMG_SIZE),
                   dtype=np.float32)
    for i, plane in enumerate(n_series_planes):
        rows = series_rows[series_rows.Anatomical_Plane == plane]
        if len(rows) == 0:
            continue
        fluid_col = "Fluid_Sensitive" if "Fluid_Sensitive" in rows.columns else None
        fluid = rows[rows[fluid_col] == 1] if fluid_col else rows.iloc[0:0]
        if len(fluid) == 0 and "Fat_Suppression" in rows.columns:
            fluid = rows[rows.Fat_Suppression == 1]
        pick = (fluid if len(fluid) else rows).sort_values("SeriesInstanceUID").iloc[0]
        series_dir = study_dir / pick.SeriesInstanceUID
        if not series_dir.exists():
            continue
        try:
            vol = load_series_dicom(series_dir)
        except Exception as e:
            print(f"  [warn] serie {pick.SeriesInstanceUID}: {e}")
            continue
        out[i] = vol.astype(np.float32) / 255.0
    return out
