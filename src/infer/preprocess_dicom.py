# -*- coding: utf-8 -*-
"""Pretraitement DICOM -> volume uint8, aligne sur la conversion NPZ.

Recette identique a `_archive/kaggle_DICOM_exctract/_gen_convert_nb.py` (grayscale)
et `_gen_convert_rgb25d_nb.py` (voisins DICOM) :
1. Lecture des .dcm, tri InstanceNumber (fallback : normale IOP).
2. RescaleSlope/Intercept, inversion MONOCHROME1.
3. Crop centre 130 mm (PixelSpacing) sinon centre 70 %.
4. Percentiles 1-99 -> [0, 1].
5. Bande profondeur 20-80 % (evite air / bords), puis linspace 24 coupes.
6. Resize IMG_SIZE (224 par defaut, 320 pour crop130_320), uint8.
7. Si rgb_neighbors : pour chaque centre k du linspace, stocker k-1, k, k+1
   dans le volume ORIGINAL (1 coupe DICOM d'ecart), pas dans les 24.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.dataset_npz import PROTOCOL_SLOTS
from src.data.laterality import laterality_from_study_dir, normalise_laterality

N_SLICES = 24
IMG_SIZE = 224
CROP_MM = 130.0
CROP_REL = 0.70
BAND = (0.20, 0.80)


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


def _spacing_mm(ds):
    ps = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    if ps is None:
        return None
    try:
        return float(ps[0]), float(ps[1])
    except (TypeError, ValueError, IndexError):
        return None


def _center_crop(img: np.ndarray, row_mm, col_mm) -> np.ndarray:
    h, w = img.shape[:2]
    if row_mm is not None and col_mm is not None and row_mm > 0 and col_mm > 0:
        ch = int(round(CROP_MM / row_mm))
        cw = int(round(CROP_MM / col_mm))
        ch = min(h, max(8, ch))
        cw = min(w, max(8, cw))
    else:
        ch = max(8, int(round(h * CROP_REL)))
        cw = max(8, int(round(w * CROP_REL)))
        ch, cw = min(h, ch), min(w, cw)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def _rescale(ds, arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def _to_u8(sl: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    sl = cv2.resize(sl, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return np.clip(sl * 255.0, 0, 255).round().astype(np.uint8)


def load_series_dicom(
    series_dir: str | Path,
    rgb_neighbors: bool = False,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """Un dossier de .dcm -> (24, S, S) ou (24, 3, S, S) uint8, S=`img_size`.

    rgb_neighbors=False : grayscale crop130 (linspace 24).
    rgb_neighbors=True  : R=k-1, G=k, B=k+1 dans le volume original.
    """
    import pydicom

    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"pas de .dcm dans {series_dir}")

    datasets = []
    for f in files:
        try:
            datasets.append(pydicom.dcmread(str(f)))
        except Exception:
            continue
    if not datasets:
        raise RuntimeError(f"aucun DICOM lisible dans {series_dir}")
    datasets.sort(key=_slice_sort_key)

    frames = []
    sp = None
    for ds in datasets:
        arr = ds.pixel_array
        if sp is None:
            sp = _spacing_mm(ds)
        arr = _rescale(ds, arr)
        if arr.ndim == 2:
            frames.append(arr)
        elif arr.ndim == 3:
            frames.extend([arr[i] for i in range(arr.shape[0])])
        else:
            raise RuntimeError(f"ndim inattendu {arr.ndim}")
    if not frames:
        raise RuntimeError("aucune coupe")

    row_mm, col_mm = (sp if sp is not None else (None, None))
    cropped = [_center_crop(im, row_mm, col_mm) for im in frames]
    h0, w0 = cropped[0].shape[:2]
    stacked = []
    for c in cropped:
        if c.shape[:2] != (h0, w0):
            c = cv2.resize(c, (w0, h0), interpolation=cv2.INTER_AREA)
        stacked.append(c)
    vol = np.stack(stacked, axis=0)

    lo, hi = np.percentile(vol, 1.0), np.percentile(vol, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)

    n = vol.shape[0]
    a = int(np.floor(BAND[0] * (n - 1)))
    b = int(np.ceil(BAND[1] * (n - 1)))
    if b - a + 1 < 8:
        a, b = 0, n - 1
    idx_band = np.linspace(0, b - a, N_SLICES).round().astype(int)
    idx_orig = np.clip(a + idx_band, 0, n - 1)

    size = int(img_size)
    if not rgb_neighbors:
        out = np.empty((N_SLICES, size, size), dtype=np.uint8)
        for i, k in enumerate(idx_orig):
            out[i] = _to_u8(vol[int(k)], size)
        return out

    out = np.empty((N_SLICES, 3, size, size), dtype=np.uint8)
    last = n - 1
    for i, k in enumerate(idx_orig):
        k = int(k)
        km = 0 if k <= 0 else k - 1
        kp = last if k >= last else k + 1
        out[i, 0] = _to_u8(vol[km], size)
        out[i, 1] = _to_u8(vol[k], size)
        out[i, 2] = _to_u8(vol[kp], size)
    return out


def _pick_plane_row(series_rows, plane: str):
    """Fluid-sensitive d'abord, sinon n'importe quelle serie du plan."""
    rows = series_rows[series_rows.Anatomical_Plane == plane]
    if len(rows) == 0:
        return None
    fluid_col = "Fluid_Sensitive" if "Fluid_Sensitive" in rows.columns else None
    fluid = rows[rows[fluid_col] == 1] if fluid_col else rows.iloc[0:0]
    if len(fluid) == 0 and "Fat_Suppression" in rows.columns:
        fluid = rows[rows.Fat_Suppression == 1]
    pick = (fluid if len(fluid) else rows).sort_values("SeriesInstanceUID").iloc[0]
    return pick


def _pick_nonfluid_row(series_rows, exclude: set[str]):
    """T1 / non-fluide pas deja utilisee. Sagittal d'abord."""
    for plane in ("Sagittal", "Coronal", "Axial"):
        rows = series_rows[series_rows.Anatomical_Plane == plane]
        if "Fluid_Sensitive" in rows.columns:
            rows = rows[rows.Fluid_Sensitive == 0]
        rows = rows[~rows.SeriesInstanceUID.isin(exclude)]
        if len(rows) == 0:
            continue
        return rows.sort_values("SeriesInstanceUID").iloc[0]
    return None


def _pick_protocol_row(series_rows, plane: str, fluid: int):
    """Un contraste exact (FS ou T1) sur un plan. Pas de repli."""
    rows = series_rows[series_rows.Anatomical_Plane == plane]
    if len(rows) == 0:
        return None
    if "Fluid_Sensitive" in rows.columns:
        rows = rows[rows.Fluid_Sensitive == int(fluid)]
    elif fluid == 1 and "Fat_Suppression" in rows.columns:
        rows = rows[rows.Fat_Suppression == 1]
    elif fluid == 0 and "Fat_Suppression" in rows.columns:
        rows = rows[rows.Fat_Suppression == 0]
    if len(rows) == 0:
        return None
    return rows.sort_values("SeriesInstanceUID").iloc[0]


def study_to_tensor(study_dir: str | Path, series_rows,
                    n_series: int = 3,
                    rgb_neighbors: bool = False,
                    img_size: int = IMG_SIZE,
                    laterality: bool = False) -> np.ndarray:
    """Une etude -> float32 dans [0, 1].

    Grayscale : (n_series, 24, S, S) avec S=`img_size`.
    rgb_neighbors : (n_series, 24, 3, S, S) R=k-1 G=k B=k+1 DICOM.

    series_rows : DataFrame des series de l'etude (colonnes SeriesInstanceUID,
    Anatomical_Plane, Fluid_Sensitive, Fat_Suppression).
    n_series=3 : sag / cor / ax, fluid-sensitive d'abord.
    n_series=4 : + T1 extra (sagittal prioritaire).
    n_series=6 : 3 FS + 3 T1 par plan, sans repli (v20b).
    laterality=True : canon gauche (flip R), meme recette que le train.
    """
    planes = ("Sagittal", "Coronal", "Axial")
    study_dir = Path(study_dir)
    size = int(img_size)
    if rgb_neighbors:
        out = np.zeros(
            (n_series, N_SLICES, 3, size, size), dtype=np.float32,
        )
    else:
        out = np.zeros((n_series, N_SLICES, size, size), dtype=np.float32)
    used: set[str] = set()
    lat = laterality_from_study_dir(study_dir) if laterality else None

    def _fill(i: int, pick, plane: str | None) -> None:
        if pick is None:
            return
        series_dir = study_dir / pick.SeriesInstanceUID
        if not series_dir.exists():
            return
        try:
            vol = load_series_dicom(
                series_dir, rgb_neighbors=rgb_neighbors, img_size=size)
        except Exception as e:
            print(f"  [warn] serie {pick.SeriesInstanceUID}: {e}")
            return
        used.add(str(pick.SeriesInstanceUID))
        vol = vol.astype(np.float32) / 255.0
        if laterality:
            pl = plane or getattr(pick, "Anatomical_Plane", None)
            vol = normalise_laterality(vol, pl, lat)
        out[i] = vol

    if n_series == 6:
        for i, (plane, fluid) in enumerate(PROTOCOL_SLOTS):
            _fill(i, _pick_protocol_row(series_rows, plane, fluid), plane)
        return out

    for i, plane in enumerate(planes):
        _fill(i, _pick_plane_row(series_rows, plane), plane)
    if n_series >= 4:
        pick = _pick_nonfluid_row(series_rows, used)
        pl = None if pick is None else str(pick.Anatomical_Plane)
        _fill(3, pick, pl)
    return out
