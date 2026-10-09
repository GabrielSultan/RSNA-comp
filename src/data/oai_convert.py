# -*- coding: utf-8 -*-
"""Serie OAI (.tar.gz ou dossier de DICOM sans extension) -> volumes crop130.

Recette de kaggle_DICOM_exctract/_gen_convert_nb.py::convert_series, verifiee
bit a bit contre src/infer/preprocess_dicom.load_series_dicom
(runpod/oai/check_sample.py), avec trois ecarts propres a l'OAI :
- tous les fichiers sont lus (noms 001, 002... sans .dcm), force=True ;
- les images sans PixelData et les images de projection des MPR
  (ImageType PROJECTION IMAGE, ./001 des COR/AX_MPR) sont ecartees ;
- une seule lecture produit 224 (crop130) et 320 (crop130_320).
Module autonome (numpy, cv2, pydicom) : pas de torch dans les workers.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pydicom

N_SLICES = 24
CROP_MM = 130.0
CROP_REL = 0.70
BAND = (0.20, 0.80)
SIZES = (224, 320)


def _sort_key(ds) -> float:
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


def _center_crop(img: np.ndarray, row_mm, col_mm) -> tuple[np.ndarray, str]:
    h, w = img.shape[:2]
    if row_mm is not None and col_mm is not None and row_mm > 0 and col_mm > 0:
        ch = int(round(CROP_MM / row_mm))
        cw = int(round(CROP_MM / col_mm))
        ch = min(h, max(8, ch))
        cw = min(w, max(8, cw))
        mode = "mm"
    else:
        ch = max(8, int(round(h * CROP_REL)))
        cw = max(8, int(round(w * CROP_REL)))
        ch, cw = min(h, ch), min(w, cw)
        mode = "rel"
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw], mode


def _rescale(ds, arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def is_slice(ds) -> bool:
    if "PixelData" not in ds:
        return False
    return "PROJECTION IMAGE" not in str(ds.get("ImageType", "")).upper()


def _read(raw: bytes):
    try:
        return pydicom.dcmread(io.BytesIO(raw), force=True)
    except Exception:
        return None


def iter_raw(src: str | Path):
    """(nom, octets) de chaque fichier d'une archive tar(.gz) ou d'un dossier."""
    src = Path(src)
    if src.is_file():
        with tarfile.open(src) as tf:
            members = sorted((m for m in tf.getmembers() if m.isfile()),
                             key=lambda m: m.name)
            for m in members:
                yield m.name, tf.extractfile(m).read()
    else:
        for p in sorted(x for x in src.rglob("*") if x.is_file()):
            yield p.name, p.read_bytes()


def read_series(src: str | Path) -> tuple[list, int]:
    """-> (coupes gardees, nombre de fichiers lus)."""
    kept, n_files = [], 0
    for _name, raw in iter_raw(src):
        n_files += 1
        ds = _read(raw)
        if ds is not None and is_slice(ds):
            kept.append(ds)
    return kept, n_files


def convert_datasets(datasets: list, sizes=SIZES) -> tuple[dict[int, np.ndarray], dict]:
    if not datasets:
        raise RuntimeError("aucune coupe DICOM")
    datasets = sorted(datasets, key=_sort_key)

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

    row_mm, col_mm = (sp if sp is not None else (None, None))
    cropped, crop_mode = [], None
    for im in frames:
        c, crop_mode = _center_crop(im, row_mm, col_mm)
        cropped.append(c)
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
    band = vol[a:b + 1]
    idx = np.linspace(0, band.shape[0] - 1, N_SLICES).round().astype(int)
    band = band[idx]

    outs: dict[int, np.ndarray] = {}
    for size in sizes:
        out = np.empty((N_SLICES, size, size), dtype=np.uint8)
        for i in range(N_SLICES):
            sl = cv2.resize(band[i], (size, size), interpolation=cv2.INTER_AREA)
            out[i] = np.clip(sl * 255.0, 0, 255).round().astype(np.uint8)
        outs[int(size)] = out

    meta = {
        "n_raw": int(n),
        "band": [a, b],
        "crop_mode": crop_mode,
        "spacing_mm": None if sp is None else [float(sp[0]), float(sp[1])],
        "src_hw": [int(h0), int(w0)],
        "n_files": len(datasets),
    }
    return outs, meta


def convert_series(src: str | Path, sizes=SIZES) -> tuple[dict[int, np.ndarray], dict]:
    datasets, n_files = read_series(src)
    outs, meta = convert_datasets(datasets, sizes)
    meta["n_skipped"] = n_files - len(datasets)
    return outs, meta


def npz_bytes(vol: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, data=vol)
    return buf.getvalue()
