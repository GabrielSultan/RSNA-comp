# -*- coding: utf-8 -*-
"""Canonicalisation L/R des genoux (DINOsaur / Head-and-shoulders).

Ce n'est PAS un hflip d'augmentation : les labels medial/lateral ne bougent
pas. On detecte le cote de l'etude (tag DICOM, sinon geometrie patient) et
on transforme les genoux droits pour qu'ils ressemblent a des gauches.

Regle native (defaut, celle des notebooks hors bras E10 legacy) :
  1. tag Laterality si L/R
  2. sinon X du *centre* de l'image en coordonnees patient (LPS :
     X>0 = gauche patient). Zone morte 20 mm -> unresolved, pas de flip.

Flip si laterality == 'R' :
  Coronal / Axial : axe W (gauche-droite dans l'image)
  Sagittal        : axe D (ordre des coupes, medial <-> lateral)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LAT_MIN_OFFSET_MM = 20.0
CANON = "L"
HDR_TAGS = (
    "Laterality",
    "PixelSpacing",
    "Rows",
    "Columns",
    "ImagePositionPatient",
    "ImageOrientationPatient",
)


def _vec(v, n: int) -> np.ndarray | None:
    if v is None:
        return None
    if isinstance(v, str):
        try:
            arr = np.array([float(x) for x in v.split("|")], dtype=np.float64)
        except ValueError:
            return None
        return arr if arr.size >= n else None
    try:
        arr = np.asarray(list(v), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.size < n or not np.isfinite(arr[:n]).all():
        return None
    return arr


def side_from_tag(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().upper()
    if s and s[0] in ("L", "R"):
        return s[0]
    return None


def image_center_x_mm(ipp, iop, ps, rows, cols) -> float | None:
    """X patient du centre de la coupe (mm). None si headers incomplets."""
    ipp_v = _vec(ipp, 3)
    iop_v = _vec(iop, 6)
    ps_v = _vec(ps, 2)
    try:
        rows_f, cols_f = float(rows), float(cols)
    except (TypeError, ValueError):
        return None
    if ipp_v is None or iop_v is None or ps_v is None:
        return None
    if rows_f <= 0 or cols_f <= 0:
        return None
    # centre = IPP + row_dir * col_spacing * cols/2 + col_dir * row_spacing * rows/2
    c = (ipp_v[:3]
         + iop_v[:3] * ps_v[1] * cols_f / 2.0
         + iop_v[3:6] * ps_v[0] * rows_f / 2.0)
    x = float(c[0])
    return x if np.isfinite(x) else None


def side_from_x(x: float | None, min_offset: float = LAT_MIN_OFFSET_MM) -> str | None:
    if x is None or not np.isfinite(x):
        return None
    if abs(float(x)) < float(min_offset):
        return None
    return "R" if float(x) < 0 else "L"


def side_from_geometry_rows(rows) -> dict[str, str | None]:
    """UID -> L/R/None a partir du median des X centre (regle native)."""
    xs: dict[str, list[float]] = {}
    for r in rows:
        uid = str(r.get("StudyInstanceUID", "") or "")
        if not uid:
            continue
        x = image_center_x_mm(
            r.get("ImagePositionPatient"), r.get("ImageOrientationPatient"),
            r.get("PixelSpacing"), r.get("Rows"), r.get("Columns"),
        )
        if x is not None:
            xs.setdefault(uid, []).append(x)
    out: dict[str, str | None] = {}
    for uid, vals in xs.items():
        out[uid] = side_from_x(float(np.median(vals)))
    return out


def lat_of_rows(rows: list[dict], *, tag: str = "", log: bool = True) -> dict[str, str | None]:
    """Cote par etude : tag Laterality, sinon geometrie native.

    `rows` : dicts avec StudyInstanceUID + HDR_TAGS (une ligne / serie).
    """
    geo = side_from_geometry_rows(rows)
    by_uid: dict[str, list[dict]] = {}
    for r in rows:
        uid = str(r.get("StudyInstanceUID", "") or "")
        if uid:
            by_uid.setdefault(uid, []).append(r)
    out: dict[str, str | None] = {}
    n_tag = n_geo = n_none = n_disagree = 0
    for uid, grp in by_uid.items():
        tags = [side_from_tag(r.get("Laterality")) for r in grp]
        tags = [t for t in tags if t is not None]
        side = tags[0] if tags else None
        g = geo.get(uid)
        if side is not None:
            n_tag += 1
            if g is not None and g != side:
                n_disagree += 1
        else:
            side = g
            n_geo += side is not None
            n_none += side is None
        out[uid] = side
    if log:
        prefix = f"{tag} " if tag else ""
        print(
            f"{prefix}laterality : {n_tag} tag, {n_geo} geo, {n_none} unresolved ; "
            f"tag vs geo disagree {n_disagree} "
            f"({n_disagree / max(n_tag, 1):.1%} des tags)",
            flush=True,
        )
    return out


def normalise_laterality(vol, plane: str | None, lat: str | None):
    """Canon gauche. vol numpy (D,H,W) / (D,3,H,W) ou torch identique."""
    if lat != "R":
        return vol
    plane_c = str(plane or "").strip().capitalize()
    if plane_c in ("Coronal", "Axial"):
        axis, dims = -1, [-1]
    elif plane_c == "Sagittal":
        axis, dims = 0, [0]
    else:
        return vol
    if isinstance(vol, np.ndarray):
        return np.ascontiguousarray(np.flip(vol, axis=axis))
    import torch
    if isinstance(vol, torch.Tensor):
        return torch.flip(vol, dims=dims)
    raise TypeError(f"volume type inconnu : {type(vol)}")


_MAP_CACHE: dict[str, dict[str, str | None]] = {}


def load_laterality_csv(path: str | Path) -> dict[str, str | None]:
    df = pd.read_csv(path)
    if "StudyInstanceUID" not in df.columns:
        raise SystemExit(f"{path} sans colonne StudyInstanceUID")
    col = "laterality" if "laterality" in df.columns else None
    if col is None:
        raise SystemExit(f"{path} sans colonne laterality")
    out: dict[str, str | None] = {}
    for uid, lat in zip(df["StudyInstanceUID"], df[col]):
        out[str(uid)] = side_from_tag(lat)
    return out


def laterality_map_from_cfg(cfg: dict | None) -> dict[str, str | None] | None:
    """None si desactive. SystemExit si laterality=true sans CSV."""
    if not cfg or not cfg.get("laterality"):
        return None
    path = Path(str(cfg.get("laterality_csv") or "data/csv/study_laterality.csv"))
    if not path.is_file():
        raise SystemExit(
            f"laterality=true mais CSV absent : {path}\n"
            "  dump Kaggle (headers only) puis placer le CSV ici :\n"
            "  python -X utf8 -m src.data.dump_laterality "
            "--out data/csv/study_laterality.csv"
        )
    key = str(path.resolve())
    if key not in _MAP_CACHE:
        _MAP_CACHE[key] = load_laterality_csv(path)
        m = _MAP_CACHE[key]
        n_l = sum(v == "L" for v in m.values())
        n_r = sum(v == "R" for v in m.values())
        n_u = sum(v is None for v in m.values())
        print(
            f"laterality {path} : {len(m)} etudes | L {n_l} | R {n_r} (flip) | "
            f"unresolved {n_u} | canon {CANON}",
            flush=True,
        )
        if len(m) == 0:
            raise SystemExit(f"{path} vide")
    return _MAP_CACHE[key]


def probe_series_header(series_dir: str | Path) -> dict | None:
    """Lit un DICOM median (stop_before_pixels). None si illisible."""
    import pydicom

    series_dir = Path(series_dir)
    files = sorted(p for p in series_dir.iterdir() if p.suffix.lower() == ".dcm")
    if not files:
        files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return None
    mid = files[len(files) // 2]
    try:
        ds = pydicom.dcmread(
            str(mid), stop_before_pixels=True, force=True,
            specific_tags=list(HDR_TAGS),
        )
    except Exception:
        return None
    row: dict = {}
    for t in HDR_TAGS:
        v = getattr(ds, t, None)
        if v is None:
            row[t] = None
        elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
            row[t] = "|".join(str(x) for x in v)
        else:
            row[t] = v if t in ("Rows", "Columns") else str(v)
    return row


def laterality_from_study_dir(study_dir: str | Path) -> str | None:
    """Cote d'une etude DICOM (test Kaggle). Un header / serie."""
    study_dir = Path(study_dir)
    if not study_dir.is_dir():
        return None
    rows = []
    for series in sorted(p for p in study_dir.iterdir() if p.is_dir()):
        rec = probe_series_header(series)
        if rec is None:
            continue
        rec["StudyInstanceUID"] = study_dir.name
        rec["SeriesInstanceUID"] = series.name
        rows.append(rec)
    if not rows:
        return None
    return lat_of_rows(rows, log=False).get(study_dir.name)
