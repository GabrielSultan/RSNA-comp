# -*- coding: utf-8 -*-
"""Genere le kernel Kaggle dump laterality (headers only).

  python -X utf8 kaggle_DICOM_exctract/_gen_dump_laterality_nb.py
  kaggle kernels push -p kaggle_DICOM_exctract/dump_laterality
  kaggle kernels output sultent/rsna-knee-dump-laterality -p data/csv
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dump_laterality"

CODE = r'''
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

KAGGLE = Path("/kaggle/input").exists()
LAT_MIN_OFFSET_MM = 20.0
HDR = [
    "Laterality", "PixelSpacing", "Rows", "Columns",
    "ImagePositionPatient", "ImageOrientationPatient",
]
N_WORKERS = 8
OUT_CSV = Path("/kaggle/working/study_laterality.csv") if KAGGLE else Path("study_laterality.csv")
OUT_JSON = Path("/kaggle/working/laterality_summary.json") if KAGGLE else Path("laterality_summary.json")


def find_comp_dir() -> Path:
    if not KAGGLE:
        p = Path("../data/csv")
        if (p / "train_series.csv").exists():
            return p
        raise FileNotFoundError("train_series.csv introuvable en local")
    data = Path("/kaggle/input")
    print("contenu /kaggle/input :", sorted(x.name for x in data.iterdir()), flush=True)
    for base in [data / "competitions", data]:
        if not base.exists():
            continue
        for p in sorted(base.iterdir()):
            if (p / "train.csv").exists() and (p / "train_series.csv").exists():
                return p
    raise FileNotFoundError("train.csv + train_series.csv introuvables sous /kaggle/input")


def find_train_series_dir(comp_dir: Path) -> Path:
    cand = comp_dir / "train_series"
    if cand.is_dir():
        return cand
    for p in comp_dir.iterdir():
        if p.is_dir() and (p / "train_series").is_dir():
            return p / "train_series"
    raise FileNotFoundError(f"train_series/ introuvable sous {comp_dir}")


def side_tag(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    return s[0] if s and s[0] in ("L", "R") else None


def _vec(v, n):
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


def center_x(ipp, iop, ps, rows, cols):
    ipp_v, iop_v, ps_v = _vec(ipp, 3), _vec(iop, 6), _vec(ps, 2)
    try:
        rows_f, cols_f = float(rows), float(cols)
    except (TypeError, ValueError):
        return None
    if ipp_v is None or iop_v is None or ps_v is None or rows_f <= 0 or cols_f <= 0:
        return None
    c = (ipp_v[:3]
         + iop_v[:3] * ps_v[1] * cols_f / 2.0
         + iop_v[3:6] * ps_v[0] * rows_f / 2.0)
    x = float(c[0])
    return x if np.isfinite(x) else None


def probe(series_dir: Path):
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return None
    try:
        ds = pydicom.dcmread(
            str(files[len(files) // 2]), stop_before_pixels=True,
            force=True, specific_tags=HDR,
        )
    except Exception:
        return None
    row = {}
    for t in HDR:
        v = getattr(ds, t, None)
        if v is None:
            row[t] = None
        elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
            row[t] = "|".join(str(x) for x in v)
        else:
            row[t] = v
    return row


COMP_DIR = find_comp_dir()
TRAIN_SERIES_CSV = COMP_DIR / "train_series.csv"
TRAIN_SERIES_DIR = find_train_series_dir(COMP_DIR)
print("COMP_DIR         ", COMP_DIR, flush=True)
print("TRAIN_SERIES_CSV ", TRAIN_SERIES_CSV, TRAIN_SERIES_CSV.exists(), flush=True)
print("TRAIN_SERIES_DIR ", TRAIN_SERIES_DIR, TRAIN_SERIES_DIR.exists(), flush=True)

series_df = pd.read_csv(TRAIN_SERIES_CSV)
print("series", len(series_df), "| studies", series_df.StudyInstanceUID.nunique(),
      flush=True)

items = []
missing_dir = 0
for r in series_df.itertuples(index=False):
    d = TRAIN_SERIES_DIR / r.StudyInstanceUID / r.SeriesInstanceUID
    if not d.is_dir():
        missing_dir += 1
        continue
    items.append((r.StudyInstanceUID, r.SeriesInstanceUID, d))
print("series avec dossier", len(items), "| sans dossier", missing_dir, flush=True)

rows = []
n_fail = 0
with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    futs = {pool.submit(probe, it[2]): it for it in items}
    for i, fut in enumerate(as_completed(futs), 1):
        uid, sid, _ = futs[fut]
        rec = fut.result()
        if rec is None:
            n_fail += 1
        else:
            rec["StudyInstanceUID"] = uid
            rec["SeriesInstanceUID"] = sid
            rows.append(rec)
        if i % 2000 == 0 or i == len(futs):
            print(f"  {i}/{len(futs)} headers ({len(rows)} ok, {n_fail} fail)",
                  flush=True)

if not rows:
    raise SystemExit("aucun header lu")

by = {}
for r in rows:
    by.setdefault(str(r["StudyInstanceUID"]), []).append(r)

out_rows = []
n_tag = n_geo = n_none = n_dis = 0
for uid, grp in sorted(by.items()):
    tags = [side_tag(r.get("Laterality")) for r in grp]
    tags = [t for t in tags if t]
    tag_side = tags[0] if tags else None
    xs = [
        center_x(
            r.get("ImagePositionPatient"), r.get("ImageOrientationPatient"),
            r.get("PixelSpacing"), r.get("Rows"), r.get("Columns"),
        )
        for r in grp
    ]
    xs = [x for x in xs if x is not None]
    geo = None
    med_x = float(np.median(xs)) if xs else None
    if med_x is not None and abs(med_x) >= LAT_MIN_OFFSET_MM:
        geo = "R" if med_x < 0 else "L"
    if tag_side:
        n_tag += 1
        if geo is not None and geo != tag_side:
            n_dis += 1
        side, source = tag_side, "tag"
    elif geo:
        n_geo += 1
        side, source = geo, "geo"
    else:
        n_none += 1
        side, source = "", "none"
    out_rows.append({
        "StudyInstanceUID": uid,
        "laterality": side,
        "source": source,
        "tag_laterality": tag_side or "",
        "geo_laterality": geo or "",
        "geo_x_mm": med_x if med_x is not None else "",
        "n_series_probed": len(grp),
        "disagree": int(tag_side is not None and geo is not None and tag_side != geo),
    })

df = pd.DataFrame(out_rows)
df.to_csv(OUT_CSV, index=False)
n_l = int((df.laterality == "L").sum())
n_r = int((df.laterality == "R").sum())
n_u = int((df.laterality == "").sum())
summary = {
    "n_studies": int(len(df)),
    "n_L": n_l,
    "n_R": n_r,
    "n_unresolved": n_u,
    "n_from_tag": n_tag,
    "n_from_geo": n_geo,
    "n_disagree_tag_geo": n_dis,
    "n_header_fail": n_fail,
    "n_missing_dir": missing_dir,
    "canon": "L",
    "rule": "tag then image-center X, deadzone 20 mm",
}
OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("laterality :", n_tag, "tag,", n_geo, "geo,", n_none, "unresolved ; disagree",
      n_dis, flush=True)
print("L", n_l, "| R", n_r, "(flip) | unresolved", n_u, flush=True)
print("ecrit", OUT_CSV, len(df), "etudes", flush=True)
print("summary", summary, flush=True)
df.head()
'''


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# RSNA Knee — dump laterality (headers only)\n\n"
            "Lit `train_series/` (competition), un DICOM median par serie "
            "(`stop_before_pixels`).\n\n"
            "Cote par etude, canon **gauche** :\n"
            "1. tag DICOM `Laterality` si L/R\n"
            "2. sinon X du **centre** de l'image (LPS, zone morte 20 mm)\n\n"
            "Sortie `/kaggle/working/` :\n"
            "- `study_laterality.csv`\n"
            "- `laterality_summary.json`\n\n"
            "Ensuite en local :\n"
            "`kaggle kernels output sultent/rsna-knee-dump-laterality -p data/csv`"
        ),
        nbf.v4.new_code_cell(CODE.strip() + "\n"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT / "dump.ipynb")
    meta = {
        "id": "sultent/rsna-knee-dump-laterality",
        "title": "RSNA Knee dump laterality",
        "code_file": "dump.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
