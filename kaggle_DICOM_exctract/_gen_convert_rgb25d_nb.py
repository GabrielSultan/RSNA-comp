# -*- coding: utf-8 -*-
"""Génère les notebooks Kaggle : DICOM -> NPZ crop 130 mm, RGB 2.5D voisin DICOM.

Pour chaque coupe linspace k (bande 20-80 %), on stocke les vraies coupes
k-1, k, k+1 du volume original (pas des 24 sous-échantillonnées).
Shape : (24, 3, 224, 224) uint8, R=k-1, G=k, B=k+1 (clamp aux bords).
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent

CODE_CONVERT = r'''
import os, io, gc, time, json, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import pydicom

# --- Recette (doit etre reproduite a l'identique au submit) ----------------
CROP_MM = 130.0
CROP_REL = 0.70          # fallback si PixelSpacing absent
BAND = (0.20, 0.80)      # profondeur : on ignore les 20 % de chaque bord
N_SLICES = 24
IMG_SIZE = 224
PLANES = ["Sagittal", "Coronal", "Axial"]
N_WORKERS = 4
# --------------------------------------------------------------------------

KAGGLE = os.path.exists("/kaggle")
OUT_DIR = Path("/kaggle/working") if KAGGLE else Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("KAGGLE", KAGGLE, "| SHARD", SHARD_ID, "/", N_SHARDS)
print("out", OUT_DIR)
'''

CODE_FIND = r'''
def find_comp_dir() -> Path:
    if not KAGGLE:
        p = Path("../data/csv")
        if (p / "train_series.csv").exists():
            return p
        raise FileNotFoundError("train_series.csv introuvable en local")
    data = Path("/kaggle/input")
    print("contenu /kaggle/input :", sorted(x.name for x in data.iterdir()))
    for base in [data / "competitions", data]:
        if not base.exists():
            continue
        for p in sorted(base.iterdir()):
            if (p / "train.csv").exists() and (p / "train_series.csv").exists():
                return p
    raise FileNotFoundError("train.csv + train_series.csv introuvables sous /kaggle/input")

COMP_DIR = find_comp_dir()
TRAIN_SERIES_CSV = COMP_DIR / "train_series.csv"
TRAIN_SERIES_DIR = COMP_DIR / "train_series"
if not TRAIN_SERIES_DIR.exists():
    for cand in COMP_DIR.iterdir():
        if cand.is_dir() and (cand / "train_series").exists():
            TRAIN_SERIES_DIR = cand / "train_series"
            break
print("COMP_DIR         ", COMP_DIR)
print("TRAIN_SERIES_CSV ", TRAIN_SERIES_CSV, TRAIN_SERIES_CSV.exists())
print("TRAIN_SERIES_DIR ", TRAIN_SERIES_DIR, TRAIN_SERIES_DIR.exists())
'''

CODE_PICK = r'''
series_df = pd.read_csv(TRAIN_SERIES_CSV)
print("series", len(series_df), "| studies", series_df.StudyInstanceUID.nunique())

def pick_three_planes(g: pd.DataFrame) -> list[pd.Series]:
    """1 serie / plan, fluid-sensitive d'abord (meme logique que StudySeriesIndex)."""
    picks = []
    for plane in PLANES:
        gp = g[g.Anatomical_Plane == plane].sort_values("SeriesInstanceUID")
        if len(gp) == 0:
            continue
        fluid = gp[gp.Fluid_Sensitive == 1] if "Fluid_Sensitive" in gp.columns else gp.iloc[0:0]
        if len(fluid) == 0 and "Fat_Suppression" in gp.columns:
            fluid = gp[gp.Fat_Suppression == 1]
        row = fluid.iloc[0] if len(fluid) else gp.iloc[0]
        picks.append(row)
    return picks

jobs = []
studies = sorted(series_df.StudyInstanceUID.unique())
my_studies = [u for i, u in enumerate(studies) if i % N_SHARDS == SHARD_ID]
print(f"etudes ce shard : {len(my_studies)} / {len(studies)}")

for uid, g in series_df[series_df.StudyInstanceUID.isin(set(my_studies))].groupby("StudyInstanceUID"):
    for row in pick_three_planes(g):
        jobs.append({
            "study": uid,
            "series": row.SeriesInstanceUID,
            "plane": row.Anatomical_Plane,
            "fluid": int(row.Fluid_Sensitive) if "Fluid_Sensitive" in row.index else -1,
        })
print("series a convertir :", len(jobs))
assert jobs, "aucune serie selectionnee"
'''

CODE_FUNCS = r'''
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


def _to_u8(sl: np.ndarray) -> np.ndarray:
    sl = cv2.resize(sl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return np.clip(sl * 255.0, 0, 255).round().astype(np.uint8)


def convert_series(series_dir: Path) -> tuple[np.ndarray, dict]:
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"pas de .dcm dans {series_dir}")
    datasets = []
    for f in files:
        try:
            datasets.append(pydicom.dcmread(str(f), stop_before_pixels=False))
        except Exception:
            continue
    if not datasets:
        raise RuntimeError("aucun DICOM lisible")
    datasets.sort(key=_sort_key)

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
    vol = np.stack(stacked, axis=0)  # (n, h, w) volume ORIGINAL trie

    lo, hi = np.percentile(vol, 1.0), np.percentile(vol, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)

    n = vol.shape[0]
    a = int(np.floor(BAND[0] * (n - 1)))
    b = int(np.ceil(BAND[1] * (n - 1)))
    if b - a + 1 < 8:
        a, b = 0, n - 1
    # memes 24 centres que crop130 (linspace dans la bande),
    # mais voisins ±1 dans le volume original, pas dans les 24
    idx_band = np.linspace(0, b - a, N_SLICES).round().astype(int)
    idx_orig = np.clip(a + idx_band, 0, n - 1)

    out = np.empty((N_SLICES, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    last = n - 1
    n_clamp = 0
    for i, k in enumerate(idx_orig):
        k = int(k)
        km = 0 if k <= 0 else k - 1
        kp = last if k >= last else k + 1
        if km == k or kp == k:
            n_clamp += 1
        out[i, 0] = _to_u8(vol[km])
        out[i, 1] = _to_u8(vol[k])
        out[i, 2] = _to_u8(vol[kp])

    meta = {
        "n_raw": int(n),
        "band": [a, b],
        "crop_mode": crop_mode,
        "spacing_mm": None if sp is None else [float(sp[0]), float(sp[1])],
        "src_hw": [int(h0), int(w0)],
        "n_files": len(datasets),
        "n_clamp": int(n_clamp),
        "rgb": "dicom_neighbors",
    }
    return out, meta


def series_path(study_uid: str, series_uid: str) -> Path:
    p = TRAIN_SERIES_DIR / study_uid / series_uid
    if p.exists():
        return p
    p2 = TRAIN_SERIES_DIR / series_uid
    return p2
'''

CODE_RUN = r'''
import zipfile

zip_path = OUT_DIR / f"volumes_crop130_rgb25d_part{SHARD_ID}.zip"
manifest_path = OUT_DIR / f"manifest_part{SHARD_ID}.csv"
recipe_path = OUT_DIR / "RECIPE.json"

recipe = {
    "crop_mm": CROP_MM,
    "crop_rel": CROP_REL,
    "band": list(BAND),
    "n_slices": N_SLICES,
    "img_size": IMG_SIZE,
    "planes": PLANES,
    "shard_id": SHARD_ID,
    "n_shards": N_SHARDS,
    "n_jobs": len(jobs),
    "shape": [N_SLICES, 3, IMG_SIZE, IMG_SIZE],
    "rgb": "dicom_neighbors",
    "note": (
        "1 serie fluid-prioritaire par plan. Crop 130 mm (PixelSpacing) sinon "
        "centre 70 %. Bande 20-80 %, linspace 24 centres. Pour chaque centre k "
        "dans le volume ORIGINAL : R=k-1, G=k, B=k+1 (1 coupe DICOM d'ecart, "
        "clamp aux bords). Resize 224, uint8. PAS les voisins des 24 linspace."
    ),
}
recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
print(json.dumps(recipe, indent=2))

def _one(job):
    sid = job["series"]
    t0 = time.time()
    try:
        p = series_path(job["study"], sid)
        vol, meta = convert_series(p)
        buf = io.BytesIO()
        np.savez_compressed(buf, data=vol)
        payload = buf.getvalue()
        rec = {
            "StudyInstanceUID": job["study"],
            "SeriesInstanceUID": sid,
            "plane": job["plane"],
            "fluid": job["fluid"],
            "ok": 1,
            "error": "",
            "nbytes": len(payload),
            "dt": round(time.time() - t0, 3),
            **meta,
        }
        return sid, payload, rec
    except Exception as e:
        rec = {
            "StudyInstanceUID": job["study"],
            "SeriesInstanceUID": sid,
            "plane": job["plane"],
            "fluid": job["fluid"],
            "ok": 0,
            "error": f"{type(e).__name__}: {e}"[:300],
            "nbytes": 0,
            "dt": round(time.time() - t0, 3),
            "n_raw": None, "band": None, "crop_mode": None,
            "spacing_mm": None, "src_hw": None, "n_files": None,
            "n_clamp": None, "rgb": None,
        }
        return sid, None, rec

t_all = time.time()
rows = []
n_ok = n_fail = 0
bytes_sum = 0

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_one, job) for job in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            sid, payload, rec = fut.result()
            rows.append(rec)
            if payload is not None:
                zf.writestr(f"{sid}.npz", payload)
                n_ok += 1
                bytes_sum += rec["nbytes"]
            else:
                n_fail += 1
            if i % 25 == 0 or i == len(jobs):
                elapsed = time.time() - t_all
                rate = i / max(elapsed, 1e-6)
                eta = (len(jobs) - i) / max(rate, 1e-6)
                print(
                    f"[{i}/{len(jobs)}] ok={n_ok} fail={n_fail} "
                    f"zip={bytes_sum/1e9:.2f}GB  {rate:.2f} s/serie  ETA {eta/60:.0f} min",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(manifest_path, index=False)

pd.DataFrame(rows).to_csv(manifest_path, index=False)
print("termine", zip_path, "size", zip_path.stat().st_size / 1e9, "GB")
print("ok", n_ok, "fail", n_fail, "en", (time.time() - t_all) / 60, "min")
print("manifest", manifest_path)
if n_fail:
    fails = [r for r in rows if not r["ok"]]
    print("exemples d'erreurs :")
    for r in fails[:8]:
        print(" ", r["SeriesInstanceUID"], r["error"])
'''

CODE_VERIFY = r'''
import zipfile as zfmod
with zfmod.ZipFile(zip_path) as zf:
    names = [n for n in zf.namelist() if n.endswith(".npz")]
    print("npz dans le zip :", len(names))
    for name in names[:3]:
        with zf.open(name) as f:
            vol = np.load(io.BytesIO(f.read()))["data"]
        assert vol.shape == (N_SLICES, 3, IMG_SIZE, IMG_SIZE), vol.shape
        assert vol.dtype == np.uint8, vol.dtype
        rg = np.abs(vol[:, 0].astype(np.float32) - vol[:, 1].astype(np.float32)).mean()
        bg = np.abs(vol[:, 2].astype(np.float32) - vol[:, 1].astype(np.float32)).mean()
        print(
            name, vol.shape, vol.dtype,
            "min", int(vol.min()), "max", int(vol.max()),
            "mean", float(vol.mean()),
            "|R-G|", round(float(rg), 2), "|B-G|", round(float(bg), 2),
        )
print("attendu : shape (24, 3, 224, 224), |R-G| et |B-G| > 0 (voisins DICOM reels)")
print("fichiers working :", sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()))
'''


def make_notebook(shard_id: int, n_shards: int) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# RSNA Knee — conversion DICOM → NPZ RGB 2.5D (voisins DICOM)\n\n"
            f"**Shard {shard_id}/{n_shards}** (CPU, internet off).\n\n"
            "Lit `train_series/` sur Kaggle, garde **1 série fluid-prioritaire par plan** "
            "(sag / cor / ax), applique :\n\n"
            "1. crop physique **130 mm** (`PixelSpacing`) ou crop centre 70 % si spacing absent\n"
            "2. bande profondeur **20–80 %**\n"
            "3. **linspace 24** pour choisir les coupes *centrales* `k`\n"
            "4. pour chaque `k`, stocker **`k-1, k, k+1` dans le volume original** "
            "(1 coupe anatomique d'écart), **pas** les voisins des 24 linspace\n"
            "5. resize **224×224**, uint8, percentiles 1–99\n\n"
            "Sortie : `{SeriesInstanceUID}.npz` → `data` de shape "
            "`(24, 3, 224, 224)` uint8, canaux **R=k-1, G=k, B=k+1** "
            "(clamp sur la première / dernière coupe DICOM).\n\n"
            "Fichiers dans `/kaggle/working/` :\n"
            f"- `volumes_crop130_rgb25d_part{shard_id}.zip`\n"
            f"- `manifest_part{shard_id}.csv`\n"
            "- `RECIPE.json`\n\n"
            "Ensuite : `kaggle kernels output` puis train local. "
            "Le submit devra recopier **la même recette**."
        ),
        nbf.v4.new_code_cell(
            f"SHARD_ID = {shard_id}\nN_SHARDS = {n_shards}\n" + CODE_CONVERT
        ),
        nbf.v4.new_code_cell(CODE_FIND),
        nbf.v4.new_code_cell(CODE_PICK),
        nbf.v4.new_code_cell(CODE_FUNCS),
        nbf.v4.new_code_cell(CODE_RUN),
        nbf.v4.new_code_cell(CODE_VERIFY),
    ]
    nb["cells"] = cells
    return nb


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2) + "\n"


def main() -> None:
    n_shards = 3
    for shard in range(n_shards):
        d = ROOT / f"convert_rgb25d_part{shard}"
        d.mkdir(parents=True, exist_ok=True)
        nb = make_notebook(shard, n_shards)
        path = d / "convert.ipynb"
        nbf.write(nb, path)
        meta = {
            "id": f"sultent/rsna-knee-convert-rgb-2-5d-part{shard}",
            "title": f"RSNA Knee Convert RGB 2.5D part{shard}",
            "code_file": "convert.ipynb",
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
        (d / "kernel-metadata.json").write_text(
            json_dumps(meta), encoding="utf-8"
        )
        print("wrote", path)


if __name__ == "__main__":
    main()
