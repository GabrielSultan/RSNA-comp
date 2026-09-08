# -*- coding: utf-8 -*-
"""Genere les kernels Kaggle : 6 slots protocole v20b, RGB 2.5D voisins DICOM.

Meme recette spatiale que convert_rgb25d_part* (130 mm, bande 20-80 %,
linspace 24 centres, R=k-1 G=k B=k+1 dans le volume ORIGINAL).

Selection = pick_protocol v20b (contraste exact, pas de repli) :

  0 SAG_FS   1 COR_FS   2 AX_FS
  3 SAG_T1   4 COR_T1   5 AX_T1

6 shards : ~autant de series/kernel que rgb25d 3-slots (3 shards x 3 series).

  python -X utf8 kaggle_DICOM_exctract/_gen_convert_rgb25d_proto_nb.py
  python -X utf8 kaggle_DICOM_exctract/_push_rgb25d_proto.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_gen_convert_rgb25d_nb", ROOT / "_gen_convert_rgb25d_nb.py",
)
_rgb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_rgb)

DIR_PREFIX = "convert_rgb25d_proto"
ZIP_PREFIX = "volumes_crop130_rgb25d_proto"
# Slug Kaggle : "2.5D" -> "2-5d" (comme convert_rgb25d_part*).
KERNEL_SLUG = "rsna-knee-convert-rgb-2-5d-proto"
KERNEL_TITLE = "RSNA Knee Convert RGB 2.5D proto"
N_SHARDS = 6

CODE_PICK = r'''
series_df = pd.read_csv(TRAIN_SERIES_CSV)
print("series", len(series_df), "| studies", series_df.StudyInstanceUID.nunique())

# Ordre = PROTOCOL_SLOTS v20b (src/data/dataset_npz.py). Pas de repli.
PROTOCOL_SLOTS = [
    ("SAG_FS", "Sagittal", 1),
    ("COR_FS", "Coronal", 1),
    ("AX_FS", "Axial", 1),
    ("SAG_T1", "Sagittal", 0),
    ("COR_T1", "Coronal", 0),
    ("AX_T1", "Axial", 0),
]

def pick_protocol_row(g: pd.DataFrame, plane: str, fluid: int):
    """Contraste exact, premier UID. Slot absent = skip (volume zero au train)."""
    gp = g[g.Anatomical_Plane == plane].sort_values("SeriesInstanceUID")
    if "Fluid_Sensitive" not in gp.columns:
        return None
    gp = gp[gp.Fluid_Sensitive == int(fluid)]
    if len(gp) == 0:
        return None
    return gp.iloc[0]

jobs = []
studies = sorted(series_df.StudyInstanceUID.unique())
my_studies = [u for i, u in enumerate(studies) if i % N_SHARDS == SHARD_ID]
print(f"etudes ce shard : {len(my_studies)} / {len(studies)}")

n_by_slot = {name: 0 for name, _p, _f in PROTOCOL_SLOTS}
n_empty = 0
for uid, g in series_df[series_df.StudyInstanceUID.isin(set(my_studies))].groupby("StudyInstanceUID"):
    n_before = len(jobs)
    for slot, plane, fluid in PROTOCOL_SLOTS:
        row = pick_protocol_row(g, plane, fluid)
        if row is None:
            continue
        jobs.append({
            "study": uid,
            "series": str(row.SeriesInstanceUID),
            "plane": row.Anatomical_Plane,
            "fluid": int(row.Fluid_Sensitive) if "Fluid_Sensitive" in row.index else int(fluid),
            "slot": slot,
        })
        n_by_slot[slot] += 1
    if len(jobs) == n_before:
        n_empty += 1

print("slots :", n_by_slot)
print("etudes sans aucun slot :", n_empty)
print("series proto rgb25d a convertir :", len(jobs))
assert jobs, "aucune serie protocole selectionnee"
'''

CODE_RUN = r'''
import zipfile

zip_path = OUT_DIR / f"volumes_crop130_rgb25d_proto_part{SHARD_ID}.zip"
manifest_path = OUT_DIR / f"manifest_rgb25d_proto_part{SHARD_ID}.csv"
recipe_path = OUT_DIR / "RECIPE.json"

recipe = {
    "crop_mm": CROP_MM,
    "crop_rel": CROP_REL,
    "band": list(BAND),
    "n_slices": N_SLICES,
    "img_size": IMG_SIZE,
    "planes": PLANES,
    "n_protocol_slots": 6,
    "slots": [
        "SAG_FS", "COR_FS", "AX_FS", "SAG_T1", "COR_T1", "AX_T1",
    ],
    "shard_id": SHARD_ID,
    "n_shards": N_SHARDS,
    "n_jobs": len(jobs),
    "shape": [N_SLICES, 3, IMG_SIZE, IMG_SIZE],
    "rgb": "dicom_neighbors",
    "note": (
        "6 slots protocole v20b (FS/T1 par plan), contraste exact, pas de repli. "
        "Crop 130 mm (PixelSpacing) sinon centre 70 %. Bande 20-80 %, linspace 24 "
        "centres. Pour chaque centre k dans le volume ORIGINAL : R=k-1, G=k, B=k+1 "
        "(1 coupe DICOM d'ecart, clamp aux bords). Resize 224, uint8. "
        "PAS les voisins des 24 linspace."
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
            "slot": job.get("slot", ""),
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
            "slot": job.get("slot", ""),
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


def make_notebook(shard_id: int, n_shards: int):
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# RSNA Knee — conversion DICOM → NPZ RGB 2.5D (6 slots proto)\n\n"
            f"**Shard {shard_id}/{n_shards}** (CPU, internet off).\n\n"
            "Lit `train_series/` sur Kaggle, garde les **6 slots protocole v20b** "
            "(FS / T1 par plan, contraste exact, **pas de repli**) :\n\n"
            "- `SAG_FS`, `COR_FS`, `AX_FS`\n"
            "- `SAG_T1`, `COR_T1`, `AX_T1`\n\n"
            "Recette **identique** a `convert_rgb25d_part*` :\n\n"
            "1. crop physique **130 mm** (`PixelSpacing`) ou crop centre 70 % si spacing absent\n"
            "2. bande profondeur **20–80 %**\n"
            "3. **linspace 24** pour choisir les coupes *centrales* `k`\n"
            "4. pour chaque `k`, stocker **`k-1, k, k+1` dans le volume original** "
            "(1 coupe anatomique d'écart), **pas** les voisins des 24 linspace\n"
            "5. resize **224×224**, uint8, percentiles 1–99\n\n"
            "Sortie : `{SeriesInstanceUID}.npz` → `data` de shape "
            "`(24, 3, 224, 224)` uint8, canaux **R=k-1, G=k, B=k+1**.\n\n"
            "Fichiers dans `/kaggle/working/` :\n"
            f"- `{ZIP_PREFIX}_part{shard_id}.zip`\n"
            f"- `manifest_rgb25d_proto_part{shard_id}.csv`\n"
            "- `RECIPE.json`\n\n"
            "Download local : `python -X utf8 kaggle_DICOM_exctract/download_rgb25d_proto.py` "
            "→ `data/crop130_rgb25d/`."
        ),
        nbf.v4.new_code_cell(
            f"SHARD_ID = {shard_id}\nN_SHARDS = {n_shards}\n" + _rgb.CODE_CONVERT
        ),
        nbf.v4.new_code_cell(_rgb.CODE_FIND),
        nbf.v4.new_code_cell(CODE_PICK),
        nbf.v4.new_code_cell(_rgb.CODE_FUNCS),
        nbf.v4.new_code_cell(CODE_RUN),
        nbf.v4.new_code_cell(_rgb.CODE_VERIFY),
    ]
    nb["cells"] = cells
    return nb


def main() -> None:
    import nbformat as nbf

    for shard in range(N_SHARDS):
        d = ROOT / f"{DIR_PREFIX}_part{shard}"
        d.mkdir(parents=True, exist_ok=True)
        nb = make_notebook(shard, N_SHARDS)
        path = d / "convert.ipynb"
        nbf.write(nb, path)
        meta = {
            "id": f"sultent/{KERNEL_SLUG}-part{shard}",
            "title": f"{KERNEL_TITLE} part{shard}",
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
            _rgb.json_dumps(meta), encoding="utf-8"
        )
        print("wrote", path)


if __name__ == "__main__":
    main()
