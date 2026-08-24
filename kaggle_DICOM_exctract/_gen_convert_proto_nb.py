# -*- coding: utf-8 -*-
"""Genere les kernels Kaggle : T1 coronal + axial extra, crop 130 mm 224.

Meme recette spatiale que convert_crop130_part* (130 mm, bande 20-80 %, 24x224).

Dans ce dataset Fluid_Sensitive == Fat_Suppression (aucun fluide sans FS).
Les 6 slots v20b deviennent donc :

  SAG/COR/AX_FS  = deja dans volumes_crop130_part*
  SAG_T1         = deja dans volumes_crop130_t1_part* (pick_nonfluid sag d'abord)
  COR_T1, AX_T1  = ce kernel (series Fluid_Sensitive==0 pas deja prises)

  python -X utf8 kaggle_DICOM_exctract/_gen_convert_proto_nb.py
  kaggle kernels push -p kaggle_DICOM_exctract/convert_crop130_proto_part0
  kaggle kernels push -p kaggle_DICOM_exctract/convert_crop130_proto_part1
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_gen_convert_nb", ROOT / "_gen_convert_nb.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

IMG_SIZE = 224
DIR_PREFIX = "convert_crop130_proto"
ZIP_PREFIX = "volumes_crop130_proto"
KERNEL_SLUG = "rsna-knee-convert-crop130-proto"
KERNEL_TITLE = "RSNA Knee Convert crop130 proto"
N_SHARDS = 2

CODE_PICK = r'''
series_df = pd.read_csv(TRAIN_SERIES_CSV)
print("series", len(series_df), "| studies", series_df.StudyInstanceUID.nunique())

def pick_fluid_sids(g: pd.DataFrame) -> set[str]:
    """Meme 3 picks que crop130 (1 fluid-prioritaire / plan)."""
    used: set[str] = set()
    for plane in PLANES:
        gp = g[g.Anatomical_Plane == plane].sort_values("SeriesInstanceUID")
        if len(gp) == 0:
            continue
        fluid = gp[gp.Fluid_Sensitive == 1] if "Fluid_Sensitive" in gp.columns else gp.iloc[0:0]
        if len(fluid) == 0 and "Fat_Suppression" in gp.columns:
            fluid = gp[gp.Fat_Suppression == 1]
        row = fluid.iloc[0] if len(fluid) else gp.iloc[0]
        used.add(str(row.SeriesInstanceUID))
    return used

def pick_t1_extra(g: pd.DataFrame, exclude: set[str]):
    """Meme 1 non-fluide que convert_crop130_t1 (sagittal d'abord)."""
    for plane in PLANES:
        gp = g[g.Anatomical_Plane == plane].sort_values("SeriesInstanceUID")
        if "Fluid_Sensitive" not in gp.columns:
            continue
        nf = gp[gp.Fluid_Sensitive == 0]
        for _, row in nf.iterrows():
            if str(row.SeriesInstanceUID) not in exclude:
                return row
    return None

def pick_first(g: pd.DataFrame, plane: str, fluid: int, exclude: set[str]):
    gp = g[g.Anatomical_Plane == plane].sort_values("SeriesInstanceUID")
    if "Fluid_Sensitive" not in gp.columns:
        return None
    gp = gp[gp.Fluid_Sensitive == fluid]
    for _, row in gp.iterrows():
        if str(row.SeriesInstanceUID) not in exclude:
            return row
    return None

jobs = []
studies = sorted(series_df.StudyInstanceUID.unique())
my_studies = [u for i, u in enumerate(studies) if i % N_SHARDS == SHARD_ID]
print(f"etudes ce shard : {len(my_studies)} / {len(studies)}")

n_skip = 0
n_cor = n_ax = 0
for uid, g in series_df[series_df.StudyInstanceUID.isin(set(my_studies))].groupby("StudyInstanceUID"):
    used = pick_fluid_sids(g)
    extra = pick_t1_extra(g, used)
    if extra is not None:
        used.add(str(extra.SeriesInstanceUID))
    n_before = len(jobs)
    for slot, plane in (("COR_T1", "Coronal"), ("AX_T1", "Axial")):
        row = pick_first(g, plane, 0, used)
        if row is None:
            continue
        sid = str(row.SeriesInstanceUID)
        used.add(sid)
        jobs.append({
            "study": uid,
            "series": sid,
            "plane": row.Anatomical_Plane,
            "fluid": int(row.Fluid_Sensitive) if "Fluid_Sensitive" in row.index else 0,
            "slot": slot,
        })
        if slot == "COR_T1":
            n_cor += 1
        else:
            n_ax += 1
    if len(jobs) == n_before:
        n_skip += 1

print("COR_T1", n_cor, "| AX_T1", n_ax, "| etudes sans extra :", n_skip)
print("series proto a convertir :", len(jobs))
assert jobs, "aucune serie proto selectionnee"
'''


def _run_cell(img_size: int, zip_prefix: str) -> str:
    src = _mod.CODE_RUN.replace(
        'f"manifest_part{SHARD_ID}.csv"',
        'f"manifest_proto_part{SHARD_ID}.csv"',
    ).replace(
        "1 serie fluid-prioritaire par plan. ",
        "1 COR_T1 + 1 AX_T1 si dispo, hors 3 picks fluid et hors T1 sag extra. ",
    ).replace(
        '"fluid": job["fluid"],',
        '"fluid": job["fluid"],\n            "slot": job.get("slot", ""),',
    )
    return _mod._fill(src, img_size, zip_prefix)


def make_notebook(shard_id: int, n_shards: int, img_size: int, zip_prefix: str):
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# RSNA Knee — conversion T1 coronal/axial → NPZ (crop 130 mm)\n\n"
            f"**Shard {shard_id}/{n_shards}** (CPU, internet off). "
            f"**{img_size}×{img_size}** uint8.\n\n"
            "Complement v20b de `convert_crop130_part*` + `convert_crop130_t1_part*` : "
            "**1 T1 coronal** et **1 T1 axial** par étude s'ils existent, "
            "hors des 3 picks fluid et hors du T1 sagittal déjà extrait. "
            "Dans ce CSV `Fluid_Sensitive == Fat_Suppression` : pas de fluide sans FS, "
            "donc AX_T1 remplace le slot vide `SAG_FLUID_NOFS`.\n\n"
            "**Même recette** que crop130 v1 :\n\n"
            "1. crop physique **130 mm** (`PixelSpacing`) ou crop centre 70 % si spacing absent\n"
            "2. bande profondeur **20–80 %**\n"
            f"3. resample **24** coupes, resize **{img_size}×{img_size}**, uint8, percentiles 1–99\n\n"
            "Sortie dans `/kaggle/working/` :\n"
            f"- `{zip_prefix}_part{shard_id}.zip` (`{{SeriesInstanceUID}}.npz` → `data`)\n"
            f"- `manifest_proto_part{shard_id}.csv`\n"
            "- `RECIPE.json`\n\n"
            "A fusionner ensuite dans `data/crop130/` (zips **en plus**). "
            "Le submit DICOM n'a pas besoin de ce zip : `preprocess_dicom` relira les slots."
        ),
        nbf.v4.new_code_cell(
            f"SHARD_ID = {shard_id}\nN_SHARDS = {n_shards}\n"
            + _mod._fill(_mod.CODE_CONVERT, img_size, zip_prefix)
        ),
        nbf.v4.new_code_cell(_mod.CODE_FIND),
        nbf.v4.new_code_cell(CODE_PICK),
        nbf.v4.new_code_cell(_mod.CODE_FUNCS),
        nbf.v4.new_code_cell(_run_cell(img_size, zip_prefix)),
        nbf.v4.new_code_cell(_mod.CODE_VERIFY),
    ]
    nb["cells"] = cells
    return nb


def main() -> None:
    import nbformat as nbf

    for shard in range(N_SHARDS):
        d = ROOT / f"{DIR_PREFIX}_part{shard}"
        d.mkdir(parents=True, exist_ok=True)
        nb = make_notebook(shard, N_SHARDS, IMG_SIZE, ZIP_PREFIX)
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
            _mod.json_dumps(meta), encoding="utf-8"
        )
        print("wrote", path)


if __name__ == "__main__":
    main()
