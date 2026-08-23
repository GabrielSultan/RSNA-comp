# -*- coding: utf-8 -*-
"""Genere les kernels Kaggle : 1 serie T1/non-fluide extra par etude, crop 130 mm 224.

Meme recette spatiale que convert_crop130_part* (130 mm, bande 20-80 %, 24x224).
Selection = StudySeriesIndex.pick_nonfluid : une serie Fluid_Sensitive==0
pas deja prise par les 3 picks fluid (sag d'abord). ~4406 series, 2 shards.

  python -X utf8 kaggle_DICOM_exctract/_gen_convert_t1_nb.py
  kaggle kernels push -p kaggle_DICOM_exctract/convert_crop130_t1_part0
  kaggle kernels push -p kaggle_DICOM_exctract/convert_crop130_t1_part1
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
DIR_PREFIX = "convert_crop130_t1"
ZIP_PREFIX = "volumes_crop130_t1"
KERNEL_SLUG = "rsna-knee-convert-crop130-t1"
KERNEL_TITLE = "RSNA Knee Convert crop130 T1"
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

def pick_t1_row(g: pd.DataFrame, exclude: set[str]):
    """1 non-fluide pas deja dans les 3 picks. Sagittal d'abord (comme pick_nonfluid)."""
    for plane in PLANES:
        gp = g[g.Anatomical_Plane == plane].sort_values("SeriesInstanceUID")
        if "Fluid_Sensitive" not in gp.columns:
            continue
        nf = gp[gp.Fluid_Sensitive == 0]
        for _, row in nf.iterrows():
            if str(row.SeriesInstanceUID) not in exclude:
                return row
    return None

jobs = []
studies = sorted(series_df.StudyInstanceUID.unique())
my_studies = [u for i, u in enumerate(studies) if i % N_SHARDS == SHARD_ID]
print(f"etudes ce shard : {len(my_studies)} / {len(studies)}")

n_skip = 0
for uid, g in series_df[series_df.StudyInstanceUID.isin(set(my_studies))].groupby("StudyInstanceUID"):
    row = pick_t1_row(g, pick_fluid_sids(g))
    if row is None:
        n_skip += 1
        continue
    jobs.append({
        "study": uid,
        "series": row.SeriesInstanceUID,
        "plane": row.Anatomical_Plane,
        "fluid": int(row.Fluid_Sensitive) if "Fluid_Sensitive" in row.index else 0,
    })
print("series T1 a convertir :", len(jobs), "| etudes sans T1 extra :", n_skip)
assert jobs, "aucune serie T1 selectionnee"
'''


def _run_cell(img_size: int, zip_prefix: str) -> str:
    src = _mod.CODE_RUN.replace(
        'f"manifest_part{SHARD_ID}.csv"',
        'f"manifest_t1_part{SHARD_ID}.csv"',
    ).replace(
        "1 serie fluid-prioritaire par plan. ",
        "1 serie T1/non-fluide extra (pas dans les 3 picks fluid). ",
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
            "# RSNA Knee — conversion T1 / non-fluide → NPZ (crop 130 mm)\n\n"
            f"**Shard {shard_id}/{n_shards}** (CPU, internet off). "
            f"**{img_size}×{img_size}** uint8.\n\n"
            "Complement de `convert_crop130_part*` : **1 série non-fluide extra** "
            "par étude (`Fluid_Sensitive==0`, pas déjà dans les 3 picks fluid, "
            "sagittal d'abord). **Même recette** que crop130 :\n\n"
            "1. crop physique **130 mm** (`PixelSpacing`) ou crop centre 70 % si spacing absent\n"
            "2. bande profondeur **20–80 %**\n"
            f"3. resample **24** coupes, resize **{img_size}×{img_size}**, uint8, percentiles 1–99\n\n"
            "Sortie dans `/kaggle/working/` :\n"
            f"- `{zip_prefix}_part{shard_id}.zip` (`{{SeriesInstanceUID}}.npz` → `data`)\n"
            f"- `manifest_t1_part{shard_id}.csv`\n"
            "- `RECIPE.json`\n\n"
            "A fusionner ensuite dans `data/crop130/` (zips **en plus** des fluid). "
            "Le submit DICOM n'a pas besoin de ce zip : `preprocess_dicom` relit le T1."
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
    for shard in range(N_SHARDS):
        d = ROOT / f"{DIR_PREFIX}_part{shard}"
        d.mkdir(parents=True, exist_ok=True)
        nb = make_notebook(shard, N_SHARDS, IMG_SIZE, ZIP_PREFIX)
        path = d / "convert.ipynb"
        import nbformat as nbf
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
