# -*- coding: utf-8 -*-
"""Kernels Kaggle : crop 100 mm @ 224, meme recette que convert_crop130_part*.

Levier magnification vs crop130 : FOV 100 mm au lieu de 130 mm, tout le reste
identique (1 serie fluid-prioritaire / plan, bande 20-80 %, 24 coupes, 224 px).
CROP_REL reste 0.70 (fallback spacing absent seulement).

  python -X utf8 kaggle_DICOM_exctract/_gen_convert_crop100_nb.py
  kaggle kernels push -p kaggle_DICOM_exctract/convert_crop100_part0
  kaggle kernels push -p kaggle_DICOM_exctract/convert_crop100_part1
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
DIR_PREFIX = "convert_crop100"
ZIP_PREFIX = "volumes_crop100"
KERNEL_SLUG = "rsna-knee-convert-crop100"
KERNEL_TITLE = "RSNA Knee Convert crop100"
N_SHARDS = 2

CODE_CONVERT = _mod.CODE_CONVERT.replace("CROP_MM = 130.0", "CROP_MM = 100.0")
CODE_RUN = _mod.CODE_RUN.replace("Crop 130 mm", "Crop 100 mm")


def make_notebook(shard_id: int, n_shards: int, img_size: int, zip_prefix: str):
    nbf = _mod.nbf
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# RSNA Knee — conversion DICOM → NPZ (crop 100 mm)\n\n"
            f"**Shard {shard_id}/{n_shards}** (CPU, internet off). "
            f"**{img_size}×{img_size}** uint8.\n\n"
            "Même pipeline que crop130 224 : 1 série fluid-prioritaire par plan "
            "(sag / cor / ax). **Seul le FOV change** (100 mm vs 130 mm) — "
            "même bande profondeur, même 24 coupes, même 224 px.\n\n"
            "1. crop physique **100 mm** (`PixelSpacing`) ou crop centre 70 % si spacing absent\n"
            "2. bande profondeur **20–80 %**\n"
            f"3. resample **24** coupes, resize **{img_size}×{img_size}**, uint8, percentiles 1–99\n\n"
            "Sortie dans `/kaggle/working/` :\n"
            f"- `{zip_prefix}_part{shard_id}.zip` (`{{SeriesInstanceUID}}.npz` → `data`)\n"
            f"- `manifest_part{shard_id}.csv`\n"
            "- `RECIPE.json`\n\n"
            "Ensuite : `kaggle kernels output` puis train local "
            "(`configs/kneenet_v17_crop100_subset.yaml`). "
            "Le submit devra recopier **la même recette** (`CROP_MM = 100`)."
        ),
        nbf.v4.new_code_cell(
            f"SHARD_ID = {shard_id}\nN_SHARDS = {n_shards}\n"
            + _mod._fill(CODE_CONVERT, img_size, zip_prefix)
        ),
        nbf.v4.new_code_cell(_mod.CODE_FIND),
        nbf.v4.new_code_cell(_mod.CODE_PICK),
        nbf.v4.new_code_cell(_mod.CODE_FUNCS),
        nbf.v4.new_code_cell(_mod._fill(CODE_RUN, img_size, zip_prefix)),
        nbf.v4.new_code_cell(_mod.CODE_VERIFY),
    ]
    nb["cells"] = cells
    return nb


def main() -> None:
    nbf = _mod.nbf
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
