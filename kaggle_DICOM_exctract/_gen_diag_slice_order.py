# -*- coding: utf-8 -*-
"""Genere le kernel Kaggle de diagnostic ordre des coupes."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SRC = REPO / "src" / "infer" / "diag_slice_order.py"
OUT = ROOT / "diag_slice_order"


def main() -> None:
    src = SRC.read_text(encoding="utf-8")
    body = src.replace('if __name__ == "__main__":\n    main()\n', "")
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
            "# RSNA Knee — diagnostic ordre des coupes\n\n"
            "Headers seulement (`stop_before_pixels`). Compare **InstanceNumber** "
            "(cle actuelle des NPZ) vs **IPP · normale** (tri geometrique).\n\n"
            "Echantillon : 800 etudes, 1 serie fluid-prioritaire / plan "
            "(meme pick que crop130). Bande 20-80 % + linspace 24.\n\n"
            "Sortie : `slice_order_diag.csv`, `slice_order_verdict.json`."
        ),
        nbf.v4.new_code_cell(body),
        nbf.v4.new_code_cell(
            "comp_dir, series_dir = find_comp_dir()\n"
            "print(\"COMP_DIR\", comp_dir)\n"
            "print(\"TRAIN_SERIES_DIR\", series_dir, series_dir.exists())\n"
            "inp = Path(\"/kaggle/input\")\n"
            "print(\"contenu input\", sorted(p.name for p in inp.iterdir()) "
            "if inp.exists() else \"n/a\")\n"
            "summary = run(\n"
            "    comp_dir, series_dir, Path(\"/kaggle/working\"),\n"
            "    n_studies=800, seed=0, workers=8,\n"
            ")\n"
            "print(\"VERDICT\", summary[\"verdict\"])\n"
            "print(\"raisons\", summary[\"reason\"])\n"
        ),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT / "diag.ipynb")
    meta = {
        "id": "sultent/rsna-knee-diag-slice-order",
        "title": "RSNA Knee diag slice order",
        "code_file": "diag.ipynb",
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
