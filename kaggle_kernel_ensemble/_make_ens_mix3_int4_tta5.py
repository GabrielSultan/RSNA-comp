# -*- coding: utf-8 -*-
"""Notebook mix3 INT4, TTA x5. Meme dataset que la version x3."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_kernel_ensemble" / "_push_ens_h025_mix3_int4" / "submit.ipynb"
DST = ROOT / "kaggle_kernel_ensemble" / "_push_ens_h025_mix3_int4_tta5"
SLUG = "rsna-knee-ens-oai3-mix3-int4-tta-x5"


def _set(cell: dict, text: str) -> None:
    text = text if text.endswith("\n") else text + "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    old = (
        "TTA_VARIANTS = (\n"
        "    (0.0, 1.00),\n"
        "    (5.0, 1.00),\n"
        "    (-5.0, 1.00),\n"
        ")"
    )
    new = (
        "TTA_VARIANTS = (\n"
        "    (0.0, 1.00),\n"
        "    (5.0, 1.00),\n"
        "    (-5.0, 1.00),\n"
        "    (0.0, 0.95),\n"
        "    (0.0, 1.05),\n"
        ")"
    )
    n_var = 0
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        text = text.replace(
            "**INT4** + **TTA x3** (identite, rotation +5°, rotation -5°).",
            "**INT4** + **TTA x5** (identite, rotation ±5°, scale 0.95 et 1.05).",
        )
        text = text.replace(
            "print('TTA             : x3 rotation ±5')",
            "print('TTA             : x5 rotation ±5 et scale 0.95/1.05')",
        )
        text = text.replace(
            "print(len(test), 'etudes | TTA x3 | pas de flip')",
            "print(len(test), 'etudes | TTA x5 | pas de flip')",
        )
        if old in text:
            text = text.replace(old, new)
            n_var += 1
        _set(cell, text)
    blob = "\n".join("".join(c["source"]) for c in nb["cells"])
    if n_var != 1 or "TTA x3" in blob or "(0.0, 1.05)" not in blob:
        raise SystemExit(f"patch incomplet n_var={n_var}")
    if "TTA x5" not in blob or "scale 0.95" not in blob:
        raise SystemExit("texte TTA x5 absent")
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
    )
    meta = {
        "id": f"sultent/{SLUG}",
        "title": SLUG,
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": ["sultent/rsna-knee-ens-oai3-mix3-int4"],
        "category_ids": [],
    }
    (DST / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("ok", DST)


if __name__ == "__main__":
    main()
