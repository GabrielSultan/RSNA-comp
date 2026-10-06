# -*- coding: utf-8 -*-
"""Telecharge les sorties T1 + proto 320 vers data/crop130_320.

Les zip fluid (volumes_crop130_320_part*) restent en place.
Les nouveaux fichiers :

  volumes_crop130_320_t1_part{0,1}.zip
  volumes_crop130_320_proto_part{0,1}.zip
  manifest_t1_part{0,1}.csv
  manifest_proto_part{0,1}.csv

  python -X utf8 kaggle_DICOM_exctract/download_crop130_320_missing.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "download_crop130_320", ROOT / "download_crop130_320.py",
)
_dl = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dl)

OUT = ROOT.parent / "data" / "crop130_320"
KERNELS = [
    "sultent/rsna-knee-convert-crop130-320-t1-part0",
    "sultent/rsna-knee-convert-crop130-320-t1-part1",
    "sultent/rsna-knee-convert-crop130-320-proto-part0",
    "sultent/rsna-knee-convert-crop130-320-proto-part1",
]


def _dest(kernel: str, file_name: str) -> Path:
    if file_name != "RECIPE.json":
        return OUT / file_name
    if "-t1-" in kernel:
        kind = "t1"
    elif "-proto-" in kernel:
        kind = "proto"
    else:
        raise SystemExit(f"kernel inattendu : {kernel}")
    suffix = "part1" if kernel.endswith("part1") else "part0"
    return OUT / f"RECIPE_{kind}_{suffix}.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    api = _dl.KaggleApi()
    api.authenticate()
    for kernel in KERNELS:
        print("==", kernel, flush=True)
        items = _dl.list_files(api, kernel)
        if not items:
            print("AUCUN fichier de sortie", flush=True)
            sys.exit(1)
        for item in items:
            dest = _dest(kernel, item.file_name)
            if dest.exists() and dest.stat().st_size > 1000:
                print("skip existant", dest, dest.stat().st_size, flush=True)
                continue
            print("download", item.file_name, "->", dest.name, flush=True)
            _dl.download_one(item.url, dest)


if __name__ == "__main__":
    main()
