# -*- coding: utf-8 -*-
"""Telecharge les sorties convert-rgb-2-5d-320 vers data/crop130_rgb25d_320."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from download_rgb25d_proto import download_one, list_files

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "crop130_rgb25d_320"
KERNELS = [
    f"sultent/rsna-knee-convert-rgb-2-5d-320-part{i}" for i in range(12)
]


def download_kernel(kernel: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("==", kernel, flush=True)
    items = list_files(api, kernel)
    if not items:
        raise RuntimeError(f"AUCUN fichier de sortie : {kernel}")
    suffix = kernel.rsplit("part", 1)[-1]
    for item in items:
        dest = OUT / item.file_name
        if item.file_name == "RECIPE.json":
            dest = OUT / f"RECIPE_320_part{suffix}.json"
        if dest.exists() and dest.stat().st_size > 1000:
            print("skip existant", dest.name, dest.stat().st_size, flush=True)
            continue
        print("download", item.file_name, "->", dest.name, flush=True)
        download_one(item.url, dest, label=dest.name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"download : {len(KERNELS)} kernels -> {OUT}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(download_kernel, k): k for k in KERNELS}
        for fut in as_completed(futs):
            fut.result()
            print("DONE", futs[fut], flush=True)


if __name__ == "__main__":
    main()
