# -*- coding: utf-8 -*-
"""URLs signees des kernels crop130_320. Ne pas committer la sortie."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_archive/kaggle_DICOM_exctract"))
from download_crop130_320 import list_files  # noqa: E402
from kaggle.api.kaggle_api_extended import KaggleApi

KERNELS = [
    "sultent/rsna-knee-convert-crop130-320-part0",
    "sultent/rsna-knee-convert-crop130-320-part1",
    "sultent/rsna-knee-convert-crop130-320-t1-part0",
    "sultent/rsna-knee-convert-crop130-320-t1-part1",
    "sultent/rsna-knee-convert-crop130-320-proto-part0",
    "sultent/rsna-knee-convert-crop130-320-proto-part1",
]


def dest_name(kernel: str, file_name: str) -> str:
    if file_name != "RECIPE.json":
        return file_name
    suffix = "part1" if kernel.endswith("part1") else "part0"
    if "-t1-" in kernel:
        return f"RECIPE_t1_{suffix}.json"
    if "-proto-" in kernel:
        return f"RECIPE_proto_{suffix}.json"
    if suffix == "part1":
        return "RECIPE_part1.json"
    return "RECIPE.json"


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    items = []
    for kernel in KERNELS:
        files = list_files(api, kernel)
        print(f"{kernel}: {len(files)} files", flush=True)
        if not files:
            print("AUCUN fichier", kernel, flush=True)
            continue
        for item in files:
            url = getattr(item, "url", None)
            if not url:
                raise SystemExit(f"no url for {kernel} {item.file_name}")
            items.append({
                "kernel": kernel,
                "name": dest_name(kernel, item.file_name),
                "url": url,
                "size": getattr(item, "total_bytes", None) or getattr(item, "size", None),
            })
    out = Path(sys.argv[1])
    out.write_text(json.dumps(items), encoding="utf-8")
    names = sorted(it["name"] for it in items)
    print("names", *names, sep="\n", flush=True)
    print("wrote", out, "n=", len(items), flush=True)


if __name__ == "__main__":
    main()
