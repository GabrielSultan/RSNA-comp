# -*- coding: utf-8 -*-
"""URLs signees des 6 kernels rgb25d proto (folds OOF ViT-k sur pod).
Sortie a dumper juste avant le demarrage du pod (URLs perimees sinon).
Ne pas committer la sortie.

    python runpod/_dump_rgb25d_urls.py urls_rgb25d.json
puis sur le pod :
    python runpod/_pod_download_from_urls.py urls_rgb25d.json /workspace/data/crop130_rgb25d
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_archive/kaggle_DICOM_exctract"))
from download_rgb25d_proto import list_files  # noqa: E402
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

KERNELS = [f"sultent/rsna-knee-convert-rgb-2-5d-proto-part{i}" for i in range(6)]


def dest_name(kernel: str, file_name: str) -> str:
    if file_name != "RECIPE.json":
        return file_name
    return f"RECIPE_proto_part{kernel.rsplit('part', 1)[1]}.json"


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
