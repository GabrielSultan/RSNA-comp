# -*- coding: utf-8 -*-
"""Dump signed Kaggle kernel-output URLs for the pod (local auth only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_DICOM_exctract"))
from download_rgb25d_proto import KERNELS, list_files  # noqa: E402
from kaggle.api.kaggle_api_extended import KaggleApi

def main() -> None:
    api = KaggleApi()
    api.authenticate()
    items = []
    for kernel in KERNELS:
        suffix = kernel.rsplit("part", 1)[-1]
        files = list_files(api, kernel)
        print(f"{kernel}: {len(files)} files", flush=True)
        for item in files:
            name = item.file_name
            if name == "RECIPE.json":
                name = f"RECIPE_proto_part{suffix}.json"
            url = getattr(item, "url", None)
            size = getattr(item, "total_bytes", None) or getattr(item, "size", None)
            if not url:
                raise SystemExit(f"no url for {kernel} {item.file_name}")
            items.append({"kernel": kernel, "name": name, "url": url, "size": size})
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kaggle_urls.json")
    out.write_text(json.dumps(items), encoding="utf-8")
    print("wrote", out, "n=", len(items), flush=True)

if __name__ == "__main__":
    main()
