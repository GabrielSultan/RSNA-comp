# -*- coding: utf-8 -*-
"""Archive code + labels pour le pod 320. Pas les zips."""
from __future__ import annotations

import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__import__("os").environ["TEMP"]) / "rsna_vit320.tar"
PATHS = [
    "src",
    "configs/kneenet_vit_k_llrd08_320_v3b_oofmask_v20h_full.yaml",
    "data/labels/coolin/targets_v3b_oofmask.csv",
    "data/csv/train.csv",
    "data/csv/train_series.csv",
    "runpod/_pod_download_from_urls.py",
    "weights/dinov3-vitb16-lvd1689m/config.json",
]


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    with tarfile.open(OUT, "w") as tar:
        for rel in PATHS:
            path = ROOT / rel
            if path.is_dir():
                for file in path.rglob("*"):
                    if "__pycache__" in file.parts or file.suffix == ".pyc":
                        continue
                    if file.is_file():
                        tar.add(file, arcname=str(file.relative_to(ROOT)).replace("\\", "/"))
            else:
                tar.add(path, arcname=rel.replace("\\", "/"))
    print(OUT, OUT.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
