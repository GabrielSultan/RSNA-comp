# -*- coding: utf-8 -*-
"""Paquet unique a envoyer au pod (local) : code, configs oai3, CSV, init_*.pt,
plus runpod/oai/expected_rsna.json (tailles exactes des zips RSNA locaux,
identiques aux sorties des kernels Kaggle). Scripts .sh passes en LF.

    python runpod/oai/bundle.py   ->  data/oai/_bundle/bundle.tar
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "oai" / "_bundle" / "bundle.tar"
FILES = [
    "data/csv/train.csv",
    "data/csv/train_series.csv",
    "data/oai/derived/oai_series.csv",
    "data/oai/derived/train_series_oai.csv",
    "data/oai/derived/targets_v3b_oofmask_oai.csv",
    "runpod/_pod_download_from_urls.py",
]
GLOBS = ["src/**/*.py", "configs/oai3/*.yaml", "runpod/oai/*.py", "runpod/oai/*.sh",
         "checkpoints/oai3_init/*.pt"]
RSNA_DIRS = ["data/crop130", "data/crop130_320"]


def expected_rsna() -> dict[str, dict[str, int]]:
    out = {}
    for d in RSNA_DIRS:
        files = sorted((ROOT / d).glob("volumes_*.zip")) + sorted((ROOT / d).glob("manifest_*.csv"))
        out[d] = {p.name: p.stat().st_size for p in files}
        if sum(n.endswith(".zip") for n in out[d]) != 6:
            raise SystemExit(f"{d} : 6 zips RSNA attendus en local")
    return out


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o755 if name.endswith(".sh") else 0o644
    tar.addfile(info, io.BytesIO(data))


def main() -> None:
    paths = [ROOT / f for f in FILES]
    for g in GLOBS:
        paths += sorted(p for p in ROOT.glob(g) if "__pycache__" not in p.parts)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"manquant : {missing}")
    if len(list(ROOT.glob("checkpoints/oai3_init/*.pt"))) != 4:
        raise SystemExit("4 init_*.pt attendus (python runpod/oai/dryrun.py --write-init)")

    exp = ROOT / "runpod/oai/expected_rsna.json"
    exp.write_text(json.dumps(expected_rsna(), indent=1), encoding="utf-8")
    if exp not in paths:
        paths.append(exp)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with tarfile.open(OUT, "w") as tar:
        for p in paths:
            name = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            if name.endswith(".sh"):
                data = data.replace(b"\r\n", b"\n")
            add_bytes(tar, name, data)
            total += len(data)
    print(f"{OUT.relative_to(ROOT)} : {len(paths)} fichiers, {total / 1e6:.0f} Mo "
          f"(tar {OUT.stat().st_size / 1e6:.0f} Mo)")


if __name__ == "__main__":
    main()
