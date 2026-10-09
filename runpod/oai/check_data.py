# -*- coding: utf-8 -*-
"""Donnees pretes pour un train (pod) : zips RSNA a la taille exacte
(expected_rsna.json), aucun .part, et les 2 shards OAI de chaque lot.

    python runpod/oai/check_data.py data/crop130 [--no-oai] [--wait]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SERIES = ROOT / "data/oai/derived/oai_series.csv"


def problems(zip_dir: str, oai: bool) -> list[str]:
    d = (ROOT / zip_dir).resolve()
    key = d.relative_to(ROOT).as_posix()
    expected = json.loads((ROOT / "runpod/oai/expected_rsna.json").read_text())[key]
    out = []
    for name, size in expected.items():
        p = d / name
        got = p.stat().st_size if p.is_file() else None
        if got != size:
            out.append(f"{name} {got} != {size}")
    out += [f"en cours : {p.name}" for p in d.glob("*.part")]
    if oai:
        prefix = "oai_crop130_320" if zip_dir.endswith("_320") else "oai_crop130"
        batches = sorted(pd.read_csv(SERIES, usecols=["batch"]).batch.unique())
        miss = [int(b) for b in batches if not (d / f"{prefix}_b{int(b):02d}.zip").is_file()]
        if miss:
            out.append(f"shards OAI manquants : {len(miss)}/{len(batches)} ({miss[:4]}...)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_dir")
    ap.add_argument("--no-oai", action="store_true")
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    last = 0.0
    while True:
        errs = problems(args.zip_dir, oai=not args.no_oai)
        if not errs:
            print(f"donnees OK : {args.zip_dir}", flush=True)
            return
        if not args.wait:
            print("\n".join(errs), flush=True)
            sys.exit(1)
        if time.time() - last > 600:
            last = time.time()
            print(f"{time.strftime('%H:%M:%S')} attente {args.zip_dir} : {errs[0]} "
                  f"(+{len(errs) - 1})", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
