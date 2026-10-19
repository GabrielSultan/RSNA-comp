# -*- coding: utf-8 -*-
"""Reprend l'attente et la soumission, sans nouveau push."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    which = sys.argv[1]
    if which == "solo":
        mod = load("cnx_solo", ROOT / "kaggle_kernel_cnx" / "_push_swa78910.py")
        gold, _by = mod._gold()
        print("REPRISE solo version 1 gold", round(gold, 4), flush=True)
        mod.wait_kernel()
        mod.submit("1", gold)
        return
    if which == "pub60":
        mod = load("cnx_pub60", ROOT / "kaggle_kernel_ensemble" / "_push_ens_cnx_pub60_tta.py")
        print("REPRISE pub60 version 2", flush=True)
        mod.wait()
        mod.submit("2")
        return
    raise SystemExit(which)


if __name__ == "__main__":
    main()
