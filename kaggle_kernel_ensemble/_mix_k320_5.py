# -*- coding: utf-8 -*-
"""Mix 5 modeles : softmax T=0.05 des AUC val subset, plancher 10 %."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

KERNEL = Path(__file__).resolve().parent


def _load_mix():
    spec = importlib.util.spec_from_file_location("mix3", KERNEL / "_mix_k320.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mix3 = _load_mix()
ROOT = mix3.ROOT
TARGETS = mix3.TARGETS
RUNS = [
    ("vit320", ROOT / "checkpoints" / "subset_vit_k_llrd08_det_320_frac025" / "history.json"),
    ("v20b", ROOT / "checkpoints" / "subset_v20b_crop130_det_frac025" / "history.json"),
    ("hdino", ROOT / "checkpoints" / "subset_v20h_crop130_dino_ep11_frac025" / "history.json"),
    ("hlat", ROOT / "checkpoints" / "subset_v20h_crop130_det_frac025" / "history.json"),
    ("vitk", ROOT / "checkpoints" / "subset_vit_k_llrd08_det_frac025" / "history.json"),
]


def mix_rows() -> list[tuple[str, tuple[float, float, float, float, float]]]:
    recs = [mix3.best(path) for _name, path in RUNS]
    out = []
    for t in TARGETS:
        auc = np.array([float(r[f"auc_{t}"]) for r in recs], dtype=np.float64)
        w = mix3.softmax_floor(auc)
        out.append((t, tuple(float(x) for x in w)))
    return out


def main() -> None:
    recs = []
    for name, path in RUNS:
        rec = mix3.best(path)
        recs.append(rec)
        print(name, "epoch", rec["epoch"], "val", round(float(rec["val_macro_auc"]), 4))
    for t, w in mix_rows():
        auc = np.array([float(r[f"auc_{t}"]) for r in recs], dtype=np.float64)
        nums = " ".join(f"{a:7.4f}" for a in auc)
        ws = " ".join(f"{x:.2f}" for x in w)
        print(f"{t:<18} {nums}  {ws}")


if __name__ == "__main__":
    main()
