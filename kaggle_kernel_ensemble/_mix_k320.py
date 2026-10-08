# -*- coding: utf-8 -*-
"""Mix par finding : AUC val du best subset, softmax T=0.05, plancher 10%."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
RUNS = [
    ("vit", ROOT / "checkpoints" / "subset_vit_k_llrd08_det_320_frac025" / "history.json"),
    ("lat", ROOT / "checkpoints" / "subset_v20b_crop130_det_frac025" / "history.json"),
    ("dino", ROOT / "checkpoints" / "subset_v20h_crop130_dino_ep11_frac025" / "history.json"),
]
TEMP = 0.05
FLOOR = 0.10


def best(path: Path) -> dict:
    hist = json.loads(path.read_text(encoding="utf-8"))
    return max(hist, key=lambda r: float(r["val_macro_auc"]))


def softmax_floor(auc: np.ndarray) -> np.ndarray:
    z = auc / TEMP
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()
    free = np.ones(len(w), dtype=bool)
    for _ in range(len(w)):
        low = (w < FLOOR - 1e-9) & free
        if not np.any(low):
            break
        w = w.copy()
        w[low] = FLOOR
        free = free & ~low
        rest = 1.0 - float(w[~free].sum())
        base = w[free]
        w[free] = rest * base / base.sum()
    r = np.round(w, 2)
    delta = round(1.0 - float(r.sum()), 2)
    r[int(np.argmax(r))] = round(float(r[int(np.argmax(r))] + delta), 2)
    if np.any(r < FLOOR - 1e-9) or abs(float(r.sum()) - 1.0) > 1e-6:
        raise SystemExit(f"mix invalide {r}")
    return r


def mix_rows() -> list[tuple[str, tuple[float, float, float]]]:
    recs = [best(path) for _name, path in RUNS]
    out = []
    for t in TARGETS:
        auc = np.array([float(r[f"auc_{t}"]) for r in recs], dtype=np.float64)
        w = softmax_floor(auc)
        out.append((t, (float(w[0]), float(w[1]), float(w[2]))))
    return out


def tilt_floor(auc: np.ndarray) -> np.ndarray:
    """Mix tranche. Ordre (vit, v20b, v20h). v20h est la base generaliste.

    Ecart tete/2e <= 0.01 : pas de switch. Plancher 10 % seulement si le
    dernier est a plus de 0.03. Sinon quasi egalite.
    0.01 < ecart <= 0.02 : tete 50 %, ou 60 % si le dernier est nettement derriere.
    0.02 < ecart <= 0.04 : tete 60 %.
    ecart > 0.04 : tete 70 %, deuxieme 20 %, dernier 10 %.
    """
    order = np.argsort(-auc)
    best, mid, last = (int(i) for i in order)
    gap = float(auc[best] - auc[mid])
    gap_last = float(auc[best] - auc[last])
    w = np.zeros(3, dtype=np.float64)
    generalist = 2
    far = gap_last > 0.03
    if gap <= 0.01:
        if far:
            w[last] = 0.10
            if generalist in (best, mid):
                w[generalist] = 0.50
                w[best if mid == generalist else mid] = 0.40
            else:
                w[best] = 0.50
                w[mid] = 0.40
        else:
            w[best], w[mid], w[last] = 0.36, 0.33, 0.31
    elif gap <= 0.02:
        if far:
            w[best], w[mid], w[last] = 0.60, 0.30, 0.10
        else:
            w[best], w[mid], w[last] = 0.50, 0.30, 0.20
    elif gap <= 0.04:
        if far:
            w[best], w[mid], w[last] = 0.60, 0.30, 0.10
        else:
            w[best], w[mid], w[last] = 0.60, 0.25, 0.15
    else:
        w[best], w[mid], w[last] = 0.70, 0.20, 0.10
    if abs(float(w.sum()) - 1.0) > 1e-9 or np.any(w < FLOOR - 1e-9):
        raise SystemExit(f"tilt invalide {w}")
    return w


def tilt_rows() -> list[tuple[str, tuple[float, float, float]]]:
    recs = [best(path) for _name, path in RUNS]
    out = []
    for t in TARGETS:
        auc = np.array([float(r[f"auc_{t}"]) for r in recs], dtype=np.float64)
        w = tilt_floor(auc)
        out.append((t, (float(w[0]), float(w[1]), float(w[2]))))
    return out


def main() -> None:
    recs = []
    for name, path in RUNS:
        rec = best(path)
        recs.append(rec)
        print(
            name, "epoch", rec["epoch"],
            "val", round(float(rec["val_macro_auc"]), 4),
            "gold", round(float(rec["gold_macro_auc"]), 4),
        )
    print(f"{'finding':<18} {'vit':>7} {'lat':>7} {'dino':>7}  {'w_vit':>6} {'w_lat':>6} {'w_dino':>6}  lead")
    rows = []
    for t in TARGETS:
        auc = np.array([float(r[f"auc_{t}"]) for r in recs], dtype=np.float64)
        w = softmax_floor(auc)
        lead = ("vit", "lat", "dino")[int(auc.argmax())]
        gap = float(auc.max() - np.partition(auc, -2)[-2])
        rows.append((t, auc, w, lead, gap))
        print(
            f"{t:<18} {auc[0]:7.4f} {auc[1]:7.4f} {auc[2]:7.4f}"
            f"  {w[0]:6.3f} {w[1]:6.3f} {w[2]:6.3f}  {lead} +{gap:.3f}"
        )
    # also print rounded weights that sum to 1
    print("--- rounded ---")
    for t, auc, w, lead, gap in rows:
        r = np.round(w, 2)
        r[int(np.argmax(w))] += round(1.0 - float(r.sum()), 2)
        r = np.round(r, 2)
        print(f"{t}: ({r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f})")


if __name__ == "__main__":
    main()
