# -*- coding: utf-8 -*-
"""Apercu des mixes hybrid025, softmax T=0.05, sans plancher."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMP = 0.05
NAMES = (
    ("ViT-k320", "subset_vit_k_llrd08_320_v3b_hybrid025"),
    ("v20b DINO", "subset_v20b_dino_ep11_v3b_hybrid025"),
    ("v20h DINO", "subset_v20h_dino_ep11_v3b_hybrid025"),
    ("CoAt-a", "subset_coat_a_llrd08_v3b_hybrid025"),
)
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def best(path: Path) -> dict:
    hist = json.loads(path.read_text(encoding="utf-8"))
    return max(hist, key=lambda r: float(r["val_macro_auc"]))


def softmax_round(auc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = auc / TEMP
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()
    r = np.round(w, 2)
    delta = round(1.0 - float(r.sum()), 2)
    r = r.copy()
    r[int(np.argmax(r))] = round(float(r[int(np.argmax(r))] + delta), 2)
    if abs(float(r.sum()) - 1.0) > 1e-6 or np.any(r < -1e-9):
        raise SystemExit(f"mix invalide {r}")
    return w, r


def _recs_and_aucs() -> tuple[list[dict], list[np.ndarray]]:
    recs = []
    for _label, name in NAMES:
        recs.append(best(ROOT / "checkpoints" / name / "history.json"))
    aucs = [
        np.array([float(r[f"auc_{t}"]) for r in recs], dtype=np.float64)
        for t in TARGETS
    ]
    return recs, aucs


def _as_rows(aucs: list[np.ndarray], idx: list[int]) -> list[tuple[str, tuple[float, ...]]]:
    out = []
    for t, row in zip(TARGETS, aucs):
        _raw, rnd = softmax_round(row[idx])
        out.append((t, tuple(float(v) for v in rnd)))
    return out


def ensemble_rows() -> dict[str, list[tuple[str, tuple[float, ...]]]]:
    """Poids arrondis valides : mix4, mix3 (sans CoAt), spec (porte 0.015, T=0.02)."""
    _recs, aucs = _recs_and_aucs()
    spec = []
    for t, row in zip(TARGETS, aucs):
        keep = row >= (float(row.max()) - 0.015 - 1e-12)
        rnd = softmax_among(row, keep, 0.02)
        spec.append((t, tuple(float(v) for v in rnd)))
    return {
        "mix4": _as_rows(aucs, [0, 1, 2, 3]),
        "mix3": _as_rows(aucs, [0, 1, 2]),
        "spec": spec,
    }


def main() -> None:
    recs = []
    for label, name in NAMES:
        rec = best(ROOT / "checkpoints" / name / "history.json")
        recs.append(rec)
        print(
            f"{label:12} best ep {int(rec['epoch'])} "
            f"val {float(rec['val_macro_auc']):.4f}"
        )
    aucs = []
    print()
    print(f"{'finding':<18}" + "".join(f"{n[0]:>12}" for n in NAMES))
    for t in TARGETS:
        row = np.array([float(r[f"auc_{t}"]) for r in recs], dtype=np.float64)
        aucs.append(row)
        print(f"{t:<18}" + "".join(f"{v:12.4f}" for v in row))

    def show(title: str, idx: list[int]) -> None:
        labels = [NAMES[i][0] for i in idx]
        print()
        print(title)
        print(f"{'finding':<18}" + "".join(f"{lb:>12}" for lb in labels))
        for t, row in zip(TARGETS, aucs):
            raw, rnd = softmax_round(row[idx])
            pct = "".join(f"{int(round(v * 100)):11d}%" for v in rnd)
            raw_s = " ".join(f"{v:.3f}" for v in raw)
            print(f"{t:<18}{pct}   raw {raw_s}")

    show("MIX 4", [0, 1, 2, 3])
    show("MIX 3 sans CoAt", [0, 1, 2])
    show_specialist(aucs)


def softmax_among(auc: np.ndarray, keep: np.ndarray, temp: float) -> np.ndarray:
    w = np.zeros(len(auc), dtype=np.float64)
    z = auc[keep] / temp
    z = z - z.max()
    e = np.exp(z)
    e = e / e.sum()
    w[keep] = e
    r = np.round(w, 2)
    delta = round(1.0 - float(r.sum()), 2)
    if abs(delta) > 0:
        alive = np.flatnonzero(r > 0)
        if len(alive) == 0:
            alive = np.flatnonzero(keep)
        j = alive[int(np.argmax(r[alive]))]
        r[j] = round(float(r[j] + delta), 2)
    if abs(float(r.sum()) - 1.0) > 1e-6 or np.any(r < -1e-9):
        raise SystemExit(f"mix specialiste invalide {r}")
    return r


def show_specialist(aucs: list[np.ndarray]) -> None:
    """Tete = meilleur AUC. Au-dela de 0.015 sous la tete : 0.
    Les survivants se partagent en softmax T=0.02."""
    gate = 0.015
    temp = 0.02
    labels = [n[0] for n in NAMES]
    print()
    print(f"MIX 4 specialiste  gate {gate}  T {temp}")
    print(f"{'finding':<18}" + "".join(f"{lb:>12}" for lb in labels))
    for t, row in zip(TARGETS, aucs):
        keep = row >= (row.max() - gate - 1e-12)
        rnd = softmax_among(row, keep, temp)
        pct = "".join(f"{int(round(v * 100)):11d}%" for v in rnd)
        print(f"{t:<18}{pct}")


if __name__ == "__main__":
    main()
