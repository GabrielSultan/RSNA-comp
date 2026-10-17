# -*- coding: utf-8 -*-
"""Meme mix5 hybrid, avec 25 % de v20b et de v20h crop130 deplaces sur ViT-k."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SHIFT = 0.25


def _load():
    path = KERNEL / "_push_ens_mix5_hybrid.py"
    spec = importlib.util.spec_from_file_location("mix5_hybrid", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tilt(rows):
    out = []
    for finding, base in rows:
        w = np.array(base, dtype=np.float64)
        w[0] = w[0] + SHIFT * w[1] + SHIFT * w[3]
        w[1] = w[1] * (1.0 - SHIFT)
        w[3] = w[3] * (1.0 - SHIFT)
        rnd = np.round(w, 2)
        delta = round(1.0 - float(rnd.sum()), 2)
        rnd = rnd.copy()
        rnd[0] = round(float(rnd[0] + delta), 2)
        if abs(float(rnd.sum()) - 1.0) > 1e-6 or np.any(rnd < -1e-9):
            raise SystemExit(f"mix invalide {finding} {rnd}")
        if rnd[0] < base[0] - 1e-9 or rnd[1] > base[1] + 1e-9 or rnd[3] > base[3] + 1e-9:
            raise SystemExit(f"tilt inverse {finding} {base} -> {rnd}")
        if abs(rnd[2] - base[2]) > 1e-9 or abs(rnd[4] - base[4]) > 1e-9:
            raise SystemExit(f"distil ou coat a bouge {finding}")
        out.append((finding, tuple(float(v) for v in rnd)))
    return out


def main() -> None:
    mix5 = _load()
    base_rows = mix5.mix_rows
    base_md = mix5._markdown
    mix5.SLUG = "rsna-knee-ens-mix5-vitk"
    mix5.PUSH = KERNEL / "_push_ens_mix5_vitk"
    mix5.OUT = KERNEL / "_out_ens_mix5_vitk"
    mix5.KAGGLE_ID = f"sultent/{mix5.SLUG}"
    mix5.MSG = (
        "ens soft hybrid T=0.05 puis 25% de v20b et 25% de v20h crop130 "
        "deplaces sur ViT-k320 : ViT OAI3 SWA 1-3 + v20b OAI3 SWA 1-3 + "
        "v20h distil crop145 SWA 7-9 + v20h crop130 SWA 6-7 + CoAt OAI3 SWA 1-3, "
        "sans TTA, pas de flip, T4"
    )
    mix5.mix_rows = lambda: tilt(base_rows())

    def _markdown(rows):
        text = base_md(rows)
        old_title = "# RSNA Knee — rsna-knee-ens-mix5-hybrid"
        new_title = "# RSNA Knee — rsna-knee-ens-mix5-vitk"
        old = (
            "Mix souple de 5 modeles. Poids = softmax des AUC val hybrid, "
            "meilleure epoch, temperature 0.05, sans plancher."
        )
        new = (
            "Mix souple de 5 modeles (softmax hybrid T=0.05), puis **25 %** du poids "
            "de v20b et **25 %** du poids de v20h crop130 sont deplaces sur ViT-k320. "
            "v20h distil crop145 et CoAt-a restent au poids hybrid. Arrondi au centieme."
        )
        if old_title not in text or old not in text:
            raise SystemExit("markdown source inattendu")
        return text.replace(old_title, new_title).replace(old, new)

    mix5._markdown = _markdown
    mix5.main()


if __name__ == "__main__":
    main()
