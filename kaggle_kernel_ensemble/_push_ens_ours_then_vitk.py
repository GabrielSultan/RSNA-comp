# -*- coding: utf-8 -*-
"""Relance rsna-knee-ens-oai3swa-ours, puis resoumet le mix5 decale vers ViT-k."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup_vitk(mix5):
    vitk = _load("mix5_vitk_setup", KERNEL / "_push_ens_mix5_vitk.py")
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
    mix5.mix_rows = lambda: vitk.tilt(base_rows())

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


def _already(exc: BaseException) -> bool:
    low = str(exc).lower()
    return "already" in low or "duplicate" in low


def main() -> None:
    ours = _load("ours", KERNEL / "_push_ens_oai3swa_ours.py")
    ours.MAX_WAIT_S = 6 * 3600
    print("REPRISE ours", flush=True)
    ours.main()

    mix5 = _load("mix5_hybrid_chain", KERNEL / "_push_ens_mix5_hybrid.py")
    mix5.MAX_WAIT_S = 6 * 3600
    _setup_vitk(mix5)
    mix5.write_nb()
    version = (mix5.PUSH / "_version.txt").read_text(encoding="utf-8").strip()
    print("ENCHAINE vitk version", version, flush=True)
    try:
        mix5.submit(version)
    except Exception as exc:
        if not _already(exc):
            raise
        print("version deja soumise, nouveau push", flush=True)
        version = mix5.push()
        mix5.wait()
        mix5.submit(version)


if __name__ == "__main__":
    main()
