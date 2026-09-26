# -*- coding: utf-8 -*-
"""Ensemble v20b ep8 + v20h ep7 + vit-k ep8. Min 10% par modele et par finding."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
# (v20b, v20h, vit-k). Softmax AUC subset / 0.05, plancher 0.10, renorm.
MIX = [
    (0.50, 0.29, 0.21),  # ACL
    (0.40, 0.50, 0.10),  # MCL
    (0.44, 0.42, 0.14),  # Medial Meniscus
    (0.28, 0.28, 0.44),  # Lateral Meniscus
    (0.33, 0.38, 0.29),  # Medial OA
    (0.38, 0.40, 0.22),  # Lateral OA
    (0.38, 0.51, 0.11),  # PF OA
    (0.30, 0.37, 0.33),  # Effusion
    (0.34, 0.31, 0.35),  # Synovitis
    (0.34, 0.32, 0.34),  # Baker's
    (0.33, 0.34, 0.33),  # Contusion
    (0.40, 0.46, 0.14),  # Fracture
]


def main() -> None:
    spec = importlib.util.spec_from_file_location("ens", KERNEL / "_make_nb_ens.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ens = mod._load_ens()
    mod._weights = lambda with_vite: MIX
    slug = "rsna-knee-ens-v20b-v20h-vitk"
    push = KERNEL / "_push_ens_v20h_vitk"
    mod._one(
        ens, True, slug, slug,
        [
            "sultent/rsna-knee-v20b-oofmask-ep8",
            "sultent/rsna-knee-v20h-ep7-lat",
            "sultent/rsna-knee-vit-k-llrd-ep8",
        ],
        push,
    )
    nb_path = push / "submit.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = src.replace("vit_e_ep7.pt", "vit_k_ep8.pt")
        src = src.replace(
            "../kaggle_dataset_ensemble/_upload/vit_k_ep8.pt",
            "../kaggle_dataset_vit_k/_upload_ep8/vit_k_ep8.pt",
        )
        src = src.replace("vit_e_crop130_oofmask_full", "vit_k_llrd08_det_oofmask_full")
        src = src.replace(
            "!= 'vit_k_llrd08_det_oofmask_full':\n"
            "        raise RuntimeError(cfg.get('run_name'))\n"
            "    if int(ckpt.get('epoch', -1)) != 7:",
            "!= 'vit_k_llrd08_det_oofmask_full':\n"
            "        raise RuntimeError(cfg.get('run_name'))\n"
            "    if int(ckpt.get('epoch', -1)) != 8:",
        )
        src = src.replace("ViT-e ne doit pas etre flipped", "ViT-k ne doit pas etre flipped")
        lines = src.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        cell["source"] = lines
    src = "".join(nb["cells"][0]["source"])
    # rewrite intro
    intro = """# RSNA Knee — Ensemble v20b ep8 + v20h lat ep7 + ViT-k ep8

Notebook **self-contained** (internet off, **T4**).

- **v20b** baseline : `v20b_crop130_oofmask_full` epoch 8. **Pas de flip.**
- **v20h** : `v20h_crop130_oofmask_v20b_lat_full` epoch 7. **Canon L.**
- **ViT-k** : `vit_k_llrd08_det_oofmask_full` epoch 8 (LLRD 0.8, `rgb_as_gray` = canal G). **Pas de flip.**

Mix = softmax des AUC val subset (notebook 08, epoch du best) avec temperature 0.05, puis plancher **10 %** et renormalisation. Les trois modeles sont sur chaque finding.

v20h voit le volume flippé. v20b et ViT-k voient le volume brut. Blend apres sigmoid.

TTA affine x5. Pas de hflip TTA.
"""
    nb["cells"][0]["source"] = [ln + "\n" for ln in intro.strip("\n").split("\n")]
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    # sanity
    import ast
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    blob = json.dumps(nb)
    assert "vit_k_llrd08_det_oofmask_full" in blob
    assert "vit_e_crop130" not in blob
    assert "0.44" in blob and "0.1" in blob
    print("ok", push)
    print("intro starts", src[:40])


if __name__ == "__main__":
    main()
