# -*- coding: utf-8 -*-
"""Mix tta-rs avec un passage en moins sur le ViT et sur le distil. Deux T4 obligatoires.

CNX cuda:0, 3 vues. ViT cuda:1, 2 vues. Distil cuda:1, 4 vues.
Pousse le kernel et verifie le commit. Pas de soumission competition.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_cnx_tta_rs" / "submit.ipynb"
SLUG = "rsna-knee-ens-cnx-tta-rs24"
PUSH = KERNEL / "_push_ens_cnx_tta_rs24"
OUT = KERNEL / "_out_ens_cnx_tta_rs24"
KAGGLE_ID = f"sultent/{SLUG}"
MSG = (
    "ens CNX-r7 SWA 7-10 + ViT-k320 OAI3 SWA 1-3 + v20h distil crop145 SWA 7-9. "
    "CNX 3 : rotation +5, rotation -5, scale 1.05. "
    "ViT 2 : rotation +5, rotation -5. "
    "Distil 4 : +5, -5, scale 1.05, scale 0.95. "
    "CNX cuda:0, ViT et distil cuda:1. Sans hflip."
)


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _replace(nb: dict, old: str, new: str, expect: int) -> None:
    found = 0
    for cell in nb["cells"]:
        src = _text(cell)
        n = src.count(old)
        if not n:
            continue
        found += n
        _set(cell, src.replace(old, new))
    if found != expect:
        raise SystemExit(f"remplacement x{found}, attendu {expect}: {old[:80]!r}")


def write_nb() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    _replace(nb, "rsna-knee-ens-cnx-tta-rs", SLUG, 1)
    _replace(
        nb,
        "  Entree 320, FOV 130 mm. **Memes 3 passages que le CNX** :\n"
        "  rotation +5, rotation -5, scale 1.05.\n",
        "  Entree 320, FOV 130 mm. **2 passages** : rotation +5, rotation -5.\n",
        1,
    )
    _replace(
        nb,
        "  Entree 224, FOV 145 mm. **5 passages** : rotation +5, rotation -5,\n"
        "  scale 1.05, scale 0.95, rotation +5 avec scale 1.05.\n",
        "  Entree 224, FOV 145 mm. **4 passages** : rotation +5, rotation -5,\n"
        "  scale 1.05, scale 0.95.\n",
        1,
    )
    _replace(
        nb,
        "print('TTA             : cnx 3, vit 3, distil 5, rotation scale, 2 GPU')",
        "print('TTA             : cnx 3 cuda:0, vit 2 cuda:1, distil 4 cuda:1')",
        1,
    )
    _replace(
        nb,
        "print(len(test), 'etudes | cnx 3 vit 3 distil 5 | rotation +5 rotation -5 "
        "scale 0.95 scale 1.05 | 2 GPU | sans hflip | FOV vit/cnx 130 mm, v20h 145 mm | "
        "cnx forward fp32')",
        "print(len(test), 'etudes | cnx 3 vit 2 distil 4 | rotation +5 rotation -5 "
        "scale 0.95 scale 1.05 | 2 GPU | cnx cuda:0 vit distil cuda:1 | sans hflip | "
        "FOV vit/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        1,
    )
    _replace(
        nb,
        "VIEWS_3 = ((5.0, 1.0), (-5.0, 1.0), (0.0, 1.05))\n"
        "VIEWS_5 = ((5.0, 1.0), (-5.0, 1.0), (0.0, 1.05), (0.0, 0.95), (5.0, 1.05))\n"
        "if len(VIEWS_3) != 3 or len(VIEWS_5) != 5:\n"
        "    raise RuntimeError((len(VIEWS_3), len(VIEWS_5)))\n"
        "print(\n"
        "    'TTA cnx 3 vit 3 distil 5 | rotation +5 rotation -5 scale 1.05 | '\n"
        "    'distil ajoute scale 0.95 et rotation +5 avec scale 1.05 | '\n"
        "    'sans hflip, sans echange de labels',\n"
        "    flush=True,\n"
        ")\n",
        "VIEWS_CNX = ((5.0, 1.0), (-5.0, 1.0), (0.0, 1.05))\n"
        "VIEWS_VIT = ((5.0, 1.0), (-5.0, 1.0))\n"
        "VIEWS_DISTIL = ((5.0, 1.0), (-5.0, 1.0), (0.0, 1.05), (0.0, 0.95))\n"
        "if (len(VIEWS_CNX), len(VIEWS_VIT), len(VIEWS_DISTIL)) != (3, 2, 4):\n"
        "    raise RuntimeError((len(VIEWS_CNX), len(VIEWS_VIT), len(VIEWS_DISTIL)))\n"
        "print(\n"
        "    'TTA cnx 3 vit 2 distil 4 | cnx cuda:0 | vit distil cuda:1 | '\n"
        "    'sans hflip, sans echange de labels',\n"
        "    flush=True,\n"
        ")\n",
        1,
    )
    _replace(
        nb,
        "N_GPU = torch.cuda.device_count() if DEVICE == 'cuda' else 1\n"
        "if DEVICE == 'cuda' and N_GPU < 2:\n"
        "    raise RuntimeError(('attendu 2 GPU', N_GPU))\n"
        "DEV_CNX = 'cuda:0' if DEVICE == 'cuda' else 'cpu'\n"
        "DEV_REST = 'cuda:1' if N_GPU >= 2 else DEV_CNX\n",
        "N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0\n"
        "if N_GPU < 2:\n"
        "    raise RuntimeError(('attendu 2 GPU', N_GPU))\n"
        "DEV_CNX = 'cuda:0'\n"
        "DEV_REST = 'cuda:1'\n",
        1,
    )
    _replace(
        nb,
        "            ps = [torch.sigmoid(model_cnx(_affine(x_cnx, deg, sc)).float()) for deg, sc in VIEWS_3]\n",
        "            ps = [torch.sigmoid(model_cnx(_affine(x_cnx, deg, sc)).float()) for deg, sc in VIEWS_CNX]\n",
        1,
    )
    _replace(
        nb,
        "        p_vit = _avg(model_vit, x_vit, VIEWS_3)\n"
        "        p_v20h = _avg(model_v20h, x_h, VIEWS_5)\n",
        "        p_vit = _avg(model_vit, x_vit, VIEWS_VIT)\n"
        "        p_v20h = _avg(model_v20h, x_h, VIEWS_DISTIL)\n",
        1,
    )
    _replace(
        nb,
        "_POOL = None\n"
        "if DEV_REST != DEV_CNX:\n"
        "    from concurrent.futures import ThreadPoolExecutor\n"
        "    _POOL = ThreadPoolExecutor(max_workers=2)\n",
        "from concurrent.futures import ThreadPoolExecutor\n"
        "_POOL = ThreadPoolExecutor(max_workers=2)\n",
        1,
    )
    _replace(
        nb,
        "        if _POOL is None:\n"
        "            p_cnx = _probs_cnx(x_224_cpu)\n"
        "            p_vit, p_v20h = _probs_rest(x_vit_cpu, x_h_cpu)\n"
        "        else:\n"
        "            fa = _POOL.submit(_probs_cnx, x_224_cpu)\n"
        "            fb = _POOL.submit(_probs_rest, x_vit_cpu, x_h_cpu)\n"
        "            p_cnx = fa.result()\n"
        "            p_vit, p_v20h = fb.result()\n",
        "        fa = _POOL.submit(_probs_cnx, x_224_cpu)\n"
        "        fb = _POOL.submit(_probs_rest, x_vit_cpu, x_h_cpu)\n"
        "        p_cnx = fa.result()\n"
        "        p_vit, p_v20h = fb.result()\n",
        1,
    )
    for cell in nb["cells"]:
        src = _text(cell)
        if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
            ast.parse(src)
    blob = "\n".join(_text(c) for c in nb["cells"])
    for needle in (
        SLUG,
        "DEV_CNX = 'cuda:0'",
        "DEV_REST = 'cuda:1'",
        "attendu 2 GPU",
        "VIEWS_CNX = ((5.0, 1.0), (-5.0, 1.0), (0.0, 1.05))",
        "VIEWS_VIT = ((5.0, 1.0), (-5.0, 1.0))",
        "VIEWS_DISTIL = ((5.0, 1.0), (-5.0, 1.0), (0.0, 1.05), (0.0, 0.95))",
        "ThreadPoolExecutor(max_workers=2)",
        "_POOL.submit(_probs_cnx",
        "_POOL.submit(_probs_rest",
        "cnx forward fp32",
        "(0.00, 0.40, 0.60),  # ACL",
        "submission.csv",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "VIEWS_3",
        "VIEWS_5",
        "DEV_REST = 'cuda:1' if",
        "_POOL is None",
        "_hflip",
        "degnonguidi",
        "laterality=True",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((KERNEL / "_push_ens_cnx_tta_rs" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KAGGLE_ID
    meta["title"] = SLUG
    meta["machine_shape"] = "NvidiaTeslaT4"
    meta["enable_gpu"] = True
    meta["enable_internet"] = False
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(nb["cells"]), flush=True)


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "push_rs", KERNEL / "_push_ens_cnx_tta_rs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    write_nb()
    if "--write" in sys.argv:
        return
    base = _load_base()
    base.SLUG = SLUG
    base.PUSH = PUSH
    base.OUT = OUT
    base.KAGGLE_ID = KAGGLE_ID
    base.MSG = MSG
    version = base.push()
    base.wait()
    OUT.mkdir(parents=True, exist_ok=True)
    print(base.kaggle("kernels", "output", KAGGLE_ID, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and SLUG in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3 gold 0.9903",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10 lat False gold 0.976",
        "cnx 3 vit 2 distil 4",
        "gpus visibles 2",
        "cnx cuda:0",
        "cuda:1",
        "lignes 3",
        "cnx forward fp32",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "gpus visibles 1" in text or "lat True" in text:
        raise SystemExit("GPU incomplet")
    print("LOG_OK", log.name, "version", version, flush=True)
    print("PAS_DE_SOUMISSION", KAGGLE_ID, flush=True)


if __name__ == "__main__":
    main()
