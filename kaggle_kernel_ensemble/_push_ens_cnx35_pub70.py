# -*- coding: utf-8 -*-
"""Meme mix que cnx35-hflip, sans TTA, rang 70/30 contre degnonguidi.

Push du kernel seulement. Pas de soumission competition.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_cnx35_hflip" / "submit.ipynb"
PUBLIC = KERNEL / "_public_degnonguidi" / "rnsa-best-public-score.ipynb"
SLUG = "rsna-knee-ens-cnx35-pub70"
PUSH = KERNEL / "_push_ens_cnx35_pub70"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
MSG = (
    "ens CNX-r7 SWA 7-10 plancher 35% + specialistes soft "
    "(ViT-k320 OAI3 SWA 1-3, v20b OAI SWA 1-3, v20h distil crop145 SWA 7-9, "
    "CoAt OAI SWA 1-3), mix lineaire 70% + public degnonguidi 30%, rang, "
    "sans TTA, pas de flip, pas de rotation, T4"
)
MD = """# RSNA Knee — rsna-knee-ens-cnx35-pub70

**Sans TTA.** Internet off. La session a **deux T4** : le CNX sur cuda:0, les autres modeles sur cuda:1.
Pas de flip, pas de rotation.

1. Meme mix que `rsna-knee-ens-cnx35-hflip`, sans le second passage CNX.
   CNX-r7 SWA 7-10 a au moins **35 %** sur chaque finding. Les 4 specialistes
   se partagent le reste (porte 0.015, softmax T=0.05).
   ViT-k320 OAI3 SWA 1-3, entree 320, FOV 130 mm.
   v20b OAI SWA 1-3, entree 224, FOV 130 mm.
   v20h distil crop145 SWA 7-9, entree 224, FOV 145 mm.
   CoAt-a OAI SWA 1-3, entree 224, FOV 130 mm.
   CNX-r7 en float32, un seul passage. Budget 3 h, puis GPU libere.
2. Notebook public `degnonguidi/rnsa-best-public-score`
   (Raptor 384 + Raptor 224, A224=0.35, puis ConvNeXt R6).
   Echeance 8.3 h depuis le debut du notebook.
3. Fusion par rangs, **70 %** le mix et **30 %** la solution publique.
"""


def _pub():
    path = KERNEL / "_push_ens_cnx_pub70.py"
    spec = importlib.util.spec_from_file_location("pub70_for_cnx35", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_nb() -> None:
    pub = _pub()
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    if len(nb["cells"]) != 10 or len(public["cells"]) != 7:
        raise SystemExit("notebooks source inattendus")
    nb["cells"][0] = pub._md(MD)
    pub._replace_nb(
        nb,
        "print('TTA             : cnx hflip x1, pas de rotation, autres x1')",
        "print('TTA             : non')",
        1,
    )
    pub._replace_nb(
        nb,
        "print(len(test), 'etudes | TTA cnx hflip x1 | pas de rotation | "
        "FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        "print(len(test), 'etudes | sans TTA | pas de flip | pas de rotation | "
        "FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        1,
    )
    pub._replace_nb(
        nb,
        "def _hflip(x):\n"
        "    \"\"\"Retourner l'image, sans rotation. Sagittal : axe D. Coronal/Axial : axe W.\"\"\"\n"
        "    if x.ndim != 5 or int(x.shape[1]) != 6:\n"
        "        raise RuntimeError(tuple(x.shape))\n"
        "    y = x.clone()\n"
        "    for s in (0, 3):\n"
        "        y[:, s] = torch.flip(y[:, s], dims=(1,))\n"
        "    for s in (1, 2, 4, 5):\n"
        "        y[:, s] = torch.flip(y[:, s], dims=(3,))\n"
        "    return y\n"
        "\n"
        "\n"
        "def _swap_medlat(p):\n"
        "    q = p.clone()\n"
        "    q[:, 2] = p[:, 3]\n"
        "    q[:, 3] = p[:, 2]\n"
        "    q[:, 4] = p[:, 5]\n"
        "    q[:, 5] = p[:, 4]\n"
        "    return q\n"
        "\n"
        "\n"
        "print(\n"
        "    'TTA cnx identite + hflip, swap menisque et OA medial/lateral, pas de rotation, autres x1',\n"
        "    flush=True,\n"
        ")\n",
        "",
        1,
    )
    pub._replace_nb(
        nb,
        "def _probs_cnx(x_224):\n"
        "    with torch.no_grad():\n"
        "        if str(DEV_CNX).startswith('cuda'):\n"
        "            torch.cuda.set_device(DEV_CNX)\n"
        "        x = x_224.to(DEV_CNX).float()\n"
        "        with torch.autocast('cuda', enabled=False):\n"
        "            p_id = torch.sigmoid(model_cnx(x).float())\n"
        "            p_flip = torch.sigmoid(model_cnx(_hflip(x)).float())\n"
        "        return (0.5 * (p_id + _swap_medlat(p_flip))).float().cpu()\n",
        "def _probs_cnx(x_224):\n"
        "    with torch.no_grad():\n"
        "        if str(DEV_CNX).startswith('cuda'):\n"
        "            torch.cuda.set_device(DEV_CNX)\n"
        "        x = x_224.to(DEV_CNX).float()\n"
        "        with torch.autocast('cuda', enabled=False):\n"
        "            p_cnx = torch.sigmoid(model_cnx(x).float())\n"
        "        return p_cnx.float().cpu()\n",
        1,
    )
    pub._replace_nb(
        nb,
        "    for i, uid in enumerate(test.StudyInstanceUID):\n",
        "    for i, uid in enumerate(test.StudyInstanceUID):\n"
        "        if time.time() - NB_T0 > MIX5_BUDGET_S:\n"
        "            print(f'MIX5_BUDGET {i}/{len(test)}', flush=True)\n"
        "            break\n",
        1,
    )
    last = pub._text(nb["cells"][-1])
    if "submission.csv au format exact" not in last:
        raise SystemExit("derniere cellule inattendue")
    nb["cells"].pop()
    cells = [nb["cells"][0], pub._code(pub.CLOCK)]
    cells.extend(nb["cells"][1:])
    save = pub.SAVE.replace(
        "if torch.cuda.is_available():\n    torch.cuda.empty_cache()\n",
        "if torch.cuda.is_available():\n"
        "    for _i in range(torch.cuda.device_count()):\n"
        "        with torch.cuda.device(_i):\n"
        "            torch.cuda.empty_cache()\n",
    )
    cells.append(pub._code(save))
    pub._patch_public(public["cells"])
    cells.extend(public["cells"])
    cells.append(pub._code(pub.BLEND))
    for cell in cells:
        src = pub._text(cell)
        if cell["cell_type"] != "code" or src.lstrip().startswith("%%"):
            continue
        ast.parse(src)
    blob = "\n".join(pub._text(c) for c in cells)
    for needle in (
        "(0.00, 0.00, 0.65, 0.00, 0.35),  # ACL",
        "(0.00, 0.32, 0.32, 0.00, 0.36),  # MCL",
        "v20b_dino_oai3_swa_ep1_3.pt",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3', 'swa-1-2-3'",
        "v20h_crop145_distil_swa_ep7_9.pt",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "cnx_r7_swa_ep7_10.pt",
        "swa-7-8-9-10",
        "cnx forward fp32",
        "gpus visibles",
        "cuda:1",
        "sans TTA",
        "pas de rotation",
        "print('TTA             : non')",
        "w[:, 4] * p_cnx",
        "MIX5_BUDGET_S = 3.0 * 3600",
        "echeance publique h 8.3",
        "DEADLINE_HOURS = 8.3",
        "A224 = 0.35",
        "T0 = NB_T0",
        "W_MIX5, W_PUBLIC = 0.70, 0.30",
        "BLEND70_OK",
        "PUBLIC_OURS_PARTIAL",
        "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt",
        "laterality=False",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "_hflip",
        "_swap_medlat",
        "p_flip",
        "v20b_dino_full_swa_ep6_9.pt",
        "swa-6-7-8-9",
        "laterality=True",
        "TTA_VARIANTS",
        "apply_affine",
        "W_MIX5, W_PUBLIC = 0.60, 0.40",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    if SLUG not in blob:
        raise SystemExit("slug absent")
    out = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": KAGGLE_ID,
        "title": SLUG,
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": ACCEL,
        "competition_sources": [COMPETITION],
        "dataset_sources": [
            "heliosli/rsna-knee-r6-frozen-fe055406-w1/1",
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
            "sultent/rsna-knee-vit-k320-oai3-swa",
            "sultent/rsna-knee-v20b-dino-oai3-swa",
            "sultent/rsna-knee-v20h-crop145-distil-swa789",
            "sultent/rsna-knee-coat-a-oai3-swa",
            "sultent/rsna-knee-cnx-r7-swa78910",
        ],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(cells), flush=True)


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        raise


def push() -> str:
    while True:
        try:
            out = kaggle("kernels", "push", "-p", str(PUSH), "--accelerator", ACCEL, "-t", "32400")
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            low = text.lower()
            if "quota" in low or "maximum batch gpu" in low:
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if "quota" in out.lower() or "maximum batch gpu" in out.lower():
            print("quota GPU, retry 60s", flush=True)
            time.sleep(60)
            continue
        if "invalid dataset" in out.lower():
            raise SystemExit(out)
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        version = m.group(1)
        (PUSH / "_version.txt").write_text(version + "\n", encoding="utf-8")
        print("kernel version", version, flush=True)
        print("PAS_DE_SOUMISSION", flush=True)
        return version


def main() -> None:
    write_nb()
    if "--write" in sys.argv:
        return
    push()


if __name__ == "__main__":
    main()
