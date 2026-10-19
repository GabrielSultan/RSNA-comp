# -*- coding: utf-8 -*-
"""Meme mix que cnx-base-spec, sans TTA, rang 70/30 contre degnonguidi."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_cnx_base_spec" / "submit.ipynb"
PUBLIC = KERNEL / "_public_degnonguidi" / "rnsa-best-public-score.ipynb"
SLUG = "rsna-knee-ens-cnx-base-pub70"
PUSH = KERNEL / "_push_ens_cnx_base_pub70"
OUT = KERNEL / "_out_ens_cnx_base_pub70"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
ACL = "vit 0.00 v20b 0.00 v20h 0.40 coat 0.00 cnx 0.60"
MSG = (
    "ens CNX-r7 SWA 7-10 baseline 60% + specialistes soft 40% "
    "(ViT-k320 OAI3 SWA 1-3, v20b full SWA 6-9, v20h distil crop145 SWA 7-9, "
    "CoAt OAI SWA 1-3), mix lineaire 70% + public degnonguidi 30%, rang, "
    "sans TTA, pas de flip, T4"
)
MD = """# RSNA Knee — rsna-knee-ens-cnx-base-pub70

**Sans TTA.** Internet off, **T4**. Pas de flip.

1. Meme mix que `rsna-knee-ens-cnx-base-spec`, sans le second passage CNX.
   CNX-r7 SWA 7-10 est la baseline **60 %**. Les 4 specialistes se partagent
   **40 %** (porte 0.015, softmax T=0.05, residu sur le CNX).
   ViT-k320 OAI3 SWA 1-3, entree 320, FOV 130 mm.
   v20b full SWA 6-9, entree 224, FOV 130 mm.
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
    spec = importlib.util.spec_from_file_location("pub70_for_base", path)
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
        "print('TTA             : cnx flip lateral x1, autres x1')",
        "print('TTA             : non')",
        1,
    )
    pub._replace_nb(
        nb,
        "print(len(test), 'etudes | TTA cnx flip lateral x1 | autres x1 | "
        "FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        "print(len(test), 'etudes | sans TTA | pas de flip | "
        "FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        1,
    )
    pub._replace_nb(
        nb,
        "def _flip_lateral(x):\n"
        "    \"\"\"Miroir lateral. Sagittal : axe D. Coronal/Axial : axe W.\"\"\"\n"
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
        "    'TTA cnx identite + flip lateral, swap menisque et OA medial/lateral, autres x1',\n"
        "    flush=True,\n"
        ")\n",
        "",
        1,
    )
    pub._replace_nb(
        nb,
        "        with torch.autocast('cuda', enabled=False):\n"
        "            p_id = torch.sigmoid(model_cnx(x_224.float()).float())\n"
        "            p_flip = torch.sigmoid(model_cnx(_flip_lateral(x_224).float()).float())\n"
        "        p_cnx = 0.5 * (p_id + _swap_medlat(p_flip))\n",
        "        with torch.autocast('cuda', enabled=False):\n"
        "            p_cnx = torch.sigmoid(model_cnx(x_224.float()).float())\n",
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
    cells.append(pub._code(pub.SAVE))
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
        "(0.00, 0.00, 0.40, 0.00, 0.60),  # ACL",
        "v20b_dino_full_swa_ep6_9.pt",
        "v20b_crop130_dino_ep11_oof_v3b_full', 'swa-6-7-8-9'",
        "v20h_crop145_distil_swa_ep7_9.pt",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "cnx_r7_swa_ep7_10.pt",
        "swa-7-8-9-10",
        "cnx forward fp32",
        "sans TTA",
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
        "_flip_lateral",
        "_swap_medlat",
        "p_flip",
        "flip lateral",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "laterality=True",
        "TTA_VARIANTS",
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
            "sultent/rsna-knee-v20b-dino-full-swa",
            "sultent/rsna-knee-v20h-crop145-distil-swa789",
            "sultent/rsna-knee-coat-a-oai3-swa",
            "sultent/rsna-knee-cnx-r7-swa78910",
        ],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    (PUSH / "_acl.txt").write_text(ACL + "\n", encoding="utf-8")
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


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def _transient(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "expecting value", "remote end closed", "connection aborted",
        "temporarily unavailable", "timed out", "timeout", "502", "503", "504",
    ))


def push() -> str:
    while True:
        try:
            out = kaggle("kernels", "push", "-p", str(PUSH), "--accelerator", ACCEL, "-t", "32400")
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text):
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            if _transient(text):
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
                continue
            raise
        print(out, flush=True)
        if _quota(out):
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
        return version


def wait() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KAGGLE_ID).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def _daily_limit(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "maximum number", "submission limit", "daily submission allowance",
        "try again tomorrow", "too many submission", "exceeded",
    ))


def submit(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KAGGLE_ID, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and SLUG in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3 gold 0.9903",
        "v20b_crop130_dino_ep11_oof_v3b_full epoch swa-6-7-8-9 lat False gold 0.9678",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "coat coat_a_llrd08_v3b_oofmask_full_oai3 epoch swa-1-2-3 lat False gold 0.9884",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10 lat False gold 0.976",
        "cnx forward fp32",
        "sans TTA",
        "TTA             : non",
        ACL,
        "MIX5_SCORED",
        "GPU mix5 libere",
        "RAPTOR224_STAGE_DONE",
        "NOTEBOOK_COMPLETE",
        "poids mix5 0.70 public 0.30",
        "BLEND70_OK",
        "lignes 3",
        "lat False",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if (
        "flip lateral" in text
        or "lat True" in text
        or "public 0.40" in text
        or "v20b_crop130_dino_ep11_oof_v3b_full_oai3" in text
    ):
        raise SystemExit("TTA, flip, v20b oai ou blend 60/40")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    waited = False
    while True:
        print("submit", KAGGLE_ID, "version", version, flush=True)
        try:
            resp = api.competition_submit_code(
                "submission.csv", MSG, COMPETITION, KAGGLE_ID, int(version), quiet=True,
            )
        except Exception as e:
            http = getattr(e, "response", None)
            body = http.text if http is not None else str(e)
            print(body, flush=True)
            if _daily_limit(body) and not waited:
                waited = True
                now = datetime.now(timezone.utc)
                target = now.replace(hour=0, minute=5, second=0, microsecond=0)
                if now >= target:
                    target = target + timedelta(days=1)
                delay = (target - now).total_seconds()
                print(f"ATTENTE_QUOTA {delay:.0f}s", flush=True)
                time.sleep(delay + 5)
                continue
            if _transient(body):
                time.sleep(20)
                continue
            raise
        print(resp, flush=True)
        print("SOUMIS", KAGGLE_ID, "version", version, flush=True)
        return


def main() -> None:
    write_nb()
    if "--write" in sys.argv:
        return
    version = push()
    wait()
    submit(version)


if __name__ == "__main__":
    main()
