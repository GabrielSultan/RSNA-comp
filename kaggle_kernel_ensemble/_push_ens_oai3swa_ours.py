# -*- coding: utf-8 -*-
"""Meme mix que oai3swa-pubbest, sans le notebook public. 100 % les 4 modeles."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_c145_soft" / "submit.ipynb"
SLUG = "rsna-knee-ens-oai3swa-ours"
PUSH = KERNEL / "_push_ens_oai3swa_ours"
OUT = KERNEL / "_out_ens_oai3swa_ours"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 3 * 3600
MSG = (
    "ens ViT-k320 v20h_full OAI3 SWA 1-3 + v20b + v20h crop145 + CoAt, "
    "mix lineaire 100% nos 4 modeles, sans public, sans TTA, pas de flip, T4"
)
DATASETS = [
    "sultent/rsna-knee-vit-k320-oai3-swa",
    "sultent/rsna-knee-v20b-dino-oai3-swa",
    "sultent/rsna-knee-v20h-crop145-swa678",
    "sultent/rsna-knee-coat-a-oai3-swa",
]


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def write_nb() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    if len(nb["cells"]) != 9:
        raise SystemExit(f"notebook source : {len(nb['cells'])} cellules")
    head = _text(nb["cells"][0])
    old = "# RSNA Knee — rsna-knee-ens-c145-soft\n"
    if not head.startswith(old):
        raise SystemExit("titre source inattendu")
    nb["cells"][0]["source"][0] = "# RSNA Knee — rsna-knee-ens-oai3swa-ours\n"
    blob = "\n".join(_text(c) for c in nb["cells"])
    for needle in (
        "vit_k320_oai3_swa_ep1_3.pt",
        "v20h_crop145_swa_ep6_8.pt",
        "v20b_dino_oai3_swa_ep1_3.pt",
        "coat_a_oai3_swa_ep1_3.pt",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "swa-1-2-3",
        "swa-6-7-8",
        "(0.09, 0.32, 0.55, 0.04)",
        "(0.25, 0.20, 0.28, 0.27)",
        "CROP_MM = 145.0",
        "laterality=False",
        "submission.csv",
        "sans TTA",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "raptor",
        "degnonguidi",
        "W_PUBLIC",
        "BLEND5",
        "TTA_VARIANTS",
        "laterality=True",
        "vitk_full_oai3",
        "swa-8-9",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    for cell in nb["cells"]:
        src = _text(cell)
        if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
            ast.parse(src)
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
    )
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
        "dataset_sources": DATASETS,
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(nb["cells"]), flush=True)


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
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
        "expecting value",
        "remote end closed",
        "connection aborted",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "502",
        "503",
        "504",
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
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3",
        "gold 0.9903",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop145_dino_ep11_oof_v3b_full_oaiv2_wd01",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "swa-6-7-8",
        "swa-1-2-3",
        "lat False",
        "v20h 145 mm",
        "vit 0.09 v20b 0.32 v20h 0.55 coat 0.04",
        "sans TTA",
        "lignes 3",
        "submission.csv",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text or "lat True" in text or "raptor" in text.lower() or "BLEND5" in text:
        raise SystemExit("TTA, flip, public ou blend")
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
