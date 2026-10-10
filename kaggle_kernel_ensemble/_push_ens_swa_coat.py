# -*- coding: utf-8 -*-
"""Pousse l'ensemble SWA ViT-k320 + v20h DINO + v20b DINO + CoAt-a.

Les quatre datasets existent deja. Pas de nouvel upload si les fichiers
sont en ligne. Soumission competition seulement s'il reste un quota.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUSH = ROOT / "kaggle_kernel_ensemble" / "_push_ens_swa_coat"
OUT = ROOT / "kaggle_kernel_ensemble" / "_out_ens_swa_coat"
KERNEL = "sultent/rsna-knee-ens-swa-k320-hd-bd-coat"
COMPETITION = "rsna-knee-abnormality-detection"
MSG = (
    "ens SWA ViT-k320 ep7-10 + v20b DINO ep6-9 + v20h DINO ep6-7 + CoAt-a ep7-10 "
    "(softmax subset T=0.05, plancher 10%, sans TTA, pas de flip, T4)"
)
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
DATASETS = (
    ("sultent/rsna-knee-vit-k320-v3b-swa", "vit_k320_v3b_swa_ep7_10.pt"),
    ("sultent/rsna-knee-v20h-dino-full-swa", "v20h_dino_full_swa_ep6_7.pt"),
    ("sultent/rsna-knee-v20b-dino-full-swa", "v20b_dino_full_swa_ep6_9.pt"),
    ("sultent/rsna-knee-coat-a-v3b-swa", "coat_a_v3b_swa_ep7_10.pt"),
)


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


def dataset_ready(dataset: str, weight: str) -> None:
    text = kaggle("datasets", "files", dataset)
    size = None
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == weight:
            size = int(parts[1])
            break
    print(dataset, weight, size, flush=True)
    if size is None or size < 80_000_000:
        raise SystemExit(f"dataset pas pret: {dataset} {weight}")


def push_kernel() -> str:
    nb = (PUSH / "submit.ipynb").read_text(encoding="utf-8")
    if "TTA_VARIANTS" in nb or "laterality=True" in nb or "sans TTA" not in nb:
        raise SystemExit("notebook incoherent")
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(PUSH),
                "--accelerator", ACCEL, "-t", "32400",
            )
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if "quota" in text.lower() or "Maximum batch GPU" in text:
                print("quota GPU, retry dans 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if "quota" in out.lower() or "Maximum batch GPU" in out:
            print("quota GPU, retry dans 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        version = m.group(1)
        (PUSH / "_version.txt").write_text(version, encoding="utf-8")
        print("kernel version", version, flush=True)
        return version


def wait_kernel() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KERNEL).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            OUT.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
            except subprocess.CalledProcessError:
                pass
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (seen or time.time() - t0 > 90):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def collect(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    log = next(OUT.glob("*.log"), None)
    csv = OUT / "submission.csv"
    if log is None or not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit("sortie incomplete")
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "sans TTA",
        "lignes 3",
        "coat_a_llrd08_v3b_oofmask_full",
        "v20h_crop130_dino_ep11_oof_v3b_full",
        "v20b_crop130_dino_ep11_oof_v3b_full",
        "swa-6-7",
        "swa-6-7-8-9",
        "swa-7-8-9-10",
        "coat ",
        "KneeNet-CoAt-a",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "lat True" in text or "TTA x" in text:
        raise SystemExit("flip ou TTA dans le log")
    if text.count("lat False") < 2:
        raise SystemExit("lat mal charge")
    print("KERNEL_PRET", KERNEL, "version", version, flush=True)


def maybe_submit(version: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    limits = api.competition_get_submission_limits(COMPETITION)
    left = int(getattr(limits, "num_allowed_now", 0) or 0)
    print("soumissions restantes", left, flush=True)
    if left <= 0:
        print("LIMITE_JOURNALIERE", KERNEL, "version", version, flush=True)
        print("kernel pret, soumission competition non faite", flush=True)
        return
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", version,
        "-f", "submission.csv", "-m", MSG,
    ), flush=True)
    print("SOUMIS", KERNEL, "version", version, flush=True)


def main() -> None:
    for dataset, weight in DATASETS:
        dataset_ready(dataset, weight)
    version = push_kernel()
    wait_kernel()
    collect(version)
    maybe_submit(version)


if __name__ == "__main__":
    main()
