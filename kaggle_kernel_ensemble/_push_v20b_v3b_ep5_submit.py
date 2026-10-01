# -*- coding: utf-8 -*-
"""Cree le dataset v20b v3b ep5, push T4, submit si submission.csv existe."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD = Path(r"C:\Users\sulta\Documents\RSNA-comp\kaggle_dataset_v20b\_upload_v3b_ep10")
PUSH = ROOT / "kaggle_kernel_ensemble" / "_push_v20b_v3b_ep10"
DATASET = "sultent/rsna-knee-v20b-v3b-subset-ep10"
KERNEL = "sultent/rsna-knee-v20b-v3b-ep10-tta-x5"
COMPETITION = "rsna-knee-abnormality-detection"
OUT = ROOT / "kaggle_kernel_ensemble" / "_out_v20b_v3b_ep10"
MSG = "v20b subset v3b epoch10 (gold 0.905, val 0.943, TTA x5, T4)"
POLL_S = 60
MAX_WAIT_S = 8 * 3600


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e.output, flush=True)
        raise


def main() -> None:
    pt = UPLOAD / "v20b_v3b_subset_ep10.pt"
    if not pt.exists() or pt.stat().st_size < 50_000_000:
        raise SystemExit(f"poids manquant: {pt}")
    try:
        st0 = kaggle("datasets", "status", DATASET).strip()
    except subprocess.CalledProcessError:
        st0 = ""
    print(st0, flush=True)
    if "ready" not in st0.lower():
        print(kaggle("datasets", "create", "-p", str(UPLOAD)), flush=True)

    t0 = time.time()
    while True:
        try:
            st = kaggle("datasets", "status", DATASET).strip()
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            print(text, flush=True)
            if "403" not in text and "404" not in text and "Forbidden" not in text:
                raise
            st = "pending"
        print(st, flush=True)
        if "ready" in st.lower() and "pending" not in st.lower():
            break
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout dataset")
        time.sleep(20)

    out = kaggle(
        "kernels", "push",
        "-p", str(PUSH),
        "--accelerator", "NvidiaTeslaT4",
        "-t", "32400",
    )
    print(out, flush=True)
    m = re.search(r"Kernel version (\d+)", out)
    if not m:
        raise SystemExit(f"version introuvable: {out}")
    version = m.group(1)

    t1 = time.time()
    while True:
        st = kaggle("kernels", "status", KERNEL).strip()
        print(st, flush=True)
        up = st.upper()
        if "ERROR" in up or "FAILED" in up or "CANCEL" in up:
            OUT.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
            except subprocess.CalledProcessError:
                pass
            raise SystemExit("kernel en erreur")
        if "RUNNING" not in up and ("COMPLETE" in up or "SUCCEEDED" in up):
            break
        if time.time() - t1 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)

    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    csv = OUT / "submission.csv"
    if not csv.exists() or csv.stat().st_size == 0:
        raise SystemExit(f"submission.csv absent: {OUT}")
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", version, "-f", "submission.csv", "-m", MSG,
    ), flush=True)
    print("OK v20b v3b ep10", version, flush=True)


if __name__ == "__main__":
    main()
