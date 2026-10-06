# -*- coding: utf-8 -*-
"""Dataset + kernel T4 ViT-k v3b epoch 7, sans TTA, puis soumission."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "kaggle_dataset_vit_k" / "_upload_v3b_ep7"
PUSH = ROOT / "kaggle_kernel_vit_e_dino_last" / "_push_v3b_ep7"
OUT = ROOT / "kaggle_kernel_vit_e_dino_last" / "_out_v3b_ep7"
DATASET = "sultent/rsna-knee-vit-k-v3b-ep7"
KERNEL = "sultent/rsna-knee-vit-k-v3b-ep7-sans-tta"
COMPETITION = "rsna-knee-abnormality-detection"
MSG = (
    "ViT-k LLRD 0.8 v3b oofmask full epoch7 "
    "(gold in-sample 0.940, sans TTA, T4)"
)
POLL_S = 30
MAX_WAIT_S = 8 * 3600
ACCEL = "NvidiaTeslaT4"
TIMEOUT_S = "32400"


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


def _status_dataset() -> str:
    try:
        return kaggle("datasets", "status", DATASET).strip()
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        low = text.lower()
        missing = ("404", "403", "not found", "notfound", "does not exist")
        if any(s in low for s in missing):
            return ""
        raise


def _ready(status: str) -> bool:
    low = status.lower()
    return "ready" in low and "pending" not in low


def ensure_dataset() -> None:
    pt = UPLOAD / "vit_k_v3b_ep7.pt"
    if not pt.is_file() or pt.stat().st_size < 100_000_000:
        raise SystemExit(f"poids manquant: {pt}")
    st = _status_dataset()
    print("[ds]", st or "(absent)", flush=True)
    if _ready(st):
        print("[ds] deja pret, nouvelle version", flush=True)
        print(kaggle(
            "datasets", "version", "-p", str(UPLOAD),
            "-m", "ViT-k v3b oofmask epoch7 sans optim",
            "--dir-mode", "zip",
        ), flush=True)
    elif st and "pending" in st.lower():
        print("[ds] creation deja en cours", flush=True)
    else:
        print(kaggle(
            "datasets", "create", "-p", str(UPLOAD), "--dir-mode", "zip",
        ), flush=True)
    t0 = time.time()
    while True:
        st = _status_dataset()
        print("[ds]", st or "(pending)", flush=True)
        if _ready(st):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout dataset")
        time.sleep(20)


def try_push() -> str | None:
    print("push", KERNEL, ACCEL, flush=True)
    try:
        out = kaggle(
            "kernels", "push", "-p", str(PUSH),
            "--accelerator", ACCEL, "-t", TIMEOUT_S,
        )
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        if "Maximum batch GPU" in text or "quota" in text.lower():
            print("quota GPU, retry", flush=True)
            return None
        raise
    print(out, flush=True)
    if "Maximum batch GPU" in out or "quota" in out.lower():
        print("quota GPU, retry", flush=True)
        return None
    m = re.search(r"Kernel version (\d+)", out)
    if not m:
        raise RuntimeError(f"version introuvable: {out!r}")
    (PUSH / "_version.txt").write_text(m.group(1), encoding="utf-8")
    print("kernel version", m.group(1), flush=True)
    return m.group(1)


def submit(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    csv = OUT / "submission.csv"
    if not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit(f"submission.csv absent: {OUT}")
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", version,
        "-f", "submission.csv", "-m", MSG,
    ), flush=True)
    print("soumis", KERNEL, "version", version, flush=True)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="")
    p.add_argument("--skip-dataset", action="store_true")
    args = p.parse_args()
    if not args.skip_dataset:
        ensure_dataset()
    if args.version:
        submit_when_done(args.version)
        return
    version = None
    t0 = time.time()
    while version is None:
        version = try_push()
        if version is None:
            if time.time() - t0 > MAX_WAIT_S:
                raise SystemExit("timeout push GPU")
            time.sleep(60)
    submit_when_done(version)


def submit_when_done(version: str) -> None:
    seen_running = False
    t1 = time.time()
    while True:
        raw = kaggle("kernels", "status", KERNEL).strip()
        up = raw.upper()
        print(raw, flush=True)
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            OUT.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
            except subprocess.CalledProcessError:
                pass
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen_running = True
        done = ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up
        if done and (seen_running or (time.time() - t1) > 90):
            submit(version)
            print("DONE", version, flush=True)
            return
        if time.time() - t1 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
