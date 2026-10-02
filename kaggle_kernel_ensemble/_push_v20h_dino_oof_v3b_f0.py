# -*- coding: utf-8 -*-
"""Dataset + kernel T4 v20h DINO OOF v3b fold0, puis soumission."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
MAKE = KERNEL_DIR / "_make_nb_v20h_dino_oof_v3b_f0.py"
UPLOAD = ROOT / "kaggle_dataset_v20h_lat" / "_upload_dino_ep11_oof_v3b_f0"
PUSH = KERNEL_DIR / "_push_v20h_dino_oof_v3b_f0"
OUT = KERNEL_DIR / "_out_v20h_dino_oof_v3b_f0"
DATASET = "sultent/rsna-knee-v20h-dino-oof-v3b-f0"
KERNEL = "sultent/rsna-knee-v20h-dino-oof-v3b-f0-solo"
COMPETITION = "rsna-knee-abnormality-detection"
WEIGHT = "v20h_dino_ep11_oof_v3b_f0.pt"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
ACCEL = "NvidiaTeslaT4"


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
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
        missing = ("404", "403", "not found", "notfound", "does not exist", "forbidden")
        if any(s in low for s in missing):
            return ""
        raise


def _ready(status: str) -> bool:
    low = status.lower()
    return "ready" in low and "pending" not in low


def ensure_dataset() -> None:
    pt = UPLOAD / WEIGHT
    if not pt.is_file() or pt.stat().st_size < 50_000_000:
        raise SystemExit(f"poids manquant: {pt}")
    st = _status_dataset()
    print("[ds]", st or "(absent)", flush=True)
    if _ready(st):
        print(kaggle(
            "datasets", "version", "-p", str(UPLOAD),
            "-m", "best.pt OOF v3b fold0, optimizer retire",
        ), flush=True)
    elif st and "pending" in st.lower():
        print("[ds] creation deja en cours", flush=True)
    else:
        print(kaggle("datasets", "create", "-p", str(UPLOAD), "--dir-mode", "zip"), flush=True)
    t0 = time.time()
    while True:
        st = _status_dataset()
        print("[ds]", st or "(pending)", flush=True)
        if _ready(st):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout dataset")
        time.sleep(20)


def push_kernel() -> str:
    if not (PUSH / "kernel-metadata.json").is_file():
        raise FileNotFoundError(PUSH)
    meta = json.loads((PUSH / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("machine_shape") != ACCEL:
        raise SystemExit(f"machine_shape={meta.get('machine_shape')!r}, attendu {ACCEL}")
    nb = (PUSH / "submit.ipynb").read_text(encoding="utf-8")
    if "submission.csv" not in nb:
        raise SystemExit("notebook sans submission.csv")
    if "LATERALITY_CANON_L = False" not in nb:
        raise SystemExit("notebook sans LATERALITY_CANON_L = False")
    print("push", KERNEL, "T4", flush=True)
    out = kaggle(
        "kernels", "push", "-p", str(PUSH),
        "--accelerator", ACCEL, "-t", "32400",
    )
    print(out, flush=True)
    m = re.search(r"Kernel version (\d+) successfully pushed", out, flags=re.I)
    if not m:
        raise RuntimeError(f"version introuvable: {out!r}")
    print("kernel version", KERNEL, m.group(1), flush=True)
    return m.group(1)


def wait_kernel() -> None:
    t0 = time.time()
    seen_running = False
    while True:
        raw = kaggle("kernels", "status", KERNEL).strip()
        up = raw.upper()
        print("[kernel]", raw, flush=True)
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
        if done and (seen_running or (time.time() - t0) > 90):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def submit(version: str) -> None:
    hist = json.loads(
        (ROOT / "checkpoints" / "v20h_crop130_dino_ep11_oof_v3b_fold0" / "history.json")
        .read_text(encoding="utf-8")
    )
    nb = json.loads((PUSH / "submit.ipynb").read_text(encoding="utf-8"))
    md = "".join(nb["cells"][0]["source"])
    m_ep = re.search(r"epoch (\d+)", md)
    epoch = m_ep.group(1) if m_ep else "?"
    rec = next((r for r in hist if str(r["epoch"]) == epoch), hist[-1])
    msg = (
        f"v20h DINO ep11 OOF v3b fold0 best epoch {epoch} "
        f"(val {float(rec['val_macro_auc']):.3f}, "
        f"gold {float(rec['gold_macro_auc']):.3f}, "
        f"pas de flip, TTA x5, T4)"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    csv = OUT / "submission.csv"
    if not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit(f"submission.csv absent: {OUT}")
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", version,
        "-f", "submission.csv", "-m", msg,
    ), flush=True)
    print("submitted", KERNEL, "version", version, csv, csv.stat().st_size, flush=True)


def main() -> None:
    subprocess.check_call([sys.executable, "-X", "utf8", str(MAKE)])
    ensure_dataset()
    version = push_kernel()
    wait_kernel()
    submit(version)
    print("DONE", version, flush=True)


if __name__ == "__main__":
    main()
