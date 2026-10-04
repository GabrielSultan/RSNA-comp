# -*- coding: utf-8 -*-
"""Datasets + kernels T4 v20h DINO full ep6 puis ep7, deux soumissions."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
MAKE = KERNEL_DIR / "_make_nb_v20h_dino_full.py"
RUN = ROOT / "checkpoints" / "v20h_crop130_dino_ep11_oof_v3b_full"
COMPETITION = "rsna-knee-abnormality-detection"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
ACCEL = "NvidiaTeslaT4"
EPOCHS = (6, 7)


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        raise


def dataset_id(epoch: int) -> str:
    return f"sultent/rsna-knee-v20h-dino-full-ep{epoch}"


def kernel_id(epoch: int) -> str:
    return f"sultent/rsna-knee-v20h-dino-full-ep{epoch}-solo"


def upload_dir(epoch: int) -> Path:
    return ROOT / "kaggle_dataset_v20h_lat" / f"_upload_dino_full_ep{epoch}"


def push_dir(epoch: int) -> Path:
    return KERNEL_DIR / f"_push_v20h_dino_full_ep{epoch}"


def out_dir(epoch: int) -> Path:
    return KERNEL_DIR / f"_out_v20h_dino_full_ep{epoch}"


def weight_name(epoch: int) -> str:
    return f"v20h_dino_full_ep{epoch}.pt"


def _status_dataset(epoch: int) -> str:
    try:
        return kaggle("datasets", "status", dataset_id(epoch)).strip()
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


def _quota(text: str) -> bool:
    low = text.lower()
    return "maximum batch gpu" in low or "quota" in low


def ensure_dataset(epoch: int) -> None:
    pt = upload_dir(epoch) / weight_name(epoch)
    if not pt.is_file() or pt.stat().st_size < 50_000_000:
        raise SystemExit(f"poids manquant: {pt}")
    st = _status_dataset(epoch)
    print(f"[ds ep{epoch}]", st or "(absent)", flush=True)
    if _ready(st):
        print(kaggle(
            "datasets", "version", "-p", str(upload_dir(epoch)),
            "-m", f"epoch{epoch}.pt optimizer retire",
        ), flush=True)
    elif st and "pending" in st.lower():
        print(f"[ds ep{epoch}] creation deja en cours", flush=True)
    else:
        print(kaggle(
            "datasets", "create", "-p", str(upload_dir(epoch)), "--dir-mode", "zip",
        ), flush=True)
    t0 = time.time()
    while True:
        st = _status_dataset(epoch)
        print(f"[ds ep{epoch}]", st or "(pending)", flush=True)
        if _ready(st):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout dataset ep{epoch}")
        time.sleep(20)


def push_kernel(epoch: int) -> str:
    push = push_dir(epoch)
    meta = json.loads((push / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("machine_shape") != ACCEL:
        raise SystemExit(f"machine_shape={meta.get('machine_shape')!r}")
    if meta.get("id") == dataset_id(epoch):
        raise SystemExit("slug kernel = slug dataset")
    nb = (push / "submit.ipynb").read_text(encoding="utf-8")
    if "submission.csv" not in nb or "LATERALITY_CANON_L = False" not in nb:
        raise SystemExit(f"notebook ep{epoch} incomplet")
    print("push", kernel_id(epoch), "T4", flush=True)
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(push),
                "--accelerator", ACCEL, "-t", "32400",
            )
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text):
                print(f"quota GPU ep{epoch}, retry dans 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if _quota(out):
            print(f"quota GPU ep{epoch}, retry dans 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+) successfully pushed", out, flags=re.I)
        if not m:
            raise RuntimeError(f"version introuvable: {out!r}")
        print("kernel version", kernel_id(epoch), m.group(1), flush=True)
        return m.group(1)


def wait_kernel(epoch: int) -> None:
    t0 = time.time()
    seen_running = False
    while True:
        raw = kaggle("kernels", "status", kernel_id(epoch)).strip()
        up = raw.upper()
        print(f"[ep{epoch}]", raw, flush=True)
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            dest = out_dir(epoch)
            dest.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", kernel_id(epoch), "-p", str(dest)), flush=True)
            except subprocess.CalledProcessError:
                pass
            raise SystemExit(f"kernel ep{epoch} en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen_running = True
        done = ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up
        if done and (seen_running or (time.time() - t0) > 90):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout kernel ep{epoch}")
        time.sleep(POLL_S)


def submit(epoch: int, version: str) -> None:
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    rec = next(r for r in hist if int(r["epoch"]) == epoch)
    gold = float(rec["gold_macro_auc"])
    msg = (
        f"v20h DINO ep11 full oofmask v3b epoch {epoch} "
        f"(gold in-sample {gold:.3f}, pas de flip, TTA x5, T4)"
    )
    dest = out_dir(epoch)
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", kernel_id(epoch), "-p", str(dest)), flush=True)
    csv = dest / "submission.csv"
    if not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit(f"submission.csv absent: {dest}")
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", kernel_id(epoch), "-v", version,
        "-f", "submission.csv", "-m", msg,
    ), flush=True)
    print("submitted", kernel_id(epoch), "version", version, csv, csv.stat().st_size, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, nargs="*", default=list(EPOCHS))
    args = p.parse_args()
    subprocess.check_call(
        [sys.executable, "-X", "utf8", str(MAKE), "--epoch", *map(str, args.epoch)]
    )
    for epoch in args.epoch:
        ensure_dataset(epoch)
        version = push_kernel(epoch)
        wait_kernel(epoch)
        submit(epoch, version)
        print("DONE", epoch, version, flush=True)


if __name__ == "__main__":
    main()
