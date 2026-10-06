# -*- coding: utf-8 -*-
"""Dataset + kernel T4 ViT-k v3b, sans TTA, puis soumission."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "rsna-knee-abnormality-detection"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
ACCEL = "NvidiaTeslaT4"
TIMEOUT_S = "32400"
GOLD = {8: "0.951", 9: "0.968"}


def paths(epoch: int) -> dict[str, Path | str]:
    return {
        "upload": ROOT / "kaggle_dataset_vit_k" / f"_upload_v3b_ep{epoch}",
        "push": ROOT / "kaggle_kernel_vit_e_dino_last" / f"_push_v3b_ep{epoch}",
        "out": ROOT / "kaggle_kernel_vit_e_dino_last" / f"_out_v3b_ep{epoch}",
        "pt": f"vit_k_v3b_ep{epoch}.pt",
        "dataset": f"sultent/rsna-knee-vit-k-v3b-ep{epoch}",
        "kernel": f"sultent/rsna-knee-vit-k-v3b-ep{epoch}-sans-tta",
        "msg": (
            f"ViT-k LLRD 0.8 v3b oofmask full epoch{epoch} "
            f"(gold in-sample {GOLD[epoch]}, sans TTA, T4)"
        ),
    }


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


def _status_dataset(dataset: str) -> str:
    try:
        return kaggle("datasets", "status", dataset).strip()
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


def ensure_dataset(epoch: int) -> None:
    cfg = paths(epoch)
    upload = cfg["upload"]
    assert isinstance(upload, Path)
    pt = upload / str(cfg["pt"])
    dataset = str(cfg["dataset"])
    if not pt.is_file() or pt.stat().st_size < 100_000_000:
        raise SystemExit(f"poids manquant: {pt}")
    st = _status_dataset(dataset)
    print("[ds]", st or "(absent)", flush=True)
    if _ready(st):
        print("[ds] deja pret, nouvelle version", flush=True)
        print(kaggle(
            "datasets", "version", "-p", str(upload),
            "-m", f"ViT-k v3b oofmask epoch{epoch} sans optim",
            "--dir-mode", "zip",
        ), flush=True)
    elif st and "pending" in st.lower():
        print("[ds] creation deja en cours", flush=True)
    else:
        print(kaggle(
            "datasets", "create", "-p", str(upload), "--dir-mode", "zip",
        ), flush=True)
    t0 = time.time()
    while True:
        st = _status_dataset(dataset)
        print("[ds]", st or "(pending)", flush=True)
        if _ready(st):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout dataset")
        time.sleep(20)


def try_push(epoch: int) -> str | None:
    cfg = paths(epoch)
    push = cfg["push"]
    kernel = str(cfg["kernel"])
    print("push", kernel, ACCEL, flush=True)
    try:
        out = kaggle(
            "kernels", "push", "-p", str(push),
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
    assert isinstance(push, Path)
    (push / "_version.txt").write_text(m.group(1), encoding="utf-8")
    print("kernel version", m.group(1), flush=True)
    return m.group(1)


def submit(epoch: int, version: str) -> None:
    cfg = paths(epoch)
    out = cfg["out"]
    assert isinstance(out, Path)
    out.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", str(cfg["kernel"]), "-p", str(out)), flush=True)
    csv = out / "submission.csv"
    if not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit(f"submission.csv absent: {out}")
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", str(cfg["kernel"]), "-v", version,
        "-f", "submission.csv", "-m", str(cfg["msg"]),
    ), flush=True)
    print("soumis", cfg["kernel"], "version", version, flush=True)


def submit_when_done(epoch: int, version: str) -> None:
    kernel = str(paths(epoch)["kernel"])
    out = paths(epoch)["out"]
    assert isinstance(out, Path)
    seen_running = False
    t1 = time.time()
    while True:
        raw = kaggle("kernels", "status", kernel).strip()
        up = raw.upper()
        print(raw, flush=True)
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            out.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", kernel, "-p", str(out)), flush=True)
            except subprocess.CalledProcessError:
                pass
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen_running = True
        done = ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up
        if done and (seen_running or (time.time() - t1) > 90):
            submit(epoch, version)
            print("DONE", version, flush=True)
            return
        if time.time() - t1 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def run(epoch: int, *, skip_dataset: bool, version: str) -> None:
    if not skip_dataset:
        ensure_dataset(epoch)
    if version:
        submit_when_done(epoch, version)
        return
    got = None
    t0 = time.time()
    while got is None:
        got = try_push(epoch)
        if got is None:
            if time.time() - t0 > MAX_WAIT_S:
                raise SystemExit("timeout push GPU")
            time.sleep(60)
    submit_when_done(epoch, got)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, required=True, choices=sorted(GOLD))
    p.add_argument("--version", default="")
    p.add_argument("--skip-dataset", action="store_true")
    args = p.parse_args()
    run(args.epoch, skip_dataset=args.skip_dataset, version=args.version)


if __name__ == "__main__":
    main()
