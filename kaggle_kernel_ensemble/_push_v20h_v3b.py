# -*- coding: utf-8 -*-
"""Datasets + kernels T4 v20h v3b lat ep 6-10, puis soumission."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
MAKE = KERNEL_DIR / "_make_nb_v20h_v3b.py"
COMPETITION = "rsna-knee-abnormality-detection"
EPOCHS = (6, 7, 8, 9, 10)
POLL_S = 45
MAX_WAIT_S = 8 * 3600
TIMEOUT_S = "32400"
ACCEL = "NvidiaTeslaT4"
GOLD = {
    6: 0.9580348958787716,
    7: 0.9650750217565953,
    8: 0.9700817779550243,
    9: 0.9713271830772822,
    10: 0.9736530688791812,
}


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
    return f"sultent/rsna-knee-v20h-v3b-lat-ep{epoch}"


def kernel_id(epoch: int) -> str:
    return f"sultent/rsna-knee-v20h-v3b-lat-solo-ep{epoch}"


def upload_dir(epoch: int) -> Path:
    return ROOT / "kaggle_dataset_v20h_lat" / f"_upload_v3b_ep{epoch}"


def push_dir(epoch: int) -> Path:
    return KERNEL_DIR / f"_push_v20h_v3b_ep{epoch}"


def out_dir(epoch: int) -> Path:
    return KERNEL_DIR / f"_out_v20h_v3b_ep{epoch}"


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


def ensure_datasets(epochs: list[int]) -> None:
    for ep in epochs:
        pt = upload_dir(ep) / f"v20h_v3b_lat_ep{ep}.pt"
        if not pt.is_file() or pt.stat().st_size < 50_000_000:
            raise SystemExit(f"poids manquant: {pt}")
        st = _status_dataset(ep)
        print(f"[ds ep{ep}]", st or "(absent)", flush=True)
        if _ready(st):
            print(f"[ds ep{ep}] deja a jour", flush=True)
        elif st and "pending" in st.lower():
            print(f"[ds ep{ep}] creation deja en cours", flush=True)
        else:
            print(kaggle("datasets", "create", "-p", str(upload_dir(ep))), flush=True)

    pending = set(epochs)
    t0 = time.time()
    while pending:
        for ep in sorted(pending):
            st = _status_dataset(ep)
            print(f"[ds ep{ep}]", st or "(pending)", flush=True)
            if _ready(st):
                pending.remove(ep)
        if not pending:
            break
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout datasets {sorted(pending)}")
        time.sleep(20)


def try_push(epoch: int) -> str | None:
    push = push_dir(epoch)
    if not (push / "kernel-metadata.json").is_file():
        raise FileNotFoundError(push)
    print("push", kernel_id(epoch), "T4", flush=True)
    try:
        out = kaggle(
            "kernels", "push", "-p", str(push),
            "--accelerator", ACCEL, "-t", TIMEOUT_S,
        )
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        if "Maximum batch GPU" in text or "quota" in text.lower():
            print("quota GPU, retry plus tard ep", epoch, flush=True)
            return None
        raise
    print(out, flush=True)
    if "Maximum batch GPU" in out or "quota" in out.lower():
        print("quota GPU, retry plus tard ep", epoch, flush=True)
        return None
    m = re.search(r"Kernel version (\d+) successfully pushed", out, flags=re.I)
    if not m:
        raise RuntimeError(f"version introuvable: {out!r}")
    print("kernel version", kernel_id(epoch), m.group(1), flush=True)
    return m.group(1)


def submit_one(epoch: int, version: str) -> None:
    msg = (
        f"v20h v3b lat full solo epoch {epoch} "
        f"(gold in-sample {GOLD[epoch]:.3f}, canon L, TTA x5, T4)"
    )
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", kernel_id(epoch), "-v", version,
        "-f", "submission.csv", "-m", msg,
    ), flush=True)
    dest = out_dir(epoch)
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", kernel_id(epoch), "-p", str(dest)), flush=True)
    csv = dest / "submission.csv"
    if not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit(f"submission.csv absent: {dest}")
    print("submitted", kernel_id(epoch), "version", version, csv, flush=True)


def wait_and_submit(versions: dict[int, str], leftover: list[int]) -> None:
    pending = set(versions)
    seen_running = {ep: False for ep in versions}
    t0 = time.time()
    last_push = 0.0
    while pending or leftover:
        if leftover and (time.time() - last_push) >= 60:
            last_push = time.time()
            still = []
            for ep in leftover:
                ver = try_push(ep)
                if ver is None:
                    still.append(ep)
                else:
                    versions[ep] = ver
                    pending.add(ep)
                    seen_running[ep] = False
            leftover = still
        progressed = False
        for ep in sorted(pending):
            raw = kaggle("kernels", "status", kernel_id(ep)).strip()
            up = raw.upper()
            print(f"[ep{ep}] {raw}", flush=True)
            if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
                dest = out_dir(ep)
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    print(kaggle(
                        "kernels", "output", kernel_id(ep), "-p", str(dest)
                    ), flush=True)
                except subprocess.CalledProcessError:
                    pass
                raise SystemExit(f"kernel ep{ep} failed")
            if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
                seen_running[ep] = True
            done = ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up
            ready = done and (seen_running[ep] or (time.time() - t0) > 120)
            if ready:
                submit_one(ep, versions[ep])
                pending.remove(ep)
                progressed = True
                last_push = 0.0
                break
        if not pending and not leftover:
            break
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout kernels pending={sorted(pending)} leftover={leftover}")
        if not progressed:
            time.sleep(POLL_S)
    print("DONE", versions, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-make", action="store_true")
    p.add_argument("--skip-datasets", action="store_true")
    p.add_argument("--epoch", type=int, nargs="*", default=list(EPOCHS))
    p.add_argument("--version", action="append", default=[],
                   help="ep:version deja pousse, ex. 7:1")
    args = p.parse_args()
    epochs = list(args.epoch)
    if not args.skip_make:
        subprocess.check_call(
            [sys.executable, "-X", "utf8", str(MAKE), "--epoch", *map(str, epochs)]
        )
    if not args.skip_datasets:
        ensure_datasets(epochs)
    versions: dict[int, str] = {}
    for item in args.version:
        ep_s, _, ver = item.partition(":")
        versions[int(ep_s)] = ver
    leftover: list[int] = []
    for ep in args.epoch:
        if ep in versions:
            continue
        ver = try_push(ep)
        if ver is None:
            leftover.append(ep)
        else:
            versions[ep] = ver
    wait_and_submit(versions, leftover)


if __name__ == "__main__":
    main()
