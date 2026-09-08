# -*- coding: utf-8 -*-
"""Push un mix, attend COMPLETE, soumet la version kernel a la competition."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
PUSH_DIR = KERNEL_DIR / "_push"
MAKE_NB = KERNEL_DIR / "_make_nb.py"
MIX_PATH = KERNEL_DIR / "mix.json"
META_SRC = KERNEL_DIR / "kernel-metadata.json"
META_DST = PUSH_DIR / "kernel-metadata.json"

KERNEL = "sultent/rsna-knee-ensemble-v20b-vit-e"
COMPETITION = "rsna-knee-abnormality-detection"
POLL_EVERY_S = 30
MAX_WAIT_S = 2 * 3600


def _kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _wait_complete() -> None:
    t0 = time.time()
    while True:
        status_out = _kaggle("kernels", "status", KERNEL).strip()
        print(status_out, flush=True)
        status_u = status_out.upper()
        if "FAILED" in status_u or "ERROR" in status_u:
            raise SystemExit("kernel failed")
        running = "RUNNING" in status_u
        done = ("COMPLETE" in status_u) or ("SUCCEEDED" in status_u)
        if done and not running:
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout waiting for kernel COMPLETE")
        time.sleep(POLL_EVERY_S)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mix", required=True, help="json mix (ex. mix_v2_90.json)")
    p.add_argument("--message", required=True)
    args = p.parse_args()

    mix_src = Path(args.mix)
    if not mix_src.is_absolute():
        mix_src = KERNEL_DIR / mix_src
    if not mix_src.is_file():
        raise FileNotFoundError(mix_src)

    shutil.copy2(mix_src, MIX_PATH)
    print("mix ->", MIX_PATH, flush=True)
    subprocess.check_call([sys.executable, "-X", "utf8", str(MAKE_NB)])
    shutil.copy2(META_SRC, META_DST)

    out = _kaggle(
        "kernels", "push",
        "-p", str(PUSH_DIR),
        "--accelerator", "NvidiaTeslaT4",
        "-t", "32400",
    )
    print(out, flush=True)
    version = None
    for token in out.replace(".", " ").split():
        if token.isdigit():
            version = token
            break
    if not version:
        raise RuntimeError(f"version kernel introuvable dans : {out!r}")
    print("kernel version", version, flush=True)

    _wait_complete()

    submit_out = _kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL,
        "-v", version,
        "-f", "submission.csv",
        "-m", args.message,
    )
    print(submit_out, flush=True)
    print("submitted version", version, flush=True)


if __name__ == "__main__":
    main()
