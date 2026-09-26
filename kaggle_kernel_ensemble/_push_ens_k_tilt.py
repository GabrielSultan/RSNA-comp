# -*- coding: utf-8 -*-
"""Attend le dataset ViT-k, pousse le mix tilt, soumet."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "sultent/rsna-knee-ens-v20b-v20h-vitk-tilt"
FOLDER = ROOT / "kaggle_kernel_ensemble" / "_push_ens_v20h_vitk_tilt"
COMPETITION = "rsna-knee-abnormality-detection"
MSG = "ens v20b ep8 + v20h ep7 + vit-k ep8 tilt, sans TTA"


def _kaggle(*args: str) -> str:
    raw = subprocess.check_output(
        [sys.executable, "-m", "kaggle", *args], stderr=subprocess.STDOUT
    )
    return raw.decode("utf-8", errors="replace")


def _wait_dataset() -> None:
    t0 = time.time()
    while True:
        try:
            out = _kaggle("datasets", "files", "-d", "sultent/rsna-knee-vit-k-llrd-ep8")
        except subprocess.CalledProcessError as e:
            out = e.output if isinstance(e.output, str) else (
                e.output.decode("utf-8", "replace") if e.output else str(e)
            )
        print(out[:300], flush=True)
        if "vit_k_ep8.pt" in out:
            return
        if time.time() - t0 > 3 * 3600:
            raise SystemExit("dataset vit-k absent")
        time.sleep(60)


def _push() -> str:
    while True:
        print("push", KERNEL, flush=True)
        try:
            out = _kaggle(
                "kernels", "push", "-p", str(FOLDER),
                "--accelerator", "NvidiaTeslaT4", "-t", "32400",
            )
        except subprocess.CalledProcessError as e:
            text = e.output.decode("utf-8", "replace") if isinstance(e.output, (bytes, bytearray)) else str(e.output or e)
            print(text, flush=True)
            if "Maximum batch GPU" in text or "already running" in text.lower():
                time.sleep(120)
                continue
            raise
        print(out, flush=True)
        m = re.search(r"Kernel version (\d+) successfully pushed", out, flags=re.I)
        if not m:
            raise SystemExit(out)
        return m.group(1)


def main() -> None:
    _wait_dataset()
    ver = _push()
    t0 = time.time()
    while True:
        try:
            raw = _kaggle("kernels", "status", KERNEL).strip()
        except subprocess.CalledProcessError as e:
            text = e.output.decode("utf-8", "replace") if isinstance(e.output, (bytes, bytearray)) else str(e.output or e)
            print("status retry", text[:300], flush=True)
            time.sleep(30)
            continue
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up:
            raise SystemExit(raw)
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up:
            break
        if time.time() - t0 > 8 * 3600:
            raise SystemExit("timeout")
        time.sleep(60)
    sub = _kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", ver, "-f", "submission.csv", "-m", MSG,
    )
    print(sub, flush=True)
    print("submitted", KERNEL, ver, flush=True)


if __name__ == "__main__":
    main()
