# -*- coding: utf-8 -*-
"""Attend COMPLETE puis soumet la version kernel a la competition."""
from __future__ import annotations

import subprocess
import sys
import time

KERNEL = "sultent/rsna-knee-v20b-distil-solo"
COMPETITION = "rsna-knee-abnormality-detection"
VERSION = "1"
POLL_EVERY_S = 30
MAX_WAIT_S = 2 * 3600
MSG = "v20b distil ViT-e lam03 subset frac025 best ep9 solo (pas d ensemble)"


def _kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def main() -> None:
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
            break
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout waiting for kernel COMPLETE")
        time.sleep(POLL_EVERY_S)

    out = _kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", VERSION, "-f", "submission.csv", "-m", MSG,
    )
    print(out, flush=True)
    print("submitted version", VERSION, flush=True)


if __name__ == "__main__":
    main()
