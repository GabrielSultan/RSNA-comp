# -*- coding: utf-8 -*-
"""Pousse puis soumet l'ensemble v20b + v20h + ViT-k."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "sultent/rsna-knee-ens-v20b-v20h-vitk"
FOLDER = ROOT / "kaggle_kernel_ensemble" / "_push_ens_v20h_vitk"
COMPETITION = "rsna-knee-abnormality-detection"
MSG = "ens v20b ep8 + v20h lat ep7 + vit-k ep8 softmax, sans TTA"


def _kaggle(*args: str) -> str:
    raw = subprocess.check_output(
        [sys.executable, "-m", "kaggle", *args], stderr=subprocess.STDOUT
    )
    return raw.decode("utf-8", errors="replace")


def main() -> None:
    print("push", KERNEL, flush=True)
    out = _kaggle(
        "kernels", "push", "-p", str(FOLDER),
        "--accelerator", "NvidiaTeslaT4", "-t", "32400",
    )
    print(out, flush=True)
    m = re.search(r"Kernel version (\d+) successfully pushed", out, flags=re.I)
    if not m:
        raise SystemExit(out)
    ver = m.group(1)
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
