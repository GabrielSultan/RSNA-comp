from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

KERNEL = "sultent/rsna-knee-vit-k-llrd-ep7-tta-x5"
COMPETITION = "rsna-knee-abnormality-detection"
OUT_DIR = Path("kaggle_kernel_vit_e_dino_last/_out")
VERSION_FILE = Path("kaggle_kernel_vit_e_dino_last/_push_version.txt")

POLL_EVERY_S = 300
MAX_WAIT_S = 8 * 3600


def _kaggle(*args: str) -> str:
    cmd = ["python", "-m", "kaggle", *args]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    version = VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else ""

    t0 = time.time()
    while True:
        status_out = _kaggle("kernels", "status", KERNEL).strip()
        status_u = status_out.upper()
        print(status_out, flush=True)

        if "FAILED" in status_u or "ERROR" in status_u:
            print("Kernel failed - aborting.", flush=True)
            sys.exit(2)
        if "RUNNING" not in status_u and (
            "COMPLETE" in status_u or "SUCCEEDED" in status_u
        ):
            break
        if time.time() - t0 > MAX_WAIT_S:
            print("Timeout waiting for kernel - aborting.", flush=True)
            sys.exit(3)
        time.sleep(POLL_EVERY_S)

    msg = "ViT-k LLRD 0.8 oofmask full epoch7 (gold in-sample 0.931, TTA x5, T4)"
    args = [
        "competitions", "submit", COMPETITION,
        "-k", KERNEL,
        "-f", "submission.csv",
        "-m", msg,
    ]
    if version:
        args.extend(["-v", version])
    _kaggle(*args)
    print("Competition submission submitted successfully.", flush=True)


if __name__ == "__main__":
    main()
