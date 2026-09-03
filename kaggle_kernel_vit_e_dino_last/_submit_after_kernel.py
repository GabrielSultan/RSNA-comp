from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


KERNEL = "sultent/rsna-knee-vit-e-dino-last-pt-tta-x5/3"
COMPETITION = "rsna-knee-abnormality-detection"
OUT_DIR = Path("kaggle_kernel_vit_e_dino_last/_out")

POLL_EVERY_S = 300  # 5 min
MAX_WAIT_S = 6 * 3600  # 6h safeguard


def _kaggle(*args: str) -> str:
    cmd = ["python", "-m", "kaggle", *args]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _kernel_status() -> str:
    # Example output:
    # sultent/<kernel> has status "KernelWorkerStatus.RUNNING"
    out = _kaggle("kernels", "status", KERNEL).strip()
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    while True:
        status_out = _kernel_status()
        status_u = status_out.upper()
        print(status_out, flush=True)

        if "SUCCEEDED" in status_u:
            break
        if "FAILED" in status_u:
            print("Kernel failed - aborting.", flush=True)
            sys.exit(2)

        if time.time() - t0 > MAX_WAIT_S:
            print("Timeout waiting for kernel - aborting.", flush=True)
            sys.exit(3)

        time.sleep(POLL_EVERY_S)

    # Download only submission.csv (written by the kernel)
    _kaggle(
        "kernels",
        "output",
        KERNEL,
        "-p",
        str(OUT_DIR),
        "--file-pattern",
        r"submission\.csv$",
    )

    submission_csv = OUT_DIR / "submission.csv"
    if not submission_csv.exists():
        raise FileNotFoundError(f"submission.csv not found in {OUT_DIR.resolve()}")

    msg = "KneeNet ViT-e DINOv3 last.pt (TTA x5) via Kaggle kernel"
    _kaggle(
        "competitions",
        "submit",
        COMPETITION,
        "-f",
        str(submission_csv),
        "-m",
        msg,
    )
    print("Competition submission submitted successfully.", flush=True)


if __name__ == "__main__":
    main()

