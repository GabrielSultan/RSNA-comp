from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

KERNEL = "sultent/rsna-knee-ensemble-v20b-vit-e"
COMPETITION = "rsna-knee-abnormality-detection"
OUT_DIR = Path("kaggle_kernel_ensemble/_out_v20b_vite")

POLL_EVERY_S = 300
MAX_WAIT_S = 8 * 3600


def _kaggle(*args: str) -> str:
    cmd = ["python", "-m", "kaggle", *args]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        status_out = _kaggle("kernels", "status", KERNEL).strip()
        status_u = status_out.upper()
        print(status_out, flush=True)

        if "SUCCEEDED" in status_u or "COMPLETE" in status_u:
            # COMPLETE can mean previous version; require not RUNNING
            if "RUNNING" in status_u:
                pass
            else:
                break
        if "FAILED" in status_u or "ERROR" in status_u:
            print("Kernel failed - aborting.", flush=True)
            sys.exit(2)
        if time.time() - t0 > MAX_WAIT_S:
            print("Timeout waiting for kernel - aborting.", flush=True)
            sys.exit(3)
        time.sleep(POLL_EVERY_S)

    # Save & Run All n'a que ~3 etudes test : ne PAS uploader ce CSV.
    # Soumission code : Kaggle relance le kernel sur le vrai test.
    msg = "Ensemble v20b distil ViT-e lam03 ep9 + ViT-e ep7 mix adaptatif 80/20"
    _kaggle(
        "competitions",
        "submit",
        COMPETITION,
        "-k",
        KERNEL,
        "-v",
        "4",
        "-f",
        "submission.csv",
        "-m",
        msg,
    )
    print("Competition submission submitted successfully.", flush=True)


if __name__ == "__main__":
    main()
