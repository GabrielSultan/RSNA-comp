"""Upload des poids, push T4 ep7+ep8, submit seulement si submission.csv existe."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD = Path(r"C:\Users\sulta\Documents\RSNA-comp\kaggle_dataset_vit_e_dino_last\_upload")
DATASET = "sultent/rsna-knee-vit-e-dino-last-weights"
COMPETITION = "rsna-knee-abnormality-detection"
OUT = ROOT / "kaggle_kernel_vit_e_dino_last" / "_out_submit"

JOBS = [
    {
        "epoch": 7,
        "kernel": "sultent/rsna-knee-vit-k-llrd-ep7-tta-x5",
        "push": ROOT / "kaggle_kernel_vit_e_dino_last" / "_push_ep7",
        "msg": "ViT-k LLRD 0.8 oofmask full epoch7 (gold in-sample 0.931, TTA x5, T4)",
    },
    {
        "epoch": 8,
        "kernel": "sultent/rsna-knee-vit-k-llrd-ep8-tta-x5",
        "push": ROOT / "kaggle_kernel_vit_e_dino_last" / "_push_ep8",
        "msg": "ViT-k LLRD 0.8 oofmask full epoch8 (gold in-sample 0.940, TTA x5, T4)",
    },
]

POLL_S = 120
MAX_WAIT_S = 10 * 3600


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("last.pt", "vit_k_ep8.pt"):
        p = UPLOAD / name
        if not p.exists() or p.stat().st_size < 100_000_000:
            raise SystemExit(f"poids manquant: {p}")

    before = kaggle("datasets", "list", "-m", "--search", "rsna-knee-vit-e-dino-last-weights")
    print(before, flush=True)
    stamp = ""
    for line in before.splitlines():
        if DATASET in line:
            m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
            stamp = m.group(0) if m else ""
    print("dataset stamp avant upload:", stamp, flush=True)

    print(kaggle(
        "datasets", "version",
        "-p", str(UPLOAD),
        "-m", "ViT-k LLRD epoch7 last.pt + epoch8 vit_k_ep8.pt",
    ), flush=True)

    t0 = time.time()
    while True:
        st = kaggle("datasets", "status", DATASET).strip()
        listing = kaggle("datasets", "list", "-m", "--search", "rsna-knee-vit-e-dino-last-weights")
        print(st, flush=True)
        print(listing, flush=True)
        new_stamp = ""
        for line in listing.splitlines():
            if DATASET in line:
                m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
                new_stamp = m.group(0) if m else ""
        ready = "ready" in st.lower() and "pending" not in st.lower()
        if ready and new_stamp and new_stamp != stamp:
            print("dataset pret:", new_stamp, flush=True)
            break
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout dataset")
        time.sleep(30)

    for job in JOBS:
        out = kaggle(
            "kernels", "push",
            "-p", str(job["push"]),
            "--accelerator", "NvidiaTeslaT4",
            "-t", "32400",
        )
        print(out, flush=True)
        m = re.search(r"Kernel version (\d+)", out)
        if not m:
            raise SystemExit(f"version introuvable pour ep{job['epoch']}: {out}")
        job["version"] = m.group(1)
        (job["push"] / "_version.txt").write_text(job["version"], encoding="utf-8")

    pending = {job["kernel"]: job for job in JOBS}
    t1 = time.time()
    while pending:
        if time.time() - t1 > MAX_WAIT_S:
            raise SystemExit(f"timeout kernels: {list(pending)}")
        done = []
        for kernel, job in pending.items():
            st = kaggle("kernels", "status", kernel).strip()
            print(st, flush=True)
            up = st.upper()
            if "ERROR" in up or "FAILED" in up or "CANCEL" in up:
                dest = OUT / f"ep{job['epoch']}"
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    print(kaggle("kernels", "output", kernel, "-p", str(dest)), flush=True)
                except subprocess.CalledProcessError as e:
                    print(e.output, flush=True)
                raise SystemExit(f"kernel ep{job['epoch']} en erreur")
            if "RUNNING" not in up and ("COMPLETE" in up or "SUCCEEDED" in up):
                done.append(kernel)
        for kernel in done:
            job = pending.pop(kernel)
            dest = OUT / f"ep{job['epoch']}"
            dest.mkdir(parents=True, exist_ok=True)
            print(kaggle("kernels", "output", kernel, "-p", str(dest)), flush=True)
            csv = dest / "submission.csv"
            if not csv.exists() or csv.stat().st_size == 0:
                raise SystemExit(f"submission.csv absent pour ep{job['epoch']} ({dest})")
            print(kaggle(
                "competitions", "submit", COMPETITION,
                "-k", kernel,
                "-v", job["version"],
                "-f", "submission.csv",
                "-m", job["msg"],
            ), flush=True)
            print(f"soumis ep{job['epoch']} version {job['version']}", flush=True)
        if pending:
            time.sleep(POLL_S)

    print("OK ep7 + ep8", flush=True)


if __name__ == "__main__":
    main()
