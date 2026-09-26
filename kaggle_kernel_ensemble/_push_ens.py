# -*- coding: utf-8 -*-
"""Pousse puis soumet les deux ensembles, l'un apres l'autre (quota GPU)."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
COMPETITION = "rsna-knee-abnormality-detection"
POLL_EVERY_S = 60
MAX_WAIT_S = 8 * 3600
TIMEOUT_S = "32400"

JOBS = [
    # Deja pousse : le titre a impose ce slug (pas l'id du metadata).
    {
        "kernel": "sultent/rsna-knee-ens-v20b-ep8-v20h-ep7",
        "folder": None,
        "version": "1",
        "msg": "ens v20b ep8 + v20h lat ep7 (smooth genereux, canon L sur v20h seul)",
    },
    {
        "kernel": "sultent/rsna-knee-ens-v20b-v20h-vite",
        "folder": KERNEL_DIR / "_push_ens_v20h_vite",
        "version": None,
        "msg": "ens v20b ep8 + v20h lat ep7 + ViT-e ep7 (ViT-e 70% Effusion seulement)",
    },
]


def _kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return raw.decode("utf-8", errors="replace")


def _parse_version(out: str) -> str:
    if "successfully pushed" not in out.lower():
        raise RuntimeError(f"push echoue : {out.strip()!r}")
    m = re.search(r"Kernel version (\d+) successfully pushed", out, flags=re.I)
    if not m:
        raise RuntimeError(f"version introuvable : {out!r}")
    return m.group(1)


def _wait(kernel: str) -> None:
    t0 = time.time()
    while True:
        raw = _kaggle("kernels", "status", kernel).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up:
            raise SystemExit(f"kernel failed {kernel}")
        if (("COMPLETE" in up) or ("SUCCEEDED" in up)) and "RUNNING" not in up:
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout {kernel}")
        time.sleep(POLL_EVERY_S)


def _push(kernel: str, folder: Path) -> str:
    while True:
        print("push", kernel, flush=True)
        try:
            out = _kaggle(
                "kernels", "push",
                "-p", str(folder),
                "--accelerator", "NvidiaTeslaT4",
                "-t", TIMEOUT_S,
            )
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            print(text, flush=True)
            if "Maximum batch GPU" in text or "already running" in text.lower():
                print("quota GPU, retry dans 120s", flush=True)
                time.sleep(120)
                continue
            raise
        print(out, flush=True)
        ver = _parse_version(out)
        print("version", kernel, ver, flush=True)
        return ver


def main() -> None:
    for job in JOBS:
        kernel = job["kernel"]
        folder = job["folder"]
        ver = job["version"]
        if folder is not None:
            if not (folder / "submit.ipynb").is_file():
                raise FileNotFoundError(folder)
            ver = _push(kernel, folder)
        if not ver:
            raise RuntimeError(kernel)
        _wait(kernel)
        out = _kaggle(
            "competitions", "submit", COMPETITION,
            "-k", kernel,
            "-v", ver,
            "-f", "submission.csv",
            "-m", job["msg"],
        )
        print(out, flush=True)
        print("submitted", kernel, ver, flush=True)


if __name__ == "__main__":
    main()
