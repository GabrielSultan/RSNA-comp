# -*- coding: utf-8 -*-
"""Pousse les 3 ensembles hybrid025. Pas de soumission competition."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
JOBS = (
    {
        "slug": "rsna-knee-ens-oai3-h025-mix4",
        "push": ROOT / "kaggle_kernel_ensemble" / "_push_ens_h025_mix4",
        "out": ROOT / "kaggle_kernel_ensemble" / "_out_ens_h025_mix4",
        "coat": True,
    },
    {
        "slug": "rsna-knee-ens-oai3-h025-mix3",
        "push": ROOT / "kaggle_kernel_ensemble" / "_push_ens_h025_mix3",
        "out": ROOT / "kaggle_kernel_ensemble" / "_out_ens_h025_mix3",
        "coat": False,
    },
    {
        "slug": "rsna-knee-ens-oai3-h025-spec",
        "push": ROOT / "kaggle_kernel_ensemble" / "_push_ens_h025_spec",
        "out": ROOT / "kaggle_kernel_ensemble" / "_out_ens_h025_spec",
        "coat": True,
    },
)


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        raise


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def push_one(job: dict) -> str:
    kernel = f"sultent/{job['slug']}"
    nb = (job["push"] / "submit.ipynb").read_text(encoding="utf-8")
    if "TTA_VARIANTS" in nb or "laterality=True" in nb or "sans TTA" not in nb:
        raise SystemExit(f"{job['slug']} notebook incoherent")
    if job["coat"] and "KneeNetCoat" not in nb:
        raise SystemExit(f"{job['slug']} sans coat")
    if not job["coat"] and "KneeNetCoat" in nb:
        raise SystemExit(f"{job['slug']} contient coat")
    print("push", kernel, flush=True)
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(job["push"]),
                "--accelerator", ACCEL, "-t", "32400",
            )
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text):
                print("quota GPU, retry dans 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if _quota(out):
            print("quota GPU, retry dans 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        version = m.group(1)
        (job["push"] / "_version.txt").write_text(version, encoding="utf-8")
        print("kernel version", kernel, version, flush=True)
        return version


def wait_one(job: dict) -> None:
    kernel = f"sultent/{job['slug']}"
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", kernel).strip()
        print(job["slug"], raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit(f"{job['slug']} en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout {job['slug']}")
        time.sleep(POLL_S)


def collect(job: dict, version: str) -> None:
    kernel = f"sultent/{job['slug']}"
    dest = job["out"]
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", kernel, "-p", str(dest)), flush=True)
    log = next(dest.glob("*.log"), None)
    csv = dest / "submission.csv"
    if log is None or not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit(f"sortie incomplete {job['slug']}")
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "sans TTA",
        "lignes 3",
        "swa-1-2-3",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop130_dino_ep11_oof_v3b_full_oai3",
    ):
        if needle not in text:
            raise SystemExit(f"{job['slug']} log sans {needle}")
    if job["coat"]:
        if "coat_a_llrd08_v3b_oofmask_full_oai3" not in text:
            raise SystemExit(f"{job['slug']} log sans coat")
    elif "coat_a_llrd08" in text:
        raise SystemExit(f"{job['slug']} a charge coat")
    if "TTA x" in text or "lat True" in text:
        raise SystemExit(f"{job['slug']} TTA ou flip")
    print("KERNEL_PRET", kernel, "version", version, "| pas de soumission competition", flush=True)


def main() -> None:
    versions = {}
    for job in JOBS:
        versions[job["slug"]] = push_one(job)
    for job in JOBS:
        wait_one(job)
        collect(job, versions[job["slug"]])
    print("PAS_DE_SOUMISSION_COMPETITION", flush=True)


if __name__ == "__main__":
    main()
