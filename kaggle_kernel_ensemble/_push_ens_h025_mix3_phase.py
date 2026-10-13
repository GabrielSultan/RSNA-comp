# -*- coding: utf-8 -*-
"""Pousse mix3 + dephasage. Pas de soumission competition."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "rsna-knee-ens-oai3-h025-mix3-phase"
KERNEL = f"sultent/{SLUG}"
PUSH = ROOT / "kaggle_kernel_ensemble" / "_push_ens_h025_mix3_phase"
OUT = ROOT / "kaggle_kernel_ensemble" / "_out_ens_h025_mix3_phase"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600


def kaggle(*args: str) -> str:
    if args and args[0] == "competitions":
        raise SystemExit("soumission competition interdite")
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


def push_one() -> str:
    nb = (PUSH / "submit.ipynb").read_text(encoding="utf-8")
    for needle in ("dephasage x2", "phase=phase", "_phase_index", "sans TTA"):
        if needle not in nb:
            raise SystemExit(f"notebook sans {needle}")
    if "TTA_VARIANTS" in nb or "laterality=True" in nb or "KneeNetCoat" in nb:
        raise SystemExit("notebook incoherent")
    print("push", KERNEL, flush=True)
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(PUSH),
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
        (PUSH / "_version.txt").write_text(version, encoding="utf-8")
        print("kernel version", KERNEL, version, flush=True)
        return version


def wait_one() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KERNEL).strip()
        print(SLUG, raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit(f"{SLUG} en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout {SLUG}")
        time.sleep(POLL_S)


def collect(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    log = next(OUT.glob("*.log"), None)
    csv = OUT / "submission.csv"
    if log is None or not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit("sortie incomplete")
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "sans TTA",
        "dephasage x2",
        "demi-pas 0.5",
        "lignes 3",
        "swa-1-2-3",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop130_dino_ep11_oof_v3b_full_oai3",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "coat_a_llrd08" in text or "TTA x" in text or "lat True" in text:
        raise SystemExit("coat, TTA ou flip")
    print(
        "KERNEL_PRET", KERNEL, "version", version,
        "| pas de soumission competition",
        flush=True,
    )


def main() -> None:
    version = push_one()
    wait_one()
    collect(version)
    print("PAS_DE_SOUMISSION_COMPETITION", flush=True)


if __name__ == "__main__":
    main()
