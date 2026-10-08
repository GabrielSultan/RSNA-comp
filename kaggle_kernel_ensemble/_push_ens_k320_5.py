# -*- coding: utf-8 -*-
"""Pousse l'ensemble T4 puis soumet a la competition."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "sultent/rsna-knee-ens-k320e9-hd6-bd8-lat7-vk7"
PUSH = ROOT / "kaggle_kernel_ensemble" / "_push_ens_k320_5"
OUT = ROOT / "kaggle_kernel_ensemble" / "_out_ens_k320_5"
DATASETS = [
    "sultent/rsna-knee-vit-k320-v3b-ep9",
    "sultent/rsna-knee-v20h-dino-full-ep6",
    "sultent/rsna-knee-v20b-dino-full-ep8",
    "sultent/rsna-knee-v20h-ep7-lat",
    "sultent/rsna-knee-vit-e-dino-last-weights",
]
COMPETITION = "rsna-knee-abnormality-detection"
MSG = (
    "ens ViT-k320 ep9 + v20h DINO ep6 + v20b DINO ep8 + v20h lat ep7 + ViT-k ep7 "
    "(softmax subset, plancher 10%, sans TTA, T4)"
)
ACCEL = "NvidiaTeslaT4"


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


def dataset_ready(ds: str) -> None:
    try:
        st = kaggle("datasets", "status", ds).strip()
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        raise SystemExit(f"dataset absent {ds}: {text[:200]}") from e
    print(ds, st, flush=True)
    low = st.lower()
    if "ready" not in low or "pending" in low:
        raise SystemExit(f"dataset pas pret: {ds} {st}")


def main() -> None:
    for ds in DATASETS:
        dataset_ready(ds)
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(PUSH),
                "--accelerator", ACCEL, "-t", "32400",
            )
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if "quota" in text.lower() or "Maximum batch GPU" in text:
                print("quota GPU, retry dans 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if "quota" in out.lower() or "Maximum batch GPU" in out:
            print("quota GPU, retry dans 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out)
        if not m:
            raise RuntimeError(out)
        version = m.group(1)
        break
    print("kernel version", version, flush=True)
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KERNEL).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            OUT.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
            except subprocess.CalledProcessError:
                pass
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (seen or time.time() - t0 > 90):
            break
        if time.time() - t0 > 8 * 3600:
            raise SystemExit("timeout")
        time.sleep(30)
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    log = next(OUT.glob("*.log"))
    text = log.read_text(encoding="utf-8", errors="replace")
    if "sans TTA" not in text or "lignes 3" not in text:
        raise SystemExit("log inattendu")
    if "v20b_crop130_dino_ep11_oof_v3b_full" not in text or "v20h_crop130_dino_ep11_oof_v3b_full" not in text:
        raise SystemExit("modeles absents du log")
    if "v20h_crop130_oofmask_v20b_lat_full" not in text or text.count("lat True") != 1:
        raise SystemExit("lat mal charge")
    if "vit_k_llrd08_det_oofmask_full" not in text or "vk " not in text:
        raise SystemExit("vit-k absent du log")
    if text.count("lat False") < 2:
        raise SystemExit("flip inattendu sur un modele non lat")
    csv = OUT / "submission.csv"
    if not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit("submission.csv absent")
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", version,
        "-f", "submission.csv", "-m", MSG,
    ), flush=True)
    print("SOUMIS", KERNEL, "version", version, flush=True)


if __name__ == "__main__":
    main()
