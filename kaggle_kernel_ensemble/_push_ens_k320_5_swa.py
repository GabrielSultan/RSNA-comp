# -*- coding: utf-8 -*-
"""Datasets SWA + kernel ensemble T4, sans TTA. Soumission competition si le quota le permet."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
UPLOAD_PY = ROOT / "kaggle_kernel_vit_e_dino_last" / "_upload_chunked.py"
PUSH = KERNEL_DIR / "_push_ens_k320_5_swa"
OUT = KERNEL_DIR / "_out_ens_k320_5_swa"
KERNEL = "sultent/rsna-knee-ens-swa-k320-hd-bd-lat-vk"
COMPETITION = "rsna-knee-abnormality-detection"
MSG = (
    "ens SWA ViT-k320 ep7-10 + v20h DINO ep6-7 + v20b DINO ep6-9 "
    "+ v20h lat ep7-12 + ViT-k ep7-10 "
    "(baseline ViT-k320 40%, plancher 10%, sans TTA, T4)"
)
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600


def _prep():
    spec = importlib.util.spec_from_file_location(
        "prep_ens_swa", KERNEL_DIR / "_prep_ens_swa.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _files_ready(dataset: str, weight: str) -> str:
    try:
        text = kaggle("datasets", "files", dataset)
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        low = text.lower()
        if any(s in low for s in ("404", "403", "not found", "notfound", "forbidden")):
            return "absent"
        raise
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == weight:
            try:
                size = int(parts[1])
            except (IndexError, ValueError):
                return "pending"
            return "ready" if size >= 80_000_000 else "pending"
    return "pending"


def ensure_dataset(job: dict) -> None:
    folder = job["upload"]
    weight = job["weight"]
    pt = folder / weight
    if not pt.is_file() or pt.stat().st_size < 80_000_000:
        raise SystemExit(f"poids manquant: {pt}")
    dataset = job["dataset"]
    state = _files_ready(dataset, weight)
    print(f"[ds {weight}]", state, flush=True)
    if state == "ready":
        print("[ds] deja pret", dataset, flush=True)
        return
    cmd = [
        sys.executable, "-X", "utf8", str(UPLOAD_PY),
        "--folder", str(folder),
        "--message", job["title"],
    ]
    if state == "pending":
        cmd.append("--version")
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    t0 = time.time()
    while True:
        state = _files_ready(dataset, weight)
        print(f"[ds {weight}]", state, flush=True)
        if state == "ready":
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout dataset {dataset}")
        time.sleep(20)


def _quota(text: str) -> bool:
    low = text.lower()
    return "maximum batch gpu" in low or ("quota" in low and "gpu" in low)


def push_kernel() -> str:
    meta = json.loads((PUSH / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("id") != KERNEL:
        raise SystemExit(meta.get("id"))
    if meta.get("enable_internet"):
        raise SystemExit("internet off requis")
    nb = (PUSH / "submit.ipynb").read_text(encoding="utf-8")
    if "TTA_VARIANTS" in nb or "sans TTA" not in nb:
        raise SystemExit("TTA incoherent")
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
        print("kernel version", version, flush=True)
        return version


def wait_kernel() -> None:
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
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def collect(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    log = next(OUT.glob("*.log"), None)
    csv = OUT / "submission.csv"
    if log is None or not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit("sortie incomplete")
    text = log.read_text(encoding="utf-8", errors="replace")
    if "sans TTA" not in text or "lignes 3" not in text:
        raise SystemExit("log inattendu")
    for needle in (
        "v20b_crop130_dino_ep11_oof_v3b_full",
        "v20h_crop130_dino_ep11_oof_v3b_full",
        "v20h_crop130_oofmask_v20b_lat_full",
        "vit_k_llrd08_det_oofmask_full",
        "swa-7-8-9-10",
        "swa-6-7-8-9",
        "swa-7-8-9-10-11-12",
        "vit 0.41",
        "vk ",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if text.count("lat True") != 1:
        raise SystemExit("lat mal charge")
    if text.count("lat False") < 2:
        raise SystemExit("flip inattendu")
    if "TTA x" in text:
        raise SystemExit("TTA dans le log")
    print("KERNEL_PRET", KERNEL, "version", version, flush=True)


def submit(version: str) -> None:
    try:
        print(kaggle(
            "competitions", "submit", COMPETITION,
            "-k", KERNEL, "-v", version,
            "-f", "submission.csv", "-m", MSG,
        ), flush=True)
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        low = text.lower()
        limited = ("maximum" in low and "submission" in low) or "daily" in low or "limit" in low
        if limited:
            print("LIMITE_JOURNALIERE", KERNEL, "version", version, flush=True)
            print("kernel pret, soumission competition non faite", flush=True)
            return
        raise
    print("SOUMIS", KERNEL, "version", version, flush=True)


def main() -> None:
    prep = _prep()
    for job in prep.PACKS:
        ensure_dataset(job)
    version = push_kernel()
    wait_kernel()
    collect(version)
    submit(version)


if __name__ == "__main__":
    main()
