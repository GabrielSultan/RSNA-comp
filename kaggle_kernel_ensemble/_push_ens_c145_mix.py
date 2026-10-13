# -*- coding: utf-8 -*-
"""Pousse les mixes crop145 soft et hard, puis soumet a la competition."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
COMPETITION = "rsna-knee-abnormality-detection"
JOBS = (
    {
        "slug": "rsna-knee-ens-c145-soft",
        "push": ROOT / "kaggle_kernel_ensemble" / "_push_ens_c145_soft",
        "out": ROOT / "kaggle_kernel_ensemble" / "_out_ens_c145_soft",
    },
    {
        "slug": "rsna-knee-ens-c145-hard",
        "push": ROOT / "kaggle_kernel_ensemble" / "_push_ens_c145_hard",
        "out": ROOT / "kaggle_kernel_ensemble" / "_out_ens_c145_hard",
    },
)
NEEDLES = (
    "sans TTA",
    "lignes 3",
    "swa-6-7-8",
    "swa-1-2-3",
    "v20h 145 mm",
    "lat False",
    "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
    "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
    "v20h_crop145_dino_ep11_oof_v3b_full_oaiv2_wd01",
    "coat_a_llrd08_v3b_oofmask_full_oai3",
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


def _daily_limit(text: str) -> bool:
    low = text.lower()
    return any(
        s in low
        for s in (
            "maximum number",
            "submission limit",
            "daily submission allowance",
            "try again tomorrow",
            "too many submission",
            "exceeded",
        )
    )


def push_one(job: dict) -> str:
    kernel = f"sultent/{job['slug']}"
    nb = (job["push"] / "submit.ipynb").read_text(encoding="utf-8")
    if "TTA_VARIANTS" in nb or "laterality=True" in nb or "sans TTA" not in nb:
        raise SystemExit(f"{job['slug']} notebook incoherent")
    if "KneeNetCoat" not in nb or "swa-6-7-8" not in nb or "CROP_MM = 145.0" not in nb:
        raise SystemExit(f"{job['slug']} incomplet")
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
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if _quota(out):
            print("quota GPU, retry 60s", flush=True)
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


def collect(job: dict) -> None:
    kernel = f"sultent/{job['slug']}"
    dest = job["out"]
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", kernel, "-p", str(dest)), flush=True)
    log = next(dest.glob("*.log"), None)
    csv = dest / "submission.csv"
    if log is None or not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit(f"sortie incomplete {job['slug']}")
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in NEEDLES:
        if needle not in text:
            raise SystemExit(f"{job['slug']} log sans {needle}")
    if "TTA x" in text or "lat True" in text:
        raise SystemExit(f"{job['slug']} TTA ou flip")
    if "v20h_crop130_dino_ep11_oof_v3b_full_oai3" in text:
        raise SystemExit(f"{job['slug']} a charge l ancien v20h")
    print("KERNEL_PRET", kernel, flush=True)


def _sleep_until_reset() -> None:
    now = datetime.now(timezone.utc)
    target = now.replace(hour=0, minute=5, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    delay = (target - now).total_seconds()
    print(f"ATTENTE_QUOTA {delay:.0f}s jusqu a {target.isoformat()}", flush=True)
    time.sleep(delay + 5)


def submit_one(job: dict, version: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    kernel = f"sultent/{job['slug']}"
    msg = (job["push"] / "_message.txt").read_text(encoding="utf-8").strip()
    api = KaggleApi()
    api.authenticate()
    print("submit", kernel, "version", version, flush=True)
    try:
        resp = api.competition_submit_code(
            "submission.csv", msg, COMPETITION, kernel, int(version), quiet=True,
        )
    except Exception as e:
        http = getattr(e, "response", None)
        body = http.text if http is not None else str(e)
        print(body, flush=True)
        err = RuntimeError(body)
        raise err
    print(resp, flush=True)
    print("SOUMIS", kernel, "version", version, flush=True)


def submit_all(versions: dict[str, str]) -> None:
    pending = list(JOBS)
    waited = False
    while pending:
        job = pending[0]
        try:
            submit_one(job, versions[job["slug"]])
        except RuntimeError as e:
            text = str(e)
            if _daily_limit(text) and not waited:
                waited = True
                _sleep_until_reset()
                continue
            raise
        pending.pop(0)


def submit_existing() -> None:
    versions = {
        job["slug"]: (job["push"] / "_version.txt").read_text(encoding="utf-8").strip()
        for job in JOBS
    }
    submit_all(versions)
    print("TERMINE", flush=True)


def main() -> None:
    versions = {}
    for job in JOBS:
        versions[job["slug"]] = push_one(job)
    for job in JOBS:
        wait_one(job)
        collect(job)
    submit_all(versions)
    print("TERMINE", flush=True)


if __name__ == "__main__":
    if "--submit-only" in sys.argv:
        submit_existing()
    else:
        main()
