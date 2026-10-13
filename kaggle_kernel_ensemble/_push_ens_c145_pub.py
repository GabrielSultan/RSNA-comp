# -*- coding: utf-8 -*-
"""Pousse le mix public. Pas de soumission competition."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUSH = ROOT / "kaggle_kernel_ensemble" / "_push_ens_c145_pub"
OUT = ROOT / "kaggle_kernel_ensemble" / "_out_ens_c145_pub"
KERNEL = "sultent/rsna-knee-ens-c145-pub"
POLL_S = 30
MAX_WAIT_S = 3 * 3600


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


def push() -> str:
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(PUSH),
                "--accelerator", "NvidiaTeslaT4", "-t", "32400",
            )
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if "quota" in text.lower() or "maximum batch gpu" in text.lower():
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if "quota" in out.lower() or "maximum batch gpu" in out.lower():
            print("quota GPU, retry 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        print("kernel version", m.group(1), flush=True)
        return m.group(1)


def wait() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KERNEL).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def collect() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    logs = [p for p in OUT.glob("*.log") if "cufile" not in p.name.lower()]
    log = next(p for p in logs if "rsna-knee-ens-c145-pub" in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "swa-6-7-8",
        "v20h 145 mm",
        "MIX4_SCORED",
        "GPU mix4 libere",
        "poids mix4 0.60 public 0.40",
        "BLEND5_OK",
        "NOTEBOOK_COMPLETE",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text or "lat True" in text:
        raise SystemExit("TTA ou flip")
    if "0.5 * pr" in text:
        raise SystemExit("ancien blend 50/50")
    print("KERNEL_PRET", KERNEL, flush=True)
    version = (PUSH / "_version.txt").read_text(encoding="utf-8").strip()
    return version


def _daily_limit(text: str) -> bool:
    low = text.lower()
    return any(
        s in low
        for s in (
            "daily submission allowance",
            "try again tomorrow",
            "submission limit",
            "maximum number",
        )
    )


def _sleep_until_reset() -> None:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    target = now.replace(hour=0, minute=5, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    delay = (target - now).total_seconds()
    print(f"ATTENTE_QUOTA {delay:.0f}s jusqu a {target.isoformat()}", flush=True)
    time.sleep(delay + 5)


def submit(version: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    msg = (
        "ens c145 mix4 60% + public 0.956 40%, rang, sans TTA, "
        "FOV v20h 145 mm, pas de flip, T4"
    )
    api = KaggleApi()
    api.authenticate()
    waited = False
    while True:
        print("submit", KERNEL, "version", version, flush=True)
        try:
            resp = api.competition_submit_code(
                "submission.csv", msg, "rsna-knee-abnormality-detection",
                KERNEL, int(version), quiet=True,
            )
        except Exception as e:
            http = getattr(e, "response", None)
            body = http.text if http is not None else str(e)
            print(body, flush=True)
            if _daily_limit(body) and not waited:
                waited = True
                _sleep_until_reset()
                continue
            raise
        print(resp, flush=True)
        print("SOUMIS", KERNEL, "version", version, flush=True)
        return


def main() -> None:
    version = push()
    (PUSH / "_version.txt").write_text(version + "\n", encoding="utf-8")
    wait()
    submit(collect())


if __name__ == "__main__":
    main()
