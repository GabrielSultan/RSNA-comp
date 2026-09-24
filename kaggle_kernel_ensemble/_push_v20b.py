# -*- coding: utf-8 -*-
"""Push kernel T4 v20b SWA ep8/9/11, attend COMPLETE, telecharge (submit optionnel)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
MAKE_NB = KERNEL_DIR / "_make_nb_v20b.py"
COMPETITION = "rsna-knee-abnormality-detection"
POLL_EVERY_S = 60
MAX_WAIT_S = 8 * 3600
TIMEOUT_S = "32400"
ACCEL = "NvidiaTeslaT4"
KERNEL = "sultent/rsna-knee-v20b-solo-swa"
PUSH = KERNEL_DIR / "_push_v20b_swa"
OUT = KERNEL_DIR / "_out_v20b_swa"


def _kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _parse_version(out: str) -> str:
    import re
    if "successfully pushed" not in out.lower():
        raise RuntimeError(f"push kernel echoue : {out.strip()!r}")
    m = re.search(r"Kernel version (\d+) successfully pushed", out, flags=re.I)
    if not m:
        raise RuntimeError(f"version kernel introuvable dans : {out!r}")
    return m.group(1)


def _status_upper(kernel: str) -> tuple[str, str]:
    raw = _kaggle("kernels", "status", kernel).strip()
    return raw, raw.upper()


def _is_failed(status_u: str) -> bool:
    return "FAILED" in status_u or "ERROR" in status_u


def _is_complete(status_u: str) -> bool:
    running = "RUNNING" in status_u
    done = ("COMPLETE" in status_u) or ("SUCCEEDED" in status_u)
    return done and not running


def try_push() -> str | None:
    meta = PUSH / "kernel-metadata.json"
    nb = PUSH / "submit.ipynb"
    if not meta.is_file() or not nb.is_file():
        raise FileNotFoundError(PUSH)
    print("push", KERNEL, "T4", TIMEOUT_S, "s", flush=True)
    try:
        out = _kaggle(
            "kernels", "push",
            "-p", str(PUSH),
            "--accelerator", ACCEL,
            "-t", TIMEOUT_S,
        )
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        if "Maximum batch GPU" in text or "500 Server Error" in text:
            print("push refuse, retry plus tard", flush=True)
            return None
        raise
    print(out, flush=True)
    ver = _parse_version(out)
    print("kernel version", KERNEL, ver, flush=True)
    return ver


def download_one() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = _kaggle("kernels", "output", KERNEL, "-p", str(OUT))
    print(out, flush=True)
    csv = OUT / "submission.csv"
    print("downloaded", csv, "exists", csv.exists(), flush=True)
    if not csv.exists():
        raise SystemExit(f"submission.csv absent dans {OUT}")


def submit_one(version: str, *, submit: bool = True) -> None:
    if submit:
        msg = "v20b oofmask SWA ep8-9-11 (moyenne de poids, pas de laterality, T4)"
        submit_out = _kaggle(
            "competitions", "submit", COMPETITION,
            "-k", KERNEL, "-v", version, "-f", "submission.csv", "-m", msg,
        )
        print(submit_out, flush=True)
        print("submitted", KERNEL, "version", version, flush=True)
    else:
        print("pas de soumission competition (quota) :", KERNEL, "v", version,
              flush=True)
    download_one()


def wait_and_submit(version: str | None, leftover: bool, *, submit: bool) -> str:
    t0 = time.time()
    seen_running = False
    last_push_try = 0.0
    pending = version is not None
    while pending or leftover:
        progressed = False
        if leftover and (time.time() - last_push_try) >= 60:
            last_push_try = time.time()
            ver = try_push()
            if ver is None:
                leftover = True
            else:
                version = ver
                leftover = False
                pending = True
                seen_running = False
                progressed = True
        if pending:
            raw, status_u = _status_upper(KERNEL)
            print(f"[swa] {raw}", flush=True)
            if _is_failed(status_u):
                raise SystemExit("kernel v20b swa failed")
            if "RUNNING" in status_u or "QUEUED" in status_u or "PENDING" in status_u:
                seen_running = True
            ready = _is_complete(status_u) and (
                seen_running or (time.time() - t0) > 120
            )
            if ready:
                submit_one(version, submit=submit)
                pending = False
                progressed = True
                last_push_try = 0.0
        if not pending and not leftover:
            break
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout waiting version={version} leftover={leftover}")
        if not progressed:
            time.sleep(POLL_EVERY_S)
    return version


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--push-only", action="store_true")
    p.add_argument("--no-submit", action="store_true",
                   help="attend COMPLETE et telecharge, sans soumettre a la competition")
    p.add_argument("--submit-only", action="store_true")
    p.add_argument("--version", default=None)
    args = p.parse_args()

    version = args.version
    leftover = False
    if not args.submit_only:
        subprocess.check_call([sys.executable, "-X", "utf8", str(MAKE_NB)])
        if version is None:
            ver = try_push()
            if ver is None:
                leftover = True
            else:
                version = ver

    if args.push_only:
        print("version", version, "leftover", leftover, flush=True)
        return

    if version is None and not leftover:
        raise SystemExit("rien a attendre ni a pousser")
    version = wait_and_submit(version, leftover, submit=not args.no_submit)
    print("DONE", {0: version}, flush=True)


if __name__ == "__main__":
    main()
