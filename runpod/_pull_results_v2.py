# -*- coding: utf-8 -*-
"""Rapatrie history.json et last.pt de IEEE/results-v2 pendant le train pod.

last.pt n'est tire que si sa taille distante est stable d'un passage a
l'autre, pour ne pas ramener un fichier encore en cours d'ecriture.

    python -X utf8 runpod/_pull_results_v2.py --host 216.243.220.170 --port 16198
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "IEEE" / "results-v2"
REMOTE_ROOT = "/workspace/IEEE/results-v2"
RUNS = (
    [f"convnext_s{s}" for s in range(44, 48)]
    + [f"vit_s{s}" for s in range(42, 48)]
)
EPOCHS = 7


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def ssh(args: argparse.Namespace, remote: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh", "-p", str(args.port), "-i", args.key,
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-o", "StrictHostKeyChecking=accept-new",
        f"root@{args.host}", remote,
    ]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def scp(args: argparse.Namespace, remote: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    cmd = [
        "scp", "-P", str(args.port), "-i", args.key,
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        f"root@{args.host}:{remote}", str(partial),
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"scp {remote} rc={proc.returncode}")
    partial.replace(dest)


def listing(args: argparse.Namespace) -> dict[str, tuple[int, float]]:
    remote = (
        f"find {REMOTE_ROOT} -mindepth 2 -maxdepth 2 "
        "\\( -name history.json -o -name last.pt \\) -printf '%p %s %T@\\n'"
    )
    proc = ssh(args, remote)
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr.strip()[:300])
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1].isdigit():
            out[parts[0]] = (int(parts[1]), float(parts[2]))
    return out


def epochs_of(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        hist = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return -1
    return len(hist) if isinstance(hist, list) else 0


def pull_history(args: argparse.Namespace, run: str, size: int, mtime: float) -> None:
    dest = DEST / run / "history.json"
    if dest.is_file() and dest.stat().st_size == size and dest.stat().st_mtime >= mtime - 2 and epochs_of(dest) > 0:
        return
    scp(args, f"{REMOTE_ROOT}/{run}/history.json", dest)
    n = epochs_of(dest)
    if n < 0:
        log(f"{run} history json illisible, on reessaiera")
        return
    log(f"{run} history {n}/{EPOCHS}")


def pull_last(args: argparse.Namespace, run: str, size: int, mtime: float) -> None:
    dest = DEST / run / "last.pt"
    if dest.is_file() and dest.stat().st_size == size and dest.stat().st_mtime >= mtime - 2:
        return
    scp(args, f"{REMOTE_ROOT}/{run}/last.pt", dest)
    if dest.stat().st_size != size:
        raise RuntimeError(f"{run} last.pt taille {dest.stat().st_size} != {size}")
    os.utime(dest, (mtime, mtime))
    log(f"{run} last.pt {size / 1e6:.0f} Mo")


def finished(args_runs: list[str]) -> bool:
    for run in args_runs:
        if epochs_of(DEST / run / "history.json") < EPOCHS:
            return False
        if not (DEST / run / "last.pt").is_file():
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--key", default=os.path.expanduser("~/.ssh/id_ed25519_runpod"))
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()
    seen: dict[str, int] = {}
    log(f"pull -> {DEST}")
    while True:
        try:
            files = listing(args)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            log(f"listing: {exc}")
            time.sleep(args.every)
            continue
        stable: dict[str, tuple[int, float]] = {}
        for run in RUNS:
            hist = f"{REMOTE_ROOT}/{run}/history.json"
            last = f"{REMOTE_ROOT}/{run}/last.pt"
            if hist in files:
                try:
                    pull_history(args, run, *files[hist])
                except RuntimeError as exc:
                    log(str(exc))
            if last in files:
                if seen.get(last) == files[last][0]:
                    stable[run] = files[last]
                seen[last] = files[last][0]
        pulled = False
        for run, (size, mtime) in stable.items():
            dest = DEST / run / "last.pt"
            if dest.is_file() and dest.stat().st_size == size and dest.stat().st_mtime >= mtime - 2:
                continue
            try:
                pull_last(args, run, size, mtime)
            except RuntimeError as exc:
                log(str(exc))
            pulled = True
            break
        local = " ".join(
            f"{run}:{epochs_of(DEST / run / 'history.json')}" for run in RUNS
        )
        log(f"local {local}")
        if finished(list(RUNS)):
            log("10 runs a 7/7 en local")
            return
        time.sleep(20 if pulled else args.every)


if __name__ == "__main__":
    main()
