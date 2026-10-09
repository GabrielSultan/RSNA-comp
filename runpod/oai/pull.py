# -*- coding: utf-8 -*-
"""Puller local unique (lecture seule sur le pod) : snapshots epoch*.pt et
history.json des runs *_oai3, puis shards OAI 224 (320 avec --shards all)
et leurs manifestes. Priorite dans cet ordre ; un fichier n'est tire que s'il
reste sa taille + 3 Go sur le disque local. Ecriture via .partial, taille
verifiee, puis rename. Un verrou empeche deux pullers de se battre sur un .partial.

    python runpod/oai/pull.py [--once] [--shards 224|all|none]
"""
from __future__ import annotations

import argparse
import msvcrt
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runpod"))
from _pod import ssh, ssh_cmd  # noqa: E402

RUNS = [
    "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
    "v20h_crop130_dino_ep11_oof_v3b_full_oai3",
    "coat_a_llrd08_v3b_oofmask_full_oai3",
    "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
]
EPOCHS = 3
N_BATCHES = 12
FREE_MARGIN = 3_000_000_000
LOCK = ROOT / "data" / "oai" / "_pull.lock"
HIST_SEEN: dict[str, tuple[int, float]] = {}
LISTING = (
    "cd /workspace && date +%s && "
    "find checkpoints -mindepth 2 -maxdepth 2 -path '*_oai3/*' "
    "\\( -name 'epoch*.pt' -o -name history.json \\) -printf '%p %s %T@\\n'; "
    "find data/crop130 data/crop130_320 -maxdepth 1 "
    "\\( -name 'oai_crop130*_b*.zip' -o -name 'oai_manifest_b*.csv' \\) -printf '%p %s %T@\\n'"
)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def remote_listing() -> tuple[float, dict[str, tuple[int, float]]]:
    proc = ssh(LISTING, timeout=60)
    lines = proc.stdout.split("\n")
    if proc.returncode not in (0, 1) or not lines or not lines[0].strip().isdigit():
        raise RuntimeError(f"listing rc={proc.returncode} {proc.stderr.strip()[:200]}")
    files = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:
            files[parts[0]] = (int(parts[1]), float(parts[2]))
    return float(lines[0]), files


def plan(now: float, files: dict[str, tuple[int, float]], shards: str) -> list[tuple[str, int]]:
    """Fichiers a tirer, par priorite ; history.json seulement s'il a change."""
    snaps, hist, sh224, sh320 = [], [], [], []
    for p, (size, mtime) in files.items():
        local = ROOT / p
        if local.is_file() and local.stat().st_size == size and not p.endswith("history.json"):
            continue
        if p.endswith(".pt"):
            snaps.append((p, size))
        elif p.endswith("history.json"):
            if now - mtime > 10 and HIST_SEEN.get(p) != (size, mtime):
                hist.append((p, size))
        elif shards != "none":
            m = re.search(r"_b(\d+)\.(zip|csv)$", p)
            d = p.rsplit("/", 1)[0]
            if d.endswith("_320") and shards != "all":
                continue
            prefix = "oai_crop130_320" if d.endswith("_320") else "oai_crop130"
            if p.endswith(".csv") and f"{d}/{prefix}_b{m.group(1)}.zip" not in files:
                continue
            (sh320 if d.endswith("_320") else sh224).append((p, size))
    key = lambda t: t[0]  # noqa: E731
    return sorted(snaps, key=key) + hist + sorted(sh224, key=key) + sorted(sh320, key=key)


def pull(rel: str, size: int) -> bool:
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest.parent).free
    if free < size + FREE_MARGIN:
        log(f"attente disque : {rel} ({size / 1e9:.2f} Go, {free / 1e9:.1f} Go libres)")
        return False
    tmp = dest.with_name(dest.name + ".partial")
    tmp.unlink(missing_ok=True)
    t0 = time.time()
    proc = subprocess.Popen(ssh_cmd(f"cat /workspace/{rel}"), stdout=subprocess.PIPE)
    assert proc.stdout is not None
    got = 0
    try:
        with open(tmp, "wb") as handle:
            while buf := proc.stdout.read(1024 * 1024):
                handle.write(buf)
                got += len(buf)
    except OSError:
        proc.kill()
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise
    rc = proc.wait()
    if rc != 0 or got != size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{rel} incomplet {got}/{size} rc={rc}")
    for _ in range(30):
        try:
            os.replace(tmp, dest)
            break
        except OSError as exc:
            log(f"rename retry {exc}")
            time.sleep(2)
    else:
        raise RuntimeError(f"rename {rel}")
    dt = max(time.time() - t0, 1e-3)
    if size > 10_000_000:
        log(f"OK {rel} ({size / 1e9:.2f} Go, {size / dt / 1e6:.0f} Mo/s)")
    return True


def complete(shards: str) -> bool:
    snaps = all((ROOT / "checkpoints" / r / f"epoch{e}.pt").is_file()
                for r in RUNS for e in range(1, EPOCHS + 1))
    n224 = len(list((ROOT / "data/crop130").glob("oai_crop130_b*.zip")))
    n320 = len(list((ROOT / "data/crop130_320").glob("oai_crop130_320_b*.zip")))
    return (snaps and (shards == "none" or n224 == N_BATCHES)
            and (shards != "all" or n320 == N_BATCHES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--shards", choices=["224", "all", "none"], default="224",
                    help="shards OAI a rapatrier (defaut : 224 seulement, place sur C:)")
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()
    shards = args.shards

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_f = open(LOCK, "a+")
    try:
        msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("un autre puller tourne deja (verrou data/oai/_pull.lock)")

    log(f"puller : {len(RUNS)} runs x {EPOCHS} epochs | shards OAI : {shards}")
    while True:
        try:
            now, files = remote_listing()
            todo = plan(now, files, shards)
            for rel, size in todo:
                if pull(rel, size):
                    if rel.endswith("history.json"):
                        HIST_SEEN[rel] = files[rel]
                elif rel.endswith(".pt"):
                    break
        except Exception as exc:  # noqa: BLE001
            log(f"retry : {type(exc).__name__} {exc}")
        if complete(shards):
            log("tout est rapatrie, fin du puller")
            return
        if args.once:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
