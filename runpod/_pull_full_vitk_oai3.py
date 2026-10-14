# -*- coding: utf-8 -*-
"""Rapatrie au fil de l'eau le full train vit_k_llrd08_320_v3b_oofmask_vitk_full_oai3 :
history.json, last.pt, best.pt et les snapshots epoch5.pt .. epoch11.pt
(cap utilisateur : 11 epochs max).

Sensible au mtime (tous les .pt ont la meme taille). Un fichier n'est tire
que lorsqu'il est stable (meme taille+mtime sur deux listings, mtime > 20 s).
Priorite a history.json puis aux snapshots dans l'ordre. S'arrete quand
history.json a 11 epochs et que epoch11.pt + last.pt locaux sont a jour.

    python runpod/_pull_full_vitk_oai3.py [--once] [--every 60]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import msvcrt
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = "vit_k_llrd08_320_v3b_oofmask_vitk_full_oai3"
REL = f"checkpoints/{RUN}"
SNAPS = tuple(f"epoch{e}.pt" for e in range(5, 12))
NAMES = ("history.json",) + SNAPS + ("last.pt", "best.pt")
EPOCHS = 11
LOCK = ROOT / "checkpoints" / RUN / "_pull_full.lock"
LISTING = (
    f"cd /workspace && find {REL} -maxdepth 1 -type f "
    "\\( -name history.json -o -name last.pt -o -name best.pt -o -name 'epoch*.pt' \\) "
    "-printf '%p %s %T@\\n'"
)

spec = importlib.util.spec_from_file_location(
    "oai_pull", ROOT / "runpod" / "oai" / "pull.py")
pullmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pullmod)
ssh = pullmod.ssh
pull = pullmod.pull
log = pullmod.log


def listing() -> dict[str, tuple[int, float]]:
    proc = ssh(LISTING, timeout=60)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"listing rc={proc.returncode} {proc.stderr.strip()[:200]}")
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].rsplit("/", 1)[-1] in NAMES:
            out[parts[0]] = (int(parts[1]), float(parts[2]))
    return out


def epochs_of(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        hist = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return len(hist) if isinstance(hist, list) else 0


def up_to_date(rel: str, size: int, mtime: float) -> bool:
    local = ROOT / rel
    if not local.is_file() or local.stat().st_size != size:
        return False
    return local.stat().st_mtime >= mtime - 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_f = open(LOCK, "a+")
    try:
        msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("puller full oai3 deja lance")

    seen: dict[str, tuple[int, float]] = {}
    files: dict[str, tuple[int, float]] = {}
    log(f"puller full {RUN} (snapshots {SNAPS[0]}..{SNAPS[-1]})")
    while True:
        try:
            files = listing()
            now = time.time()
            todo = sorted(files.items(),
                          key=lambda kv: (NAMES.index(kv[0].rsplit('/', 1)[-1])
                                          if kv[0].rsplit('/', 1)[-1] in NAMES else 99))
            for rel, (size, mtime) in todo:
                prev = seen.get(rel)
                seen[rel] = (size, mtime)
                if prev != (size, mtime) or now - mtime < 20:
                    continue
                if up_to_date(rel, size, mtime):
                    continue
                pull(rel, size)
                if rel.endswith("history.json"):
                    log(f"history : {epochs_of(ROOT / rel)}/{EPOCHS} epochs")
        except Exception as exc:  # noqa: BLE001
            log(f"retry : {type(exc).__name__} {exc}")
        done_snaps = f"{REL}/epoch{EPOCHS}.pt" in files and up_to_date(
            f"{REL}/epoch{EPOCHS}.pt", *files.get(f"{REL}/epoch{EPOCHS}.pt", (0, 0)))
        done_last = f"{REL}/last.pt" in files and up_to_date(
            f"{REL}/last.pt", *files.get(f"{REL}/last.pt", (0, 0)))
        if epochs_of(ROOT / REL / "history.json") >= EPOCHS and done_snaps and done_last:
            log("full train rapatrie, fin du puller")
            return
        if args.once:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
