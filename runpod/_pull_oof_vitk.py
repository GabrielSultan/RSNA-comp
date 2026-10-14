# -*- coding: utf-8 -*-
"""Rapatrie history.json, last.pt, best.pt et oof_preds.csv des 3 folds OOF
vit_k_llrd08_oof_v3b pousses a 5 epochs sur pod.

Sensible au mtime : tous les .pt ont la meme taille, une taille identique ne
suffit pas a sauter un fichier. Un fichier n'est tire que lorsqu'il est
stable (meme taille+mtime sur deux listings, mtime > 20 s). S'arrete quand
chaque fold a history.json >= 5 epochs et que last.pt / best.pt /
oof_preds.csv locaux sont a jour (taille + mtime distants).

    python runpod/_pull_oof_vitk.py [--once] [--every 45]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import msvcrt
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = "vit_k_llrd08_oof_v3b"
FOLDS = (0, 1, 2)
NAMES = ("history.json", "last.pt", "best.pt", "oof_preds.csv")
EPOCHS = 5
LOCK = ROOT / "checkpoints" / "_pull_oof_vitk.lock"
LISTING = (
    "cd /workspace && find checkpoints -mindepth 2 -maxdepth 2 "
    "\\( -path '*vit_k_llrd08_oof_v3b_fold*' \\) "
    "\\( -name history.json -o -name last.pt -o -name best.pt -o -name oof_preds.csv \\) "
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
        if len(parts) == 3 and parts[0].startswith("checkpoints/"):
            if parts[0].rsplit("/", 1)[-1] in NAMES:
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


def fold_done(k: int, files: dict[str, tuple[int, float]]) -> bool:
    rel = f"checkpoints/{RUN}_fold{k}"
    if epochs_of(ROOT / rel / "history.json") < EPOCHS:
        return False
    return all(
        f"{rel}/{name}" in files and up_to_date(f"{rel}/{name}", *files[f"{rel}/{name}"])
        for name in ("last.pt", "best.pt", "oof_preds.csv")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=45)
    args = ap.parse_args()

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_f = open(LOCK, "a+")
    try:
        msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("puller folds OOF deja lance")

    seen: dict[str, tuple[int, float]] = {}
    files: dict[str, tuple[int, float]] = {}
    log(f"puller folds {RUN} -> {EPOCHS} epochs")
    while True:
        try:
            files = listing()
            now = time.time()
            for rel, (size, mtime) in sorted(files.items()):
                prev = seen.get(rel)
                seen[rel] = (size, mtime)
                if prev != (size, mtime) or now - mtime < 20:
                    continue
                if up_to_date(rel, size, mtime):
                    continue
                pull(rel, size)
                if rel.endswith("history.json"):
                    log(f"{rel} : {epochs_of(ROOT / rel)}/{EPOCHS} epochs")
        except Exception as exc:  # noqa: BLE001
            log(f"retry : {type(exc).__name__} {exc}")
        if files and all(fold_done(k, files) for k in FOLDS):
            log("folds rapatries, fin du puller")
            return
        if args.once:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
