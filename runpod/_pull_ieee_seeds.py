# -*- coding: utf-8 -*-
"""Rapatrie les 9 seeds ISBI dans IEEE/results (pas dans checkpoints/).

history.json a chaque epoch, last.pt / best.pt des qu'ils sont stables,
puis les CSV et JSON du score OAI. S'arrete quand les 9 runs ont 7 epochs
et que les 9 paires OAI sont locales et a jour.

    python runpod/_pull_ieee_seeds.py [--once] [--every 45]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import msvcrt
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "IEEE" / "results"
MODELS = ("v20h_rsna", "cnx_r7_rsna", "vitk224_rsna")
SEEDS = (42, 43, 44)
EPOCHS = 7
RUNS = tuple(f"{m}_s{s}" for m in MODELS for s in SEEDS)
LOCK = DEST / "_pull.lock"
LISTING = (
    "cd /workspace && "
    "find checkpoints/ieee_seeds -mindepth 2 -maxdepth 2 -type f "
    "\\( -name history.json -o -name last.pt -o -name best.pt \\) "
    "-printf '%p %s %T@\\n'; "
    "find IEEE/seeds/preds -maxdepth 1 -type f "
    "\\( -name 'oai__*.csv' -o -name 'oai_auc__*.json' \\) "
    "-printf '%p %s %T@\\n'"
)

spec = importlib.util.spec_from_file_location(
    "oai_pull", ROOT / "runpod" / "oai" / "pull.py")
pullmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pullmod)
ssh = pullmod.ssh
pull = pullmod.pull
log = pullmod.log


def local_of(rel: str) -> Path:
    if rel.startswith("checkpoints/ieee_seeds/"):
        return DEST / rel[len("checkpoints/ieee_seeds/"):]
    if rel.startswith("IEEE/seeds/preds/"):
        return DEST / "preds" / rel[len("IEEE/seeds/preds/"):]
    raise ValueError(rel)


def listing() -> dict[str, tuple[int, float]]:
    proc = ssh(LISTING, timeout=60)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"listing rc={proc.returncode} {proc.stderr.strip()[:200]}")
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and (
            parts[0].startswith("checkpoints/ieee_seeds/")
            or parts[0].startswith("IEEE/seeds/preds/oai")
        ):
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
    local = local_of(rel)
    if not local.is_file() or local.stat().st_size != size:
        return False
    return local.stat().st_mtime >= mtime - 2


def pull_into(rel: str, size: int) -> None:
    """Tire vers un chemin temporaire sous la destination, puis le pull
    standard ecrit a la racine du repo : on deplace ensuite."""
    dest = local_of(rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = ROOT / rel
    if staging.resolve() == dest.resolve():
        pull(rel, size)
        return
    pull(rel, size)
    if not staging.is_file():
        raise RuntimeError(f"pull sans fichier {staging}")
    staging.replace(dest)
    # retire les dossiers vides crees par le pull sous checkpoints/ ou IEEE/seeds/preds
    parent = staging.parent
    while parent != ROOT and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def run_done(run: str, files: dict[str, tuple[int, float]]) -> bool:
    base = f"checkpoints/ieee_seeds/{run}"
    if epochs_of(DEST / run / "history.json") < EPOCHS:
        return False
    if not all(
        f"{base}/{name}" in files and up_to_date(f"{base}/{name}", *files[f"{base}/{name}"])
        for name in ("last.pt", "best.pt")
    ):
        return False
    short = run  # v20h_rsna_s42
    csv = f"IEEE/seeds/preds/oai__{short}.csv"
    js = f"IEEE/seeds/preds/oai_auc__{short}.json"
    return all(p in files and up_to_date(p, *files[p]) for p in (csv, js))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=45)
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    lock_f = open(LOCK, "a+")
    try:
        msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("puller ieee seeds deja lance")

    seen: dict[str, tuple[int, float]] = {}
    files: dict[str, tuple[int, float]] = {}
    log(f"puller ieee seeds -> {DEST.relative_to(ROOT)}")
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
                pull_into(rel, size)
                if rel.endswith("history.json"):
                    log(f"{rel} : {epochs_of(local_of(rel))}/{EPOCHS} epochs")
        except Exception as exc:  # noqa: BLE001
            log(f"retry : {type(exc).__name__} {exc}")
        if files and all(run_done(r, files) for r in RUNS):
            log("9 seeds rapatries, fin du puller")
            return
        if args.once:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
