# -*- coding: utf-8 -*-
"""Rapatrie history.json, last.pt et best.pt du hybrid ViT-k 320.

Lecture seule sur le pod. Un fichier n'est tire que taille stable, via
.partial, avec 3 Go de marge sur C:. S'arrete quand history.json a 7
epochs et que last.pt / best.pt locaux ont la taille distante.
"""
from __future__ import annotations

import importlib.util
import json
import msvcrt
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = "subset_vit_k_llrd08_320_v3b_hybrid025"
REL = f"checkpoints/{RUN}"
NAMES = ("history.json", "last.pt", "best.pt")
LOCK = ROOT / "checkpoints" / RUN / "_pull.lock"
EPOCHS = 7

spec = importlib.util.spec_from_file_location(
    "oai_pull", ROOT / "runpod" / "oai" / "pull.py")
pullmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pullmod)
ssh = pullmod.ssh
pull = pullmod.pull
log = pullmod.log


def listing() -> dict[str, tuple[int, float]]:
    remote = (
        f"find /workspace/{REL} -maxdepth 1 -type f "
        "\\( -name history.json -o -name last.pt -o -name best.pt \\) "
        "-printf '%f %s %T@\\n'"
    )
    proc = ssh(remote, timeout=40)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"listing rc={proc.returncode} {proc.stderr.strip()[:200]}")
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in NAMES:
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


def main() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_f = open(LOCK, "a+")
    try:
        msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("puller hybrid deja lance")
    seen: dict[str, tuple[int, float]] = {}
    stable: dict[str, tuple[int, float]] = {}
    files: dict[str, tuple[int, float]] = {}
    log(f"puller hybrid {RUN}")
    while True:
        try:
            files = listing()
            now = time.time()
            for name in NAMES:
                if name not in files:
                    continue
                size, mtime = files[name]
                prev = seen.get(name)
                seen[name] = (size, mtime)
                if prev != (size, mtime) or now - mtime < 20:
                    continue
                stable[name] = (size, mtime)
                local = ROOT / REL / name
                # tailles identiques entre epochs : la taille seule ne suffit
                # pas, il faut aussi que le mtime local suive le distant.
                if (name != "history.json" and local.is_file()
                        and local.stat().st_size == size
                        and local.stat().st_mtime >= mtime - 2):
                    continue
                if name == "history.json" and local.is_file() and local.stat().st_size == size:
                    if epochs_of(local) >= EPOCHS and stable.get("last.pt") and stable.get("best.pt"):
                        continue
                    if local.stat().st_mtime >= mtime - 2 and epochs_of(local) > 0:
                        continue
                pull(f"{REL}/{name}", size)
                if name == "history.json":
                    log(f"history : {epochs_of(ROOT / REL / name)}/{EPOCHS} epochs")
        except Exception as exc:  # noqa: BLE001
            log(f"retry : {type(exc).__name__} {exc}")
        local_hist = ROOT / REL / "history.json"
        n = epochs_of(local_hist)
        if n >= EPOCHS and all(
            (ROOT / REL / name).is_file()
            and name in files
            and (ROOT / REL / name).stat().st_size == files[name][0]
            and (ROOT / REL / name).stat().st_mtime >= files[name][1] - 2
            for name in ("last.pt", "best.pt")
        ):
            log("hybrid rapatrie, fin du puller")
            return
        time.sleep(45)


if __name__ == "__main__":
    main()
