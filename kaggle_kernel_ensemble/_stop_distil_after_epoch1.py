# -*- coding: utf-8 -*-
"""Attend last.pt de l'epoch 1, verifie le contenu, arrete le train."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "checkpoints" / "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full"
HIST = RUN / "history.json"
LAST = RUN / "last.pt"
TRAIN_NEEDLE = "rsna_distil_main_full"
WATCH_NEEDLE = "_watch_v20h_distil_swa789.py"
WAIT_S = 50 * 60


def say(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _cmds() -> list[tuple[psutil.Process, str]]:
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        out.append((p, cmd))
    return out


def _stop_matching(needle: str) -> list[int]:
    killed = []
    for p, cmd in _cmds():
        if needle not in cmd:
            continue
        if "src.train.train" not in cmd and needle != WATCH_NEEDLE:
            continue
        tree = [p, *p.children(recursive=True)]
        for proc in reversed(tree):
            try:
                proc.terminate()
                killed.append(proc.pid)
            except psutil.NoSuchProcess:
                pass
    gone = set()
    deadline = time.time() + 20
    while time.time() < deadline:
        alive = []
        for pid in killed:
            if pid in gone:
                continue
            if psutil.pid_exists(pid):
                try:
                    psutil.Process(pid).kill()
                except psutil.NoSuchProcess:
                    gone.add(pid)
                else:
                    alive.append(pid)
            else:
                gone.add(pid)
        if not alive:
            break
        time.sleep(0.5)
    return killed


def _history_epoch1() -> bool:
    if not HIST.is_file():
        return False
    try:
        hist = json.loads(HIST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return any(int(r.get("epoch", -1)) == 1 for r in hist)


def _last_stable() -> bool:
    if not LAST.is_file() or LAST.stat().st_size < 80_000_000:
        return False
    if LAST.stat().st_mtime + 1 < HIST.stat().st_mtime:
        return False
    size = LAST.stat().st_size
    time.sleep(8)
    return LAST.is_file() and LAST.stat().st_size == size


def main() -> None:
    watchers = _stop_matching(WATCH_NEEDLE)
    say(f"watcher coupe {watchers or '-'}")
    t0 = time.time()
    while True:
        if _history_epoch1() and _last_stable():
            break
        if time.time() - t0 > WAIT_S:
            raise SystemExit("timeout avant last.pt epoch 1")
        say(
            f"attente | history {HIST.is_file()} | "
            f"last {LAST.stat().st_size if LAST.is_file() else 0}"
        )
        time.sleep(15)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    ckpt = torch.load(LAST, map_location="cpu", weights_only=False)
    ep = int(ckpt.get("epoch", -1))
    step = int(ckpt.get("step", -1))
    if ep != 1:
        raise SystemExit(f"last.pt epoch {ep}, attendu 1")
    if "model" not in ckpt or "optim" not in ckpt:
        raise SystemExit("last.pt incomplet")
    say(f"last.pt OK epoch {ep} step {step} {LAST.stat().st_size} o")
    killed = _stop_matching(TRAIN_NEEDLE)
    say(f"train coupe {killed}")
    still = [
        pid for p, cmd in _cmds()
        if "src.train.train" in cmd and TRAIN_NEEDLE in cmd
    ]
    if still:
        raise SystemExit(f"train encore vivant {still}")
    say("STOP epoch 1")


if __name__ == "__main__":
    main()
