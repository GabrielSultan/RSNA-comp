# -*- coding: utf-8 -*-
"""Enchaine les 4 v20h crop145 esv2. Saute un run deja a 4 epochs.

Reprend sur last.pt s'il existe. S'arrete au premier code non nul.

    python -X utf8 -u -m src.train.run_esv2_queue
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [
    "configs/archive/kneenet_v20h_crop145_dino_ep11_v3b_rsna1687_esv2_soft.yaml",
    "configs/archive/kneenet_v20h_crop145_dino_ep11_v3b_rsna1687_esv2_calibrated.yaml",
    "configs/archive/kneenet_v20h_crop145_dino_ep11_v3b_rsna1687_esv2_accuracy.yaml",
    "configs/archive/kneenet_v20h_crop145_dino_ep11_v3b_rsna1687_esv2_strictmri.yaml",
]
EPOCHS = 4


def _run_name(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("run_name:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"run_name absent : {path}")


def _log(path: Path, msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    queue_log = ROOT / "logs" / "v20h_esv2_queue.log"
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    _log(queue_log, "QUEUE START")
    for cfg in CONFIGS:
        cfg_path = ROOT / cfg
        name = _run_name(cfg_path)
        hist = ROOT / "checkpoints" / name / "history.json"
        if hist.is_file():
            n = len(json.loads(hist.read_text(encoding="utf-8")))
            if n >= EPOCHS:
                _log(queue_log, f"SKIP {name} history={n}")
                continue
        last = ROOT / "checkpoints" / name / "last.pt"
        cmd = [
            sys.executable, "-X", "utf8", "-u", "-m", "src.train.train",
            "--config", cfg,
        ]
        if last.is_file():
            cmd.append("--resume")
        run_log = ROOT / "logs" / f"{name}.log"
        _log(queue_log, f"START {name} resume={last.is_file()}")
        with run_log.open("a", encoding="utf-8") as out:
            out.write(
                f"\n==== START {datetime.now().isoformat(timespec='seconds')} "
                f"{' '.join(cmd)} ====\n"
            )
            out.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        _log(queue_log, f"EXIT {name} code={proc.returncode}")
        if proc.returncode != 0:
            _log(queue_log, "STOP queue")
            raise SystemExit(proc.returncode)
    _log(queue_log, "QUEUE DONE")


if __name__ == "__main__":
    main()
