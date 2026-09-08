# -*- coding: utf-8 -*-
"""Push les 6 kernels convert-rgb25d-proto et lance un run CPU 9 h chacun.

Kaggle limite a 5 sessions CPU batch. Les shards au-dela sont reessais
quand un slot se libere (--wait-quota).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
N_SHARDS = 6
TIMEOUT_S = 32400  # 9 h
SLUG = "sultent/rsna-knee-convert-rgb-2-5d-proto-part"
POLL_S = 180


def _kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _push(d: Path) -> str:
    cmd = [sys.executable, "-m", "kaggle", "kernels", "push",
           "-p", str(d), "-t", str(TIMEOUT_S)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or _quota_full(out) or "kernel push error" in out.lower():
        err = subprocess.CalledProcessError(proc.returncode or 1, cmd, output=out)
        raise err
    return out


def _quota_full(msg: str) -> bool:
    m = msg.lower()
    return "session count" in m or "maximum batch" in m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default="", help="ex. 5 ou 3,4,5 (defaut : 0-5)")
    p.add_argument("--wait-quota", action="store_true",
                   help="reessayer si le quota CPU 5/5 est plein")
    args = p.parse_args()
    if args.shards.strip():
        shards = [int(x) for x in args.shards.split(",")]
    else:
        shards = list(range(N_SHARDS))

    pending = list(shards)
    while pending:
        shard = pending[0]
        d = ROOT / f"convert_rgb25d_proto_part{shard}"
        if not (d / "convert.ipynb").is_file() or not (d / "kernel-metadata.json").is_file():
            raise SystemExit(f"manque convert.ipynb / metadata dans {d}")
        print("== push", d.name, flush=True)
        try:
            print(_push(d), flush=True)
            pending.pop(0)
            time.sleep(8)
        except subprocess.CalledProcessError as e:
            msg = e.output or ""
            print(msg, flush=True)
            if args.wait_quota and _quota_full(msg):
                print(f"quota CPU plein, retry part{shard} dans {POLL_S}s", flush=True)
                time.sleep(POLL_S)
                continue
            raise

    print("status :", flush=True)
    for shard in shards:
        slug = f"{SLUG}{shard}"
        try:
            print(_kaggle("kernels", "status", slug).strip(), flush=True)
        except subprocess.CalledProcessError as e:
            print(slug, "status error:", e.output, flush=True)


if __name__ == "__main__":
    main()
