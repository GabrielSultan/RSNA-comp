# -*- coding: utf-8 -*-
"""Attend epoch9.pt du ViT en local, puis lance le full train v20b sur le pod."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

HOST = "root@103.196.86.127"
PORT = "15596"
KEY = os.path.expanduser(r"~/.ssh/id_ed25519_runpod")
ROOT = Path(__file__).resolve().parents[1]
VIT_LOCAL = ROOT / "checkpoints" / "vit_k_llrd08_v3b_oofmask_full"
VIT_REMOTE = "/workspace/checkpoints/vit_k_llrd08_v3b_oofmask_full"
V20B_LOCAL = ROOT / "checkpoints" / "v20b_crop130_dino_ep11_oof_v3b_full"
V20B_REMOTE = "/workspace/checkpoints/v20b_crop130_dino_ep11_oof_v3b_full"
ZIPS = {
    "volumes_crop130_part0.zip": 6369063023,
    "volumes_crop130_part1.zip": 6356536064,
    "volumes_crop130_proto_part0.zip": 1931346236,
    "volumes_crop130_proto_part1.zip": 1931914317,
    "volumes_crop130_t1_part0.zip": 2266705956,
    "volumes_crop130_t1_part1.zip": 2266404234,
}
ENCODER = 94343719


def ssh(remote: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh", "-p", PORT, "-i", KEY,
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=15",
        HOST, remote,
    ]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def remote_size(path: str) -> int | None:
    proc = ssh(f"stat -c %s {path} 2>/dev/null || true")
    text = proc.stdout.strip()
    if not text.isdigit():
        return None
    return int(text)


def wait_epoch9() -> None:
    deadline = time.time() + 8 * 3600
    while time.time() < deadline:
        remote = remote_size(f"{VIT_REMOTE}/epoch9.pt")
        local = VIT_LOCAL / "epoch9.pt"
        local_n = local.stat().st_size if local.is_file() else 0
        print(f"vit epoch9 remote={remote} local={local_n}", flush=True)
        if remote and local_n == remote:
            print("epoch9 local complet", flush=True)
            return
        time.sleep(30)
    raise SystemExit("epoch9 pas arrive")


def wait_zips() -> None:
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        proc = ssh(
            "find /workspace/data/crop130 -name '*.part' | wc -l; "
            "stat -c '%n %s' /workspace/data/crop130/volumes_crop130*.zip 2>/dev/null || true"
        )
        print(proc.stdout.strip(), flush=True)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        parts = lines[0] if lines else "?"
        got = {}
        for ln in lines[1:]:
            name, size = ln.rsplit(" ", 1)
            got[os.path.basename(name)] = int(size)
        if parts.strip() == "0" and all(got.get(n) == s for n, s in ZIPS.items()):
            enc = remote_size(
                "/workspace/checkpoints/v20h_ssl_dino_screen/encoder_ep11.pt"
            )
            if enc == ENCODER:
                print("crop130 et encodeur prets", flush=True)
                return
            print(f"encodeur size {enc}", flush=True)
        time.sleep(20)
    raise SystemExit("zips crop130 incomplets")


def stop_vit() -> None:
    remote = r"""
set -euo pipefail
pids=$(ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}')
if [ -n "$pids" ]; then
  kill $pids || true
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    left=$(ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}')
    if [ -z "$left" ]; then break; fi
    sleep 1
  done
  left=$(ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}')
  if [ -n "$left" ]; then kill -9 $left || true; fi
fi
sleep 2
ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print}' || true
echo VIT_STOPPED
"""
    proc = ssh(remote, timeout=60)
    print(proc.stdout, proc.stderr, flush=True)
    if "VIT_STOPPED" not in proc.stdout:
        raise SystemExit("arret vit incertain")
    if "src.train.train" in proc.stdout.split("VIT_STOPPED")[-1]:
        raise SystemExit("vit encore vivant")


def start_v20b() -> None:
    remote = r"""
set -euo pipefail
cd /workspace
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
rm -f /workspace/kaggle_urls_crop130.json
nohup python -X utf8 -m src.train.train --config configs/kneenet_v20b_crop130_dino_ep11_oof_v3b_full.yaml >> /workspace/train_v20b_full.log 2>&1 &
sleep 12
ps -eo pid,etime,comm,args | awk '$3=="python" && /src.train.train/'
echo '---TAIL---'
tail -n 25 /workspace/train_v20b_full.log || true
"""
    proc = ssh(remote, timeout=40)
    print(proc.stdout, proc.stderr, flush=True)
    if "src.train.train" not in proc.stdout:
        raise SystemExit("v20b non demarre")


def pull_v20b() -> None:
    V20B_LOCAL.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 36 * 3600
    seen: set[str] = set()
    while time.time() < deadline:
        proc = ssh(
            f"find {V20B_REMOTE} -maxdepth 1 -type f -name 'epoch*.pt' -printf '%f %s\\n'"
        )
        if proc.returncode != 0:
            print("list fail", proc.stderr.strip(), flush=True)
            time.sleep(30)
            continue
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            name, size_s = line.split()
            size = int(size_s)
            dest = V20B_LOCAL / name
            if name in seen and dest.is_file() and dest.stat().st_size == size:
                continue
            print(f"download {name} ({size/1e9:.2f} GB)", flush=True)
            tmp = V20B_LOCAL / f"{name}.partial"
            cmd = [
                "ssh", "-p", PORT, "-i", KEY,
                "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
                HOST, f"cat {V20B_REMOTE}/{name}",
            ]
            child = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            assert child.stdout is not None
            got = 0
            with open(tmp, "wb") as handle:
                while True:
                    buf = child.stdout.read(1024 * 1024)
                    if not buf:
                        break
                    handle.write(buf)
                    got += len(buf)
            rc = child.wait()
            if rc != 0 or got != size:
                tmp.unlink(missing_ok=True)
                print(f"incomplet {name} {got} != {size}", flush=True)
                continue
            tmp.replace(dest)
            seen.add(name)
            print(f"OK {dest}", flush=True)
        hist = ssh(f"cat {V20B_REMOTE}/history.json")
        if hist.returncode == 0 and hist.stdout.strip():
            (V20B_LOCAL / "history.json").write_text(hist.stdout, encoding="utf-8")
        if (V20B_LOCAL / "epoch12.pt").is_file():
            print("epoch12 v20b present, stop", flush=True)
            return
        time.sleep(60)
    raise SystemExit("deadline v20b")


def main() -> None:
    wait_epoch9()
    wait_zips()
    stop_vit()
    start_v20b()
    pull_v20b()


if __name__ == "__main__":
    main()
