# -*- coding: utf-8 -*-
"""Telecharge les snapshots epoch*.pt du full train ViT-k 320."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

HOST = "root@157.254.50.91"
PORT = "11242"
KEY = os.path.expanduser(r"~/.ssh/id_ed25519_runpod")
REMOTE = "/workspace/checkpoints/vit_k_llrd08_320_v3b_oofmask_v20h_full"
LOCAL = Path("checkpoints/vit_k_llrd08_320_v3b_oofmask_v20h_full")
# Marge laissee sur C: : le pagefile Windows grossit pendant les trains
# locaux, et un disque a 0 casse aussi leurs sauvegardes.
FREE_MARGIN = 3_000_000_000


def ssh(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh", "-p", PORT, "-i", KEY,
        "-o", "ConnectTimeout=20",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=15",
        HOST, *args,
    ]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def remote_listing() -> list[tuple[str, int]]:
    proc = ssh(f"find {REMOTE} -maxdepth 1 -type f -name 'epoch*.pt' -printf '%f %s\\n'")
    if proc.returncode != 0:
        print("list fail", proc.stderr.strip(), flush=True)
        return []
    out = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        name, size = line.split()
        out.append((name, int(size)))
    return out


def pull(name: str, size: int) -> bool:
    dest = LOCAL / name
    tmp = LOCAL / f"{name}.partial"
    if dest.is_file() and dest.stat().st_size == size:
        return True
    tmp.unlink(missing_ok=True)
    free = shutil.disk_usage(LOCAL).free
    if free < size + FREE_MARGIN:
        print(f"disque plein : {name} attend ({free/1e9:.1f} GB libres, "
              f"besoin {(size + FREE_MARGIN)/1e9:.1f} GB), reste sur le pod",
              flush=True)
        return False
    print(f"download {name} ({size/1e9:.2f} GB)", flush=True)
    cmd = [
        "ssh", "-p", PORT, "-i", KEY,
        "-o", "ConnectTimeout=20",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=15",
        HOST, f"cat {REMOTE}/{name}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    got = 0
    try:
        with open(tmp, "wb") as handle:
            while True:
                buf = proc.stdout.read(1024 * 1024)
                if not buf:
                    break
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
        raise RuntimeError(f"{name} incomplet {got} != {size} rc={rc}")
    for _ in range(30):
        try:
            tmp.replace(dest)
            break
        except OSError as exc:
            print("rename retry", exc, flush=True)
            time.sleep(2)
    else:
        raise RuntimeError(f"rename {name}")
    print(f"OK {dest}", flush=True)
    return True


def main() -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    deadline = time.time() + 48 * 3600
    while time.time() < deadline:
        try:
            for name, size in remote_listing():
                if name in seen and (LOCAL / name).is_file() and (LOCAL / name).stat().st_size == size:
                    continue
                if pull(name, size):
                    seen.add(name)
            hist = ssh(f"cat {REMOTE}/history.json")
            if hist.returncode == 0 and hist.stdout.strip():
                (LOCAL / "history.json").write_text(hist.stdout, encoding="utf-8")
            if (LOCAL / "epoch15.pt").is_file():
                print("epoch15 present, stop", flush=True)
                return
        except Exception as exc:
            print("retry", type(exc).__name__, exc, flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
