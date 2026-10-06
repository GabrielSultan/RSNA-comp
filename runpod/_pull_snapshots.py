# -*- coding: utf-8 -*-
"""Telecharge les snapshots epoch*.pt du full train ViT-k au fil de l'eau."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

HOST = "root@103.196.86.127"
PORT = "15596"
KEY = os.path.expanduser(r"~/.ssh/id_ed25519_runpod")
REMOTE = "/workspace/checkpoints/vit_k_llrd08_v3b_oofmask_full"
LOCAL = Path("checkpoints/vit_k_llrd08_v3b_oofmask_full")


def ssh(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh", "-p", PORT, "-i", KEY,
        "-o", "ConnectTimeout=20",
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
        name, size = line.split()
        out.append((name, int(size)))
    return out


def pull(name: str, size: int) -> None:
    dest = LOCAL / name
    tmp = LOCAL / f"{name}.partial"
    if dest.is_file() and dest.stat().st_size == size:
        return
    print(f"download {name} ({size/1e9:.2f} GB)", flush=True)
    cmd = [
        "ssh", "-p", PORT, "-i", KEY,
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=15",
        HOST, f"cat {REMOTE}/{name}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    got = 0
    with open(tmp, "wb") as handle:
        while True:
            buf = proc.stdout.read(1024 * 1024)
            if not buf:
                break
            handle.write(buf)
            got += len(buf)
    rc = proc.wait()
    if rc != 0 or got != size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name} incomplet {got} != {size} rc={rc}")
    tmp.replace(dest)
    print(f"OK {dest}", flush=True)


def main() -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    deadline = time.time() + 36 * 3600
    while time.time() < deadline:
        try:
            for name, size in remote_listing():
                if name in seen and (LOCAL / name).is_file() and (LOCAL / name).stat().st_size == size:
                    continue
                pull(name, size)
                seen.add(name)
            hist = ssh(f"cat {REMOTE}/history.json")
            if hist.returncode == 0 and hist.stdout.strip():
                (LOCAL / "history.json").write_text(hist.stdout, encoding="utf-8")
            if (LOCAL / "epoch12.pt").is_file():
                print("epoch12 present, stop", flush=True)
                return
        except Exception as exc:
            print("retry", type(exc).__name__, exc, flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
