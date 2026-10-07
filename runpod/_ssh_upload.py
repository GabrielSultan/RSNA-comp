# -*- coding: utf-8 -*-
"""Envoie un fichier local vers le pod via ssh cat. Ne logge pas la cle."""
from __future__ import annotations

import os
import subprocess
import sys
import time

HOST = "root@157.254.50.91"
PORT = "11242"
KEY = os.path.expanduser(r"~/.ssh/id_ed25519_runpod")


def upload(local: str, remote: str) -> None:
    size = os.path.getsize(local)
    cmd = [
        "ssh", "-p", PORT, "-i", KEY,
        "-o", "ConnectTimeout=20",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=15",
        HOST, f"cat > {remote}",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sent = 0
    t0 = time.time()
    last = t0
    assert proc.stdin is not None
    with open(local, "rb") as handle:
        while True:
            buf = handle.read(1024 * 1024)
            if not buf:
                break
            proc.stdin.write(buf)
            sent += len(buf)
            now = time.time()
            if now - last >= 5 or sent == size:
                last = now
                rate = sent / max(now - t0, 1e-6) / 1e6
                print(
                    f"{os.path.basename(local)} {sent/1e6:.1f}/{size/1e6:.1f} MB "
                    f"({rate:.2f} MB/s)",
                    flush=True,
                )
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ssh cat {local} -> {remote} rc={rc}")
    print(f"OK {remote}", flush=True)


if __name__ == "__main__":
    upload(sys.argv[1], sys.argv[2])
