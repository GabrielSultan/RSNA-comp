# -*- coding: utf-8 -*-
"""Envoie un fichier local vers le pod via ssh cat (pas de SCP sur le proxy
RunPod), puis compare la taille distante. Ne logge pas la cle."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pod import ssh, ssh_cmd  # noqa: E402


def upload(local: str, remote: str) -> None:
    size = os.path.getsize(local)
    proc = subprocess.Popen(ssh_cmd(f"cat > {remote}.partial"), stdin=subprocess.PIPE)
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
    got = ssh(f"stat -c %s {remote}.partial && mv -f {remote}.partial {remote}")
    if got.returncode != 0 or int(got.stdout.split()[0]) != size:
        raise SystemExit(f"taille distante {got.stdout.strip()} != {size} pour {remote}")
    print(f"OK {remote} ({size} octets)", flush=True)


if __name__ == "__main__":
    upload(sys.argv[1], sys.argv[2])
