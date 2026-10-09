# -*- coding: utf-8 -*-
"""Acces SSH au pod : hote, port et cle lus dans .env (le port change a
chaque redemarrage). Ne logge jamais le contenu de .env."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def target() -> tuple[str, str, str]:
    env = _env()
    host = env["RUNPOD_HOST"]
    if "@" not in host:
        host = f"root@{host}"
    key = os.path.expanduser(env.get("RUNPOD_SSH_KEY") or "~/.ssh/id_ed25519_runpod")
    return host, env["RUNPOD_PORT"], key


def ssh_cmd(*remote: str) -> list[str]:
    host, port, key = target()
    return [
        "ssh", "-p", port, "-i", key,
        "-o", "ConnectTimeout=20",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=8",
        host, *remote,
    ]


def ssh(remote: str, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(ssh_cmd(remote), check=False, capture_output=True,
                          text=True, timeout=timeout)
