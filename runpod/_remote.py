# -*- coding: utf-8 -*-
"""Lance une commande distante. Argument: la commande bash."""
from __future__ import annotations

import os
import subprocess
import sys

KEY = os.path.expanduser(r"~/.ssh/id_ed25519_runpod")


def main() -> None:
    remote = open(sys.argv[1], encoding="utf-8").read().replace("\r", "").strip()
    cmd = [
        "ssh", "-p", "11242", "-i", KEY,
        "-o", "ConnectTimeout=20",
        "-o", "StrictHostKeyChecking=accept-new",
        "root@157.254.50.91",
        remote,
    ]
    proc = subprocess.run(cmd, text=True, timeout=90)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
