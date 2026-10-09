# -*- coding: utf-8 -*-
"""Lance une commande distante lue dans un fichier (evite le quoting
PowerShell). Usage : python runpod/_remote.py cmd.txt [timeout_s]"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pod import ssh_cmd  # noqa: E402


def main() -> None:
    remote = open(sys.argv[1], encoding="utf-8").read().replace("\r", "").strip()
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    proc = subprocess.run(ssh_cmd(remote), text=True, timeout=timeout)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
