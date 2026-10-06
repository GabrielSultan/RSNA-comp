# -*- coding: utf-8 -*-
"""Affiche IP et ports publics du pod. Ne logge pas la cle."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path


def load_key() -> str:
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("pas de cle")


def main() -> None:
    key = load_key()
    query = """
    query {
      myself {
        pods {
          id
          name
          desiredStatus
          volumeInGb
          runtime {
            uptimeInSeconds
            ports { ip isIpPublic privatePort publicPort type }
          }
        }
      }
    }
    """
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    pods = body["data"]["myself"]["pods"]
    for pod in pods:
        if pod["id"] != "l2dxyaxcuves8m":
            continue
        print(json.dumps(pod, indent=2))


if __name__ == "__main__":
    main()
