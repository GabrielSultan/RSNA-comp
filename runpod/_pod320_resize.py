# -*- coding: utf-8 -*-
"""Passe le volume du pod 320 a 120 Go. Ne logge pas la cle."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

POD = "gqewyn885g1kv0"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"


def load_key() -> str:
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("pas de cle")


def gql(key: str, query: str) -> dict:
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
        return json.loads(resp.read().decode())


def main() -> None:
    key = load_key()
    query = """
    mutation {
      podEditJob(input: {
        podId: "%s",
        imageName: "%s",
        containerDiskInGb: 50,
        volumeInGb: 120
      }) {
        id
        desiredStatus
        volumeInGb
        imageName
      }
    }
    """ % (POD, IMAGE)
    body = gql(key, query)
    print(json.dumps(body.get("data") or body.get("errors"), indent=2))


if __name__ == "__main__":
    main()
