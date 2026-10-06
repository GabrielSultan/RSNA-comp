# -*- coding: utf-8 -*-
"""Download Kaggle kernel outputs from a URL dump (run on the pod)."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OUT = Path("/workspace/data/crop130_rgb25d")
CHUNK = 8 * 1024 * 1024
RETRIES = 200


def download_one(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while True:
        attempt += 1
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                if r.status_code == 416:
                    break
                if have and r.status_code == 200:
                    have = 0
                    mode = "wb"
                else:
                    r.raise_for_status()
                    mode = "ab" if have and r.status_code == 206 else "wb"
                    if mode == "wb":
                        have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                last = time.time()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - last >= 3:
                            last = now
                            pct = 100.0 * have / total if total else 0
                            print(
                                f"  {dest.name}: {have/1e9:.2f}/{max(total,1)/1e9:.2f} GB ({pct:.0f}%)",
                                flush=True,
                            )
            break
        except Exception as e:
            print(f"  retry {attempt}/{RETRIES} {dest.name}: {type(e).__name__}: {e}", flush=True)
            if attempt >= RETRIES:
                raise
            time.sleep(min(60, 2 ** min(attempt, 6)))
    tmp.replace(dest)
    print(f"OK {dest} ({dest.stat().st_size/1e9:.2f} GB)", flush=True)


def main() -> None:
    dump = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    items = json.loads(dump.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for it in items:
        dest = out / it["name"]
        if dest.exists() and dest.stat().st_size > 1000:
            print("skip", dest.name, dest.stat().st_size, flush=True)
            continue
        todo.append((it["url"], dest))
    print(f"download {len(todo)}/{len(items)} files -> {out}", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(download_one, url, dest) for url, dest in todo]
        for fut in as_completed(futs):
            fut.result()
    zips = list(out.glob("*.zip"))
    print(f"done zips={len(zips)} bytes={sum(p.stat().st_size for p in zips)}", flush=True)


if __name__ == "__main__":
    main()
