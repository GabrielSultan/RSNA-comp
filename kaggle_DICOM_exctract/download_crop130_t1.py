# -*- coding: utf-8 -*-
"""Telecharge les sorties convert-crop130-t1 vers data/crop130 (zips T1 en plus)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest, KaggleApi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "crop130"
KERNELS = [
    "sultent/rsna-knee-convert-crop130-t1-part0",
    "sultent/rsna-knee-convert-crop130-t1-part1",
]
CHUNK = 8 * 1024 * 1024
RETRIES = 12


def list_files(api: KaggleApi, kernel: str):
    owner, slug, _ver = api.parse_kernel_string(kernel)
    with api.build_kaggle_client() as kg:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        resp = kg.kernels.kernels_api_client.list_kernel_session_output(req)
    return list(resp.files or [])


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
                            print(f"  {dest.name}: {have/1e9:.2f}/{max(total,1)/1e9:.2f} GB ({pct:.0f}%)",
                                  flush=True)
            break
        except Exception as e:
            print(f"  retry {attempt}/{RETRIES} {dest.name}: {type(e).__name__}: {e}", flush=True)
            if attempt >= RETRIES:
                raise
            time.sleep(min(60, 2 ** attempt))
    tmp.replace(dest)
    print(f"OK {dest} ({dest.stat().st_size/1e9:.2f} GB)", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    for kernel in KERNELS:
        print("==", kernel, flush=True)
        items = list_files(api, kernel)
        if not items:
            print("AUCUN fichier de sortie", flush=True)
            sys.exit(1)
        for item in items:
            dest = OUT / item.file_name
            if item.file_name == "RECIPE.json":
                suffix = "part1" if kernel.endswith("part1") else "part0"
                dest = OUT / f"RECIPE_t1_{suffix}.json"
            if dest.exists() and dest.stat().st_size > 1000:
                print("skip existant", dest, dest.stat().st_size, flush=True)
                continue
            print("download", item.file_name, "->", dest.name, flush=True)
            download_one(item.url, dest)


if __name__ == "__main__":
    main()
