# -*- coding: utf-8 -*-
"""Telecharge les sorties convert-rgb-2-5d-proto (6 slots) vers data/crop130_rgb25d."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest, KaggleApi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "crop130_rgb25d"
KERNELS = [
    "sultent/rsna-knee-convert-rgb-2-5d-proto-part0",
    "sultent/rsna-knee-convert-rgb-2-5d-proto-part1",
    "sultent/rsna-knee-convert-rgb-2-5d-proto-part2",
    "sultent/rsna-knee-convert-rgb-2-5d-proto-part3",
    "sultent/rsna-knee-convert-rgb-2-5d-proto-part4",
    "sultent/rsna-knee-convert-rgb-2-5d-proto-part5",
]
CHUNK = 8 * 1024 * 1024
RETRIES = 200


def list_files(api: KaggleApi, kernel: str):
    owner, slug, _ver = api.parse_kernel_string(kernel)
    with api.build_kaggle_client() as kg:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        resp = kg.kernels.kernels_api_client.list_kernel_session_output(req)
    return list(resp.files or [])


def download_one(url: str, dest: Path, label: str | None = None) -> None:
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
                            tag = label or dest.name
                            print(
                                f"  {tag}: {have/1e9:.2f}/{max(total,1)/1e9:.2f} GB ({pct:.0f}%)",
                                flush=True,
                            )
            break
        except Exception as e:
            print(f"  retry {attempt}/{RETRIES} {dest.name}: {type(e).__name__}: {e}", flush=True)
            if attempt >= RETRIES:
                raise
            time.sleep(min(60, 2 ** min(attempt, 6)))
    for i in range(8):
        try:
            tmp.replace(dest)
            break
        except PermissionError:
            time.sleep(1 + i)
    else:
        tmp.replace(dest)
    print(f"OK {dest} ({dest.stat().st_size/1e9:.2f} GB)", flush=True)


def download_kernel(kernel: str) -> None:
    api = KaggleApi()
    api.authenticate()
    print("==", kernel, flush=True)
    items = list_files(api, kernel)
    if not items:
        raise RuntimeError(f"AUCUN fichier de sortie : {kernel}")
    suffix = kernel.rsplit("part", 1)[-1]
    for item in items:
        dest = OUT / item.file_name
        if item.file_name == "RECIPE.json":
            dest = OUT / f"RECIPE_proto_part{suffix}.json"
        if dest.exists() and dest.stat().st_size > 1000:
            print("skip existant", dest, dest.stat().st_size, flush=True)
            continue
        print("download", item.file_name, "->", dest.name, flush=True)
        download_one(item.url, dest, label=dest.name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(KERNELS)
    n_workers = min(2, n)
    print(f"download parallele : {n} kernels, {n_workers} workers -> {OUT}", flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(download_kernel, k): k for k in KERNELS}
        for fut in as_completed(futs):
            kernel = futs[fut]
            fut.result()
            print("DONE", kernel, flush=True)


if __name__ == "__main__":
    main()
