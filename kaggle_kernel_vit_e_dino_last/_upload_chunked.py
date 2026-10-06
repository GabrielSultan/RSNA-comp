# -*- coding: utf-8 -*-
"""Cree un dataset Kaggle avec un PUT GCS decoupe (le PUT unique se bloque)."""
from __future__ import annotations

import argparse
import os
import time

import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.models.kaggle_models_extended import ResumableUploadResult

CHUNK = 512 * 1024


def _uploaded(response: requests.Response) -> int | None:
    if response.status_code in (200, 201):
        return None
    if response.status_code in (429, 500, 503):
        return -1
    if response.status_code != 308:
        raise RuntimeError(f"upload {response.status_code}: {response.text[:300]}")
    range_val = response.headers.get("Range")
    if not range_val:
        return 0
    return int(range_val.split("-")[-1]) + 1


def _probe(url: str, file_size: int) -> int | None:
    probe = requests.put(
        url,
        headers={
            "Content-Length": "0",
            "Content-Range": f"bytes */{file_size}",
        },
        timeout=(20, 30),
    )
    got = _uploaded(probe)
    if got == -1:
        raise requests.RequestException(f"probe {probe.status_code}")
    if probe.status_code in (200, 201):
        return None
    return got or 0


def upload_complete(self, path, url, quiet, resume=False):
    file_size = os.path.getsize(path)
    start = 0
    if resume:
        start = _probe(url, file_size)
        if start is None:
            print("upload deja complet", flush=True)
            return ResumableUploadResult.COMPLETE
        print(f"reprise a {start}/{file_size}", flush=True)
    stalls = 0
    with open(path, "rb") as handle:
        while start < file_size:
            end = min(start + CHUNK, file_size) - 1
            handle.seek(start)
            data = handle.read(end - start + 1)
            headers = {
                "Content-Length": str(len(data)),
                "Content-Range": f"bytes {start}-{end}/{file_size}",
            }
            try:
                response = requests.put(url, data=data, headers=headers, timeout=(20, 40))
            except requests.RequestException as exc:
                print(f"chunk {start}: {exc}", flush=True)
                try:
                    probed = _probe(url, file_size)
                except requests.RequestException as probe_exc:
                    print(f"probe {start}: {probe_exc}", flush=True)
                    stalls += 1
                    if stalls >= 8:
                        return ResumableUploadResult.INCOMPLETE
                    time.sleep(2)
                    continue
                if probed is None:
                    print(f"upload complet {file_size}", flush=True)
                    return ResumableUploadResult.COMPLETE
                if probed > start:
                    start = probed
                    stalls = 0
                    print(f"{start}/{file_size} ({100 * start / file_size:.0f}%)", flush=True)
                    continue
                stalls += 1
                if stalls >= 8:
                    print(f"pas de progres a {start}", flush=True)
                    return ResumableUploadResult.INCOMPLETE
                time.sleep(2)
                continue
            if response.status_code in (200, 201) and end + 1 >= file_size:
                print(f"upload complet {file_size}", flush=True)
                return ResumableUploadResult.COMPLETE
            nxt = _uploaded(response)
            if nxt == -1:
                print(f"chunk {start}: HTTP {response.status_code}, retry", flush=True)
                stalls += 1
                if stalls >= 12:
                    return ResumableUploadResult.INCOMPLETE
                time.sleep(min(30, 2 * stalls))
                continue
            if nxt is None or nxt <= start:
                print(f"pas de progres {response.status_code} a {start}", flush=True)
                return ResumableUploadResult.INCOMPLETE
            start = nxt
            stalls = 0
            if start == file_size or (start // CHUNK) % 16 == 0:
                print(f"{start}/{file_size} ({100 * start / file_size:.0f}%)", flush=True)
    return ResumableUploadResult.COMPLETE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--message", default="create")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    KaggleApi.upload_complete = upload_complete
    api = KaggleApi()
    api.authenticate()
    api.MAX_UPLOAD_RESUME_ATTEMPTS = 40
    if args.version:
        api.dataset_create_version_cli(args.folder, args.message, dir_mode="skip")
    else:
        api.dataset_create_new_cli(args.folder, dir_mode="skip")
    print("DATASET_OK", flush=True)


if __name__ == "__main__":
    main()
