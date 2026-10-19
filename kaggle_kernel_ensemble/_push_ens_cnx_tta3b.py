# -*- coding: utf-8 -*-
"""tta3 + zoom ViT + hflip distil. Deux T4 obligatoires. Soumission."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_cnx_tta3" / "submit.ipynb"
SLUG = "rsna-knee-ens-cnx-tta3b"
PUSH = KERNEL / "_push_ens_cnx_tta3b"
OUT = KERNEL / "_out_ens_cnx_tta3b"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 10 * 3600
ACL = "vit 0.00 v20h 0.40 cnx 0.60"
MSG = (
    "ens CNX-r7 SWA 7-10 baseline 60% TTA x3 "
    "(hflip, center-crop zoom 1.05, center-crop zoom 1.10) "
    "+ ViT-k320 OAI3 SWA 1-3 identite + hflip + zoom 1.05 "
    "+ v20h distil crop145 SWA 7-9 identite + hflip, "
    "2 T4, pas de rotation"
)


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _replace(nb: dict, old: str, new: str, expect: int) -> None:
    found = 0
    for cell in nb["cells"]:
        src = _text(cell)
        n = src.count(old)
        if not n:
            continue
        found += n
        _set(cell, src.replace(old, new))
    if found != expect:
        raise SystemExit(f"remplacement x{found}, attendu {expect}: {old[:80]!r}")


def write_nb() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    _replace(nb, "rsna-knee-ens-cnx-tta3", SLUG, 1)
    _replace(
        nb,
        "  Entree 320, FOV 130 mm. **TTA x1** : identite + hflip, meme echange de labels.\n",
        "  Entree 320, FOV 130 mm. **Identite + hflip + center-crop zoom 1.05**.\n"
        "  Echange de labels seulement apres le hflip. Pas de rotation.\n",
        1,
    )
    _replace(
        nb,
        "  Entree 224, FOV 145 mm. Un seul passage.\n",
        "  Entree 224, FOV 145 mm. **Identite + hflip**, meme echange de labels.\n",
        1,
    )
    _replace(
        nb,
        "print('TTA             : cnx x3, vit hflip x1, distil x1')",
        "print('TTA             : cnx x3, vit hflip + zoom 1.05, distil hflip, 2 GPU')",
        1,
    )
    _replace(
        nb,
        "print(len(test), 'etudes | TTA cnx x3 | vit hflip x1 | distil x1 | pas de rotation | "
        "FOV vit/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        "print(len(test), 'etudes | TTA cnx x3 | vit hflip + zoom 1.05 | distil hflip | "
        "2 GPU | pas de rotation | FOV vit/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        1,
    )
    _replace(
        nb,
        "    'vit identite + hflip, distil x1, pas de rotation',\n",
        "    'vit identite + hflip + zoom 1.05, distil identite + hflip, pas de rotation',\n",
        1,
    )
    _replace(
        nb,
        "N_GPU = torch.cuda.device_count() if DEVICE == 'cuda' else 1\n"
        "DEV_CNX = 'cuda:0' if DEVICE == 'cuda' else 'cpu'\n"
        "DEV_REST = 'cuda:1' if N_GPU >= 2 else DEV_CNX\n",
        "N_GPU = torch.cuda.device_count() if DEVICE == 'cuda' else 1\n"
        "if DEVICE == 'cuda' and N_GPU < 2:\n"
        "    raise RuntimeError(('attendu 2 GPU', N_GPU))\n"
        "DEV_CNX = 'cuda:0' if DEVICE == 'cuda' else 'cpu'\n"
        "DEV_REST = 'cuda:1' if N_GPU >= 2 else DEV_CNX\n",
        1,
    )
    _replace(
        nb,
        "        p_vit_id = torch.sigmoid(_forward(model_vit, x_vit).float())\n"
        "        p_vit_flip = torch.sigmoid(_forward(model_vit, _hflip(x_vit)).float())\n"
        "        p_vit = 0.5 * (p_vit_id + _swap_medlat(p_vit_flip))\n"
        "        p_v20h = torch.sigmoid(_forward(model_v20h, x_h).float())\n"
        "        return p_vit.float().cpu(), p_v20h.float().cpu()\n",
        "        p_vit_id = torch.sigmoid(_forward(model_vit, x_vit).float())\n"
        "        p_vit_flip = torch.sigmoid(_forward(model_vit, _hflip(x_vit)).float())\n"
        "        p_vit_zoom = torch.sigmoid(_forward(model_vit, _center_zoom(x_vit, 1.05)).float())\n"
        "        p_vit = (p_vit_id + _swap_medlat(p_vit_flip) + p_vit_zoom) / 3.0\n"
        "        p_h_id = torch.sigmoid(_forward(model_v20h, x_h).float())\n"
        "        p_h_flip = torch.sigmoid(_forward(model_v20h, _hflip(x_h)).float())\n"
        "        p_v20h = 0.5 * (p_h_id + _swap_medlat(p_h_flip))\n"
        "        return p_vit.float().cpu(), p_v20h.float().cpu()\n",
        1,
    )
    for cell in nb["cells"]:
        src = _text(cell)
        if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
            ast.parse(src)
    blob = "\n".join(_text(c) for c in nb["cells"])
    for needle in (
        SLUG,
        "attendu 2 GPU",
        "DEV_REST = 'cuda:1'",
        "model_cnx.to(DEV_CNX)",
        "model_vit.to(DEV_REST)",
        "model_v20h.to(DEV_REST)",
        "_center_zoom(x_vit, 1.05)",
        "_center_zoom(x_cnx, 1.05)",
        "_center_zoom(x_cnx, 1.10)",
        "(_swap_medlat(p_flip) + p_z1 + p_z2) / 3.0",
        "(p_vit_id + _swap_medlat(p_vit_flip) + p_vit_zoom) / 3.0",
        "0.5 * (p_h_id + _swap_medlat(p_h_flip))",
        "enabled=False",
        "cnx forward fp32",
        "(0.00, 0.40, 0.60),  # ACL",
        "pas de rotation",
        "submission.csv",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "distil x1",
        "vit hflip x1",
        "apply_affine",
        "TTA_VARIANTS",
        "model_v20b",
        "model_coat",
        "degnonguidi",
        "laterality=True",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((KERNEL / "_push_ens_cnx_tta3" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KAGGLE_ID
    meta["title"] = SLUG
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    (PUSH / "_acl.txt").write_text(ACL + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(nb["cells"]), flush=True)


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        raise


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def _transient(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "expecting value", "remote end closed", "connection aborted",
        "temporarily unavailable", "timed out", "timeout", "502", "503", "504",
    ))


def push() -> str:
    while True:
        try:
            out = kaggle("kernels", "push", "-p", str(PUSH), "--accelerator", ACCEL, "-t", "32400")
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text):
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            if _transient(text):
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
                continue
            raise
        print(out, flush=True)
        if _quota(out):
            print("quota GPU, retry 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        version = m.group(1)
        (PUSH / "_version.txt").write_text(version + "\n", encoding="utf-8")
        print("kernel version", version, flush=True)
        return version


def wait() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KAGGLE_ID).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def _daily_limit(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "maximum number", "submission limit", "daily submission allowance",
        "try again tomorrow", "too many submission", "exceeded",
    ))


def submit(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KAGGLE_ID, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and SLUG in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3 gold 0.9903",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10 lat False gold 0.976",
        "cnx forward fp32",
        "gpus visibles 2",
        "cuda:1",
        "vit identite + hflip + zoom 1.05",
        "distil identite + hflip",
        ACL,
        "lignes 3",
        "lat False",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "lat True" in text or "gpus visibles 1" in text or "distil x1" in text:
        raise SystemExit("un seul GPU ou TTA incomplet")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    waited = False
    while True:
        print("submit", KAGGLE_ID, "version", version, flush=True)
        try:
            resp = api.competition_submit_code(
                "submission.csv", MSG, COMPETITION, KAGGLE_ID, int(version), quiet=True,
            )
        except Exception as e:
            http = getattr(e, "response", None)
            body = http.text if http is not None else str(e)
            print(body, flush=True)
            if _daily_limit(body) and not waited:
                waited = True
                now = datetime.now(timezone.utc)
                target = now.replace(hour=0, minute=5, second=0, microsecond=0)
                if now >= target:
                    target = target + timedelta(days=1)
                delay = (target - now).total_seconds()
                print(f"ATTENTE_QUOTA {delay:.0f}s", flush=True)
                time.sleep(delay + 5)
                continue
            if _transient(body):
                time.sleep(20)
                continue
            raise
        print(resp, flush=True)
        print("SOUMIS", KAGGLE_ID, "version", version, flush=True)
        return


def main() -> None:
    write_nb()
    if "--write" in sys.argv:
        return
    version = push()
    wait()
    submit(version)


if __name__ == "__main__":
    main()
