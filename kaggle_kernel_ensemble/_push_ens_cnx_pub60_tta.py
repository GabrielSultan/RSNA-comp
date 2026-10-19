# -*- coding: utf-8 -*-
"""Meme mix 5 que cnx-pub70, rang 60/40, TTA +5 deg seulement sur v20h distil."""
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
SRC = KERNEL / "_push_ens_cnx_pub70" / "submit.ipynb"
SLUG = "rsna-knee-ens-cnx-pub60-tta"
PUSH = KERNEL / "_push_ens_cnx_pub60_tta"
OUT = KERNEL / "_out_ens_cnx_pub60_tta"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
MSG = (
    "ens CNX-r7 SWA 7-10 + ViT-k320 v20h_full OAI3 SWA 1-3 + v20b OAI SWA + "
    "v20h distil crop145 SWA 7-9 + CoAt OAI SWA, mix lineaire 60% + "
    "public degnonguidi 40%, rang, TTA +5 deg uniquement sur v20h distil, "
    "pas de flip, T4"
)
DATASETS = [
    "heliosli/rsna-knee-r6-frozen-fe055406-w1",
    "nartaa/rsna-knee-publication-swa-weights-20261007",
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
    "sultent/rsna-knee-vit-k320-oai3-swa",
    "sultent/rsna-knee-v20b-dino-oai3-swa",
    "sultent/rsna-knee-v20h-crop145-distil-swa789",
    "sultent/rsna-knee-coat-a-oai3-swa",
    "sultent/rsna-knee-cnx-r7-swa78910",
]

AFFINE = '''
def apply_affine(x, angle_deg=0.0, scale=1.0):
    """Meme convention que src/data/transforms.py. Pas de flip."""
    if float(angle_deg) == 0.0 and float(scale) == 1.0:
        return x
    import torch.nn.functional as F
    if x.ndim != 5:
        raise RuntimeError(tuple(x.shape))
    b, s, d, h, w = x.shape
    xf = x.reshape(b * s * d, 1, h, w)
    angle = x.new_tensor(float(angle_deg) * 3.141592653589793 / 180.0)
    sc = x.new_tensor(float(scale))
    cos, sin = torch.cos(angle) / sc, torch.sin(angle) / sc
    theta = xf.new_zeros(b, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta = theta.repeat_interleave(s * d, dim=0)
    grid = F.affine_grid(theta, xf.shape, align_corners=False)
    xf = F.grid_sample(xf, grid, align_corners=False, padding_mode="zeros")
    return xf.reshape(b, s, d, h, w)


'''


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
        raise SystemExit(f"remplacement x{found}, attendu {expect}: {old[:70]!r}")


def write_nb() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    _replace(nb, "rsna-knee-ens-cnx-pub70", SLUG, 1)
    _replace(
        nb,
        "**Sans TTA.** Internet off, **T4**. Pas de flip.",
        "**TTA +5 deg seulement sur v20h distil** (identite et rotation +5, moyenne). "
        "Les 4 autres modeles : un seul passage. Internet off, **T4**. Pas de flip.",
        1,
    )
    _replace(nb, "Budget 3 h, puis la memoire GPU est liberee.", "Budget 3.5 h, puis la memoire GPU est liberee.", 1)
    _replace(
        nb,
        "3. Fusion par rangs, **70 %** le mix des 5 modeles et **30 %** la solution publique.",
        "3. Fusion par rangs, **60 %** le mix des 5 modeles et **40 %** la solution publique.",
        1,
    )
    _replace(nb, "MIX5_BUDGET_S = 3.0 * 3600", "MIX5_BUDGET_S = 3.5 * 3600", 1)
    _replace(nb, "print('TTA             : non')", "print('TTA             : v20h distil +5 deg, autres x1')", 1)
    _replace(
        nb,
        "print(len(test), 'etudes | sans TTA | pas de flip | FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        "print(len(test), 'etudes | TTA v20h distil +5 | pas de flip | FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        1,
    )
    _replace(
        nb,
        "rows = []\n",
        AFFINE + "print('TTA v20h distil identite + rotation +5 deg, scale 1, pas de flip', flush=True)\nrows = []\n",
        1,
    )
    _replace(
        nb,
        "        p_v20h = torch.sigmoid(_forward(model_v20h, x_h).float())\n",
        "        p_v20h_id = torch.sigmoid(_forward(model_v20h, x_h).float())\n"
        "        p_v20h_p5 = torch.sigmoid(_forward(model_v20h, apply_affine(x_h, 5.0, 1.0)).float())\n"
        "        p_v20h = 0.5 * (p_v20h_id + p_v20h_p5)\n",
        1,
    )
    _replace(nb, "W_MIX5, W_PUBLIC = 0.70, 0.30", "W_MIX5, W_PUBLIC = 0.60, 0.40", 1)
    _replace(nb, "poids mix5 0.70 public 0.30", "poids mix5 0.60 public 0.40", 1)
    _replace(nb, "BLEND70", "BLEND60", 2)
    _replace(nb, "# Rang 70/30 : 70 % mix5, 30 % submission publique.", "# Rang 60/40 : 60 % mix5, 40 % submission publique.", 1)
    for cell in nb["cells"]:
        src = _text(cell)
        if cell["cell_type"] != "code" or src.lstrip().startswith("%%"):
            continue
        ast.parse(src)
    blob = "\n".join(_text(c) for c in nb["cells"])
    for needle in (
        "apply_affine(x_h, 5.0, 1.0)",
        "0.5 * (p_v20h_id + p_v20h_p5)",
        "W_MIX5, W_PUBLIC = 0.60, 0.40",
        "TTA v20h distil identite + rotation +5 deg",
        "cnx forward fp32",
        "enabled=False",
        "cnx_r7_swa_ep7_10.pt",
        "swa-7-8-9-10",
        "v20h_crop145_distil_swa_ep7_9.pt",
        "laterality=False",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "rsna-knee-ens-cnx-pub70",
        "W_MIX5, W_PUBLIC = 0.70, 0.30",
        "print('TTA             : non')",
        "Sans TTA",
        "laterality=True",
        "(-5.0, 1.00)",
        "(0.0, 1.05)",
        "(0.0, 0.95)",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": KAGGLE_ID,
        "title": SLUG,
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": ACCEL,
        "competition_sources": [COMPETITION],
        "dataset_sources": DATASETS,
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    acl = (KERNEL / "_push_ens_cnx_pub70" / "_acl.txt").read_text(encoding="utf-8")
    (PUSH / "_acl.txt").write_text(acl, encoding="utf-8")
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
    acl = (PUSH / "_acl.txt").read_text(encoding="utf-8").strip()
    for needle in (
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3",
        "gold 0.9903",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10 lat False gold 0.976",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "TTA v20h distil identite + rotation +5 deg, scale 1, pas de flip",
        "TTA v20h distil +5",
        "cnx forward fp32",
        acl,
        "poids mix5 0.60 public 0.40",
        "BLEND60_OK",
        "MIX5_SCORED",
        "lignes 3",
        "lat False",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "lat True" in text or "rotation -5" in text or "public 0.30" in text:
        raise SystemExit("flip, TTA -5 ou blend 70/30")
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
