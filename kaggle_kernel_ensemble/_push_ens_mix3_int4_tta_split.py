# -*- coding: utf-8 -*-
"""Mix3 INT4 : TTA x3 sur v20h et v20b, ViT en un seul passage."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_kernel_ensemble" / "_push_ens_h025_mix3_int4" / "submit.ipynb"
DST = ROOT / "kaggle_kernel_ensemble" / "_push_ens_h025_mix3_int4_tta_split"
OUT = ROOT / "kaggle_kernel_ensemble" / "_out_ens_h025_mix3_int4_tta_split"
SLUG = "rsna-knee-ens-oai3-mix3-int4-tta-split"
KERNEL = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600

OLD_VAR = (
    "TTA_VARIANTS = (\n"
    "    (0.0, 1.00),\n"
    "    (5.0, 1.00),\n"
    "    (-5.0, 1.00),\n"
    ")"
)
NEW_VAR = (
    "TTA_X3 = (\n"
    "    (0.0, 1.00),\n"
    "    (5.0, 1.00),\n"
    "    (-5.0, 1.00),\n"
    ")"
)
OLD_LOOP = """        w = mix_w.to(device=x_vit.device)
        acc = None
        for angle_deg, scale in TTA_VARIANTS:
            xv = apply_affine(x_vit, angle_deg, scale)
            xr = apply_affine(x_224, angle_deg, scale)
            p_vit = torch.sigmoid(_forward(model_vit, xv).float())
            p_v20b = torch.sigmoid(_forward(model_v20b, xr).float())
            p_v20h = torch.sigmoid(_forward(model_v20h, xr).float())
            blended = w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h
            acc = blended if acc is None else acc + blended
        prob = (acc / len(TTA_VARIANTS)).cpu().numpy()[0]
"""
NEW_LOOP = """        def _mean_tta(model, x):
            acc = None
            for angle_deg, scale in TTA_X3:
                p = torch.sigmoid(_forward(model, apply_affine(x, angle_deg, scale)).float())
                acc = p if acc is None else acc + p
            return acc / len(TTA_X3)

        w = mix_w.to(device=x_vit.device)
        p_vit = torch.sigmoid(_forward(model_vit, x_vit).float())
        p_v20b = _mean_tta(model_v20b, x_224)
        p_v20h = _mean_tta(model_v20h, x_224)
        prob = (w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h).cpu().numpy()[0]
"""


def _set(cell: dict, text: str) -> None:
    text = text if text.endswith("\n") else text + "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def write_notebook() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    n_loop = n_var = 0
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        text = text.replace(
            "**INT4** + **TTA x3** (identite, rotation +5°, rotation -5°).",
            "**INT4**. TTA x3 (identite, rotation ±5°) sur v20b et v20h. ViT-k320 en un seul passage.",
        )
        text = text.replace(
            "print('TTA             : x3 rotation ±5')",
            "print('TTA             : vit x1, v20b x3, v20h x3')",
        )
        text = text.replace(
            "print(len(test), 'etudes | TTA x3 | pas de flip')",
            "print(len(test), 'etudes | TTA vit x1 v20b x3 v20h x3 | pas de flip')",
        )
        if OLD_VAR in text:
            text = text.replace(OLD_VAR, NEW_VAR)
            n_var += 1
        if OLD_LOOP in text:
            text = text.replace(OLD_LOOP, NEW_LOOP)
            n_loop += 1
        _set(cell, text)
    blob = "\n".join("".join(c["source"]) for c in nb["cells"])
    if n_var != 1 or n_loop != 1:
        raise SystemExit(f"patch n_var={n_var} n_loop={n_loop}")
    if "TTA_VARIANTS" in blob or "TTA x3**" in blob:
        raise SystemExit("ancien TTA encore present")
    if "TTA_X3" not in blob or "_mean_tta" not in blob:
        raise SystemExit("TTA par modele absent")
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
    )
    meta = {
        "id": KERNEL,
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
        "dataset_sources": ["sultent/rsna-knee-ens-oai3-mix3-int4"],
        "category_ids": [],
    }
    (DST / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("ok", DST, flush=True)


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        raise


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def push_kernel() -> str:
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(DST),
                "--accelerator", ACCEL, "-t", "32400",
            )
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text):
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
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
        print("kernel version", m.group(1), flush=True)
        return m.group(1)


def wait_kernel() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KERNEL).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            OUT.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
            except subprocess.CalledProcessError:
                pass
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


def submit(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    log = next(OUT.glob("*.log"))
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "TTA vit x1 v20b x3 v20h x3",
        "sans plancher" if False else "lignes 3",
        "swa-1-2-3",
        "INT4",
        "kneenet_int4_v1" if False else "INT4",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x5" in text or "coat_a_llrd08" in text:
        raise SystemExit("log inattendu")
    msg = (
        "ens mix3 OAI3 SWA INT4, TTA v20h x3 + v20b x3 + ViT x1, "
        "softmax hybrid025 T=0.05 sans plancher, pas de flip, T4"
    )
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", version,
        "-f", "submission.csv", "-m", msg,
    ), flush=True)
    print("SOUMIS", KERNEL, "version", version, flush=True)


def main() -> None:
    write_notebook()
    version = push_kernel()
    wait_kernel()
    submit(version)


if __name__ == "__main__":
    main()
