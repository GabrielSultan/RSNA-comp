# -*- coding: utf-8 -*-
"""Mix 4 sans TTA : ViT-k320 SWA ep8-9 + v20b/CoAt OAI 1-3 + v20h crop145 6-8.

Les poids par finding sont ceux de l'ensemble c145 soft. Seul le ViT change.
"""
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
SRC = KERNEL / "_push_ens_c145_soft" / "submit.ipynb"
SLUG = "rsna-knee-ens-vitk89-c145"
PUSH = KERNEL / "_push_ens_vitk89_c145"
OUT = KERNEL / "_out_ens_vitk89_c145"
KERNEL_ID = f"sultent/{SLUG}"
VIT_DS = "sultent/rsna-knee-vit-k320-vitk-oai3-swa89"
DATASETS = [
    VIT_DS,
    "sultent/rsna-knee-v20b-dino-oai3-swa",
    "sultent/rsna-knee-v20h-crop145-swa678",
    "sultent/rsna-knee-coat-a-oai3-swa",
]
RUN_VIT = "vit_k_llrd08_320_v3b_oofmask_vitk_full_oai3"
OLD_VIT = "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3"
WEIGHT = "vit_k320_vitk_oai3_swa_ep8_9.pt"
OLD_WEIGHT = "vit_k320_oai3_swa_ep1_3.pt"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
MSG = (
    "ens ViT-k320 vitk OAI3 SWA ep8-9 + v20b OAI3 + v20h crop145 SWA 6-8 "
    "+ CoAt OAI3, poids par finding, sans TTA, pas de flip, T4"
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
        raise SystemExit(f"remplacement {old!r} x{found}, attendu {expect}")


def _gold() -> float:
    hist = json.loads(
        (ROOT / "checkpoints" / RUN_VIT / "history.json").read_text(encoding="utf-8")
    )
    vals = []
    for epoch in (8, 9):
        rec = next(r for r in hist if int(r["epoch"]) == epoch)
        vals.append(float(rec["gold_macro_auc"]))
    return sum(vals) / len(vals)


def write_notebook() -> None:
    gold = _gold()
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    _replace(nb, OLD_WEIGHT, WEIGHT, 2)
    _replace(
        nb,
        "kaggle_dataset_vit_k/_upload_k320_oai3_swa/" + WEIGHT,
        "kaggle_dataset_vit_k/_upload_k320_vitk_oai3_swa89/" + WEIGHT,
        1,
    )
    _replace(
        nb,
        f"if str(cfg.get('run_name')) != '{OLD_VIT}':",
        f"if str(cfg.get('run_name')) != '{RUN_VIT}':",
        1,
    )
    _replace(
        nb,
        "_expect_swa(ckpt, 'swa-1-2-3', [1, 2, 3])\n"
        "    model = KneeNetVitE(",
        "_expect_swa(ckpt, 'swa-8-9', [8, 9])\n"
        "    model = KneeNetVitE(",
        1,
    )
    _replace(
        nb,
        "# RSNA Knee — rsna-knee-ens-c145-soft",
        f"# RSNA Knee — {SLUG}",
        1,
    )
    _replace(
        nb,
        "ViT, v20b et CoAt : SWA OAI epochs 1-3, FOV 130 mm.\n"
        "v20h : SWA epochs 6-8, FOV 145 mm.\n"
        "softmax des AUC val hybrid025, temperature 0.05, sans plancher. "
        "v20h : subset crop145 wd01, best epoch.\n"
        "Subsets : `subset_vit_k_llrd08_320_v3b_hybrid025`, "
        "`subset_v20b_dino_ep11_v3b_hybrid025`, "
        "`subset_v20h_dino_ep11_v3b_hybrid025_wd01_crop145`, "
        "`subset_coat_a_llrd08_v3b_hybrid025`.",
        "ViT-k320 : SWA epochs 8-9, FOV 130 mm.\n"
        "v20b et CoAt : SWA OAI epochs 1-3, FOV 130 mm.\n"
        "v20h : SWA epochs 6-8, FOV 145 mm.",
        1,
    )
    _replace(
        nb,
        f"- **ViT-k320** : `{OLD_VIT}` SWA epochs 1-3 "
        "(gold in-sample moyen 0.990). Entree **320**. Pas de flip.",
        f"- **ViT-k320** : `{RUN_VIT}` SWA epochs 8-9 "
        f"(gold in-sample moyen {gold:.3f}). Entree **320**. Pas de flip.",
        1,
    )
    blob = "\n".join(_text(c) for c in nb["cells"])
    for needle in (
        RUN_VIT, WEIGHT, "swa-8-9", "[8, 9]",
        "swa-6-7-8", "[6, 7, 8]", "swa-1-2-3", "[1, 2, 3]",
        "CROP_MM = 145.0", "sans TTA", "v20h 145 mm",
        "(0.09, 0.32, 0.55, 0.04),  # ACL",
        "(0.25, 0.20, 0.28, 0.27),  # Fracture",
        "laterality=False",
    ):
        if needle not in blob:
            raise SystemExit(f"notebook incomplet: {needle}")
    for banned in (OLD_VIT, OLD_WEIGHT, "TTA_VARIANTS", "laterality=True"):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(_text(cell))
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
    )
    meta = {
        "id": KERNEL_ID,
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
    (PUSH / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("ecrit", PUSH, "gold vit", round(gold, 4), flush=True)


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


def push() -> str:
    print("push", KERNEL_ID, flush=True)
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(PUSH),
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
        version = m.group(1)
        (PUSH / "_version.txt").write_text(version + "\n", encoding="utf-8")
        print("kernel version", version, flush=True)
        return version


def wait() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KERNEL_ID).strip()
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
    return any(
        s in low
        for s in (
            "maximum number",
            "submission limit",
            "daily submission allowance",
            "try again tomorrow",
            "too many submission",
            "exceeded",
        )
    )


def submit(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL_ID, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower())
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "sans TTA",
        "lignes 3",
        "swa-8-9",
        "swa-6-7-8",
        "swa-1-2-3",
        "v20h 145 mm",
        "lat False",
        RUN_VIT,
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop145_dino_ep11_oof_v3b_full_oaiv2_wd01",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "vit 0.09 v20b 0.32 v20h 0.55 coat 0.04",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if OLD_VIT in text or "TTA x" in text or "lat True" in text:
        raise SystemExit("ancien ViT, TTA ou flip")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    waited = False
    while True:
        print("submit", KERNEL_ID, "version", version, flush=True)
        try:
            resp = api.competition_submit_code(
                "submission.csv", MSG, COMPETITION, KERNEL_ID, int(version), quiet=True,
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
            raise
        print(resp, flush=True)
        print("SOUMIS", KERNEL_ID, "version", version, flush=True)
        return


def main() -> None:
    write_notebook()
    version = push()
    wait()
    submit(version)


if __name__ == "__main__":
    main()
