# -*- coding: utf-8 -*-
"""CNX-r7 SWA 7-10 baseline 60 % + mix specialiste soft 40 %.

TTA : un passage de plus sur le CNX seulement (flip lateral, puis
echange menisque et OA medial/lateral). Les 4 autres : un seul passage.
Pas de notebook public.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
OURS = KERNEL / "_push_ens_c145_soft" / "submit.ipynb"
SLUG = "rsna-knee-ens-cnx-base-spec"
PUSH = KERNEL / "_push_ens_cnx_base_spec"
OUT = KERNEL / "_out_ens_cnx_base_spec"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
TEMP = 0.05
GATE = 0.015
W_CNX = 0.60
W_SPEC = 0.40
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
SPEC = [
    "subset_vit_k_llrd08_320_v3b_hybrid025",
    "subset_v20b_dino_ep11_v3b_hybrid025",
    "subset_v20h_dino_ep11_v3b_rsna1687_distil_main_crop145",
    "subset_coat_a_llrd08_v3b_hybrid025",
]
MSG = (
    "ens CNX-r7 SWA 7-10 baseline 60% + specialistes soft 40% "
    "(ViT-k320 OAI3 SWA 1-3, v20b full SWA 6-9, v20h distil crop145 SWA 7-9, "
    "CoAt OAI SWA 1-3), porte 0.015 T=0.05, TTA flip lateral x1 sur CNX, T4"
)

MD = """# RSNA Knee — rsna-knee-ens-cnx-base-spec

Internet off, **T4**. Pas de notebook public.

**Baseline 60 %** : `cnx_r7_in22k_v3b_oofmask_full` SWA 7-10 (gold 0.976).
Entree 224, FOV 130 mm. ConvNeXt en float32.
**TTA x1 sur le CNX seulement** : identite + flip lateral
(sagittal sur l'axe des coupes, coronal/axial sur la largeur).
Apres le flip, echange Medial/Lateral Meniscus et Medial/Lateral OA.
Moyenne des deux sigmoides.

**40 % specialistes soft**, par finding : porte 0.015 sous la meilleure
AUC val hybrid, puis softmax T=0.05. Residual d'arrondi sur le CNX.

- ViT-k320 : `vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3` SWA 1-3. Entree 320, FOV 130 mm. Un passage.
- v20b : `v20b_crop130_dino_ep11_oof_v3b_full` SWA 6-9. Entree 224, FOV 130 mm. Un passage.
- v20h distil : `v20h_crop145_dino_ep11_v3b_rsna_distil_main_full` SWA 7-9. Entree 224, FOV 145 mm. Un passage.
- CoAt-a : `coat_a_llrd08_v3b_oofmask_full_oai3` SWA 1-3. Entree 224, FOV 130 mm. Un passage.
"""

FLIP = """
def _flip_lateral(x):
    \"\"\"Miroir lateral. Sagittal : axe D. Coronal/Axial : axe W.\"\"\"
    if x.ndim != 5 or int(x.shape[1]) != 6:
        raise RuntimeError(tuple(x.shape))
    y = x.clone()
    for s in (0, 3):
        y[:, s] = torch.flip(y[:, s], dims=(1,))
    for s in (1, 2, 4, 5):
        y[:, s] = torch.flip(y[:, s], dims=(3,))
    return y


def _swap_medlat(p):
    q = p.clone()
    q[:, 2] = p[:, 3]
    q[:, 3] = p[:, 2]
    q[:, 4] = p[:, 5]
    q[:, 5] = p[:, 4]
    return q


print(
    'TTA cnx identite + flip lateral, swap menisque et OA medial/lateral, autres x1',
    flush=True,
)
"""


def _pub():
    path = KERNEL / "_push_ens_cnx_pub70.py"
    spec = importlib.util.spec_from_file_location("pub70_cnx", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _best(name: str) -> dict:
    hist = json.loads((ROOT / "checkpoints" / name / "history.json").read_text(encoding="utf-8"))
    return max(hist, key=lambda r: float(r.get("val_macro_auc") or -1.0))


def mix_rows() -> list[tuple[float, ...]]:
    if abs(W_CNX + W_SPEC - 1.0) > 1e-9:
        raise SystemExit("parts baseline/specialistes")
    recs = [_best(name) for name in SPEC]
    for name, rec in zip(SPEC, recs):
        if int(rec["epoch"]) != 7:
            raise SystemExit(f"{name} epoch {rec.get('epoch')}")
    rows = []
    for t, recs_t in ((t, recs) for t in TARGETS):
        auc = np.array([float(r[f"auc_{t}"]) for r in recs_t], dtype=np.float64)
        keep = auc >= (float(auc.max()) - GATE - 1e-12)
        z = np.full(auc.shape, -1.0e9, dtype=np.float64)
        z[keep] = auc[keep] / TEMP
        z = z - z.max()
        e = np.exp(z)
        e = e / e.sum()
        raw = np.concatenate([e * W_SPEC, np.array([W_CNX])])
        rnd = np.round(raw, 2)
        delta = round(1.0 - float(rnd.sum()), 2)
        rnd = rnd.copy()
        rnd[4] = round(float(rnd[4] + delta), 2)
        got = tuple(float(v) for v in rnd)
        if abs(sum(got) - 1.0) > 1e-6 or min(got) < -1e-9:
            raise SystemExit(f"mix invalide {t} {got}")
        if got[4] <= max(got[:4]):
            raise SystemExit(f"cnx pas baseline {t} {got}")
        rows.append(got)
    return rows


def _mix_src(rows: list[tuple[float, ...]]) -> str:
    lines = [
        "# Ordre : ViT-k320, v20b full, v20h distil, CoAt-a, CNX-r7.",
        "# CNX baseline 0.60. Les 4 se partagent 0.40 (porte 0.015, softmax T=0.05).",
        "N_MODELS = 5",
        "MIX = [",
    ]
    for t, w in zip(TARGETS, rows):
        nums = ", ".join(f"{v:.2f}" for v in w)
        lines.append(f"    ({nums}),  # {t}")
    lines.append("]")
    return "\n".join(lines) + "\n"


def write_nb() -> str:
    pub = _pub()
    rows = mix_rows()
    nb = json.loads(OURS.read_text(encoding="utf-8"))
    if len(nb["cells"]) != 9:
        raise SystemExit("notebook source inattendu")
    pub._patch_v20h(nb)
    nb["cells"][0] = pub._md(MD)
    pub._replace_nb(nb, "v20b_dino_oai3_swa_ep1_3.pt", "v20b_dino_full_swa_ep6_9.pt", 2)
    pub._replace_nb(
        nb,
        "kaggle_dataset_v20b/_upload_dino_oai3_swa",
        "kaggle_dataset_v20b/_upload_dino_full_swa",
        1,
    )
    pub._replace_nb(
        nb,
        "'v20b_crop130_dino_ep11_oof_v3b_full_oai3', 'swa-1-2-3', [1, 2, 3],",
        "'v20b_crop130_dino_ep11_oof_v3b_full', 'swa-6-7-8-9', [6, 7, 8, 9],",
        1,
    )
    pub._replace_nb(nb, "print('TTA             : non')", "print('TTA             : cnx flip lateral x1, autres x1')", 1)
    old_mix = (
        "# Ordre : ViT-k320, v20b DINO, v20h DINO, CoAt-a. Somme 1, plancher 0.\n"
        "N_MODELS = 4\n"
        "MIX = [\n"
        "    (0.09, 0.32, 0.55, 0.04),  # ACL\n"
        "    (0.03, 0.44, 0.52, 0.01),  # MCL\n"
        "    (0.27, 0.29, 0.30, 0.14),  # Medial Meniscus\n"
        "    (0.18, 0.36, 0.41, 0.05),  # Lateral Meniscus\n"
        "    (0.17, 0.32, 0.33, 0.18),  # Medial OA\n"
        "    (0.21, 0.29, 0.29, 0.21),  # Lateral OA\n"
        "    (0.14, 0.32, 0.42, 0.12),  # PF OA\n"
        "    (0.28, 0.23, 0.22, 0.27),  # Effusion\n"
        "    (0.30, 0.26, 0.24, 0.20),  # Synovitis\n"
        "    (0.26, 0.27, 0.31, 0.16),  # Baker's\n"
        "    (0.19, 0.31, 0.35, 0.15),  # Contusion\n"
        "    (0.25, 0.20, 0.28, 0.27),  # Fracture\n"
        "]\n"
    )
    pub._replace_nb(nb, old_mix, _mix_src(rows), 1)
    pub._replace_nb(
        nb,
        "    WEIGHTS_COAT = _find_weight('coat_a_oai3_swa_ep1_3.pt')\n",
        "    WEIGHTS_COAT = _find_weight('coat_a_oai3_swa_ep1_3.pt')\n"
        "    WEIGHTS_CNX = _find_weight('cnx_r7_swa_ep7_10.pt')\n",
        1,
    )
    pub._replace_nb(
        nb,
        "    WEIGHTS_COAT = Path('kaggle_dataset_coat/_upload_oai3_swa/coat_a_oai3_swa_ep1_3.pt')\n",
        "    WEIGHTS_COAT = Path('kaggle_dataset_coat/_upload_oai3_swa/coat_a_oai3_swa_ep1_3.pt')\n"
        "    WEIGHTS_CNX = Path('kaggle_dataset_cnx/_upload_swa78910/cnx_r7_swa_ep7_10.pt')\n",
        1,
    )
    pub._replace_nb(
        nb,
        "print('poids coat      :', WEIGHTS_COAT, '| existe :', WEIGHTS_COAT.exists())\n",
        "print('poids coat      :', WEIGHTS_COAT, '| existe :', WEIGHTS_COAT.exists())\n"
        "print('poids cnx       :', WEIGHTS_CNX, '| existe :', WEIGHTS_CNX.exists())\n",
        1,
    )
    pub._replace_nb(
        nb,
        "model_coat = load_coat(WEIGHTS_COAT)\n",
        "model_coat = load_coat(WEIGHTS_COAT)\n" + pub.LOAD_CNX,
        1,
    )
    pub._replace_nb(nb, "or len(w) != 4:", "or len(w) != 5:", 1)
    pub._replace_nb(
        nb,
        "    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f}')\n",
        "    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f} cnx {w[4]:.2f}')\n",
        1,
    )
    pub._replace_nb(
        nb,
        "print(len(test), 'etudes | sans TTA | pas de flip | FOV vit/v20b/coat 130 mm, v20h 145 mm')\n",
        "print(len(test), 'etudes | TTA cnx flip lateral x1 | autres x1 | FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32')\n",
        1,
    )
    pub._replace_nb(nb, "rows = []\n", FLIP + "rows = []\n", 1)
    pub._replace_nb(
        nb,
        "        p_coat = torch.sigmoid(_forward(model_coat, x_224).float())\n"
        "        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)\n"
        "        prob = (\n"
        "            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h + w[:, 3] * p_coat\n"
        "        ).cpu().numpy()[0]\n",
        "        p_coat = torch.sigmoid(_forward(model_coat, x_224).float())\n"
        "        with torch.autocast('cuda', enabled=False):\n"
        "            p_id = torch.sigmoid(model_cnx(x_224.float()).float())\n"
        "            p_flip = torch.sigmoid(model_cnx(_flip_lateral(x_224).float()).float())\n"
        "        p_cnx = 0.5 * (p_id + _swap_medlat(p_flip))\n"
        "        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)\n"
        "        prob = (\n"
        "            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h"
        " + w[:, 3] * p_coat + w[:, 4] * p_cnx\n"
        "        ).cpu().numpy()[0]\n",
        1,
    )
    nb["cells"].insert(7, pub._cnx_cell())
    for cell in nb["cells"]:
        src = pub._text(cell)
        if cell["cell_type"] != "code" or src.lstrip().startswith("%%"):
            continue
        ast.parse(src)
    blob = "\n".join(pub._text(c) for c in nb["cells"])
    acl = rows[0]
    acl_line = (
        f"vit {acl[0]:.2f} v20b {acl[1]:.2f} v20h {acl[2]:.2f} "
        f"coat {acl[3]:.2f} cnx {acl[4]:.2f}"
    )
    for needle in (
        "v20b_dino_full_swa_ep6_9.pt",
        "v20b_crop130_dino_ep11_oof_v3b_full', 'swa-6-7-8-9'",
        "v20h_crop145_distil_swa_ep7_9.pt",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "cnx_r7_swa_ep7_10.pt",
        "swa-7-8-9-10",
        "N_MODELS = 5",
        f"({acl[0]:.2f}, {acl[1]:.2f}, {acl[2]:.2f}, {acl[3]:.2f}, {acl[4]:.2f})",
        "def _flip_lateral",
        "def _swap_medlat",
        "0.5 * (p_id + _swap_medlat(p_flip))",
        "cnx forward fp32",
        "TTA cnx identite + flip lateral",
        "enabled=False",
        "laterality=False",
        "submission.csv",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "v20b_dino_oai3_swa_ep1_3.pt",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop145_swa_ep6_8.pt",
        "oaiv2_wd01",
        "print('TTA             : non')",
        "laterality=True",
        "TTA_VARIANTS",
        "W_PUBLIC",
        "degnonguidi",
        "N_MODELS = 4",
        "(0.09, 0.32, 0.55, 0.04)",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    out = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": nb["cells"],
    }
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
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
        "dataset_sources": [
            "sultent/rsna-knee-vit-k320-oai3-swa",
            "sultent/rsna-knee-v20b-dino-full-swa",
            "sultent/rsna-knee-v20h-crop145-distil-swa789",
            "sultent/rsna-knee-coat-a-oai3-swa",
            "sultent/rsna-knee-cnx-r7-swa78910",
        ],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    (PUSH / "_acl.txt").write_text(acl_line + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(nb["cells"]), flush=True)
    for t, w in zip(TARGETS, rows):
        print(f"  {t:<18} {w}", flush=True)
    return acl_line


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
    acl_line = (PUSH / "_acl.txt").read_text(encoding="utf-8").strip()
    for needle in (
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3 gold 0.9903",
        "v20b_crop130_dino_ep11_oof_v3b_full epoch swa-6-7-8-9 lat False gold 0.9678",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "coat coat_a_llrd08_v3b_oofmask_full_oai3 epoch swa-1-2-3 lat False gold 0.9884",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10 lat False gold 0.976",
        "cnx forward fp32",
        "TTA cnx identite + flip lateral, swap menisque et OA medial/lateral, autres x1",
        "TTA             : cnx flip lateral x1, autres x1",
        acl_line,
        "lignes 3",
        "lat False",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "lat True" in text or "v20b_crop130_dino_ep11_oof_v3b_full_oai3" in text or "public 0." in text:
        raise SystemExit("flip train, v20b oai ou blend public")
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
