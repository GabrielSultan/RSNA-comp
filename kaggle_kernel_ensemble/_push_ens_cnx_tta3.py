# -*- coding: utf-8 -*-
"""CNX baseline 60 % TTA x3 + ViT hflip + distil sans TTA. Soumission."""
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
SRC = KERNEL / "_push_ens_cnx35_hflip" / "submit.ipynb"
SLUG = "rsna-knee-ens-cnx-tta3"
PUSH = KERNEL / "_push_ens_cnx_tta3"
OUT = KERNEL / "_out_ens_cnx_tta3"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 10 * 3600
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
    "subset_v20h_dino_ep11_v3b_rsna1687_distil_main_crop145",
]
MSG = (
    "ens CNX-r7 SWA 7-10 baseline 60% TTA x3 "
    "(hflip, center-crop zoom 1.05, center-crop zoom 1.10) "
    "+ ViT-k320 OAI3 SWA 1-3 identite+hflip "
    "+ v20h distil crop145 SWA 7-9 sans TTA, pas de rotation, T4"
)
MD = """# RSNA Knee — rsna-knee-ens-cnx-tta3

Internet off. La session a **deux T4** : le CNX sur cuda:0, le ViT et le distil sur cuda:1.
Pas de notebook public. Pas de rotation.

**Baseline 60 %** : `cnx_r7_in22k_v3b_oofmask_full` SWA 7-10 (gold 0.976).
Entree 224, FOV 130 mm. ConvNeXt en float32.
**TTA x3, trois passages, sans l'identite** : hflip, center-crop zoom 1.05,
center-crop zoom 1.10. Le hflip retourne l'image (sagittal sur l'axe des
coupes, coronal/axial sur la largeur) puis echange menisque et OA
medial/lateral. Les zooms recadrent le centre et redimensionnent, sans
echange de labels. Moyenne des trois sigmoides.

**40 % specialistes soft** (porte 0.015, softmax T=0.05). Residu d'arrondi
sur le CNX.

- ViT-k320 : `vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3` SWA 1-3.
  Entree 320, FOV 130 mm. **TTA x1** : identite + hflip, meme echange de labels.
- v20h distil : `v20h_crop145_dino_ep11_v3b_rsna_distil_main_full` SWA 7-9.
  Entree 224, FOV 145 mm. Un seul passage.
"""

TTA = """
def _hflip(x):
    \"\"\"Retourner l'image, sans rotation. Sagittal : axe D. Coronal/Axial : axe W.\"\"\"
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


def _center_zoom(x, scale):
    \"\"\"Zoom avant par center-crop puis resize. Pas de rotation.\"\"\"
    if x.ndim != 5 or float(scale) <= 1.0:
        raise RuntimeError((tuple(x.shape), scale))
    b, s, d, h, w = x.shape
    ch = int(round(h / float(scale)))
    cw = int(round(w / float(scale)))
    if ch < 8 or cw < 8 or ch >= h or cw >= w:
        raise RuntimeError((ch, cw, h, w, scale))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = x[:, :, :, y0:y0 + ch, x0:x0 + cw]
    flat = crop.reshape(b * s * d, 1, ch, cw)
    up = torch.nn.functional.interpolate(flat, size=(h, w), mode='bilinear', align_corners=False)
    return up.reshape(b, s, d, h, w)


print(
    'TTA cnx x3 hflip + zoom 1.05 + zoom 1.10, swap seulement apres hflip, '
    'vit identite + hflip, distil x1, pas de rotation',
    flush=True,
)
"""


def _tools():
    path = KERNEL / "_push_ens_cnx_pub70.py"
    spec = importlib.util.spec_from_file_location("pub70_tta3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _best(name: str) -> dict:
    hist = json.loads((ROOT / "checkpoints" / name / "history.json").read_text(encoding="utf-8"))
    return max(hist, key=lambda r: float(r.get("val_macro_auc") or -1.0))


def mix_rows() -> list[tuple[float, ...]]:
    recs = [_best(name) for name in SPEC]
    for name, rec in zip(SPEC, recs):
        if int(rec["epoch"]) != 7:
            raise SystemExit(f"{name} epoch {rec.get('epoch')}")
    rows = []
    for t in TARGETS:
        auc = np.array([float(r[f"auc_{t}"]) for r in recs], dtype=np.float64)
        keep = auc >= (float(auc.max()) - GATE - 1e-12)
        z = np.full(auc.shape, -1.0e9, dtype=np.float64)
        z[keep] = auc[keep] / TEMP
        z = z - z.max()
        e = np.exp(z)
        e = e / e.sum()
        raw = np.concatenate([e * W_SPEC, np.array([W_CNX])])
        cents = np.rint(raw * 100.0).astype(int)
        cents[2] += 100 - int(cents.sum())
        if int(cents[2]) < 60:
            raise SystemExit(f"cnx sous 0.60 {t} {cents.tolist()}")
        got = tuple(int(v) / 100.0 for v in cents)
        if sum(cents) != 100 or int(cents.min()) < 0 or got[2] <= max(got[:2]):
            raise SystemExit(f"mix invalide {t} {got}")
        rows.append(got)
    return rows


def _mix_src(rows: list[tuple[float, ...]]) -> str:
    lines = [
        "# Ordre : ViT-k320, v20h distil, CNX-r7.",
        "# CNX baseline 0.60. ViT et distil se partagent 0.40 (porte 0.015, softmax T=0.05).",
        "N_MODELS = 3",
        "MIX = [",
    ]
    for t, w in zip(TARGETS, rows):
        nums = ", ".join(f"{v:.2f}" for v in w)
        lines.append(f"    ({nums}),  # {t}")
    lines.append("]")
    return "\n".join(lines) + "\n"


def write_nb() -> str:
    pub = _tools()
    rows = mix_rows()
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    kept = []
    for cell in nb["cells"]:
        src = pub._text(cell)
        if src.startswith("# --- KneeNet v20b DINO ---") or src.startswith("# --- KneeNet CoAt-a ---"):
            continue
        kept.append(cell)
    nb["cells"] = kept
    if len(nb["cells"]) != 8:
        raise SystemExit(f"cells {len(nb['cells'])}")
    nb["cells"][0] = pub._md(MD)
    pub._replace_nb(
        nb,
        "    WEIGHTS_V20B = _find_weight('v20b_dino_oai3_swa_ep1_3.pt')\n"
        "    WEIGHTS_COAT = _find_weight('coat_a_oai3_swa_ep1_3.pt')\n",
        "",
        1,
    )
    pub._replace_nb(
        nb,
        "    WEIGHTS_V20B = Path('kaggle_dataset_v20b/_upload_dino_oai3_swa/v20b_dino_oai3_swa_ep1_3.pt')\n"
        "    WEIGHTS_COAT = Path('kaggle_dataset_coat/_upload_oai3_swa/coat_a_oai3_swa_ep1_3.pt')\n",
        "",
        1,
    )
    pub._replace_nb(
        nb,
        "print('poids v20b dino :', WEIGHTS_V20B, '| existe :', WEIGHTS_V20B.exists())\n"
        "print('poids coat      :', WEIGHTS_COAT, '| existe :', WEIGHTS_COAT.exists())\n",
        "",
        1,
    )
    old_mix = pub._text(nb["cells"][1])
    start = old_mix.index("# Ordre : ViT-k320, v20b OAI, v20h distil, CoAt-a, CNX-r7.")
    end = old_mix.index("]\n", start) + 2
    pub._set(nb["cells"][1], old_mix[:start] + _mix_src(rows) + old_mix[end:])
    pub._replace_nb(
        nb,
        "print('TTA             : cnx hflip x1, pas de rotation, autres x1')",
        "print('TTA             : cnx x3, vit hflip x1, distil x1')",
        1,
    )
    infer = pub._text(nb["cells"][-2])
    a = infer.find("model_v20b = load_resnet(")
    b = infer.find("def load_cnx(")
    if a < 0 or b < a or "model_coat = load_coat(WEIGHTS_COAT)" not in infer[a:b]:
        raise SystemExit("bloc v20b/coat introuvable")
    pub._set(nb["cells"][-2], infer[:a] + infer[b:])
    pub._replace_nb(nb, "or len(w) != 5:", "or len(w) != 3:", 1)
    pub._replace_nb(
        nb,
        "    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f} cnx {w[4]:.2f}')\n",
        "    print(f'  {t:<18} vit {w[0]:.2f} v20h {w[1]:.2f} cnx {w[2]:.2f}')\n",
        1,
    )
    pub._replace_nb(
        nb,
        "print(len(test), 'etudes | TTA cnx hflip x1 | pas de rotation | "
        "FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        "print(len(test), 'etudes | TTA cnx x3 | vit hflip x1 | distil x1 | pas de rotation | "
        "FOV vit/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        1,
    )
    pub._replace_nb(
        nb,
        "def _hflip(x):\n"
        "    \"\"\"Retourner l'image, sans rotation. Sagittal : axe D. Coronal/Axial : axe W.\"\"\"\n"
        "    if x.ndim != 5 or int(x.shape[1]) != 6:\n"
        "        raise RuntimeError(tuple(x.shape))\n"
        "    y = x.clone()\n"
        "    for s in (0, 3):\n"
        "        y[:, s] = torch.flip(y[:, s], dims=(1,))\n"
        "    for s in (1, 2, 4, 5):\n"
        "        y[:, s] = torch.flip(y[:, s], dims=(3,))\n"
        "    return y\n"
        "\n"
        "\n"
        "def _swap_medlat(p):\n"
        "    q = p.clone()\n"
        "    q[:, 2] = p[:, 3]\n"
        "    q[:, 3] = p[:, 2]\n"
        "    q[:, 4] = p[:, 5]\n"
        "    q[:, 5] = p[:, 4]\n"
        "    return q\n"
        "\n"
        "\n"
        "print(\n"
        "    'TTA cnx identite + hflip, swap menisque et OA medial/lateral, pas de rotation, autres x1',\n"
        "    flush=True,\n"
        ")\n",
        TTA,
        1,
    )
    pub._replace_nb(
        nb,
        "model_v20b.to(DEV_REST)\nmodel_v20h.to(DEV_REST)\nmodel_coat.to(DEV_REST)\n",
        "model_v20h.to(DEV_REST)\n",
        1,
    )
    pub._replace_nb(
        nb,
        "def _probs_cnx(x_224):\n"
        "    with torch.no_grad():\n"
        "        if str(DEV_CNX).startswith('cuda'):\n"
        "            torch.cuda.set_device(DEV_CNX)\n"
        "        x = x_224.to(DEV_CNX).float()\n"
        "        with torch.autocast('cuda', enabled=False):\n"
        "            p_id = torch.sigmoid(model_cnx(x).float())\n"
        "            p_flip = torch.sigmoid(model_cnx(_hflip(x)).float())\n"
        "        return (0.5 * (p_id + _swap_medlat(p_flip))).float().cpu()\n",
        "def _probs_cnx(x_224):\n"
        "    with torch.no_grad():\n"
        "        if str(DEV_CNX).startswith('cuda'):\n"
        "            torch.cuda.set_device(DEV_CNX)\n"
        "        x_cnx = x_224.to(DEV_CNX).float()\n"
        "        with torch.autocast('cuda', enabled=False):\n"
        "            p_flip = torch.sigmoid(model_cnx(_hflip(x_cnx)).float())\n"
        "            p_z1 = torch.sigmoid(model_cnx(_center_zoom(x_cnx, 1.05)).float())\n"
        "            p_z2 = torch.sigmoid(model_cnx(_center_zoom(x_cnx, 1.10)).float())\n"
        "        return ((_swap_medlat(p_flip) + p_z1 + p_z2) / 3.0).float().cpu()\n",
        1,
    )
    pub._replace_nb(
        nb,
        "def _probs_rest(x_vit, x_224, x_h):\n"
        "    with torch.no_grad():\n"
        "        if str(DEV_REST).startswith('cuda'):\n"
        "            torch.cuda.set_device(DEV_REST)\n"
        "        x_vit = x_vit.to(DEV_REST)\n"
        "        x_224 = x_224.to(DEV_REST)\n"
        "        x_h = x_h.to(DEV_REST)\n"
        "        p_vit = torch.sigmoid(_forward(model_vit, x_vit).float())\n"
        "        p_v20b = torch.sigmoid(_forward(model_v20b, x_224).float())\n"
        "        p_v20h = torch.sigmoid(_forward(model_v20h, x_h).float())\n"
        "        p_coat = torch.sigmoid(_forward(model_coat, x_224).float())\n"
        "        return tuple(t.float().cpu() for t in (p_vit, p_v20b, p_v20h, p_coat))\n",
        "def _probs_rest(x_vit, x_h):\n"
        "    with torch.no_grad():\n"
        "        if str(DEV_REST).startswith('cuda'):\n"
        "            torch.cuda.set_device(DEV_REST)\n"
        "        x_vit = x_vit.to(DEV_REST)\n"
        "        x_h = x_h.to(DEV_REST)\n"
        "        p_vit_id = torch.sigmoid(_forward(model_vit, x_vit).float())\n"
        "        p_vit_flip = torch.sigmoid(_forward(model_vit, _hflip(x_vit)).float())\n"
        "        p_vit = 0.5 * (p_vit_id + _swap_medlat(p_vit_flip))\n"
        "        p_v20h = torch.sigmoid(_forward(model_v20h, x_h).float())\n"
        "        return p_vit.float().cpu(), p_v20h.float().cpu()\n",
        1,
    )
    pub._replace_nb(
        nb,
        "        if _POOL is None:\n"
        "            p_cnx = _probs_cnx(x_224_cpu)\n"
        "            p_vit, p_v20b, p_v20h, p_coat = _probs_rest(x_vit_cpu, x_224_cpu, x_h_cpu)\n"
        "        else:\n"
        "            fa = _POOL.submit(_probs_cnx, x_224_cpu)\n"
        "            fb = _POOL.submit(_probs_rest, x_vit_cpu, x_224_cpu, x_h_cpu)\n"
        "            p_cnx = fa.result()\n"
        "            p_vit, p_v20b, p_v20h, p_coat = fb.result()\n"
        "        w = torch.tensor(MIX, dtype=torch.float32)\n"
        "        prob = (\n"
        "            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h + w[:, 3] * p_coat + w[:, 4] * p_cnx\n"
        "        ).numpy()[0]\n",
        "        if _POOL is None:\n"
        "            p_cnx = _probs_cnx(x_224_cpu)\n"
        "            p_vit, p_v20h = _probs_rest(x_vit_cpu, x_h_cpu)\n"
        "        else:\n"
        "            fa = _POOL.submit(_probs_cnx, x_224_cpu)\n"
        "            fb = _POOL.submit(_probs_rest, x_vit_cpu, x_h_cpu)\n"
        "            p_cnx = fa.result()\n"
        "            p_vit, p_v20h = fb.result()\n"
        "        w = torch.tensor(MIX, dtype=torch.float32)\n"
        "        prob = (\n"
        "            w[:, 0] * p_vit + w[:, 1] * p_v20h + w[:, 2] * p_cnx\n"
        "        ).numpy()[0]\n",
        1,
    )
    for cell in nb["cells"]:
        src = pub._text(cell)
        if cell["cell_type"] != "code" or src.lstrip().startswith("%%"):
            continue
        ast.parse(src)
    blob = "\n".join(pub._text(c) for c in nb["cells"])
    acl = rows[0]
    acl_line = f"vit {acl[0]:.2f} v20h {acl[1]:.2f} cnx {acl[2]:.2f}"
    for needle in (
        f"({acl[0]:.2f}, {acl[1]:.2f}, {acl[2]:.2f}),  # ACL",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "v20h_crop145_distil_swa_ep7_9.pt",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full",
        "cnx_r7_swa_ep7_10.pt",
        "swa-7-8-9-10",
        "N_MODELS = 3",
        "def _hflip",
        "def _center_zoom",
        "def _swap_medlat",
        "_center_zoom(x_cnx, 1.05)",
        "_center_zoom(x_cnx, 1.10)",
        "(_swap_medlat(p_flip) + p_z1 + p_z2) / 3.0",
        "0.5 * (p_vit_id + _swap_medlat(p_vit_flip))",
        "cnx forward fp32",
        "gpus visibles",
        "cuda:1",
        "pas de rotation",
        "enabled=False",
        "laterality=False",
        "submission.csv",
        "lignes",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "KneeNetV20b",
        "KneeNetCoat",
        "WEIGHTS_V20B",
        "WEIGHTS_COAT",
        "model_v20b",
        "model_coat",
        "v20b_dino_oai3_swa_ep1_3.pt",
        "coat_a_oai3_swa_ep1_3.pt",
        "apply_affine",
        "TTA_VARIANTS",
        "laterality=True",
        "degnonguidi",
        "W_PUBLIC",
        "N_MODELS = 5",
        "p_id =",
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
            "sultent/rsna-knee-v20h-crop145-distil-swa789",
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


def submit(version: str, acl_line: str) -> None:
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
        "TTA cnx x3 hflip + zoom 1.05 + zoom 1.10",
        "TTA             : cnx x3, vit hflip x1, distil x1",
        acl_line,
        "lignes 3",
        "lat False",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "lat True" in text or "v20b_crop130_dino" in text or "coat coat_a" in text:
        raise SystemExit("mauvais modele dans le log")
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
    acl = write_nb()
    if "--write" in sys.argv:
        return
    version = push()
    wait()
    submit(version, acl)


if __name__ == "__main__":
    main()
