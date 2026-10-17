# -*- coding: utf-8 -*-
"""Mix soft hybrid T=0.05 de 5 SWA. Sans TTA, sans notebook public."""
from __future__ import annotations

import ast
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
SRC = KERNEL / "_push_ens_c145_soft" / "submit.ipynb"
SLUG = "rsna-knee-ens-mix5-hybrid"
PUSH = KERNEL / "_push_ens_mix5_hybrid"
OUT = KERNEL / "_out_ens_mix5_hybrid"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 3 * 3600
TEMP = 0.05
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
HYBRIDS = (
    "subset_vit_k_llrd08_320_v3b_hybrid025",
    "subset_v20b_dino_ep11_v3b_hybrid025",
    "subset_v20h_dino_ep11_v3b_rsna1687_distil_main_crop145",
    "subset_v20h_dino_ep11_v3b_hybrid025",
    "subset_coat_a_llrd08_v3b_hybrid025",
)
DATASETS = [
    ("sultent/rsna-knee-vit-k320-oai3-swa", "vit_k320_oai3_swa_ep1_3.pt"),
    ("sultent/rsna-knee-v20b-dino-oai3-swa", "v20b_dino_oai3_swa_ep1_3.pt"),
    ("sultent/rsna-knee-v20h-crop145-distil-swa789", "v20h_crop145_distil_swa_ep7_9.pt"),
    ("sultent/rsna-knee-v20h-dino-full-swa", "v20h_dino_full_swa_ep6_7.pt"),
    ("sultent/rsna-knee-coat-a-oai3-swa", "coat_a_oai3_swa_ep1_3.pt"),
]
MSG = (
    "ens soft hybrid T=0.05 : ViT-k320 v20h_full OAI3 SWA 1-3 + "
    "v20b OAI3 SWA 1-3 + v20h distil crop145 SWA 7-9 + "
    "v20h crop130 full SWA 6-7 + CoAt OAI3 SWA 1-3, sans TTA, pas de flip, T4"
)
ACL_LOG = "vit 0.08 v20b 0.19 h145 0.44 h130 0.27 coat 0.02"


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _md(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


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
        raise SystemExit(f"remplacement x{found}, attendu {expect}: {old[:90]!r}")


def _best(name: str) -> dict:
    hist = json.loads((ROOT / "checkpoints" / name / "history.json").read_text(encoding="utf-8"))
    return max(hist, key=lambda r: float(r["val_macro_auc"]))


def mix_rows() -> list[tuple[str, tuple[float, ...]]]:
    recs = [_best(name) for name in HYBRIDS]
    for rec in recs:
        if int(rec["epoch"]) != 7:
            raise SystemExit(f"hybrid epoch {rec.get('epoch')}")
    rows = []
    for t in TARGETS:
        auc = np.array([float(r[f"auc_{t}"]) for r in recs], dtype=np.float64)
        z = auc / TEMP
        z = z - z.max()
        w = np.exp(z)
        w = w / w.sum()
        rnd = np.round(w, 2)
        delta = round(1.0 - float(rnd.sum()), 2)
        rnd = rnd.copy()
        rnd[int(np.argmax(rnd))] = round(float(rnd[int(np.argmax(rnd))] + delta), 2)
        if abs(float(rnd.sum()) - 1.0) > 1e-6 or np.any(rnd < -1e-9) or len(rnd) != 5:
            raise SystemExit(f"mix invalide {t} {rnd}")
        rows.append((t, tuple(float(v) for v in rnd)))
    return rows


def _mix_src(rows: list[tuple[str, tuple[float, ...]]]) -> str:
    lines = [
        "# Ordre : ViT-k320, v20b, v20h distil 145, v20h crop130, CoAt-a. Somme 1, plancher 0.",
        "N_MODELS = 5",
        "MIX = [",
    ]
    for t, w in rows:
        lines.append(
            f"    ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}, {w[4]:.2f}),  # {t}"
        )
    lines.append("]")
    return "\n".join(lines) + "\n"


def _markdown(rows: list[tuple[str, tuple[float, ...]]]) -> str:
    labels = ("ViT-k320", "v20b", "v20h distil", "v20h 130", "CoAt-a")
    lines = []
    for t, w in rows:
        bits = " / ".join(f"{lab} {v:.0%}" for lab, v in zip(labels, w))
        lines.append(f"- `{t}` : {bits}")
    body = "\n".join(lines)
    return f"""# RSNA Knee — rsna-knee-ens-mix5-hybrid

Notebook **self-contained** (internet off, **T4**). **Sans TTA.**

Mix souple de 5 modeles. Poids = softmax des AUC val hybrid, meilleure epoch, temperature 0.05, sans plancher.
Ordre : ViT-k320 / v20b DINO / v20h distil crop145 / v20h crop130 / CoAt-a.

- **ViT-k320** : `vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3` SWA epochs 1-3 (gold in-sample moyen 0.9903). Entree **320**, FOV 130 mm. Pas de flip. Hybrid `subset_vit_k_llrd08_320_v3b_hybrid025` epoch 7.
- **v20b DINO** : `v20b_crop130_dino_ep11_oof_v3b_full_oai3` SWA epochs 1-3 (gold in-sample moyen 0.9700). Entree 224, FOV 130 mm. Pas de flip. Hybrid `subset_v20b_dino_ep11_v3b_hybrid025` epoch 7.
- **v20h distil** : `v20h_crop145_dino_ep11_v3b_rsna_distil_main_full` SWA epochs 7-9 (gold in-sample moyen 0.9695). Entree 224, FOV 145 mm. Pas de flip. Hybrid `subset_v20h_dino_ep11_v3b_rsna1687_distil_main_crop145` epoch 7.
- **v20h crop130** : `v20h_crop130_dino_ep11_oof_v3b_full` SWA epochs 6-7 (gold in-sample moyen 0.9665). Entree 224, FOV 130 mm. Pas de flip. Hybrid `subset_v20h_dino_ep11_v3b_hybrid025` epoch 7.
- **CoAt-a** : `coat_a_llrd08_v3b_oofmask_full_oai3` SWA epochs 1-3 (gold in-sample moyen 0.9884). Entree 224, FOV 130 mm. Pas de flip. Hybrid `subset_coat_a_llrd08_v3b_hybrid025` epoch 7.

**Mix** :
{body}

Aucun flip de lateralite. Un modele peut valoir 0 %. Blend apres sigmoid.
"""


def write_nb() -> None:
    global ACL_LOG
    rows = mix_rows()
    a = rows[0][1]
    ACL_LOG = (
        f"vit {a[0]:.2f} v20b {a[1]:.2f} h145 {a[2]:.2f} "
        f"h130 {a[3]:.2f} coat {a[4]:.2f}"
    )
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    if len(nb["cells"]) != 9:
        raise SystemExit(f"notebook source : {len(nb['cells'])} cellules")
    nb["cells"][0] = _md(_markdown(rows))
    _replace(
        nb,
        "    WEIGHTS_V20H = _find_weight('v20h_crop145_swa_ep6_8.pt')\n",
        "    WEIGHTS_V20H = _find_weight('v20h_crop145_distil_swa_ep7_9.pt')\n"
        "    WEIGHTS_V20H130 = _find_weight('v20h_dino_full_swa_ep6_7.pt')\n",
        1,
    )
    _replace(
        nb,
        "    WEIGHTS_V20H = Path('kaggle_dataset_v20h_lat/_upload_crop145_swa678/v20h_crop145_swa_ep6_8.pt')\n",
        "    WEIGHTS_V20H = Path('kaggle_dataset_v20h_lat/_upload_crop145_distil_swa789/v20h_crop145_distil_swa_ep7_9.pt')\n"
        "    WEIGHTS_V20H130 = Path('kaggle_dataset_v20h_lat/_upload_dino_full_swa/v20h_dino_full_swa_ep6_7.pt')\n",
        1,
    )
    cfg = _text(nb["cells"][1])
    start = cfg.index("# Ordre :")
    end = cfg.index("print('competition dir")
    cfg = cfg[:start] + _mix_src(rows) + "\n" + cfg[end:]
    cfg = cfg.replace(
        "print('poids v20h dino :', WEIGHTS_V20H, '| existe :', WEIGHTS_V20H.exists())\n",
        "print('poids v20h 145  :', WEIGHTS_V20H, '| existe :', WEIGHTS_V20H.exists())\n"
        "print('poids v20h 130  :', WEIGHTS_V20H130, '| existe :', WEIGHTS_V20H130.exists())\n",
        1,
    )
    _set(nb["cells"][1], cfg)
    _replace(
        nb,
        "model_v20h = load_resnet(\n"
        "    WEIGHTS_V20H, KneeNetV20h, 'kneenet_v20h',\n"
        "    'v20h_crop145_dino_ep11_oof_v3b_full_oaiv2_wd01', 'swa-6-7-8', [6, 7, 8],\n"
        ")\n",
        "model_v20h = load_resnet(\n"
        "    WEIGHTS_V20H, KneeNetV20h, 'kneenet_v20h',\n"
        "    'v20h_crop145_dino_ep11_v3b_rsna_distil_main_full', 'swa-7-8-9', [7, 8, 9],\n"
        ")\n"
        "model_v20h130 = load_resnet(\n"
        "    WEIGHTS_V20H130, KneeNetV20h, 'kneenet_v20h',\n"
        "    'v20h_crop130_dino_ep11_oof_v3b_full', 'swa-6-7', [6, 7],\n"
        ")\n",
        1,
    )
    _replace(
        nb,
        "print(len(test), 'etudes | sans TTA | pas de flip | FOV vit/v20b/coat 130 mm, v20h 145 mm')\n",
        "print(len(test), 'etudes | sans TTA | pas de flip | FOV vit/v20b/v20h130/coat 130 mm, v20h distil 145 mm')\n",
        1,
    )
    _replace(
        nb,
        "    if abs(sum(w) - 1.0) > 1e-6 or min(w) < -1e-9 or len(w) != 4:\n"
        "        raise RuntimeError((t, w))\n"
        "    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f}')\n",
        "    if abs(sum(w) - 1.0) > 1e-6 or min(w) < -1e-9 or len(w) != 5:\n"
        "        raise RuntimeError((t, w))\n"
        "    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} h145 {w[2]:.2f} h130 {w[3]:.2f} coat {w[4]:.2f}')\n",
        1,
    )
    _replace(
        nb,
        "        p_v20h = torch.sigmoid(_forward(model_v20h, x_h).float())\n"
        "        p_coat = torch.sigmoid(_forward(model_coat, x_224).float())\n"
        "        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)\n"
        "        prob = (\n"
        "            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h + w[:, 3] * p_coat\n"
        "        ).cpu().numpy()[0]\n",
        "        p_v20h = torch.sigmoid(_forward(model_v20h, x_h).float())\n"
        "        p_v20h130 = torch.sigmoid(_forward(model_v20h130, x_224).float())\n"
        "        p_coat = torch.sigmoid(_forward(model_coat, x_224).float())\n"
        "        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)\n"
        "        prob = (\n"
        "            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h\n"
        "            + w[:, 3] * p_v20h130 + w[:, 4] * p_coat\n"
        "        ).cpu().numpy()[0]\n",
        1,
    )
    blob = "\n".join(_text(c) for c in nb["cells"])
    acl = rows[0][1]
    for needle in (
        "vit_k320_oai3_swa_ep1_3.pt",
        "v20b_dino_oai3_swa_ep1_3.pt",
        "v20h_crop145_distil_swa_ep7_9.pt",
        "v20h_dino_full_swa_ep6_7.pt",
        "coat_a_oai3_swa_ep1_3.pt",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full",
        "v20h_crop130_dino_ep11_oof_v3b_full",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "swa-1-2-3",
        "swa-7-8-9",
        "swa-6-7",
        "CROP_MM = 145.0",
        "CROP_MM = 130.0",
        "laterality=False",
        "model_v20h130",
        "N_MODELS = 5",
        f"({acl[0]:.2f}, {acl[1]:.2f}, {acl[2]:.2f}, {acl[3]:.2f}, {acl[4]:.2f})",
        "submission.csv",
        "sans TTA",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "oaiv2",
        "swa-6-7-8",
        "v20h_crop145_swa_ep6_8",
        "v20h_crop130_dino_ep11_oof_v3b_full_oai3",
        "raptor",
        "degnonguidi",
        "W_PUBLIC",
        "TTA_VARIANTS",
        "laterality=True",
        "vitk_full_oai3",
        "swa-8-9",
        "len(w) != 4",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    for cell in nb["cells"]:
        src = _text(cell)
        if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
            ast.parse(src)
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
    )
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
        "dataset_sources": [d for d, _w in DATASETS],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(nb["cells"]), flush=True)
    for t, w in rows:
        print(f"  {t:<18} {w}", flush=True)


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


def _transient(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "expecting value", "remote end closed", "connection aborted",
        "temporarily unavailable", "timed out", "timeout", "502", "503", "504",
    ))


def datasets_ready() -> None:
    for dataset, weight in DATASETS:
        try:
            text = kaggle("datasets", "files", dataset)
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if "403" not in text and "404" not in text:
                raise
            status = kaggle("datasets", "status", dataset).strip().lower()
            print(dataset, "status", status, flush=True)
            if "ready" not in status:
                raise SystemExit(f"dataset pas pret: {dataset}")
            continue
        size = None
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0] == weight:
                size = int(parts[1])
                break
        print(dataset, weight, size, flush=True)
        if size is None or size < 80_000_000:
            raise SystemExit(f"dataset pas pret: {dataset} {weight}")


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
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3",
        "gold 0.9903",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "v20h_crop130_dino_ep11_oof_v3b_full epoch swa-6-7 lat False gold 0.9665",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "swa-1-2-3",
        "lat False",
        "v20h distil 145 mm",
        ACL_LOG,
        "sans TTA",
        "lignes 3",
        "submission.csv",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text or "lat True" in text or "oaiv2" in text or "raptor" in text.lower():
        raise SystemExit("TTA, flip, ancien v20h ou public")
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
            raise
        print(resp, flush=True)
        print("SOUMIS", KAGGLE_ID, "version", version, flush=True)
        return


def main() -> None:
    write_nb()
    if "--write" in sys.argv:
        return
    datasets_ready()
    version = push()
    wait()
    submit(version)


if __name__ == "__main__":
    main()
