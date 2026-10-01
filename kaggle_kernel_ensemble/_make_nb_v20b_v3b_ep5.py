# -*- coding: utf-8 -*-
"""Poids + notebook T4 : v20b subset v3b epoch 5, sans toucher au train."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

from src.models.kneenet_v20b import KneeNet as KneeNetV20b  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "make_nb_v20b", ROOT / "kaggle_kernel_ensemble" / "_make_nb_v20b.py"
)
nb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb)

COPY = ROOT / "checkpoints" / "subset_v20b_crop130_det_v3b_frac025" / "_epoch10_copy.pt"
UPLOAD = ROOT / "kaggle_dataset_v20b" / "_upload_v3b_ep10"
WEIGHT = "v20b_v3b_subset_ep10.pt"
DATASET = "sultent/rsna-knee-v20b-v3b-subset-ep10"
SLUG = "rsna-knee-v20b-v3b-ep10-tta-x5"
RUN = "subset_v20b_crop130_det_v3b_frac025"
EPOCH = 10


def strip() -> dict:
    ckpt = torch.load(COPY, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    epoch = int(ckpt.get("epoch", -1))
    if cfg.get("run_name") != RUN:
        raise SystemExit(f"run inattendu {cfg.get('run_name')}")
    if epoch != EPOCH:
        raise SystemExit(f"epoch {epoch}, attendu {EPOCH}")
    if str(cfg.get("model")) != "kneenet_v20b":
        raise SystemExit(cfg.get("model"))
    if cfg.get("laterality"):
        raise SystemExit("laterality inattendue")
    hist = json.loads(
        (COPY.parent / "history.json").read_text(encoding="utf-8")
    )
    gold = float(next(r["gold_macro_auc"] for r in hist if int(r["epoch"]) == EPOCH))
    out = {
        "model": ckpt["model"],
        "config": cfg,
        "epoch": epoch,
        "val_macro_auc": ckpt.get("val_macro_auc"),
        "gold_macro_auc": gold,
    }
    UPLOAD.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD / WEIGHT
    torch.save(out, dest)
    model = KneeNetV20b(
        pretrained=False,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=int(cfg.get("encode_chunk", 8)),
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    )
    model.load_state_dict(out["model"], strict=True)
    print(
        "strip", dest.name, dest.stat().st_size,
        "epoch", epoch,
        "val", round(float(out["val_macro_auc"]), 4),
        "gold", round(gold, 4),
        "n_series", model.n_series,
        flush=True,
    )
    meta = {
        "title": f"RSNA Knee v20b v3b subset ep{EPOCH}",
        "id": DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (UPLOAD / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return out


def write_nb(gold: float, val: float) -> None:
    md = f"""# RSNA Knee — v20b subset v3b epoch {EPOCH}, **solo**

Notebook **self-contained** (internet off, GPU **T4**), un seul modèle :
DICOM test → crop 130 mm + bande 20–80 % → volumes (24, 224, 224)
→ **KneeNet v20b** (`{RUN}`, epoch {EPOCH},
gold in-sample {gold:.3f}, val macro-AUC {val:.3f})
→ TTA affine ×5 → `submission.csv`.

Soft labels Coolin targets_v3b, split det 924. Pas de canon latéralité.
Poids `{WEIGHT}` (optimizer retiré), dataset `{DATASET}`.

TTA : identité + rotation ±5° + scale 0.95/1.05. Pas de hflip.
Le kernel **doit** écrire `submission.csv` (cwd / `/kaggle/working`),
mêmes colonnes que sample_submission.
"""
    config = f"""import os
from pathlib import Path

KAGGLE = os.path.exists('/kaggle')
WEIGHT_NAME = {WEIGHT!r}
if KAGGLE:
    DATA_DIR = Path('/kaggle/input')
    print('contenu /kaggle/input :', sorted(p.name for p in DATA_DIR.iterdir()))
    COMP_DIR = None
    for base in [DATA_DIR / 'competitions', DATA_DIR]:
        if base.exists():
            for p in sorted(base.iterdir()):
                if (p / 'test.csv').exists():
                    COMP_DIR = p
                    break
        if COMP_DIR is not None:
            break
    if COMP_DIR is None:
        raise FileNotFoundError('test.csv introuvable sous /kaggle/input')
    search_roots = []
    if (DATA_DIR / 'datasets').exists():
        search_roots.append(DATA_DIR / 'datasets')
    search_roots.append(DATA_DIR)

    def _find_weight(name):
        for root in search_roots:
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
        raise FileNotFoundError(f'{{name}} introuvable (dataset v20b v3b attache ?)')

    WEIGHTS_PATH = _find_weight(WEIGHT_NAME)
else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_PATH = Path({str(UPLOAD.relative_to(ROOT)).replace(chr(92), '/')!r}) / WEIGHT_NAME

TEST_CSV = COMP_DIR / 'test.csv'
TEST_SERIES_CSV = COMP_DIR / 'test_series.csv'
TEST_SERIES_DIR = COMP_DIR / 'test_series'
SAMPLE_SUB = COMP_DIR / 'sample_submission.csv'

import torch

def _pick_device():
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        m = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
        _ = m(torch.zeros(1, 3, 32, 32, device='cuda')).sum().item()
        return 'cuda'
    except Exception as e:
        print('CUDA inutilisable, repli CPU :', str(e)[:120])
        return 'cpu'

DEVICE = _pick_device()
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

AMP_DTYPE = (torch.bfloat16
             if (DEVICE == 'cuda' and torch.cuda.is_bf16_supported())
             else torch.float16)
DEBUG_COMPARE_NPZ = True
NPZ_MEAN_U8, NPZ_STD_U8 = 42.0, 49.5

TTA_VARIANTS = (
    (0.0, 1.00),
    (5.0, 1.00),
    (-5.0, 1.00),
    (0.0, 0.95),
    (0.0, 1.05),
)
LATERALITY_CANON_L = False

print('competition dir :', COMP_DIR)
print('poids           :', WEIGHTS_PATH, '| existe :', WEIGHTS_PATH.exists())
print('device          :', DEVICE)
print('TTA variantes   :', TTA_VARIANTS)
print('laterality      :', LATERALITY_CANON_L)
print('modele          : v20b subset v3b epoch {EPOCH}')
"""
    infer = nb.LOAD_INFER.replace(
        "print('modele          : v20b oofmask SWA ep8/9/11')",
        f"print('modele          : v20b subset v3b epoch {EPOCH}')",
    ).replace(
        "print(f'{len(test)} etudes a predire | TTA x{len(TTA_VARIANTS)} | v20b swa solo')",
        f"print(f'{{len(test)}} etudes a predire | TTA x{{len(TTA_VARIANTS)}} | v20b v3b ep{EPOCH}')",
    )
    v20b_code = nb._patch_v20b(
        (ROOT / "src" / "models" / "kneenet_v20b.py").read_text(encoding="utf-8")
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": [
            nb.cell_md(md),
            nb.cell_code(config),
            nb.cell_code(nb.PREPROCESS_CODE),
            nb.cell_code(
                "# --- KneeNet v20b (copie self-contained de src/models/kneenet_v20b.py) ---\n"
                + v20b_code
            ),
            nb.cell_code(nb.NPZ_CODE),
            nb.cell_code(infer),
            nb.cell_code(nb.SUBMISSION_CODE),
        ],
    }
    push = ROOT / "kaggle_kernel_ensemble" / f"_push_v20b_v3b_ep{EPOCH}"
    push.mkdir(parents=True, exist_ok=True)
    (push / "submit.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (push / "kernel-metadata.json").write_text(
        json.dumps(nb._metadata(SLUG, DATASET), indent=2) + "\n", encoding="utf-8"
    )
    print("ecrit", push / "submit.ipynb", flush=True)


def main() -> None:
    info = strip()
    write_nb(float(info["gold_macro_auc"]), float(info["val_macro_auc"]))


if __name__ == "__main__":
    main()
