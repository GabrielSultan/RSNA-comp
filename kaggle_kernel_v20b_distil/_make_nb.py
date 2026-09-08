# -*- coding: utf-8 -*-
"""Kernel Kaggle solo : v20b distille ViT-e lambda 0.3 (pas d'ensemble)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_v20b_distil"
OUTS = [
    KERNEL_DIR / "submit.ipynb",
    KERNEL_DIR / "_push" / "submit.ipynb",
]


def _load_ensemble_nb():
    p = ROOT / "kaggle_kernel_ensemble" / "_make_nb.py"
    spec = importlib.util.spec_from_file_location("ens_make_nb", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ens = _load_ensemble_nb()
cell_md = ens.cell_md
cell_code = ens.cell_code
PREPROCESS_CODE = ens.PREPROCESS_CODE
NPZ_CODE = ens.NPZ_CODE
SUBMISSION_CODE = ens.SUBMISSION_CODE
_patch_v20b = ens._patch_v20b


MD0 = """# RSNA Knee — v20b distillé ViT-e (λ=0.3), **solo**

Notebook **self-contained** pour Kaggle (internet off), **un seul modèle** :
DICOM test → crop 130 mm + bande 20–80 % → volumes (24, 224, 224) →
**KneeNet v20b** (`subset_v20b_distil_vite_lam03_frac025/best.pt`, epoch 9,
gold macro-AUC 0.882) → TTA affine ×5 → `submission.csv`.

Pas d'ensemble. Distillation λ=0.3 sur MCL / ménisque médial / épanchement
(teacher ViT-e), subset 25 %.

Poids (dataset `sultent/rsna-knee-v20b-distil-lam03`) :
- `v20b_distil_lam03.pt` — optimizer retiré, 6 slots FS/T1, ResNet-50 RadImageNet

TTA : identité + rotation ±5° + scale 0.95/1.05. **Pas de hflip**.
Le kernel **doit** écrire `submission.csv` (cwd / `/kaggle/working`).
"""


CONFIG_CODE = r"""import os
from pathlib import Path

KAGGLE = os.path.exists('/kaggle')
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
        raise FileNotFoundError(f'{name} introuvable (dataset distil attache ?)')

    WEIGHTS_PATH = _find_weight('v20b_distil_lam03.pt')
else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_PATH = Path('../kaggle_dataset_ensemble/_upload_distil/v20b_distil_lam03.pt')

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

print('competition dir :', COMP_DIR)
print('poids           :', WEIGHTS_PATH, '| existe :', WEIGHTS_PATH.exists())
print('device          :', DEVICE)
print('TTA variantes   :', TTA_VARIANTS)
print('modele          : v20b distil solo (pas d ensemble)')
"""


LOAD_INFER = r"""import pandas as pd

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']


def load_v20b(weights_path):
    ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
    cfg = ckpt.get('config', {})
    model = KneeNetV20b(
        backbone=cfg.get('backbone', 'resnet50'),
        n_series=int(cfg.get('n_series', 6)),
        head_dim=int(cfg.get('head_dim', 512)),
        dropout=float(cfg.get('dropout', 0.4)),
        encode_chunk=max(8, int(cfg.get('encode_chunk', 8))),
        lstm_hidden=int(cfg.get('lstm_hidden', 256)),
        pretrained=False,
    )
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(DEVICE).eval()
    print('modele', cfg.get('model'), '| epoch', ckpt.get('epoch'),
          '| val macro-AUC :', round(float(ckpt.get('val_macro_auc', float('nan'))), 4),
          '| gold macro-AUC :', round(float(ckpt.get('gold_macro_auc', float('nan'))), 4),
          '| n_series', model.n_series)
    return model, cfg


model, cfg = load_v20b(WEIGHTS_PATH)
assert int(model.n_series) == 6, model.n_series

# --- Inference : TTA, un seul modele --------------------------------------
import time
import torch.nn.functional as F

test = pd.read_csv(TEST_CSV)
test_series = pd.read_csv(TEST_SERIES_CSV)
print(f'{len(test)} etudes a predire | TTA x{len(TTA_VARIANTS)} | v20b distil solo')


def apply_affine(x, angle_deg=0.0, scale=1.0):
    if angle_deg == 0.0 and scale == 1.0:
        return x
    b, s, d, h, w = x.shape
    xf = x.reshape(b * s * d, 1, h, w)
    angle = x.new_tensor(angle_deg * 3.141592653589793 / 180.0)
    sc = x.new_tensor(scale)
    cos, sin = torch.cos(angle) / sc, torch.sin(angle) / sc
    theta = xf.new_zeros(b, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta = theta.repeat_interleave(s * d, dim=0)
    grid = F.affine_grid(theta, xf.shape, align_corners=False)
    xf = F.grid_sample(xf, grid, align_corners=False, padding_mode='zeros')
    return xf.reshape(b, s, d, h, w)


def _forward(x, use_amp=True):
    global DEVICE, model
    try:
        if use_amp and DEVICE == 'cuda':
            with torch.autocast('cuda', dtype=AMP_DTYPE):
                return model(x)
        return model(x)
    except RuntimeError as e:
        print('forward CUDA echoue, cudnn off puis CPU :', str(e)[:80])
        torch.backends.cudnn.enabled = False
        try:
            if use_amp and DEVICE == 'cuda':
                with torch.autocast('cuda', dtype=AMP_DTYPE):
                    return model(x)
            return model(x)
        except RuntimeError:
            DEVICE = 'cpu'
            model = model.to('cpu')
            return model(x.cpu())


rows = []
t0 = time.time()
n_series = int(model.n_series)
with torch.no_grad():
    for i, uid in enumerate(test.StudyInstanceUID):
        _SERIES_CACHE.clear()
        srows = test_series[test_series.StudyInstanceUID == uid]
        sdir = TEST_SERIES_DIR / uid
        vol = study_to_tensor(sdir, srows, n_series=n_series)
        x = torch.from_numpy(vol).unsqueeze(0).to(DEVICE)
        ps = []
        for angle_deg, scale in TTA_VARIANTS:
            xa = apply_affine(x, angle_deg, scale)
            ps.append(torch.sigmoid(_forward(xa, use_amp=True).float()))
        prob = torch.stack(ps, dim=0).mean(dim=0).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(test)} | {(time.time()-t0)/60:.1f} min', flush=True)

pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
pred.head()
"""


def main() -> None:
    v20b_code = _patch_v20b(
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
            cell_md(MD0),
            cell_code(CONFIG_CODE),
            cell_code(PREPROCESS_CODE),
            cell_code(
                "# --- KneeNet v20b (copie self-contained de src/models/kneenet_v20b.py) ---\n"
                + v20b_code
            ),
            cell_code(NPZ_CODE),
            cell_code(LOAD_INFER),
            cell_code(SUBMISSION_CODE),
        ],
    }
    text = json.dumps(notebook, indent=1, ensure_ascii=False)
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print("ecrit", out)


if __name__ == "__main__":
    main()
