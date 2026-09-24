# -*- coding: utf-8 -*-
"""Kernel Kaggle solo : v20h oofmask + canon laterality L (flip R)."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
GOLD = {
    7: 0.9584800543601317,
    8: 0.9596848377450008,
    9: 0.9648646124385674,
    10: 0.9677026625181401,
    11: 0.9719422376200465,
    12: 0.9736202304208729,
}


def _load_ensemble_nb():
    p = KERNEL_DIR / "_make_nb.py"
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
_patch_v20h = ens._patch_v20h


def _md(tag: str, gold: float | None, weight: str, dataset: str) -> str:
    gold_txt = (
        f", gold in-sample macro-AUC {gold:.3f}" if gold is not None else ""
    )
    extra = ""
    if tag.startswith("SWA"):
        extra = (
            "\n**SWA** : moyenne uniforme des poids ep7+ep8+ep9+ep10 "
            "(BN gelée, pas de re-estimation). Un seul modèle, pas un "
            "ensemble de prédictions.\n"
        )
    return f"""# RSNA Knee — v20h oofmask + laterality L, **solo** {tag}

Notebook **self-contained** pour Kaggle (internet off, **T4**), **un seul modèle** :
DICOM test → crop 130 mm + bande 20–80 % → **canon laterality L** (flip des
genoux droits : Coronal/Axial axe W, Sagittal axe D) → volumes (24, 224, 224)
→ **KneeNet v20h** (`v20h_crop130_oofmask_v20b_lat_full`, {tag}{gold_txt})
→ TTA affine ×5 → `submission.csv`.
{extra}
Pas d'ensemble ViT-e : ce checkpoint a ete train **avec** flip L/R.

Poids (dataset `{dataset}`) :
- `{weight}` — optimizer retire, 6 slots FS/T1, ResNet-50 RadImageNet

TTA : identite + rotation ±5° + scale 0.95/1.05. **Pas de hflip** (ce n'est
pas la canon laterality).
Le kernel **doit** ecrire `submission.csv` (cwd / `/kaggle/working`).
"""


def _config_code(weight: str, local_rel: str, tag: str) -> str:
    return f"""import os
from pathlib import Path

KAGGLE = os.path.exists('/kaggle')
WEIGHT_NAME = {weight!r}
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
        raise FileNotFoundError(f'{{name}} introuvable (dataset v20h lat attache ?)')

    WEIGHTS_PATH = _find_weight(WEIGHT_NAME)
else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_PATH = Path({local_rel!r}) / WEIGHT_NAME

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
# Ce run a ete train avec laterality: true (canon L, flip R). Obligatoire.
LATERALITY_CANON_L = True

print('competition dir :', COMP_DIR)
print('poids           :', WEIGHTS_PATH, '| existe :', WEIGHTS_PATH.exists())
print('device          :', DEVICE)
print('TTA variantes   :', TTA_VARIANTS)
print('laterality      :', LATERALITY_CANON_L, '(flip R -> canon L)')
print('modele          : v20h oofmask lat {tag}')
"""


LOAD_INFER = r"""import pandas as pd
from collections import Counter

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']


def load_v20h(weights_path):
    ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
    cfg = ckpt.get('config', {})
    if not cfg.get('laterality'):
        raise RuntimeError(f'checkpoint sans laterality=true : {weights_path}')
    if str(cfg.get('model')) != 'kneenet_v20h':
        raise RuntimeError(f"model={cfg.get('model')!r}, attendu kneenet_v20h")
    model = KneeNetV20h(
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
          '| swa', ckpt.get('swa_epochs'),
          '| val macro-AUC :', round(float(ckpt.get('val_macro_auc', float('nan'))), 4),
          '| gold macro-AUC :', round(float(ckpt.get('gold_macro_auc', float('nan'))), 4),
          '| n_series', model.n_series,
          '| laterality', cfg.get('laterality'))
    return model, cfg


model, cfg = load_v20h(WEIGHTS_PATH)
assert int(model.n_series) == 6, model.n_series
assert LATERALITY_CANON_L is True

# --- Inference : TTA, un seul modele, pretreat = train (flip R) ----------
import time
import torch.nn.functional as F

test = pd.read_csv(TEST_CSV)
test_series = pd.read_csv(TEST_SERIES_CSV)
print(f'{len(test)} etudes a predire | TTA x{len(TTA_VARIANTS)} | v20h lat solo')


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
lat_counts = Counter()
t0 = time.time()
n_series = int(model.n_series)
with torch.no_grad():
    for i, uid in enumerate(test.StudyInstanceUID):
        _SERIES_CACHE.clear()
        srows = test_series[test_series.StudyInstanceUID == uid]
        sdir = TEST_SERIES_DIR / uid
        lat = laterality_from_study_dir(sdir)
        lat_counts[lat or 'none'] += 1
        vol = study_to_tensor(sdir, srows, n_series=n_series,
                              laterality=LATERALITY_CANON_L)
        x = torch.from_numpy(vol).unsqueeze(0).to(DEVICE)
        ps = []
        for angle_deg, scale in TTA_VARIANTS:
            xa = apply_affine(x, angle_deg, scale)
            ps.append(torch.sigmoid(_forward(xa, use_amp=True).float()))
        prob = torch.stack(ps, dim=0).mean(dim=0).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(test)} | {(time.time()-t0)/60:.1f} min | '
                  f'lat {dict(lat_counts)}', flush=True)

print('laterality counts (tag puis geo, deadzone 20 mm) :', dict(lat_counts))
pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
pred.head()
"""


def _metadata(slug: str, dataset: str) -> dict:
    return {
        "id": f"sultent/{slug}",
        "title": slug,
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [dataset],
        "category_ids": [],
    }


def build_notebook(tag: str, gold: float | None, weight: str,
                   dataset: str, local_rel: str) -> dict:
    v20h_code = _patch_v20h(
        (ROOT / "src" / "models" / "kneenet_v20h.py").read_text(encoding="utf-8")
    )
    return {
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
            cell_md(_md(tag, gold, weight, dataset)),
            cell_code(_config_code(weight, local_rel, tag)),
            cell_code(PREPROCESS_CODE),
            cell_code(
                "# --- KneeNet v20h (copie self-contained de src/models/kneenet_v20h.py) ---\n"
                + v20h_code
            ),
            cell_code(NPZ_CODE),
            cell_code(LOAD_INFER),
            cell_code(SUBMISSION_CODE),
        ],
    }


def _write(slug: str, notebook: dict) -> list[Path]:
    text = json.dumps(notebook, indent=1, ensure_ascii=False)
    push = KERNEL_DIR / f"_push_{slug.replace('rsna-knee-', '')}"
    if slug == "rsna-knee-v20h-lat-solo-swa":
        push = KERNEL_DIR / "_push_v20h_swa"
        outs = [
            KERNEL_DIR / "submit_v20h_swa.ipynb",
            push / "submit.ipynb",
        ]
    else:
        ep = slug.rsplit("ep", 1)[-1]
        push = KERNEL_DIR / f"_push_v20h_ep{ep}"
        outs = [
            KERNEL_DIR / f"submit_v20h_ep{ep}.ipynb",
            push / "submit.ipynb",
        ]
    push.mkdir(parents=True, exist_ok=True)
    written = []
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print("ecrit", out)
        written.append(out)
    return written, push


def write_epoch(epoch: int) -> list[Path]:
    if epoch not in GOLD:
        raise ValueError(f"epoch {epoch} hors sondes {sorted(GOLD)}")
    if epoch == 7:
        dataset = "sultent/rsna-knee-v20h-ep7-lat"
        local = "../kaggle_dataset_v20h_lat/_upload_ep7"
    elif epoch in (11, 12):
        dataset = "sultent/rsna-knee-v20h-ep11-12-lat"
        local = "../kaggle_dataset_v20h_lat/_upload_ep11_12"
    else:
        dataset = "sultent/rsna-knee-v20h-oofmask-lat"
        local = "../kaggle_dataset_v20h_lat/_upload"
    weight = f"v20h_ep{epoch}.pt"
    slug = f"rsna-knee-v20h-lat-solo-ep{epoch}"
    notebook = build_notebook(
        f"epoch {epoch}", GOLD[epoch], weight, dataset, local,
    )
    written, push = _write(slug, notebook)
    meta = json.dumps(_metadata(slug, dataset), indent=2) + "\n"
    (push / "kernel-metadata.json").write_text(meta, encoding="utf-8")
    return written


def write_swa() -> list[Path]:
    dataset = "sultent/rsna-knee-v20h-swa-lat"
    weight = "v20h_swa_ep7_10.pt"
    slug = "rsna-knee-v20h-lat-solo-swa"
    notebook = build_notebook(
        "SWA ep7-10", None, weight, dataset,
        "../kaggle_dataset_v20h_lat/_upload_swa",
    )
    written, push = _write(slug, notebook)
    meta = json.dumps(_metadata(slug, dataset), indent=2) + "\n"
    (push / "kernel-metadata.json").write_text(meta, encoding="utf-8")
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, nargs="*", default=None)
    p.add_argument("--swa", action="store_true")
    args = p.parse_args()
    if args.swa:
        write_swa()
        return
    epochs = args.epoch if args.epoch else [8, 9, 10]
    for ep in epochs:
        write_epoch(ep)


if __name__ == "__main__":
    main()
