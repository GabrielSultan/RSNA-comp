# -*- coding: utf-8 -*-
"""Deux ensembles Kaggle : v20b ep8 + v20h ep7, puis le meme + ViT-e ep7.

v20h est train en canon L (flip R). v20b et ViT-e ne le sont pas :
deux tenseurs, un blend de probabilites.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# Part v20h. Reste = v20b. Plancher 0.35 : v20h ep7 colle le LB de v20b,
# le smooth peut etre genereux. Plus haut la ou le subset 6 slots gagne.
W_H = {
    "ACL": 0.35,
    "MCL": 0.65,
    "Medial Meniscus": 0.35,
    "Lateral Meniscus": 0.35,
    "Medial OA": 0.35,
    "Lateral OA": 0.35,
    "PF OA": 0.75,
    "Effusion": 0.60,
    "Synovitis": 0.35,
    "Baker's": 0.35,
    "Contusion": 0.35,
    "Fracture": 0.35,
}
# ViT-e specialiste pure : seulement Effusion (delta subset +0.034 vs v20b ep10).
# 0.70, pas 0.90. Le reliquat garde le ratio v20b/v20h de l'ensemble 1 (0.40/0.60).
W_VITE_EFF = 0.70


def _load_ens():
    p = KERNEL_DIR / "_make_nb.py"
    spec = importlib.util.spec_from_file_location("ens_make_nb", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _weights(with_vite: bool) -> list[tuple[float, float, float]]:
    rows = []
    for t in TARGETS:
        wh = W_H[t]
        wb = 1.0 - wh
        wv = 0.0
        if with_vite and t == "Effusion":
            wv = W_VITE_EFF
            rest = 1.0 - wv
            wb = rest * (1.0 - W_H[t])
            wh = rest * W_H[t]
        rows.append((round(wb, 4), round(wh, 4), round(wv, 4)))
    return rows


def _mix_lines(rows: list[tuple[float, float, float]], with_vite: bool) -> str:
    lines = []
    for t, (wb, wh, wv) in zip(TARGETS, rows):
        if with_vite:
            lines.append(f"- `{t}` : v20b {wb:.0%} / v20h {wh:.0%} / ViT-e {wv:.0%}")
        else:
            lines.append(f"- `{t}` : v20b {wb:.0%} / v20h {wh:.0%}")
    return "\n".join(lines)


def _md(with_vite: bool, rows) -> str:
    mix = _mix_lines(rows, with_vite)
    vite = ""
    if with_vite:
        vite = (
            "\n+ **KneeNet ViT-e** (`vit_e_crop130_oofmask_full` epoch 7, "
            "spécialiste pure, **sans** flip).\n"
            "ViT-e n'entre que sur **Effusion** (0,70). Le reliquat 0,30 "
            "garde le ratio v20b/v20h de l'ensemble sans ViT (0,40 / 0,60).\n"
        )
    return f"""# RSNA Knee — Ensemble v20b ep8 + v20h lat ep7{" + ViT-e ep7" if with_vite else ""}

Notebook **self-contained** (internet off, **T4**).

- **v20b** baseline : `v20b_crop130_oofmask_full` epoch 8 (gold in-sample 0,959, public LB solo ~0,945). **Pas de flip.**
- **v20h** spécialiste : `v20h_crop130_oofmask_v20b_lat_full` epoch 7 (gold in-sample 0,958). **Canon L** (flip des genoux droits). LB public du même ordre que v20b, donc le smooth est généreux.
{vite}
Deux tenseurs : v20b{" et ViT-e" if with_vite else ""} voient le volume brut, v20h voit le volume flippé. Blend **après** sigmoid, par finding.

**Mix** :
{mix}

TTA affine ×5 (identité, ±5°, scale 0.95/1.05). Pas de hflip TTA.
Le kernel écrit `submission.csv`.
"""


def _config(with_vite: bool, rows: list[tuple[float, float, float]]) -> str:
    vite_find = ""
    vite_else = ""
    vite_print = ""
    if with_vite:
        vite_find = "    WEIGHTS_VITE = _find_weight('vit_e_ep7.pt')\n"
        vite_else = "    WEIGHTS_VITE = Path('../kaggle_dataset_ensemble/_upload/vit_e_ep7.pt')\n"
        vite_print = "print('poids vit_e     :', WEIGHTS_VITE, '| existe :', WEIGHTS_VITE.exists())\n"
    return f"""import os
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
        raise FileNotFoundError(name + ' introuvable')

    WEIGHTS_V20B = _find_weight('v20b_oofmask_ep8.pt')
    WEIGHTS_V20H = _find_weight('v20h_ep7.pt')
{vite_find}else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_V20B = Path('../kaggle_dataset_v20b/_upload_ep8/v20b_oofmask_ep8.pt')
    WEIGHTS_V20H = Path('../kaggle_dataset_v20h_lat/_upload_ep7/v20h_ep7.pt')
{vite_else}
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

TTA_VARIANTS = (
    (0.0, 1.00),
    (5.0, 1.00),
    (-5.0, 1.00),
    (0.0, 0.95),
    (0.0, 1.05),
)
WITH_VITE = {with_vite}
MIX = {rows!r}

print('competition dir :', COMP_DIR)
print('poids v20b      :', WEIGHTS_V20B, '| existe :', WEIGHTS_V20B.exists())
print('poids v20h      :', WEIGHTS_V20H, '| existe :', WEIGHTS_V20H.exists())
{vite_print}print('device          :', DEVICE)
print('with vite       :', WITH_VITE)
"""


def _load_infer(with_vite: bool) -> str:
    vite_load = ""
    vite_fwd = ""
    vite_blend = "0.0"
    if with_vite:
        vite_load = """
def load_vite(weights_path):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    if cfg.get('laterality'):
        raise RuntimeError('ViT-e ne doit pas etre flipped')
    if str(cfg.get('run_name')) != 'vit_e_crop130_oofmask_full':
        raise RuntimeError(cfg.get('run_name'))
    if int(ckpt.get('epoch', -1)) != 7:
        raise RuntimeError(ckpt.get('epoch'))
    model = KneeNetVitE(
        n_series=int(cfg.get('n_series', 6)),
        head_dim=int(cfg.get('head_dim', 512)),
        dropout=float(cfg.get('dropout', 0.4)),
        encode_chunk=int(cfg.get('encode_chunk', 4)),
        lstm_hidden=int(cfg.get('lstm_hidden', 256)),
        pretrained=False,
    )
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(DEVICE).eval()
    print('vit_e epoch', ckpt.get('epoch'), 'run', cfg.get('run_name'))
    return model

model_vite = load_vite(WEIGHTS_VITE)
"""
        vite_fwd = """
            pes.append(torch.sigmoid(_forward(model_vite, xa, use_amp=True).float()))"""
        vite_blend = "torch.stack(pes, 0).mean(0)"
    models_global = "model_v20b, model_v20h" + (", model_vite" if with_vite else "")
    return f"""import pandas as pd
import time
import torch.nn.functional as F
from collections import Counter

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']


def load_v20b(weights_path):
    ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
    cfg = ckpt.get('config', {{}})
    if cfg.get('laterality'):
        raise RuntimeError('v20b full ne doit pas etre flipped')
    if str(cfg.get('run_name')) != 'v20b_crop130_oofmask_full':
        raise RuntimeError(cfg.get('run_name'))
    if int(ckpt.get('epoch', -1)) != 8:
        raise RuntimeError(ckpt.get('epoch'))
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
    print('v20b epoch', ckpt.get('epoch'), 'gold', ckpt.get('gold_macro_auc'), 'run', cfg.get('run_name'))
    return model


def load_v20h(weights_path):
    ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
    cfg = ckpt.get('config', {{}})
    if not cfg.get('laterality'):
        raise RuntimeError('v20h sans laterality')
    if str(cfg.get('run_name')) != 'v20h_crop130_oofmask_v20b_lat_full':
        raise RuntimeError(cfg.get('run_name'))
    if int(ckpt.get('epoch', -1)) != 7:
        raise RuntimeError(ckpt.get('epoch'))
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
    print('v20h epoch', ckpt.get('epoch'), 'gold', ckpt.get('gold_macro_auc'), 'lat', cfg.get('laterality'))
    return model

{vite_load}
model_v20b = load_v20b(WEIGHTS_V20B)
model_v20h = load_v20h(WEIGHTS_V20H)
assert int(model_v20b.n_series) == 6 and int(model_v20h.n_series) == 6

test = pd.read_csv(TEST_CSV)
test_series = pd.read_csv(TEST_SERIES_CSV)
print(len(test), 'etudes | TTA', len(TTA_VARIANTS))
for t, (wb, wh, wv) in zip(TARGETS, MIX):
    s = wb + wh + wv
    if abs(s - 1.0) > 1e-3:
        raise RuntimeError((t, s))
    print(f'  {{t:<18}} v20b {{wb:.2f}} v20h {{wh:.2f}} vite {{wv:.2f}}')


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


def _forward(model, x, use_amp=True):
    global DEVICE, {models_global}
    try:
        if use_amp and DEVICE == 'cuda':
            with torch.autocast('cuda', dtype=AMP_DTYPE):
                return model(x)
        return model(x)
    except RuntimeError as e:
        print('forward CUDA echoue :', str(e)[:80])
        torch.backends.cudnn.enabled = False
        try:
            if use_amp and DEVICE == 'cuda':
                with torch.autocast('cuda', dtype=AMP_DTYPE):
                    return model(x)
            return model(x)
        except RuntimeError:
            DEVICE = 'cpu'
            model_v20b = model_v20b.to('cpu')
            model_v20h = model_v20h.to('cpu')
            {"model_vite = model_vite.to('cpu')" if with_vite else ""}
            return model.to('cpu')(x.cpu())


rows = []
lat_counts = Counter()
t0 = time.time()
n_series = 6
with torch.no_grad():
    for i, uid in enumerate(test.StudyInstanceUID):
        _SERIES_CACHE.clear()
        srows = test_series[test_series.StudyInstanceUID == uid]
        sdir = TEST_SERIES_DIR / uid
        lat = laterality_from_study_dir(sdir)
        lat_counts[lat or 'none'] += 1
        vol_b = study_to_tensor(sdir, srows, n_series=n_series, laterality=False)
        vol_h = study_to_tensor(sdir, srows, n_series=n_series, laterality=True)
        xb = torch.from_numpy(vol_b).unsqueeze(0).to(DEVICE)
        xh = torch.from_numpy(vol_h).unsqueeze(0).to(DEVICE)
        pbs, phs, pes = [], [], []
        for angle_deg, scale in TTA_VARIANTS:
            pbs.append(torch.sigmoid(_forward(model_v20b, apply_affine(xb, angle_deg, scale), use_amp=True).float()))
            phs.append(torch.sigmoid(_forward(model_v20h, apply_affine(xh, angle_deg, scale), use_amp=True).float()))
            {"pes.append(torch.sigmoid(_forward(model_vite, apply_affine(xb, angle_deg, scale), use_amp=True).float()))" if with_vite else "pass"}
        pb = torch.stack(pbs, 0).mean(0)
        ph = torch.stack(phs, 0).mean(0)
        pe = {vite_blend}
        w = torch.tensor(MIX, dtype=torch.float32, device=pb.device)
        prob = (w[:, 0] * pb + w[:, 1] * ph + w[:, 2] * pe).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{{i+1}}/{{len(test)}} | {{(time.time()-t0)/60:.1f}} min | lat {{dict(lat_counts)}}', flush=True)

print('laterality (v20h seulement) :', dict(lat_counts))
pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
pred.head()
"""


def _meta(slug: str, title: str, datasets: list[str]) -> dict:
    return {
        "id": f"sultent/{slug}",
        "title": title,
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
        "dataset_sources": datasets,
        "category_ids": [],
    }


def _one(ens, with_vite: bool, slug: str, title: str, datasets: list[str], push: Path) -> None:
    rows = _weights(with_vite)
    v20b_code = ens._patch_v20b((ROOT / "src" / "models" / "kneenet_v20b.py").read_text(encoding="utf-8"))
    v20h_code = ens._patch_v20h((ROOT / "src" / "models" / "kneenet_v20h.py").read_text(encoding="utf-8"))
    cells = [
        ens.cell_md(_md(with_vite, rows)),
        ens.cell_code(_config(with_vite, rows)),
        ens.cell_code(ens.PREPROCESS_CODE),
        ens.cell_code("# --- KneeNet v20b ---\n" + v20b_code),
        ens.cell_code("# --- KneeNet v20h ---\n" + v20h_code),
    ]
    if with_vite:
        vite_code = ens._patch_vite((ROOT / "src" / "models" / "kneenet_vit_e.py").read_text(encoding="utf-8"))
        cells.append(ens.cell_code("# --- KneeNet ViT-e ---\n" + vite_code))
    cells.append(ens.cell_code(_load_infer(with_vite)))
    cells.append(ens.cell_code(ens.SUBMISSION_CODE))
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }
    push.mkdir(parents=True, exist_ok=True)
    text = json.dumps(nb, indent=1, ensure_ascii=False)
    (push / "submit.ipynb").write_text(text, encoding="utf-8")
    (push / "kernel-metadata.json").write_text(
        json.dumps(_meta(slug, title, datasets), indent=2) + "\n", encoding="utf-8"
    )
    print("ecrit", push, "cells", len(cells))


def main() -> None:
    ens = _load_ens()
    base_ds = [
        "sultent/rsna-knee-v20b-oofmask-ep8",
        "sultent/rsna-knee-v20h-ep7-lat",
    ]
    _one(
        ens, False,
        "rsna-knee-ens-v20b-v20h",
        "rsna-knee-ens-v20b-v20h",
        base_ds,
        KERNEL_DIR / "_push_ens_v20h",
    )
    _one(
        ens, True,
        "rsna-knee-ens-v20b-v20h-vite",
        "rsna-knee-ens-v20b-v20h-vite",
        base_ds + ["sultent/rsna-knee-ensemble-weights"],
        KERNEL_DIR / "_push_ens_v20h_vite",
    )
    # sanity: effusion carve
    a = _weights(False)
    b = _weights(True)
    assert abs(sum(a[7]) - 1) < 1e-6 and abs(sum(b[7]) - 1) < 1e-6
    assert b[7][2] == 0.7
    assert all(x[2] == 0 for i, x in enumerate(b) if i != 7)
    print("effusion sans vite", a[7], "avec vite", b[7])


if __name__ == "__main__":
    main()
