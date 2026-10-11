# -*- coding: utf-8 -*-
"""Ensemble sans TTA : les 4 SWA OAI ep1-3 (ViT-k320, v20b, v20h, CoAt-a).

Meme mix par finding que l'ensemble RSNA SWA. Seuls les poids changent.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SLUG = "rsna-knee-ens-oai3-swa-sans-tta"
PUSH = KERNEL / "_push_ens_oai3_swa"
DATASETS = [
    "sultent/rsna-knee-vit-k320-oai3-swa",
    "sultent/rsna-knee-v20h-dino-oai3-swa",
    "sultent/rsna-knee-v20b-dino-oai3-swa",
    "sultent/rsna-knee-coat-a-oai3-swa",
]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gold_mean(run: str, epochs: tuple[int, ...]) -> float:
    hist = json.loads((ROOT / "checkpoints" / run / "history.json").read_text(encoding="utf-8"))
    vals = []
    for epoch in epochs:
        rec = next(r for r in hist if int(r["epoch"]) == epoch)
        vals.append(float(rec["gold_macro_auc"]))
    return sum(vals) / len(vals)


def _coat_code() -> str:
    text = (ROOT / "src" / "models" / "kneenet_coat_a.py").read_text(encoding="utf-8")
    needle = "import timm\n"
    if needle not in text:
        raise SystemExit("import timm introuvable")
    text = text.replace(
        needle, "import timm\nprint('timm', timm.__version__, flush=True)\n", 1,
    )
    old = "class KneeNet(nn.Module):"
    if text.count(old) != 1:
        raise SystemExit("classe KneeNet CoAt inattendue")
    text = text.replace(old, "class KneeNetCoat(nn.Module):", 1)
    # Les cellules du notebook partagent les globales. CoAt ne doit pas
    # ecraser TIMM_NAME / _weights_file du ViT.
    renames = {
        "TIMM_NAME": "COAT_TIMM_NAME",
        "DEFAULT_WEIGHTS": "COAT_DEFAULT_WEIGHTS",
        "N_TARGETS": "COAT_N_TARGETS",
        "LSTM_HIDDEN": "COAT_LSTM_HIDDEN",
        "N_PROTOCOL_SLOTS": "COAT_N_PROTOCOL_SLOTS",
        "_weights_file": "_coat_weights_file",
    }
    for src, dst in renames.items():
        text = re.sub(rf"\b{src}\b", dst, text)
    if "from src." in text or re.search(r"\bTIMM_NAME\b", text):
        raise SystemExit("renommage CoAt incomplet")
    return text


def _md(g_vit: float, g_v20b: float, g_v20h: float, g_coat: float, rows) -> str:
    lines = []
    for t, (wv, wb, wh, wc) in rows:
        lines.append(
            f"- `{t}` : ViT-k320 {wv:.0%} / v20b DINO {wb:.0%} / "
            f"v20h DINO {wh:.0%} / CoAt-a {wc:.0%}"
        )
    mix = "\n".join(lines)
    return f"""# RSNA Knee — Ensemble SWA OAI ep1-3 : ViT-k320 + v20b DINO + v20h DINO + CoAt-a

Notebook **self-contained** (internet off, **T4**). **Sans TTA.**

Meme mix que l'ensemble RSNA SWA. Poids = SWA des 3 epochs OAI.

- **ViT-k320** : `vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3` SWA epochs 1-3 (gold in-sample moyen {g_vit:.3f}). Entrée **320**. Pas de flip.
- **v20b DINO** : `v20b_crop130_dino_ep11_oof_v3b_full_oai3` SWA epochs 1-3 (gold in-sample moyen {g_v20b:.3f}). Entrée 224. Pas de flip.
- **v20h DINO** : `v20h_crop130_dino_ep11_oof_v3b_full_oai3` SWA epochs 1-3 (gold in-sample moyen {g_v20h:.3f}). Entrée 224. Pas de flip.
- **CoAt-a** : `coat_a_llrd08_v3b_oofmask_full_oai3` SWA epochs 1-3 (gold in-sample moyen {g_coat:.3f}). Entrée 224. Pas de flip.

Mix par finding = softmax des AUC val du **best** subset (température 0.05), puis plancher **10 %**.
Subsets : `subset_vit_k_llrd08_det_320_frac025`, `subset_v20b_crop130_det_frac025`, `subset_v20h_crop130_dino_ep11_frac025`, `subset_coat_a_llrd08_det_frac025`.

**Mix** (ViT-k320 / v20b DINO / v20h DINO / CoAt-a) :
{mix}

Aucun flip de latéralité. Blend après sigmoid.
"""


def _config(rows) -> str:
    mix = ",\n    ".join(
        f"({wv:.2f}, {wb:.2f}, {wh:.2f}, {wc:.2f}),  # {t}"
        for t, (wv, wb, wh, wc) in rows
    )
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
        raise FileNotFoundError(name)

    WEIGHTS_VIT = _find_weight('vit_k320_oai3_swa_ep1_3.pt')
    WEIGHTS_V20H = _find_weight('v20h_dino_oai3_swa_ep1_3.pt')
    WEIGHTS_V20B = _find_weight('v20b_dino_oai3_swa_ep1_3.pt')
    WEIGHTS_COAT = _find_weight('coat_a_oai3_swa_ep1_3.pt')
else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_VIT = Path('kaggle_dataset_vit_k/_upload_k320_oai3_swa/vit_k320_oai3_swa_ep1_3.pt')
    WEIGHTS_V20H = Path('kaggle_dataset_v20h_lat/_upload_dino_oai3_swa/v20h_dino_oai3_swa_ep1_3.pt')
    WEIGHTS_V20B = Path('kaggle_dataset_v20b/_upload_dino_oai3_swa/v20b_dino_oai3_swa_ep1_3.pt')
    WEIGHTS_COAT = Path('kaggle_dataset_coat/_upload_oai3_swa/coat_a_oai3_swa_ep1_3.pt')

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

# (ViT-k320, v20b DINO, v20h DINO, CoAt-a). Somme 1, plancher 0.10.
MIX = [
    {mix}
]

print('competition dir :', COMP_DIR)
print('poids vit       :', WEIGHTS_VIT, '| existe :', WEIGHTS_VIT.exists())
print('poids v20h dino :', WEIGHTS_V20H, '| existe :', WEIGHTS_V20H.exists())
print('poids v20b dino :', WEIGHTS_V20B, '| existe :', WEIGHTS_V20B.exists())
print('poids coat      :', WEIGHTS_COAT, '| existe :', WEIGHTS_COAT.exists())
print('device          :', DEVICE)
print('TTA             : non')
"""


def _infer() -> str:
    return r"""import pandas as pd
import time

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']


def _cap_chunk(cfg, cap):
    raw = int(cfg.get('encode_chunk', cap))
    chunk = cap if raw > cap else raw
    if chunk != raw:
        print(f'encode_chunk {raw} -> {chunk} (T4)')
    return chunk


def _expect_swa(ckpt, label, epochs):
    if str(ckpt.get('epoch')) != label:
        raise RuntimeError(ckpt.get('epoch'))
    if [int(e) for e in (ckpt.get('swa_epochs') or [])] != list(epochs):
        raise RuntimeError(ckpt.get('swa_epochs'))


def load_vit(weights_path):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    if str(cfg.get('run_name')) != 'vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3':
        raise RuntimeError(cfg.get('run_name'))
    if str(cfg.get('model')) != 'kneenet_vit_k320':
        raise RuntimeError(cfg.get('model'))
    if int(cfg.get('img_size', 0)) != 320:
        raise RuntimeError(cfg.get('img_size'))
    if cfg.get('laterality'):
        raise RuntimeError('vit-k320 ne doit pas etre flipped')
    _expect_swa(ckpt, 'swa-1-2-3', [1, 2, 3])
    model = KneeNetVitE(
        n_series=int(cfg.get('n_series', 6)),
        head_dim=int(cfg.get('head_dim', 512)),
        dropout=float(cfg.get('dropout', 0.4)),
        encode_chunk=_cap_chunk(cfg, 2),
        lstm_hidden=int(cfg.get('lstm_hidden', 256)),
        pretrained=False,
    )
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(DEVICE).eval()
    print('vit epoch', ckpt.get('epoch'), 'gold', round(float(ckpt.get('gold_macro_auc')), 4))
    return model


def load_resnet(weights_path, cls, model_name, run_name, swa_label, swa_epochs):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    if str(cfg.get('run_name')) != run_name:
        raise RuntimeError(cfg.get('run_name'))
    if str(cfg.get('model')) != model_name:
        raise RuntimeError(cfg.get('model'))
    if cfg.get('laterality'):
        raise RuntimeError('ce run ne doit pas etre flipped')
    _expect_swa(ckpt, swa_label, swa_epochs)
    model = cls(
        backbone=cfg.get('backbone', 'resnet50'),
        n_series=int(cfg.get('n_series', 6)),
        head_dim=int(cfg.get('head_dim', 512)),
        dropout=float(cfg.get('dropout', 0.4)),
        encode_chunk=_cap_chunk(cfg, 8),
        lstm_hidden=int(cfg.get('lstm_hidden', 256)),
        pretrained=False,
    )
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(DEVICE).eval()
    print(run_name, 'epoch', ckpt.get('epoch'), 'lat', cfg.get('laterality'),
          'gold', round(float(ckpt.get('gold_macro_auc')), 4))
    return model


model_vit = load_vit(WEIGHTS_VIT)
model_v20h = load_resnet(
    WEIGHTS_V20H, KneeNetV20h, 'kneenet_v20h',
    'v20h_crop130_dino_ep11_oof_v3b_full_oai3', 'swa-1-2-3', [1, 2, 3],
)
model_v20b = load_resnet(
    WEIGHTS_V20B, KneeNetV20b, 'kneenet_v20b',
    'v20b_crop130_dino_ep11_oof_v3b_full_oai3', 'swa-1-2-3', [1, 2, 3],
)


def load_coat(weights_path):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    if str(cfg.get('run_name')) != 'coat_a_llrd08_v3b_oofmask_full_oai3':
        raise RuntimeError(cfg.get('run_name'))
    if str(cfg.get('model')) != 'kneenet_coat_a':
        raise RuntimeError(cfg.get('model'))
    if str(cfg.get('backbone')) != 'coatnet_0_rw_224.sw_in1k':
        raise RuntimeError(cfg.get('backbone'))
    if cfg.get('laterality'):
        raise RuntimeError('coat ne doit pas etre flipped')
    if cfg.get('img_size') not in (None, 224):
        raise RuntimeError(cfg.get('img_size'))
    _expect_swa(ckpt, 'swa-1-2-3', [1, 2, 3])
    model = KneeNetCoat(
        backbone=str(cfg.get('backbone')),
        n_series=int(cfg.get('n_series', 6)),
        head_dim=int(cfg.get('head_dim', 512)),
        dropout=float(cfg.get('dropout', 0.4)),
        encode_chunk=_cap_chunk(cfg, 8),
        lstm_hidden=int(cfg.get('lstm_hidden', 256)),
        pretrained=False,
        img_size=224,
    )
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(DEVICE).eval()
    print('coat', cfg.get('run_name'), 'epoch', ckpt.get('epoch'),
          'lat', cfg.get('laterality'),
          'gold', round(float(ckpt.get('gold_macro_auc')), 4))
    return model


model_coat = load_coat(WEIGHTS_COAT)

test = pd.read_csv(TEST_CSV)
test_series = pd.read_csv(TEST_SERIES_CSV)
print(len(test), 'etudes | sans TTA | pas de flip')
for t, w in zip(TARGETS, MIX):
    if abs(sum(w) - 1.0) > 1e-6 or min(w) < 0.10 - 1e-6 or len(w) != 4:
        raise RuntimeError((t, w))
    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f}')


def _forward(model, x):
    if DEVICE == 'cuda':
        with torch.autocast('cuda', dtype=AMP_DTYPE):
            return model(x)
    return model(x)


rows = []
t0 = time.time()
with torch.no_grad():
    for i, uid in enumerate(test.StudyInstanceUID):
        _SERIES_CACHE.clear()
        srows = test_series[test_series.StudyInstanceUID == uid]
        sdir = TEST_SERIES_DIR / uid
        vol_224 = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)
        vol_vit = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=320)
        if vol_vit.shape[-1] != 320 or vol_224.shape[-1] != 224:
            raise RuntimeError((vol_vit.shape, vol_224.shape))
        x_224 = torch.from_numpy(vol_224).unsqueeze(0).to(DEVICE)
        x_vit = torch.from_numpy(vol_vit).unsqueeze(0).to(DEVICE)
        p_vit = torch.sigmoid(_forward(model_vit, x_vit).float())
        p_v20b = torch.sigmoid(_forward(model_v20b, x_224).float())
        p_v20h = torch.sigmoid(_forward(model_v20h, x_224).float())
        p_coat = torch.sigmoid(_forward(model_coat, x_224).float())
        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)
        prob = (
            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h + w[:, 3] * p_coat
        ).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(test)} | {(time.time()-t0)/60:.1f} min', flush=True)

pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
pred.head()
"""


def main() -> None:
    ens = _load("ens_nb", KERNEL / "_make_nb.py")
    ens4 = _load("ens4", KERNEL / "_make_nb_ens_k320_4.py")
    mix = _load("mix_swa_coat", KERNEL / "_mix_swa_coat.py")
    rows = mix.mix_rows()
    g_vit = _gold_mean("vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3", (1, 2, 3))
    g_v20h = _gold_mean("v20h_crop130_dino_ep11_oof_v3b_full_oai3", (1, 2, 3))
    g_v20b = _gold_mean("v20b_crop130_dino_ep11_oof_v3b_full_oai3", (1, 2, 3))
    g_coat = _gold_mean("coat_a_llrd08_v3b_oofmask_full_oai3", (1, 2, 3))
    v20b_code = ens._patch_v20b(
        (ROOT / "src" / "models" / "kneenet_v20b.py").read_text(encoding="utf-8")
    )
    v20h_code = ens._patch_v20h(
        (ROOT / "src" / "models" / "kneenet_v20h.py").read_text(encoding="utf-8")
    )
    vite_code = ens._patch_vite(
        (ROOT / "src" / "models" / "kneenet_vit_e.py").read_text(encoding="utf-8")
    )
    cells = [
        ens.cell_md(_md(g_vit, g_v20b, g_v20h, g_coat, rows)),
        ens.cell_code(_config(rows)),
        ens.cell_code(ens4._preprocess(ens.PREPROCESS_CODE)),
        ens.cell_code("# --- KneeNet v20b DINO ---\n" + v20b_code),
        ens.cell_code("# --- KneeNet v20h DINO ---\n" + v20h_code),
        ens.cell_code("# --- KneeNet ViT-e, poids ViT-k320 ---\n" + vite_code),
        ens.cell_code("# --- KneeNet CoAt-a ---\n" + _coat_code()),
        ens.cell_code(_infer()),
        ens.cell_code(ens.SUBMISSION_CODE),
    ]
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }
    infer = "".join(cells[7]["source"])
    cfg_src = "".join(cells[1]["source"])
    if "TTA_VARIANTS" in infer or "laterality=True" in infer:
        raise SystemExit("TTA ou flip present")
    if "sans TTA" not in infer or "img_size=320" not in infer:
        raise SystemExit("inference incomplete")
    if infer.count("laterality=False") != 2:
        raise SystemExit("volumes inattendus")
    for needle in (
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "v20h_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "swa-1-2-3",
        "KneeNetCoat",
        "pretrained=False",
    ):
        if needle not in infer:
            raise SystemExit(f"infer sans {needle}")
    if "swa-7-8-9-10" in infer or "swa-6-7" in infer:
        raise SystemExit("garde un SWA RSNA")
    if "coat_a_oai3_swa_ep1_3.pt" not in cfg_src:
        raise SystemExit("poids coat absent")
    if "v20h_crop130_oofmask_v20b_lat_full" in infer:
        raise SystemExit("lat encore present")
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "id": f"sultent/{SLUG}",
        "title": SLUG,
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
        "dataset_sources": DATASETS,
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("ok", PUSH)
    print(
        "gold", round(g_vit, 3), round(g_v20b, 3), round(g_v20h, 3), round(g_coat, 3),
    )
    for t, w in rows:
        print(f"  {t:<18} {w[0]:.2f} {w[1]:.2f} {w[2]:.2f} {w[3]:.2f}")


if __name__ == "__main__":
    main()
