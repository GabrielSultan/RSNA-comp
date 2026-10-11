# -*- coding: utf-8 -*-
"""Trois ensembles hybrid025, SWA OAI ep1-3, sans TTA, sans plancher.

mix4 : softmax T=0.05 des 4 modeles.
mix3 : idem, CoAt exclu du softmax et du notebook.
spec : porte 0.015 sous la tete, puis softmax T=0.02.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"

DATA_VIT = "sultent/rsna-knee-vit-k320-oai3-swa"
DATA_V20B = "sultent/rsna-knee-v20b-dino-oai3-swa"
DATA_V20H = "sultent/rsna-knee-v20h-dino-oai3-swa"
DATA_COAT = "sultent/rsna-knee-coat-a-oai3-swa"

JOBS = (
    {
        "tag": "mix4",
        "slug": "rsna-knee-ens-oai3-h025-mix4",
        "with_coat": True,
        "rule": (
            "softmax des AUC val hybrid025 (epoch 7), temperature 0.05, "
            "sans plancher"
        ),
    },
    {
        "tag": "mix3",
        "slug": "rsna-knee-ens-oai3-h025-mix3",
        "with_coat": False,
        "rule": (
            "softmax des AUC val hybrid025 (epoch 7), temperature 0.05, "
            "sans plancher, CoAt exclu"
        ),
    },
    {
        "tag": "spec",
        "slug": "rsna-knee-ens-oai3-h025-spec",
        "with_coat": True,
        "rule": (
            "specialiste : AUC a plus de 0.015 sous la tete = 0, "
            "puis softmax temperature 0.02 sur les survivants"
        ),
    },
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _md(golds: dict[str, float], rows, job: dict) -> str:
    lines = []
    for t, w in rows:
        if job["with_coat"]:
            wv, wb, wh, wc = w
            lines.append(
                f"- `{t}` : ViT-k320 {wv:.0%} / v20b DINO {wb:.0%} / "
                f"v20h DINO {wh:.0%} / CoAt-a {wc:.0%}"
            )
        else:
            wv, wb, wh = w
            lines.append(
                f"- `{t}` : ViT-k320 {wv:.0%} / v20b DINO {wb:.0%} / "
                f"v20h DINO {wh:.0%}"
            )
    coat_line = ""
    if job["with_coat"]:
        coat_line = (
            f"- **CoAt-a** : `coat_a_llrd08_v3b_oofmask_full_oai3` "
            f"SWA epochs 1-3 (gold in-sample moyen {golds['coat']:.3f}). "
            "Entree 224. Pas de flip.\n"
        )
    who = "ViT-k320 / v20b DINO / v20h DINO / CoAt-a" if job["with_coat"] else (
        "ViT-k320 / v20b DINO / v20h DINO"
    )
    return f"""# RSNA Knee — {job['slug']}

Notebook **self-contained** (internet off, **T4**). **Sans TTA.**

Poids = SWA OAI epochs 1-3. {job['rule']}.
Subsets : `subset_vit_k_llrd08_320_v3b_hybrid025`, `subset_v20b_dino_ep11_v3b_hybrid025`, `subset_v20h_dino_ep11_v3b_hybrid025`{', `subset_coat_a_llrd08_v3b_hybrid025`' if job['with_coat'] else ''}.

- **ViT-k320** : `vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3` SWA epochs 1-3 (gold in-sample moyen {golds['vit']:.3f}). Entree **320**. Pas de flip.
- **v20b DINO** : `v20b_crop130_dino_ep11_oof_v3b_full_oai3` SWA epochs 1-3 (gold in-sample moyen {golds['v20b']:.3f}). Entree 224. Pas de flip.
- **v20h DINO** : `v20h_crop130_dino_ep11_oof_v3b_full_oai3` SWA epochs 1-3 (gold in-sample moyen {golds['v20h']:.3f}). Entree 224. Pas de flip.
{coat_line}
**Mix** ({who}) :
{chr(10).join(lines)}

Aucun flip de lateralite. Un modele peut valoir 0 %. Blend apres sigmoid.
"""


def _config(rows, with_coat: bool) -> str:
    if with_coat:
        mix = ",\n    ".join(
            f"({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}),  # {t}"
            for t, w in rows
        )
        find_coat = "    WEIGHTS_COAT = _find_weight('coat_a_oai3_swa_ep1_3.pt')\n"
        local_coat = (
            "    WEIGHTS_COAT = Path('kaggle_dataset_coat/_upload_oai3_swa/"
            "coat_a_oai3_swa_ep1_3.pt')\n"
        )
        print_coat = "print('poids coat      :', WEIGHTS_COAT, '| existe :', WEIGHTS_COAT.exists())\n"
    else:
        mix = ",\n    ".join(
            f"({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}),  # {t}"
            for t, w in rows
        )
        find_coat = ""
        local_coat = ""
        print_coat = ""
    n = 4 if with_coat else 3
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
{find_coat}else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_VIT = Path('kaggle_dataset_vit_k/_upload_k320_oai3_swa/vit_k320_oai3_swa_ep1_3.pt')
    WEIGHTS_V20H = Path('kaggle_dataset_v20h_lat/_upload_dino_oai3_swa/v20h_dino_oai3_swa_ep1_3.pt')
    WEIGHTS_V20B = Path('kaggle_dataset_v20b/_upload_dino_oai3_swa/v20b_dino_oai3_swa_ep1_3.pt')
{local_coat}
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

# Ordre : ViT-k320, v20b DINO, v20h DINO{', CoAt-a' if with_coat else ''}. Somme 1, plancher 0.
N_MODELS = {n}
MIX = [
    {mix}
]

print('competition dir :', COMP_DIR)
print('poids vit       :', WEIGHTS_VIT, '| existe :', WEIGHTS_VIT.exists())
print('poids v20h dino :', WEIGHTS_V20H, '| existe :', WEIGHTS_V20H.exists())
print('poids v20b dino :', WEIGHTS_V20B, '| existe :', WEIGHTS_V20B.exists())
{print_coat}print('device          :', DEVICE)
print('TTA             : non')
"""


def _infer(with_coat: bool) -> str:
    n = 4 if with_coat else 3
    coat_load = ""
    coat_call = ""
    coat_term = ""
    coat_print = ""
    if with_coat:
        coat_load = '''
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
'''
        coat_call = "        p_coat = torch.sigmoid(_forward(model_coat, x_224).float())\n"
        coat_term = " + w[:, 3] * p_coat"
        coat_print = " coat {w[3]:.2f}"
    return f'''import pandas as pd
import time

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']


def _cap_chunk(cfg, cap):
    raw = int(cfg.get('encode_chunk', cap))
    chunk = cap if raw > cap else raw
    if chunk != raw:
        print(f'encode_chunk {{raw}} -> {{chunk}} (T4)')
    return chunk


def _expect_swa(ckpt, label, epochs):
    if str(ckpt.get('epoch')) != label:
        raise RuntimeError(ckpt.get('epoch'))
    if [int(e) for e in (ckpt.get('swa_epochs') or [])] != list(epochs):
        raise RuntimeError(ckpt.get('swa_epochs'))


def load_vit(weights_path):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {{}})
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
    print('vit', cfg.get('run_name'), 'epoch', ckpt.get('epoch'),
          'gold', round(float(ckpt.get('gold_macro_auc')), 4))
    return model


def load_resnet(weights_path, cls, model_name, run_name, swa_label, swa_epochs):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {{}})
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
{coat_load}
test = pd.read_csv(TEST_CSV)
test_series = pd.read_csv(TEST_SERIES_CSV)
print(len(test), 'etudes | sans TTA | pas de flip')
for t, w in zip(TARGETS, MIX):
    if abs(sum(w) - 1.0) > 1e-6 or min(w) < -1e-9 or len(w) != {n}:
        raise RuntimeError((t, w))
    print(f'  {{t:<18}} vit {{w[0]:.2f}} v20b {{w[1]:.2f}} v20h {{w[2]:.2f}}{coat_print}')


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
{coat_call}        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)
        prob = (
            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h{coat_term}
        ).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{{i+1}}/{{len(test)}} | {{(time.time()-t0)/60:.1f}} min', flush=True)

pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
pred.head()
'''


def _check_rows(rows, n: int) -> None:
    if len(rows) != 12:
        raise SystemExit(f"{len(rows)} findings")
    for t, w in rows:
        if len(w) != n or abs(sum(w) - 1.0) > 1e-6 or min(w) < -1e-9:
            raise SystemExit(f"poids invalides {t} {w}")


def build(job: dict, rows, golds: dict[str, float], ens, ens4, coat_src: str) -> Path:
    n = 4 if job["with_coat"] else 3
    _check_rows(rows, n)
    cells = [
        ens.cell_md(_md(golds, rows, job)),
        ens.cell_code(_config(rows, job["with_coat"])),
        ens.cell_code(ens4._preprocess(ens.PREPROCESS_CODE)),
        ens.cell_code("# --- KneeNet v20b DINO ---\n" + ens._patch_v20b(
            (ROOT / "src" / "models" / "kneenet_v20b.py").read_text(encoding="utf-8")
        )),
        ens.cell_code("# --- KneeNet v20h DINO ---\n" + ens._patch_v20h(
            (ROOT / "src" / "models" / "kneenet_v20h.py").read_text(encoding="utf-8")
        )),
        ens.cell_code("# --- KneeNet ViT-e, poids ViT-k320 ---\n" + ens._patch_vite(
            (ROOT / "src" / "models" / "kneenet_vit_e.py").read_text(encoding="utf-8")
        )),
    ]
    if job["with_coat"]:
        cells.append(ens.cell_code("# --- KneeNet CoAt-a ---\n" + coat_src))
    cells.append(ens.cell_code(_infer(job["with_coat"])))
    cells.append(ens.cell_code(ens.SUBMISSION_CODE))
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    infer = "".join(cells[-2]["source"])
    cfg_src = "".join(cells[1]["source"])
    if "TTA_VARIANTS" in infer or "laterality=True" in infer:
        raise SystemExit("TTA ou flip")
    if "sans TTA" not in infer or "min(w) < -1e-9" not in infer:
        raise SystemExit("garde un plancher ou le TTA")
    if job["with_coat"]:
        if "coat_a_oai3_swa_ep1_3.pt" not in cfg_src or "KneeNetCoat" not in infer:
            raise SystemExit("coat absent")
    else:
        if "coat_a" in cfg_src or "KneeNetCoat" in infer or "model_coat" in infer:
            raise SystemExit("coat encore present")
    push = KERNEL / f"_push_ens_h025_{job['tag']}"
    push.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }
    (push / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
    )
    datasets = [DATA_VIT, DATA_V20B, DATA_V20H]
    if job["with_coat"]:
        datasets.append(DATA_COAT)
    meta = {
        "id": f"sultent/{job['slug']}",
        "title": job["slug"],
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
    (push / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("ok", push)
    for t, w in rows:
        print(f"  {t:<18} " + " ".join(f"{v:.2f}" for v in w))
    return push


def main() -> None:
    ens = _load("ens_nb", KERNEL / "_make_nb.py")
    ens4 = _load("ens4", KERNEL / "_make_nb_ens_k320_4.py")
    base = _load("oai3_ens", KERNEL / "_make_nb_ens_oai3_swa.py")
    preview = _load("h025", KERNEL / "_mix_hybrid025_preview.py")
    mixes = preview.ensemble_rows()
    golds = {
        "vit": base._gold_mean("vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3", (1, 2, 3)),
        "v20b": base._gold_mean("v20b_crop130_dino_ep11_oof_v3b_full_oai3", (1, 2, 3)),
        "v20h": base._gold_mean("v20h_crop130_dino_ep11_oof_v3b_full_oai3", (1, 2, 3)),
        "coat": base._gold_mean("coat_a_llrd08_v3b_oofmask_full_oai3", (1, 2, 3)),
    }
    coat_src = base._coat_code()
    for job in JOBS:
        print(job["tag"], job["rule"])
        build(job, mixes[job["tag"]], golds, ens, ens4, coat_src)


if __name__ == "__main__":
    main()
