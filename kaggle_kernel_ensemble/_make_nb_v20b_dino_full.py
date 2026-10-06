# -*- coding: utf-8 -*-
"""Poids + notebook T4 : v20b DINO ep11 full oofmask v3b, TTA x5, sans flip."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_v20b import KneeNet as KneeNetV20b  # noqa: E402

KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
RUN = ROOT / "checkpoints" / "v20b_crop130_dino_ep11_oof_v3b_full"
RUN_NAME = "v20b_crop130_dino_ep11_oof_v3b_full"
KEEP = ("model", "config", "epoch", "gold_macro_auc")
# Le yaml d'entrainement met 576 (GPU 96 Go). Sur T4, 8 suffit et
# ne change pas les logits (decoupe du forward encodeur seulement).
INFER_CHUNK = 8


def _load_nb():
    p = KERNEL_DIR / "_make_nb_v20b.py"
    spec = importlib.util.spec_from_file_location("make_nb_v20b", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ids(epoch: int) -> tuple[Path, str, str, str]:
    upload = ROOT / "kaggle_dataset_v20b" / f"_upload_dino_full_ep{epoch}"
    weight = f"v20b_dino_full_ep{epoch}.pt"
    dataset = f"sultent/rsna-knee-v20b-dino-full-ep{epoch}"
    slug = f"rsna-knee-v20b-dino-full-ep{epoch}-tta-x5"
    return upload, weight, dataset, slug


def strip_epoch(epoch: int) -> float:
    src = RUN / f"epoch{epoch}.pt"
    if not src.is_file():
        raise FileNotFoundError(src)
    upload, weight, dataset, _slug = _ids(epoch)
    upload.mkdir(parents=True, exist_ok=True)
    dest = upload / weight
    print("load", src.name, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    if str(cfg.get("model")) != "kneenet_v20b":
        raise SystemExit(f"model={cfg.get('model')!r}")
    if cfg.get("run_name") != RUN_NAME:
        raise SystemExit(f"run inattendu {cfg.get('run_name')}")
    if cfg.get("laterality"):
        raise SystemExit("ce checkpoint a laterality=true, notebook a revoir")
    got = int(ckpt.get("epoch"))
    if got != epoch:
        raise SystemExit(f"{src.name} epoch {got}, attendu {epoch}")
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    rec = next(r for r in hist if int(r["epoch"]) == epoch)
    gold = float(rec["gold_macro_auc"])
    cfg["laterality"] = False
    out = {k: ckpt[k] for k in KEEP if k in ckpt and k != "gold_macro_auc"}
    out["config"] = cfg
    out["epoch"] = epoch
    out["gold_macro_auc"] = gold
    torch.save(out, dest)
    model = KneeNetV20b(
        pretrained=False,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=INFER_CHUNK,
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    )
    model.load_state_dict(out["model"], strict=True)
    print(
        "strip", dest.name, dest.stat().st_size,
        "epoch", epoch, "gold", round(gold, 4),
        "laterality", False, flush=True,
    )
    meta = {
        "title": f"RSNA Knee v20b DINO full ep{epoch}",
        "id": dataset,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (upload / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return gold


def write_notebook(epoch: int, gold: float) -> Path:
    nb = _load_nb()
    upload, weight, dataset, slug = _ids(epoch)
    local = str(upload.relative_to(ROOT)).replace("\\", "/")
    md = f"""# RSNA Knee — v20b DINO ep11 full oofmask v3b, **solo** epoch {epoch}

Notebook **self-contained** (internet off, GPU **T4**), un seul modèle :
DICOM test → crop 130 mm + bande 20–80 % → volumes (24, 224, 224),
**sans flip** gauche/droite → **KneeNet v20b** (`{RUN_NAME}`, encodeur DINO,
epoch {epoch}, gold in-sample macro-AUC {gold:.3f})
→ TTA affine ×5 → `submission.csv`.

Full train apres masque OOF, labels Coolin targets_v3b_oofmask,
gold dans le train (poids x4). Pas de canon laterality.

Poids `{weight}` (optimizer retire), dataset `{dataset}`.

TTA : identite + rotation ±5° + scale 0.95/1.05. Pas de hflip.
Le kernel **doit** ecrire `submission.csv` (cwd / `/kaggle/working`).
"""
    config = f"""import os
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
        raise FileNotFoundError(f'{{name}} introuvable (dataset v20b dino full ep{epoch} attache ?)')

    WEIGHTS_PATH = _find_weight(WEIGHT_NAME)
else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_PATH = Path({local!r}) / WEIGHT_NAME

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
print('laterality      :', LATERALITY_CANON_L, '(pas de flip)')
print('modele          : v20b dino full ep{epoch}')
"""
    infer = nb.LOAD_INFER.replace(
        "encode_chunk=max(8, int(cfg.get('encode_chunk', 8))),",
        f"encode_chunk={INFER_CHUNK},",
    ).replace(
        """    print('modele', cfg.get('model'), '| epoch', ckpt.get('epoch'),
          '| swa', ckpt.get('swa_epochs'),
          '| val macro-AUC :', round(float(ckpt.get('val_macro_auc', float('nan'))), 4),
          '| gold_by_epoch :', ckpt.get('gold_by_epoch'),
          '| n_series', model.n_series,
          '| laterality', cfg.get('laterality'))
""",
        """    gold = ckpt.get('gold_macro_auc', float('nan'))
    print('modele', cfg.get('model'), '| run', cfg.get('run_name'),
          '| epoch', ckpt.get('epoch'),
          '| gold macro-AUC :', None if gold is None else round(float(gold), 4),
          '| n_series', model.n_series,
          '| encode_chunk', model.encode_chunk,
          '| laterality', cfg.get('laterality'))
""",
    ).replace(
        "print(f'{len(test)} etudes a predire | TTA x{len(TTA_VARIANTS)} | v20b swa solo')",
        f"print(f'{{len(test)}} etudes a predire | TTA x{{len(TTA_VARIANTS)}} | v20b dino full ep{epoch}')",
    )
    if f"encode_chunk={INFER_CHUNK}" not in infer or "TTA x" not in infer:
        raise SystemExit("notebook inference incomplet")
    if "LATERALITY_CANON_L is False" not in infer:
        raise SystemExit("assert laterality manquant")
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
    text = json.dumps(notebook, indent=1, ensure_ascii=False)
    push = KERNEL_DIR / f"_push_v20b_dino_full_ep{epoch}"
    push.mkdir(parents=True, exist_ok=True)
    (push / "submit.ipynb").write_text(text, encoding="utf-8")
    (KERNEL_DIR / f"submit_v20b_dino_full_ep{epoch}.ipynb").write_text(
        text, encoding="utf-8"
    )
    (push / "kernel-metadata.json").write_text(
        json.dumps(nb._metadata(slug, dataset), indent=2) + "\n",
        encoding="utf-8",
    )
    print("ecrit", push / "submit.ipynb", flush=True)
    return push


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, nargs="+", required=True)
    args = p.parse_args()
    for epoch in args.epoch:
        gold = strip_epoch(epoch)
        write_notebook(epoch, gold)


if __name__ == "__main__":
    main()
