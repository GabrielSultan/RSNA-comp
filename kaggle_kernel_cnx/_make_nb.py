# -*- coding: utf-8 -*-
"""Notebook Kaggle CNX-r7 IN-22k, un forward, sans TTA. FOV 130 mm."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "cnx_r7_in22k_v3b_oofmask_full"
BACKBONE = "convnext_small.fb_in22k_ft_in1k"
MODEL_NAME = "kneenet_cnx_r7"


def cell_md(text: str) -> dict:
    src = text if text.endswith("\n") else text + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in src.rstrip("\n").split("\n")],
    }


def cell_code(text: str) -> dict:
    src = text if text.endswith("\n") else text + "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in src.rstrip("\n").split("\n")],
    }


def _md(gold: float, weight: str, dataset: str) -> str:
    return f"""# RSNA Knee — CNX-r7 IN-22k v3b — T4 — sans TTA

Notebook **self-contained** (internet off, GPU **T4**). **Sans TTA.**

- DICOM test → volumes crop **130 mm**, bande 20–80 %, (6, 24, 224, 224)
- protocole 6 slots FS/T1. Pas de flip de latéralité
- KneeNet **CNX-r7** : ConvNeXt-Small `{BACKBONE}`, tête R7
  train `{RUN_NAME}` **SWA** epochs 7, 8, 9 et 10
  (gold in-sample moyen {gold:.3f})
- poids `{weight}`, dataset `{dataset}`
- un seul forward
- écrit `/kaggle/working/submission.csv`
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

    def _find_weight(names):
        for name in names:
            for root in search_roots:
                hits = list(root.rglob(name))
                if hits:
                    return hits[0]
        raise FileNotFoundError(f'{names} introuvable (dataset weights attache ?)')

    WEIGHTS_PATH = _find_weight(('__WEIGHT__',))
else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_PATH = Path('__LOCAL__/__WEIGHT__')

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

print('competition dir :', COMP_DIR)
print('poids            :', WEIGHTS_PATH, '| existe :', WEIGHTS_PATH.exists())
print('device           :', DEVICE)
"""


INFER_CODE = r"""import numpy as np
import pandas as pd

import torch

from pathlib import Path

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']

N_TARGETS = len(TARGETS)


@torch.no_grad()
def _predict_logits(model, x: torch.Tensor) -> torch.Tensor:
    # ConvNeXt depthwise : cuDNN T4 ne trouve pas d'algo en bf16/fp16.
    with torch.autocast("cuda", enabled=False):
        return model(x.float())


def load_model(weights_path: str | Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if str(cfg.get("model")) != "kneenet_cnx_r7":
        raise RuntimeError(f"model inattendu {cfg.get('model')}")
    if str(cfg.get("run_name")) != "__RUN__":
        raise RuntimeError(f"run inattendu {cfg.get('run_name')}")
    if str(cfg.get("backbone")) != "__BACKBONE__":
        raise RuntimeError(f"backbone inattendu {cfg.get('backbone')}")
    if cfg.get("laterality"):
        raise RuntimeError("laterality inattendue")
    if int(cfg.get("hidden", -1)) != 384:
        raise RuntimeError(f"hidden inattendu {cfg.get('hidden')}")
    if abs(float(cfg.get("dropout", -1.0)) - 0.1) > 1e-6:
        raise RuntimeError(f"dropout inattendu {cfg.get('dropout')}")
    if str(ckpt.get("epoch")) != "swa-7-8-9-10":
        raise RuntimeError(f"epoch inattendue {ckpt.get('epoch')}")
    if list(ckpt.get("swa_epochs") or []) != [7, 8, 9, 10]:
        raise RuntimeError(f"swa_epochs inattendu {ckpt.get('swa_epochs')}")
    n_series = int(cfg.get("n_series", 6))
    if n_series != 6:
        raise RuntimeError(f"n_series {n_series}")
    encode_chunk_cfg = int(cfg.get("encode_chunk", 16))
    encode_chunk = min(4, encode_chunk_cfg)
    if encode_chunk != encode_chunk_cfg:
        print(f"encode_chunk {encode_chunk_cfg} -> {encode_chunk} (T4 fp32)", flush=True)
    dropout = float(cfg.get("dropout", 0.1))
    max_depth = int(cfg.get("max_depth", 64) or 64)
    model = KneeNet(
        backbone=str(cfg.get("backbone")),
        n_series=n_series,
        n_targets=N_TARGETS,
        hidden=int(cfg.get("hidden", 384)),
        dropout=dropout,
        pretrained=False,
        encode_chunk=encode_chunk,
        max_depth=max_depth,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    print("load strict OK", flush=True)
    gold = ckpt.get("gold_macro_auc")
    gold_txt = "n/a" if gold is None else round(float(gold), 4)
    print(
        "modele", cfg.get("model"), "| run", cfg.get("run_name"),
        "| epoch", ckpt.get("epoch"), "| gold in-sample", gold_txt,
        "| swa", ckpt.get("swa_epochs"),
        "| n_series", n_series, "| img 224 | crop", CROP_MM, "mm | sans TTA",
        flush=True,
    )
    model.to(DEVICE).eval()
    return model, cfg


test = pd.read_csv(TEST_CSV)
series = pd.read_csv(TEST_SERIES_CSV)
test_series_dir = Path(TEST_SERIES_DIR)

series_by_uid = {uid: g for uid, g in series.groupby("StudyInstanceUID", sort=False)}

model, _cfg = load_model(WEIGHTS_PATH)
n_series = int(getattr(model, "n_series", 6))

rows = []
print(f"{len(test)} etudes a predire | sans TTA | crop {CROP_MM} mm | cnx forward fp32", flush=True)
for i, uid in enumerate(test.StudyInstanceUID):
    study_dir = test_series_dir / uid
    srows = series_by_uid.get(uid, series.iloc[0:0])

    vol = study_to_tensor(
        study_dir,
        srows,
        n_series=n_series,
        rgb_neighbors=False,
        img_size=224,
    )
    if vol.shape != (n_series, 24, 224, 224):
        raise RuntimeError(f"volume inattendu {vol.shape}")
    x0 = torch.from_numpy(vol).unsqueeze(0).to(device=DEVICE, dtype=torch.float32)
    logits = _predict_logits(model, x0)
    prob = torch.sigmoid(logits.float()).cpu().numpy()[0]
    rows.append([uid] + prob.tolist())
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(test)} ok", flush=True)

pred = pd.DataFrame(rows, columns=["StudyInstanceUID"] + TARGETS)

sample = pd.read_csv(SAMPLE_SUB)
sub = pred.set_index("StudyInstanceUID").reindex(sample.StudyInstanceUID).reset_index()
sub = sub[list(sample.columns)]
prior = sub[TARGETS].mean()
sub[TARGETS] = sub[TARGETS].fillna(prior)

out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
out_csv = out_dir / "submission.csv"
sub.to_csv(out_csv, index=False)
cwd_csv = Path("submission.csv")
if cwd_csv.resolve() != out_csv.resolve():
    sub.to_csv(cwd_csv, index=False)

print("ecrit", out_csv.resolve(), "| lignes", len(sub), "| existe", out_csv.exists())
print(sub.shape, "| colonnes OK :", list(sub.columns) == list(sample.columns))
print(sub[TARGETS].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
assert out_csv.exists() and out_csv.stat().st_size > 0, "submission.csv manquant"
assert not sub[TARGETS].isna().any().any(), "NaN dans la submission !"
assert list(sub.columns) == list(sample.columns), "colonnes != sample_submission"
"""


def _preprocess() -> str:
    text = (ROOT / "src" / "infer" / "preprocess_dicom.py").read_text(encoding="utf-8")
    old_slots = "from src.data.dataset_npz import PROTOCOL_SLOTS"
    new_slots = (
        "PROTOCOL_SLOTS: list[tuple[str, int]] = [\n"
        '    ("Sagittal", 1),\n'
        '    ("Coronal", 1),\n'
        '    ("Axial", 1),\n'
        '    ("Sagittal", 0),\n'
        '    ("Coronal", 0),\n'
        '    ("Axial", 0),\n'
        "]\n"
    )
    if old_slots not in text:
        raise SystemExit("import PROTOCOL_SLOTS introuvable")
    text = text.replace(old_slots, new_slots)
    old_lat = (
        "from src.data.laterality import laterality_from_study_dir, normalise_laterality"
    )
    new_lat = (
        "def laterality_from_study_dir(study_dir):\n"
        "    raise RuntimeError('laterality off sur ce kernel')\n"
        "\n"
        "def normalise_laterality(vol, plane, lat):\n"
        "    raise RuntimeError('laterality off sur ce kernel')\n"
    )
    if old_lat not in text:
        raise SystemExit("import laterality introuvable")
    text = text.replace(old_lat, new_lat)
    if "from src." in text:
        raise SystemExit("import src restant dans le pretraitement")
    if "CROP_MM = 130.0" not in text:
        raise SystemExit("CROP_MM 130 absent")
    return text


def _model() -> str:
    text = (ROOT / "src" / "models" / "kneenet_cnx_r7.py").read_text(encoding="utf-8")
    needle = "import timm\n"
    if needle not in text:
        raise SystemExit("import timm introuvable")
    text = text.replace(
        needle,
        "import timm\nprint('timm', timm.__version__, flush=True)\n",
        1,
    )
    if "from src." in text:
        raise SystemExit("import src restant dans le modele")
    return text


def build(
    *,
    gold: float,
    weight: str,
    dataset: str,
    local_dir: str,
    out_dir: Path,
    kernel_id: str,
    title: str,
) -> Path:
    infer = (
        INFER_CODE.replace("__RUN__", RUN_NAME).replace("__BACKBONE__", BACKBONE)
    )
    nb_path = out_dir / "submit.ipynb"
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
            cell_md(_md(gold, weight, dataset)),
            cell_code(
                CONFIG_CODE.replace("__WEIGHT__", weight).replace("__LOCAL__", local_dir)
            ),
            cell_code(_preprocess()),
            cell_code(_model()),
            cell_code(infer),
        ],
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": kernel_id,
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
        "dataset_sources": [dataset],
        "category_ids": [],
    }
    (nb_path.parent / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    blob = nb_path.read_text(encoding="utf-8")
    required = (
        "sans TTA",
        MODEL_NAME,
        BACKBONE,
        "pretrained=False",
        "img_size=224",
        "CROP_MM = 130.0",
        weight,
        RUN_NAME,
        "swa-7-8-9-10",
        "[7, 8, 9, 10]",
        "load strict OK",
        "rgb_neighbors=False",
        "cnx forward fp32",
        "enabled=False",
    )
    for needle in required:
        if needle not in blob:
            raise SystemExit(f"notebook incomplet, manque {needle}")
    banned = ("TTA_VARIANTS", "apply_affine", "img_size=320")
    for needle in banned:
        if needle in blob:
            raise SystemExit(f"notebook interdit : {needle}")
    if "laterality=True" in infer:
        raise SystemExit("inference avec laterality=True")
    print("ecrit", nb_path, flush=True)
    return nb_path.parent
