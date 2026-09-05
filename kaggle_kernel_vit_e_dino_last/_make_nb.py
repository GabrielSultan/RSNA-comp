from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUT_PUSH = ROOT / "kaggle_kernel_vit_e_dino_last" / "_push" / "submit.ipynb"


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


MD0 = """# RSNA Knee — ViT-e DINOv3 (last.pt oofmask_full) — TTA x5

Notebook Kaggle **self-contained** (internet off) :
- lecture DICOM test → volumes crop 130 mm + bande 20–80 % (24, 224, 224)
- protocole 6 slots (FS/T1, identique v20b)
- KneeNet **ViT-e** : `checkpoints/vit_e_crop130_oofmask_full/epoch9.pt`
- TTA affine (rotation ±5° + scale 0.95/1.05, hflip off) ×5
- écrit `/kaggle/working/submission.csv` (cwd aussi)

Le kernel **doit** écrire `submission.csv`.
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

    WEIGHTS_PATH = _find_weight(('last.pt',))
else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_PATH = Path('../kaggle_dataset_vit_e_dino_last/_upload/last.pt')

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


# --- CODE INLINÉ car Kaggle ne voit pas directement notre repo ---
# On recolle :
# - le pré-traitement DICOM -> tensor (copie de src/infer/preprocess_dicom.py)
# - le modèle ViT-e (copie de src/models/kneenet_vit_e.py)


def main() -> None:
    model_code = (ROOT / "src" / "models" / "kneenet_vit_e.py").read_text(
        encoding="utf-8"
    )

    preprocess_code = (ROOT / "src" / "infer" / "preprocess_dicom.py").read_text(
        encoding="utf-8"
    )

    preprocess_code = preprocess_code.replace(
        "from src.data.dataset_npz import PROTOCOL_SLOTS",
        'PROTOCOL_SLOTS: list[tuple[str, int]] = [\n'
        '    ("Sagittal", 1),\n'
        '    ("Coronal", 1),\n'
        '    ("Axial", 1),\n'
        '    ("Sagittal", 0),\n'
        '    ("Coronal", 0),\n'
        '    ("Axial", 0),\n'
        ']\n',
    )

    INFER_CODE = r"""import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from pathlib import Path

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']

N_TARGETS = len(TARGETS)

# TTA affine : identité + rotation ±5° + scale 0.95/1.05 (hflip off)
TTA_VARIANTS = (
    (0.0, 1.00),
    (5.0, 1.00),
    (-5.0, 1.00),
    (0.0, 0.95),
    (0.0, 1.05),
)


def apply_affine(
    x: torch.Tensor,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> torch.Tensor:
    # x : (B, S, D, H, W) dans [0, 1]. Meme affine sur les plans.
    if angle_deg == 0.0 and scale == 1.0 and tx == 0.0 and ty == 0.0:
        return x
    if x.ndim != 5:
        raise ValueError(f"attendu 5D (B,S,D,H,W), recu {tuple(x.shape)}")
    b, s, d, h, w = x.shape
    c = 1

    xf = x.reshape(b * s * d, c, h, w)
    angle = xf.new_tensor(angle_deg * np.pi / 180.0)
    sc = xf.new_tensor(scale)
    cos, sin = torch.cos(angle) / sc, torch.sin(angle) / sc
    theta = xf.new_zeros(b, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty
    theta = theta.repeat_interleave(s * d, dim=0)
    grid = F.affine_grid(theta, xf.shape, align_corners=False)
    xf = F.grid_sample(xf, grid, align_corners=False, padding_mode="zeros")
    return xf.reshape(b, s, d, h, w)


@torch.no_grad()
def _predict_logits(model, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        return model(x)


def load_model(weights_path: str | Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    n_series = int(cfg.get("n_series", 6))
    encode_chunk = int(cfg.get("encode_chunk", 4))
    head_dim = int(cfg.get("head_dim", 512))
    dropout = float(cfg.get("dropout", 0.2))
    max_depth = int(cfg.get("max_depth", 64) or 64)
    lstm_hidden = int(cfg.get("lstm_hidden", 256))
    backbone_weights = str(cfg.get("backbone_weights", "weights/dinov3-vitb16-lvd1689m"))

    # IMPORTANT : pretrained=False pour éviter tout download DINOv3 backbone
    model = KneeNet(
        n_series=n_series,
        n_targets=N_TARGETS,
        head_dim=head_dim,
        dropout=dropout,
        pretrained=False,
        encode_chunk=encode_chunk,
        max_depth=max_depth,
        backbone_weights=backbone_weights,
        lstm_hidden=lstm_hidden,
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print('load_state_dict missing :', len(missing), '| unexpected :', len(unexpected))
    print('modele', cfg.get('model'), '| epoch', ckpt.get('epoch'),
          '| val macro-AUC :', round(float(ckpt.get('val_macro_auc', float('nan'))), 4),
          '| n_series', n_series)
    model.to(DEVICE).eval()
    return model, cfg
"""

    infer_code_rest = r"""test = pd.read_csv(TEST_CSV)
series = pd.read_csv(TEST_SERIES_CSV)
test_series_dir = Path(TEST_SERIES_DIR)

# prefiltres DataFrame pour accélérer l'extraction des lignes par StudyInstanceUID
series_by_uid = {uid: g for uid, g in series.groupby("StudyInstanceUID", sort=False)}

model, _cfg = load_model(WEIGHTS_PATH)
n_series = int(getattr(model, "n_series", 6))

rows = []
print(f"{len(test)} etudes a predire | TTA x{len(TTA_VARIANTS)}")
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
    x0 = torch.from_numpy(vol).unsqueeze(0).to(device=DEVICE, dtype=torch.float32)

    probs = []
    for angle_deg, scale in TTA_VARIANTS:
        xv = apply_affine(x0, angle_deg=angle_deg, scale=scale)
        logits = _predict_logits(model, xv)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy()[0])

    prob = np.stack(probs, axis=0).mean(axis=0)
    rows.append([uid] + prob.tolist())
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(test)} ok", flush=True)

pred = pd.DataFrame(rows, columns=["StudyInstanceUID"] + TARGETS)

# --- submission.csv au format exact de sample_submission --------------------
sample = pd.read_csv(SAMPLE_SUB)
sub = pred.set_index('StudyInstanceUID').reindex(sample.StudyInstanceUID).reset_index()
sub = sub[list(sample.columns)]
prior = sub[TARGETS].mean()
sub[TARGETS] = sub[TARGETS].fillna(prior)

out_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
out_csv = out_dir / 'submission.csv'
sub.to_csv(out_csv, index=False)
cwd_csv = Path('submission.csv')
if cwd_csv.resolve() != out_csv.resolve():
    sub.to_csv(cwd_csv, index=False)

print('ecrit', out_csv.resolve(), '| lignes', len(sub), '| existe', out_csv.exists())
print(sub.shape, '| colonnes OK :', list(sub.columns) == list(sample.columns))
print(sub[TARGETS].describe().loc[['mean', 'std', 'min', 'max']].round(3).to_string())
assert out_csv.exists() and out_csv.stat().st_size > 0, 'submission.csv manquant'
assert not sub[TARGETS].isna().any().any(), 'NaN dans la submission !'
assert list(sub.columns) == list(sample.columns), 'colonnes != sample_submission'
"""

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
            cell_code(preprocess_code),
            cell_code(model_code),
            cell_code(INFER_CODE + "\n" + infer_code_rest),
        ],
    }

    OUT_PUSH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PUSH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print("ecrit", OUT_PUSH)


if __name__ == "__main__":
    main()

