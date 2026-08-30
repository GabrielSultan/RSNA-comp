# -*- coding: utf-8 -*-
"""Genere notebooks/12_efficiency_int4_kaggle.ipynb et kaggle_kernel_int4/submit.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "notebooks" / "12_efficiency_int4_kaggle.ipynb"
OUT_SUBMIT = ROOT / "kaggle_kernel_int4" / "submit.ipynb"


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


MD0 = """# RSNA Knee — Efficiency INT4 (v20b, sans TTA)

Notebook **self-contained** pour Kaggle (internet off), piste **efficiency** :
DICOM test → crop 130 mm + bande 20–80 % → volumes (24, 224, 224) → KneeNet **v20b**
(6 slots FS/T1, ResNet-50 dilated CAM 14×14 + BiLSTM) en **INT4 group-wise** →
un seul forward **sans TTA** → `submission.csv`.

Poids : `v20b_ep8_int4.pt` (copie quantifiée de `v20b_ep8.pt`, epoch 8, public LB 0.945 en fp32+TTA).
Backbone INT4 (groupes de 64, symmetric −8..7). LSTM / tête / embeds restent fp16.

Leviers vitesse : pas de TTA (×5), un seul modèle, fp16 GPU, `encode_chunk=64`,
prefetch DICOM, chargement parallèle des 6 slots.

Le kernel **doit** écrire `submission.csv` (cwd / `/kaggle/working`).
"""

CODE1 = r"""import os
from pathlib import Path

# --- Configuration ---------------------------------------------------------
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
        raise FileNotFoundError(f'{names} introuvable (dataset int4 attache ?)')

    WEIGHTS_PATH = _find_weight(('v20b_ep8_int4.pt', 'v20b_ep8.pt', 'best.pt'))
else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_PATH = Path('../kaggle_dataset_int4/_upload/v20b_ep8_int4.pt')

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

ENCODE_CHUNK = 64

print('competition dir :', COMP_DIR)
print('poids           :', WEIGHTS_PATH, '| existe :', WEIGHTS_PATH.exists())
print('device          :', DEVICE)
print('encode_chunk    :', ENCODE_CHUNK)
print('TTA             : off')
"""

CODE2 = r"""# --- Pretraitement DICOM -> volume (24, 224, 224) uint8 --------------------
# Recette identique a la conversion NPZ / aux submits precedents.
import cv2
import numpy as np
import pydicom
from concurrent.futures import ThreadPoolExecutor

cv2.setNumThreads(1)

N_SLICES, IMG_SIZE = 24, 224
CROP_MM, CROP_REL, BAND = 130.0, 0.70, (0.20, 0.80)
N_LOAD_WORKERS = 4


def _slice_sort_key(ds):
    inst = getattr(ds, 'InstanceNumber', None)
    if inst is not None:
        try:
            return float(inst)
        except (TypeError, ValueError):
            pass
    ipp = getattr(ds, 'ImagePositionPatient', None)
    iop = getattr(ds, 'ImageOrientationPatient', None)
    if ipp is not None and iop is not None:
        row = np.array(iop[:3], dtype=np.float64)
        col = np.array(iop[3:], dtype=np.float64)
        normal = np.cross(row, col)
        return float(np.dot(np.array(ipp, dtype=np.float64), normal))
    return float(ipp[2]) if ipp is not None else 0.0


def _spacing_mm(ds):
    ps = getattr(ds, 'PixelSpacing', None) or getattr(ds, 'ImagerPixelSpacing', None)
    if ps is None:
        return None
    try:
        return float(ps[0]), float(ps[1])
    except (TypeError, ValueError, IndexError):
        return None


def _center_crop(img, row_mm, col_mm):
    h, w = img.shape[:2]
    if row_mm is not None and col_mm is not None and row_mm > 0 and col_mm > 0:
        ch = min(h, max(8, int(round(CROP_MM / row_mm))))
        cw = min(w, max(8, int(round(CROP_MM / col_mm))))
    else:
        ch = min(h, max(8, int(round(h * CROP_REL))))
        cw = min(w, max(8, int(round(w * CROP_REL))))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def load_series_dicom(series_dir):
    files = sorted(Path(series_dir).glob('*.dcm'))
    datasets = []
    for f in files:
        try:
            datasets.append(pydicom.dcmread(str(f)))
        except Exception:
            continue
    if not datasets:
        raise RuntimeError(f'aucun DICOM lisible dans {series_dir}')
    datasets.sort(key=_slice_sort_key)

    frames, sp = [], None
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, 'RescaleSlope', 1.0) or 1.0) \
                  + float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
        if getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
            arr = arr.max() - arr
        if sp is None:
            sp = _spacing_mm(ds)
        if arr.ndim == 2:
            frames.append(arr)
        elif arr.ndim == 3:
            frames.extend([arr[i] for i in range(arr.shape[0])])
        else:
            raise RuntimeError(f'ndim inattendu {arr.ndim}')
    if not frames:
        raise RuntimeError('aucune coupe')

    row_mm, col_mm = (sp if sp is not None else (None, None))
    cropped = [_center_crop(im, row_mm, col_mm) for im in frames]
    h0, w0 = cropped[0].shape[:2]
    stacked = []
    for c in cropped:
        if c.shape[:2] != (h0, w0):
            c = cv2.resize(c, (w0, h0), interpolation=cv2.INTER_AREA)
        stacked.append(c)
    vol = np.stack(stacked, axis=0)
    lo, hi = np.percentile(vol, 1.0), np.percentile(vol, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)

    n = vol.shape[0]
    a = int(np.floor(BAND[0] * (n - 1)))
    b = int(np.ceil(BAND[1] * (n - 1)))
    if b - a + 1 < 8:
        a, b = 0, n - 1
    band = vol[a:b + 1]
    idx = np.linspace(0, band.shape[0] - 1, N_SLICES).round().astype(int)
    band = band[idx]
    out = np.empty((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i in range(N_SLICES):
        sl = cv2.resize(band[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        out[i] = np.clip(sl * 255.0, 0, 255).round().astype(np.uint8)
    return out


PROTOCOL_SLOTS = [
    ('Sagittal', 1),
    ('Coronal', 1),
    ('Axial', 1),
    ('Sagittal', 0),
    ('Coronal', 0),
    ('Axial', 0),
]


def _pick_protocol_row(series_rows, plane, fluid):
    # Un contraste exact (FS ou T1) sur un plan. Pas de repli.
    rows = series_rows[series_rows.Anatomical_Plane == plane]
    if len(rows) == 0:
        return None
    if 'Fluid_Sensitive' in rows.columns:
        rows = rows[rows.Fluid_Sensitive == int(fluid)]
    elif fluid == 1 and 'Fat_Suppression' in rows.columns:
        rows = rows[rows.Fat_Suppression == 1]
    elif fluid == 0 and 'Fat_Suppression' in rows.columns:
        rows = rows[rows.Fat_Suppression == 0]
    if len(rows) == 0:
        return None
    return rows.sort_values('SeriesInstanceUID').iloc[0]


def _load_slot(study_dir, pick):
    if pick is None:
        return None
    sdir = Path(study_dir) / pick.SeriesInstanceUID
    if not sdir.exists():
        return None
    try:
        return load_series_dicom(sdir).astype(np.float32) / 255.0
    except Exception as e:
        print(f'  [warn] {pick.SeriesInstanceUID}: {e}')
        return None


def study_to_tensor(study_dir, series_rows, n_series=6):
    # (n_series, 24, 224, 224) ; 6 slots protocole FS/T1, sans repli.
    out = np.zeros((n_series, N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    picks = [_pick_protocol_row(series_rows, plane, fluid)
             for plane, fluid in PROTOCOL_SLOTS[:n_series]]
    jobs = [(i, picks[i]) for i in range(n_series) if picks[i] is not None]
    if not jobs:
        return out

    def _one(job):
        i, pick = job
        return i, _load_slot(study_dir, pick)

    workers = min(N_LOAD_WORKERS, len(jobs))
    if workers <= 1:
        results = [_one(j) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_one, jobs))
    for i, vol in results:
        if vol is not None:
            out[i] = vol
    return out
"""

CODE3 = r"""# --- Modele v20b + chargeur INT4 (eval only, pas de checkpoint) ------------
import torch
import torch.nn as nn
from torchvision.models import resnet50

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']
N_TARGETS = 12
MAP_HW = 14
LSTM_HIDDEN = 256
N_PROTOCOL_SLOTS = 6
GROUP_SIZE = 64


def _dilate_layer4(layer4, dilate=2):
    for i, block in enumerate(layer4):
        if i == 0:
            block.conv2.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        block.conv2.dilation = (dilate, dilate)
        block.conv2.padding = (dilate, dilate)


class _ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            base = resnet50(weights=None)
        except TypeError:
            base = resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        _dilate_layer4(self.backbone[7])
        self.num_features = 2048

    def forward(self, x):
        return self.backbone(x)


class KneeNet(nn.Module):
    arch_id = 'kneenet_v20b'

    def __init__(self, backbone='resnet50', n_series=N_PROTOCOL_SLOTS,
                 n_targets=N_TARGETS, head_dim=512, dropout=0.2,
                 pretrained=False, encode_chunk=16, max_depth=64,
                 lstm_hidden=LSTM_HIDDEN, **_kw):
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.map_hw = MAP_HW
        self.lstm_hidden = int(lstm_hidden)
        self.encoder = _ResNet50Encoder()
        feat_dim = self.encoder.num_features
        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        self.attn_score = nn.Linear(feat_dim, n_targets)
        self.spatial_score = nn.Conv2d(feat_dim, n_targets, kernel_size=1)
        self.slice_lstm = nn.LSTM(
            input_size=feat_dim, hidden_size=self.lstm_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.lstm_proj = nn.Linear(2 * self.lstm_hidden, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.stem_mix = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self.register_buffer('img_mean', torch.zeros(1, 3, 1, 1), persistent=True)
        self.register_buffer('img_std', torch.ones(1, 3, 1, 1), persistent=True)

    def train(self, mode=True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def _encode_chunks(self, rgb):
        n = rgb.size(0)
        c = self.encoder.num_features
        if n == 0:
            return rgb.new_zeros(0, c, MAP_HW, MAP_HW)
        chunk = max(1, self.encode_chunk)
        if n <= chunk:
            return self.encoder(rgb)
        return torch.cat([self.encoder(rgb[i:i + chunk]) for i in range(0, n, chunk)], dim=0)

    def _spatial_pool(self, maps):
        n, c, h, w = maps.shape
        gap = maps.mean(dim=(-2, -1))
        logits = self.spatial_score(maps)
        attn = torch.softmax(logits.float().reshape(n, self.n_targets, h * w), dim=-1).to(maps.dtype)
        local = torch.einsum('nth,nch->ntc', attn, maps.reshape(n, c, h * w))
        return gap, local

    def _stem_rgb(self, x, idx):
        _b, s, d, _h, _w = x.shape
        bs = idx // (s * d)
        rest = idx % (s * d)
        ss = rest // d
        dd = rest % d
        d_m = (dd - 1).clamp(0, d - 1)
        d_p = (dd + 1).clamp(0, d - 1)
        center = x[bs, ss, dd]
        rgb = torch.stack(
            [0.5 + 0.5 * (center - x[bs, ss, d_m]), center, 0.5 + 0.5 * (x[bs, ss, d_p] - center)],
            dim=1,
        )
        rgb = self.stem_mix(rgb)
        return (rgb - self.img_mean) / self.img_std

    def _temporal_delta(self, gaps, b, s, d):
        c = gaps.size(-1)
        seq = gaps.reshape(b * s, d, c)
        with torch.autocast('cuda', enabled=False):
            h, _ = self.slice_lstm(seq.float())
            delta = self.lstm_proj(h.float())
        return delta.to(dtype=gaps.dtype).reshape(b, s, d, c)

    def _slice_pool(self, valid, gaps, local):
        b, s, d = valid.shape
        delta = self._temporal_delta(gaps, b, s, d)
        pe = self.plane_embed[:s].view(1, s, 1, -1)
        se = self.slice_embed[:d].view(1, 1, d, -1)
        feats = gaps.reshape(b, s, d, -1) + delta + pe + se
        scores = self.attn_score(feats)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        scores = scores.reshape(b, s * d, self.n_targets)
        attn = torch.softmax(scores.float(), dim=1).to(feats.dtype)
        tokens = local.reshape(b, s, d, self.n_targets, -1)
        tokens = tokens + delta.unsqueeze(3) + pe.unsqueeze(3) + se.unsqueeze(3)
        tokens = tokens.reshape(b, s * d, self.n_targets, -1)
        pooled = torch.einsum('bnt,bntc->btc', attn, tokens)
        logits = self.head(self.norm(pooled)).squeeze(-1)
        return logits

    def forward(self, x):
        b, s, d, _h, _w = x.shape
        valid = x.amax(dim=(-2, -1)) > 1e-6
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        gaps = x.new_zeros(n_tok, feat_dim)
        local = x.new_zeros(n_tok, self.n_targets, feat_dim)
        if idx.numel() > 0:
            maps = self._encode_chunks(self._stem_rgb(x, idx))
            if gaps.dtype != maps.dtype:
                gaps = gaps.to(dtype=maps.dtype)
                local = local.to(dtype=maps.dtype)
            gap, loc = self._spatial_pool(maps)
            gaps[idx] = gap
            local[idx] = loc
        return self._slice_pool(valid, gaps, local)


def _unpack_int4(packed, numel):
    packed = packed.reshape(-1).to(torch.uint8)
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    q = torch.empty(packed.numel() * 2, dtype=torch.int8)
    q[0::2] = lo.to(torch.int8)
    q[1::2] = hi.to(torch.int8)
    return q[:numel]


def _dequant_entry(entry, dtype=torch.float32):
    shape = tuple(entry['shape'])
    out_ch = int(shape[0])
    rest = int(torch.tensor(shape[1:]).prod().item())
    q = _unpack_int4(entry['packed'], int(entry['numel']))
    scale = entry['scale'].float()
    if scale.ndim == 1:
        rows = q.to(torch.float32).reshape(out_ch, -1)[:, :rest] * scale.reshape(-1, 1)
        return rows.reshape(shape).to(dtype)
    g = int(entry.get('group_size', GROUP_SIZE))
    rest = int(entry.get('rest', rest))
    n_g = int(scale.shape[1])
    grouped = q.to(torch.float32).reshape(out_ch, n_g, g) * scale.unsqueeze(-1)
    rows = grouped.reshape(out_ch, n_g * g)[:, :rest]
    return rows.reshape(shape).to(dtype)


def dequantize_state(blob, dtype=torch.float32):
    out = {}
    for k, v in blob.items():
        if isinstance(v, dict) and 'packed' in v:
            out[k] = _dequant_entry(v, dtype=dtype)
        elif torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


ckpt = torch.load(str(WEIGHTS_PATH), map_location='cpu', weights_only=False)
cfg = ckpt.get('config', {})
if ckpt.get('format') == 'kneenet_int4_v1':
    sd = dequantize_state(ckpt['model_int4'], dtype=torch.float32)
    print('checkpoint INT4', ckpt.get('quant'))
else:
    sd = ckpt['model']
    print('checkpoint fp32 (pas int4)')

model = KneeNet(
    backbone=cfg.get('backbone', 'resnet50'),
    n_series=int(cfg.get('n_series', 6)),
    head_dim=int(cfg.get('head_dim', 512)),
    dropout=float(cfg.get('dropout', 0.4)),
    encode_chunk=ENCODE_CHUNK,
    lstm_hidden=int(cfg.get('lstm_hidden', 256)),
    pretrained=False,
)
model.load_state_dict(sd, strict=True)
model.to(DEVICE).eval()
if DEVICE == 'cuda':
    model.half()
    model.slice_lstm.float()
    model.lstm_proj.float()
print('modele', cfg.get('model'), '| epoch', ckpt.get('epoch'),
      '| n_series', model.n_series, '| dtype', next(model.encoder.parameters()).dtype)
"""

CODE4 = r"""# --- Inference : un forward / etude, prefetch DICOM, pas de TTA ------------
import time
import pandas as pd

test = pd.read_csv(TEST_CSV)
test_series = pd.read_csv(TEST_SERIES_CSV)
empty_rows = test_series.iloc[0:0]
series_map = {uid: g for uid, g in test_series.groupby('StudyInstanceUID', sort=False)}
uids = list(test.StudyInstanceUID)
print(f'{len(uids)} etudes a predire | TTA off | prefetch DICOM')

in_dtype = torch.float16 if DEVICE == 'cuda' else torch.float32


def _prep(uid):
    return study_to_tensor(
        TEST_SERIES_DIR / uid,
        series_map.get(uid, empty_rows),
        n_series=model.n_series,
    )


def _forward(x):
    global DEVICE, model, in_dtype
    try:
        return model(x)
    except RuntimeError as e:
        print('forward CUDA echoue, cudnn off puis CPU :', str(e)[:80])
        torch.backends.cudnn.enabled = False
        try:
            return model(x)
        except RuntimeError:
            DEVICE = 'cpu'
            in_dtype = torch.float32
            model = model.float().to('cpu')
            return model(x.float().cpu())


if DEVICE == 'cuda':
    with torch.inference_mode():
        _ = model(torch.zeros(1, model.n_series, N_SLICES, IMG_SIZE, IMG_SIZE,
                              device=DEVICE, dtype=in_dtype))

rows = []
t0 = time.time()
pre = ThreadPoolExecutor(max_workers=1)
fut = pre.submit(_prep, uids[0]) if uids else None
with torch.inference_mode():
    for i, uid in enumerate(uids):
        vol = fut.result()
        if i + 1 < len(uids):
            fut = pre.submit(_prep, uids[i + 1])
        x = torch.from_numpy(vol).unsqueeze(0).to(device=DEVICE, dtype=in_dtype, non_blocking=True)
        logits = _forward(x)
        prob = torch.sigmoid(logits.float()).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(uids)} | {(time.time()-t0)/60:.1f} min', flush=True)
pre.shutdown(wait=True)

pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
print(f'{len(pred)} predictions | {(time.time()-t0)/60:.2f} min')
pred.head()
"""

CODE5 = r"""# --- submission.csv au format exact de sample_submission --------------------
from pathlib import Path as _Out

sample = pd.read_csv(SAMPLE_SUB)
sub = pred.set_index('StudyInstanceUID').reindex(sample.StudyInstanceUID).reset_index()
sub = sub[list(sample.columns)]
prior = sub[TARGETS].mean()
sub[TARGETS] = sub[TARGETS].fillna(prior)

out_dir = _Out('/kaggle/working') if _Out('/kaggle/working').exists() else _Out('.')
out_csv = out_dir / 'submission.csv'
sub.to_csv(out_csv, index=False)
cwd_csv = _Out('submission.csv')
if cwd_csv.resolve() != out_csv.resolve():
    sub.to_csv(cwd_csv, index=False)

print('ecrit', out_csv.resolve(), '| lignes', len(sub), '| existe', out_csv.exists())
print(sub.shape, '| colonnes OK :', list(sub.columns) == list(sample.columns))
print(sub[TARGETS].describe().loc[['mean', 'std', 'min', 'max']].round(3).to_string())
assert out_csv.exists() and out_csv.stat().st_size > 0, 'submission.csv manquant'
assert not sub[TARGETS].isna().any().any(), 'NaN dans la submission !'
assert list(sub.columns) == list(sample.columns), 'colonnes != sample_submission'
sub.head()
"""


def main() -> None:
    nb = {
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
            cell_code(CODE1),
            cell_code(CODE2),
            cell_code(CODE3),
            cell_code(CODE4),
            cell_code(CODE5),
        ],
    }
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    OUT_SUBMIT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("ecrit", OUT_NB)
    print("ecrit", OUT_SUBMIT)


if __name__ == "__main__":
    main()
