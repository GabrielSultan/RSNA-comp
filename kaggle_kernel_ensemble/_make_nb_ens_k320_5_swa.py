# -*- coding: utf-8 -*-
"""Ensemble sans TTA, memes poids de mix, snapshots SWA a la place des epochs."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SLUG = "rsna-knee-ens-swa-k320-hd-bd-lat-vk"
PUSH = KERNEL / "_push_ens_k320_5_swa"
DATASETS = [
    "sultent/rsna-knee-vit-k320-v3b-swa",
    "sultent/rsna-knee-v20h-dino-full-swa",
    "sultent/rsna-knee-v20b-dino-full-swa",
    "sultent/rsna-knee-v20h-v3b-lat-swa",
    "sultent/rsna-knee-vit-k-det-swa",
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


def _preprocess(src: str) -> str:
    old_head = (
        "def load_series_dicom(series_dir):\n"
        "    key = str(Path(series_dir))\n"
        "    if key in _SERIES_CACHE:\n"
        "        return _SERIES_CACHE[key]\n"
    )
    new_head = (
        "def load_series_dicom(series_dir, img_size=224):\n"
        "    key = str(Path(series_dir))\n"
        "    if key not in _SERIES_CACHE:\n"
        "        _SERIES_CACHE[key] = _series_band(series_dir)\n"
        "    band = _SERIES_CACHE[key]\n"
        "    size = int(img_size)\n"
        "    out = np.empty((band.shape[0], size, size), dtype=np.uint8)\n"
        "    for i in range(band.shape[0]):\n"
        "        sl = band[i]\n"
        "        if sl.shape != (size, size):\n"
        "            sl = cv2.resize(sl, (size, size), interpolation=cv2.INTER_AREA)\n"
        "        out[i] = np.clip(np.asarray(sl, dtype=np.float32) * 255.0, 0, 255).round().astype(np.uint8)\n"
        "    return out\n"
        "\n"
        "\n"
        "def _series_band(series_dir):\n"
    )
    old_tail = (
        "    out = np.empty((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)\n"
        "    for i in range(N_SLICES):\n"
        "        sl = cv2.resize(band[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)\n"
        "        out[i] = np.clip(sl * 255.0, 0, 255).round().astype(np.uint8)\n"
        "    _SERIES_CACHE[key] = out\n"
        "    return out\n"
    )
    old_sig = (
        "def study_to_tensor(study_dir, series_rows, n_series=3, laterality=False):\n"
        "    '''(n_series, 24, 224, 224) ; n_series=6 : 3 FS + 3 T1, sans repli.'''\n"
        "    planes = ['Sagittal', 'Coronal', 'Axial']\n"
        "    out = np.zeros((n_series, N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)\n"
    )
    new_sig = (
        "def study_to_tensor(study_dir, series_rows, n_series=3, laterality=False, img_size=224):\n"
        "    '''(n_series, 24, img_size, img_size) ; n_series=6 : 3 FS + 3 T1, sans repli.'''\n"
        "    planes = ['Sagittal', 'Coronal', 'Axial']\n"
        "    size = int(img_size)\n"
        "    out = np.zeros((n_series, N_SLICES, size, size), dtype=np.float32)\n"
    )
    old_load = "vol = load_series_dicom(sdir).astype(np.float32) / 255.0"
    new_load = "vol = load_series_dicom(sdir, img_size=size).astype(np.float32) / 255.0"
    for label, old in (
        ("head", old_head), ("tail", old_tail), ("sig", old_sig), ("load", old_load),
    ):
        if old not in src:
            raise SystemExit(f"preprocess: marqueur {label} absent")
    src = src.replace(old_head, new_head, 1).replace(old_tail, "    return band\n", 1)
    src = src.replace(old_sig, new_sig, 1).replace(old_load, new_load, 1)
    return src


def _md(g_vit: float, g_v20b: float, g_v20h: float, g_lat: float, g_vk: float, rows) -> str:
    lines = []
    for t, (wv, wb, wh, wl, wk) in rows:
        lines.append(
            f"- `{t}` : ViT-k320 {wv:.0%} / v20b DINO {wb:.0%} / "
            f"v20h DINO {wh:.0%} / v20h lat {wl:.0%} / ViT-k {wk:.0%}"
        )
    mix = "\n".join(lines)
    return f"""# RSNA Knee — Ensemble SWA ViT-k320 ep7-10 + v20h DINO ep6-7 + v20b DINO ep6-9 + v20h lat ep7-12 + ViT-k ep7-10

Notebook **self-contained** (internet off, **T4**). **Sans TTA.**

- **ViT-k320** : `vit_k_llrd08_320_v3b_oofmask_v20h_full` SWA epochs 7-10 (gold in-sample moyen {g_vit:.3f}). Entrée **320**. Pas de flip.
- **v20h DINO** : `v20h_crop130_dino_ep11_oof_v3b_full` SWA epochs 6-7 (gold in-sample moyen {g_v20h:.3f}). Entrée 224. Pas de flip.
- **v20b DINO** : `v20b_crop130_dino_ep11_oof_v3b_full` SWA epochs 6-9 (gold in-sample moyen {g_v20b:.3f}). Entrée 224. Pas de flip.
- **v20h lat** : `v20h_crop130_oofmask_v20b_lat_full` SWA epochs 7-12 (gold in-sample moyen {g_lat:.3f}). Entrée 224. **Canon L** (flip des genoux droits).
- **ViT-k** : `vit_k_llrd08_det_oofmask_full` SWA epochs 7-10 (gold in-sample moyen {g_vk:.3f}). Entrée 224. Pas de flip.

**ViT-k320 est la baseline** : au moins **40 %** sur chaque finding. La masse libre (20 %) suit le softmax des AUC val du best subset (température 0.05). Les quatre autres gardent au moins **10 %**.
Subsets : `subset_vit_k_llrd08_det_320_frac025`, `subset_v20b_crop130_det_frac025`, `subset_v20h_crop130_dino_ep11_frac025`, `subset_v20h_crop130_det_frac025`, `subset_vit_k_llrd08_det_frac025`.

**Mix** (ViT-k320 / v20b DINO / v20h DINO / v20h lat / ViT-k) :
{mix}

Seul v20h lat voit le volume flippé. Blend après sigmoid.
"""


def _config(rows) -> str:
    mix = ",\n    ".join(
        f"({wv:.2f}, {wb:.2f}, {wh:.2f}, {wl:.2f}, {wk:.2f}),  # {t}"
        for t, (wv, wb, wh, wl, wk) in rows
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

    WEIGHTS_VIT = _find_weight('vit_k320_v3b_swa_ep7_10.pt')
    WEIGHTS_V20H = _find_weight('v20h_dino_full_swa_ep6_7.pt')
    WEIGHTS_V20B = _find_weight('v20b_dino_full_swa_ep6_9.pt')
    WEIGHTS_LAT = _find_weight('v20h_v3b_lat_swa_ep7_12.pt')
    WEIGHTS_VITK = _find_weight('vit_k_det_swa_ep7_10.pt')
else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_VIT = Path('kaggle_dataset_vit_k/_upload_k320_swa/vit_k320_v3b_swa_ep7_10.pt')
    WEIGHTS_V20H = Path('kaggle_dataset_v20h_lat/_upload_dino_full_swa/v20h_dino_full_swa_ep6_7.pt')
    WEIGHTS_V20B = Path('kaggle_dataset_v20b/_upload_dino_full_swa/v20b_dino_full_swa_ep6_9.pt')
    WEIGHTS_LAT = Path('kaggle_dataset_v20h_lat/_upload_v3b_lat_swa/v20h_v3b_lat_swa_ep7_12.pt')
    WEIGHTS_VITK = Path('kaggle_dataset_vit_e_dino_last/_upload_det_swa/vit_k_det_swa_ep7_10.pt')

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

# (ViT-k320, v20b DINO, v20h DINO, v20h lat, ViT-k 224).
# ViT-k320 baseline >= 0.40, les autres >= 0.10, somme 1.
MIX = [
    {mix}
]

print('competition dir :', COMP_DIR)
print('poids vit       :', WEIGHTS_VIT, '| existe :', WEIGHTS_VIT.exists())
print('poids v20h dino :', WEIGHTS_V20H, '| existe :', WEIGHTS_V20H.exists())
print('poids v20b dino :', WEIGHTS_V20B, '| existe :', WEIGHTS_V20B.exists())
print('poids v20h lat  :', WEIGHTS_LAT, '| existe :', WEIGHTS_LAT.exists())
print('poids vit-k     :', WEIGHTS_VITK, '| existe :', WEIGHTS_VITK.exists())
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


def load_vit(weights_path):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    if str(cfg.get('run_name')) != 'vit_k_llrd08_320_v3b_oofmask_v20h_full':
        raise RuntimeError(cfg.get('run_name'))
    if str(cfg.get('model')) != 'kneenet_vit_k320':
        raise RuntimeError(cfg.get('model'))
    if int(cfg.get('img_size', 0)) != 320:
        raise RuntimeError(cfg.get('img_size'))
    if cfg.get('laterality'):
        raise RuntimeError('vit-k320 ne doit pas etre flipped')
    if str(ckpt.get('epoch')) != 'swa-7-8-9-10':
        raise RuntimeError(ckpt.get('epoch'))
    if [int(e) for e in (ckpt.get('swa_epochs') or [])] != [7, 8, 9, 10]:
        raise RuntimeError(ckpt.get('swa_epochs'))
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
    if str(ckpt.get('epoch')) != swa_label:
        raise RuntimeError(ckpt.get('epoch'))
    if [int(e) for e in (ckpt.get('swa_epochs') or [])] != list(swa_epochs):
        raise RuntimeError(ckpt.get('swa_epochs'))
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
    'v20h_crop130_dino_ep11_oof_v3b_full', 'swa-6-7', [6, 7],
)
model_v20b = load_resnet(
    WEIGHTS_V20B, KneeNetV20b, 'kneenet_v20b',
    'v20b_crop130_dino_ep11_oof_v3b_full', 'swa-6-7-8-9', [6, 7, 8, 9],
)


def load_lat(weights_path):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    if str(cfg.get('run_name')) != 'v20h_crop130_oofmask_v20b_lat_full':
        raise RuntimeError(cfg.get('run_name'))
    if str(cfg.get('model')) != 'kneenet_v20h':
        raise RuntimeError(cfg.get('model'))
    if not cfg.get('laterality'):
        raise RuntimeError('v20h lat doit etre flipped')
    if str(ckpt.get('epoch')) != 'swa-7-8-9-10-11-12':
        raise RuntimeError(ckpt.get('epoch'))
    if [int(e) for e in (ckpt.get('swa_epochs') or [])] != [7, 8, 9, 10, 11, 12]:
        raise RuntimeError(ckpt.get('swa_epochs'))
    model = KneeNetV20h(
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
    print('v20h lat', cfg.get('run_name'), 'epoch', ckpt.get('epoch'),
          'lat', cfg.get('laterality'),
          'gold', round(float(ckpt.get('gold_macro_auc')), 4))
    return model


model_lat = load_lat(WEIGHTS_LAT)


def load_vitk(weights_path):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    if str(cfg.get('run_name')) != 'vit_k_llrd08_det_oofmask_full':
        raise RuntimeError(cfg.get('run_name'))
    if str(cfg.get('model')) != 'kneenet_vit_e':
        raise RuntimeError(cfg.get('model'))
    if cfg.get('laterality'):
        raise RuntimeError('vit-k ne doit pas etre flipped')
    if str(ckpt.get('epoch')) != 'swa-7-8-9-10':
        raise RuntimeError(ckpt.get('epoch'))
    if [int(e) for e in (ckpt.get('swa_epochs') or [])] != [7, 8, 9, 10]:
        raise RuntimeError(ckpt.get('swa_epochs'))
    if int(cfg.get('img_size', 224)) != 224:
        raise RuntimeError(cfg.get('img_size'))
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
    print('vit-k', cfg.get('run_name'), 'epoch', ckpt.get('epoch'),
          'lat', cfg.get('laterality'),
          'gold', round(float(ckpt.get('gold_macro_auc')), 4))
    return model


model_vitk = load_vitk(WEIGHTS_VITK)

test = pd.read_csv(TEST_CSV)
test_series = pd.read_csv(TEST_SERIES_CSV)
print(len(test), 'etudes | sans TTA | v20h lat canon L')
for t, w in zip(TARGETS, MIX):
    if abs(sum(w) - 1.0) > 1e-6 or min(w) < 0.10 - 1e-6 or w[0] < 0.40 - 1e-6 or len(w) != 5:
        raise RuntimeError((t, w))
    print(
        f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f}'
        f' lat {w[3]:.2f} vk {w[4]:.2f}'
    )


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
        vol_lat = study_to_tensor(sdir, srows, n_series=6, laterality=True, img_size=224)
        vol_vit = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=320)
        if vol_vit.shape[-1] != 320 or vol_224.shape[-1] != 224 or vol_lat.shape[-1] != 224:
            raise RuntimeError((vol_vit.shape, vol_224.shape, vol_lat.shape))
        x_224 = torch.from_numpy(vol_224).unsqueeze(0).to(DEVICE)
        x_lat = torch.from_numpy(vol_lat).unsqueeze(0).to(DEVICE)
        x_vit = torch.from_numpy(vol_vit).unsqueeze(0).to(DEVICE)
        p_vit = torch.sigmoid(_forward(model_vit, x_vit).float())
        p_v20b = torch.sigmoid(_forward(model_v20b, x_224).float())
        p_v20h = torch.sigmoid(_forward(model_v20h, x_224).float())
        p_lat = torch.sigmoid(_forward(model_lat, x_lat).float())
        p_vitk = torch.sigmoid(_forward(model_vitk, x_224).float())
        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)
        prob = (
            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h
            + w[:, 3] * p_lat + w[:, 4] * p_vitk
        ).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(test)} | {(time.time()-t0)/60:.1f} min', flush=True)

pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
pred.head()
"""


def main() -> None:
    ens = _load("ens_nb", KERNEL / "_make_nb.py")
    mix = _load("mix_k320_5", KERNEL / "_mix_k320_5.py")
    rows = mix.mix_rows()
    expected = {
        "ACL": (0.41, 0.14, 0.20, 0.13, 0.12),
        "MCL": (0.41, 0.16, 0.16, 0.16, 0.11),
        "Medial Meniscus": (0.42, 0.15, 0.16, 0.15, 0.12),
        "Lateral Meniscus": (0.43, 0.12, 0.18, 0.13, 0.14),
        "Medial OA": (0.44, 0.14, 0.15, 0.14, 0.13),
        "Lateral OA": (0.44, 0.14, 0.15, 0.14, 0.13),
        "PF OA": (0.43, 0.14, 0.17, 0.15, 0.11),
        "Effusion": (0.45, 0.13, 0.15, 0.14, 0.13),
        "Synovitis": (0.46, 0.13, 0.14, 0.13, 0.14),
        "Baker's": (0.43, 0.14, 0.15, 0.14, 0.14),
        "Contusion": (0.43, 0.14, 0.15, 0.14, 0.14),
        "Fracture": (0.43, 0.15, 0.15, 0.15, 0.12),
    }
    for t, w in rows:
        got = tuple(round(float(x), 2) for x in w)
        if got != expected[t]:
            raise SystemExit(f"mix change {t} {got} != {expected[t]}")
    g_vit = _gold_mean("vit_k_llrd08_320_v3b_oofmask_v20h_full", (7, 8, 9, 10))
    g_v20h = _gold_mean("v20h_crop130_dino_ep11_oof_v3b_full", (6, 7))
    g_v20b = _gold_mean("v20b_crop130_dino_ep11_oof_v3b_full", (6, 7, 8, 9))
    g_lat = _gold_mean("v20h_crop130_oofmask_v20b_lat_full", (7, 8, 9, 10, 11, 12))
    g_vk = _gold_mean("vit_k_llrd08_det_oofmask_full", (7, 8, 9, 10))
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
        ens.cell_md(_md(g_vit, g_v20b, g_v20h, g_lat, g_vk, rows)),
        ens.cell_code(_config(rows)),
        ens.cell_code(_preprocess(ens.PREPROCESS_CODE)),
        ens.cell_code("# --- KneeNet v20b DINO ---\n" + v20b_code),
        ens.cell_code("# --- KneeNet v20h DINO ---\n" + v20h_code),
        ens.cell_code("# --- KneeNet ViT-e, poids ViT-k320 ---\n" + vite_code),
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
    blob = json.dumps(nb)
    infer = "".join(cells[6]["source"])
    if "TTA_VARIANTS" in infer:
        raise SystemExit("TTA encore present")
    if "sans TTA" not in infer or "img_size=320" not in infer:
        raise SystemExit("inference incomplete")
    if infer.count("laterality=True") != 1 or infer.count("laterality=False") != 2:
        raise SystemExit("flip mal place")
    cfg_src = "".join(cells[1]["source"])
    if "v20h_crop130_oofmask_v20b_lat_full" not in infer or "v20h_v3b_lat_swa_ep7_12.pt" not in cfg_src:
        raise SystemExit("lat absent")
    if "vit_k_llrd08_det_oofmask_full" not in infer or "vit_k_det_swa_ep7_10.pt" not in cfg_src:
        raise SystemExit("vit-k absent")
    if "kneenet_v20b" not in infer or "v20h_dino_full_swa_ep6_7" not in "".join(cells[1]["source"]):
        raise SystemExit("mauvais poids")
    if "swa-7-8-9-10" not in infer or "swa-6-7" not in infer or "swa-7-8-9-10-11-12" not in infer:
        raise SystemExit("labels SWA absents")
    if "epoch 9" in infer or "!= 7" in infer or "!= 6" in infer or "!= 8" in infer:
        raise SystemExit("garde d'epoch simple encore presente")
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
        "gold", round(g_vit, 3), round(g_v20h, 3), round(g_v20b, 3),
        round(g_lat, 3), round(g_vk, 3),
    )
    for t, w in rows:
        print(f"  {t:<18} {w[0]:.2f} {w[1]:.2f} {w[2]:.2f} {w[3]:.2f} {w[4]:.2f}")


if __name__ == "__main__":
    main()
