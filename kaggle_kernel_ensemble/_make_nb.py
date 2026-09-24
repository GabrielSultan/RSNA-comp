from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIX_PATH = ROOT / "kaggle_kernel_ensemble" / "mix.json"
OUTS = [
    ROOT / "notebooks" / "07_ensemble_kaggle.ipynb",
    ROOT / "kaggle_kernel_ensemble" / "ensemble.ipynb",
    ROOT / "kaggle_kernel_ensemble" / "_push" / "ensemble.ipynb",
]


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


PREPROCESS_CODE = r"""# --- Pretraitement DICOM -> volume (24, 224, 224) uint8 --------------------
# (copie self-contained de src/infer/preprocess_dicom.py)
# crop 130 mm + bande 20-80 % (air / bords), identique a la conversion NPZ
import cv2
import numpy as np
import pydicom

N_SLICES, IMG_SIZE = 24, 224
CROP_MM, CROP_REL, BAND = 130.0, 0.70, (0.20, 0.80)


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


_SERIES_CACHE = {}

def load_series_dicom(series_dir):
    key = str(Path(series_dir))
    if key in _SERIES_CACHE:
        return _SERIES_CACHE[key]
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
    _SERIES_CACHE[key] = out
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
    '''Un contraste exact (FS ou T1) sur un plan. Pas de repli.'''
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


LAT_MIN_OFFSET_MM = 20.0


def _lat_side_tag(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    return s[0] if s and s[0] in ('L', 'R') else None


def _lat_center_x(ipp, iop, ps, rows, cols):
    try:
        ipp = np.asarray(list(ipp), dtype=np.float64)
        iop = np.asarray(list(iop), dtype=np.float64)
        ps = np.asarray(list(ps), dtype=np.float64)
        rows, cols = float(rows), float(cols)
    except (TypeError, ValueError):
        return None
    if ipp.size < 3 or iop.size < 6 or ps.size < 2:
        return None
    c = ipp[:3] + iop[:3] * ps[1] * cols / 2.0 + iop[3:6] * ps[0] * rows / 2.0
    x = float(c[0])
    return x if np.isfinite(x) else None


def laterality_from_study_dir(study_dir):
    '''Canon L : tag Laterality, sinon X centre (zone morte 20 mm).'''
    study_dir = Path(study_dir)
    xs, tags = [], []
    if not study_dir.is_dir():
        return None
    for series in sorted(p for p in study_dir.iterdir() if p.is_dir()):
        files = sorted(series.glob('*.dcm'))
        if not files:
            continue
        try:
            ds = pydicom.dcmread(str(files[len(files) // 2]),
                                 stop_before_pixels=True, force=True)
        except Exception:
            continue
        t = _lat_side_tag(getattr(ds, 'Laterality', None))
        if t:
            tags.append(t)
        x = _lat_center_x(
            getattr(ds, 'ImagePositionPatient', None),
            getattr(ds, 'ImageOrientationPatient', None),
            getattr(ds, 'PixelSpacing', None),
            getattr(ds, 'Rows', None), getattr(ds, 'Columns', None))
        if x is not None:
            xs.append(x)
    if tags:
        return tags[0]
    if not xs:
        return None
    m = float(np.median(xs))
    if abs(m) < LAT_MIN_OFFSET_MM:
        return None
    return 'R' if m < 0 else 'L'


def normalise_laterality(vol, plane, lat):
    if lat != 'R':
        return vol
    plane_c = str(plane or '').strip().capitalize()
    if plane_c in ('Coronal', 'Axial'):
        return np.ascontiguousarray(np.flip(vol, axis=-1))
    if plane_c == 'Sagittal':
        return np.ascontiguousarray(np.flip(vol, axis=0))
    return vol


def study_to_tensor(study_dir, series_rows, n_series=3, laterality=False):
    '''(n_series, 24, 224, 224) ; n_series=6 : 3 FS + 3 T1, sans repli.'''
    planes = ['Sagittal', 'Coronal', 'Axial']
    out = np.zeros((n_series, N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    used = set()
    lat = laterality_from_study_dir(study_dir) if laterality else None

    def _fill(i, pick, plane):
        if pick is None:
            return
        sdir = Path(study_dir) / pick.SeriesInstanceUID
        if not sdir.exists():
            return
        try:
            vol = load_series_dicom(sdir).astype(np.float32) / 255.0
            if laterality:
                vol = normalise_laterality(vol, plane, lat)
            out[i] = vol
            used.add(str(pick.SeriesInstanceUID))
        except Exception as e:
            print(f'  [warn] {pick.SeriesInstanceUID}: {e}')

    if n_series == 6:
        for i, (plane, fluid) in enumerate(PROTOCOL_SLOTS):
            _fill(i, _pick_protocol_row(series_rows, plane, fluid), plane)
        return out

    for i, plane in enumerate(planes):
        rows = series_rows[series_rows.Anatomical_Plane == plane]
        if len(rows) == 0:
            continue
        fluid = rows[rows.Fluid_Sensitive == 1] if 'Fluid_Sensitive' in rows else rows.iloc[0:0]
        if len(fluid) == 0 and 'Fat_Suppression' in rows:
            fluid = rows[rows.Fat_Suppression == 1]
        pick = (fluid if len(fluid) else rows).sort_values('SeriesInstanceUID').iloc[0]
        _fill(i, pick, plane)
    if n_series >= 4:
        t1 = None
        for plane in planes:
            rows = series_rows[series_rows.Anatomical_Plane == plane]
            if 'Fluid_Sensitive' in rows:
                rows = rows[rows.Fluid_Sensitive == 0]
            rows = rows[~rows.SeriesInstanceUID.isin(used)]
            if len(rows):
                t1 = rows.sort_values('SeriesInstanceUID').iloc[0]
                break
        pl = None if t1 is None else str(t1.Anatomical_Plane)
        _fill(3, t1, pl)
    return out
"""


NPZ_CODE = r"""# --- Validation du pretraitement : nos volumes DICOM vs stats NPZ ----------
# Au scoring, train_series/ est remplace par test_series/ ; en debug Kaggle,
# train_series/ est disponible : on compare les moments a ceux des NPZ locaux.
import pandas as pd

if DEBUG_COMPARE_NPZ:
    ref_dir = COMP_DIR / 'train_series'
    ref_csv = COMP_DIR / 'train_series.csv'
    if ref_dir.exists() and ref_csv.exists():
        ref = pd.read_csv(ref_csv)
        sample = ref.sample(min(5, len(ref)), random_state=0)
        for _, r in sample.iterrows():
            sdir = ref_dir / r.StudyInstanceUID / r.SeriesInstanceUID
            try:
                v = load_series_dicom(sdir)
                print(f'{r.SeriesInstanceUID[:40]}... shape {v.shape} '
                      f'mean {v.mean():.1f} (NPZ~{NPZ_MEAN_U8}) std {v.std():.1f} (NPZ~{NPZ_STD_U8})')
            except Exception as e:
                print('  [warn]', e)
    else:
        print('train_series/ absent (run de scoring) — comparaison ignoree')
"""


SUBMISSION_CODE = r"""# --- submission.csv au format exact de sample_submission --------------------
# Obligatoire pour le scoring Kaggle : ecrire /kaggle/working/submission.csv
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


def _patch_resnet50(src: str, filename: str) -> str:
    old = "        base = resnet50(weights=None)"
    new = (
        "        try:\n"
        "            base = resnet50(weights=None)\n"
        "        except TypeError:\n"
        "            base = resnet50(pretrained=False)"
    )
    if old not in src:
        raise RuntimeError(f"patch resnet50 introuvable dans {filename}")
    return src.replace(old, new, 1)


def _patch_v20b(src: str) -> str:
    src = src.replace("class KneeNet(nn.Module):", "class KneeNetV20b(nn.Module):")
    return _patch_resnet50(src, "kneenet_v20b.py")


def _patch_v20h(src: str) -> str:
    src = src.replace("class KneeNet(nn.Module):", "class KneeNetV20h(nn.Module):")
    return _patch_resnet50(src, "kneenet_v20h.py")


def _patch_vite(src: str) -> str:
    src = src.replace("class KneeNet(nn.Module):", "class KneeNetVitE(nn.Module):")
    return src


TARGETS_ORDER = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def main() -> None:
    mix = json.loads(MIX_PATH.read_text(encoding="utf-8"))
    v20b_targets = tuple(mix["v20b_targets"])
    vite_targets = tuple(mix["vit_e_targets"])
    v20b_ep = int(mix["v20b_epoch"])
    vite_ep = int(mix["vit_e_epoch"])
    ev = mix.get("specialist_evidence", {}).get("table", {})
    w_vite = mix["weights_vite"]
    missing = [t for t in TARGETS_ORDER if t not in w_vite]
    if missing:
        raise KeyError(f"weights_vite incomplet : {missing}")
    w_tuple = tuple(float(w_vite[t]) for t in TARGETS_ORDER)
    blend_name = str(mix.get("blend_name", "blend"))

    rows = []
    for t in TARGETS_ORDER:
        wv = float(w_vite[t])
        w20 = 1.0 - wv
        winner = "vit_e" if wv >= w20 else "v20b"
        rec = ev.get(t, {})
        extra = ""
        if rec:
            extra = (
                f" ; gold vs v17 Δ {rec['delta']:+.3f}"
            )
        rows.append(
            f"- `{t}` : **{winner}** {w20:.0%} v20b / {wv:.0%} ViT-e{extra}"
        )
    mix_lines = "\n".join(rows)

    md0 = f"""# RSNA Knee — Ensemble lissé v20b + ViT-e (`{blend_name}`)

Notebook **self-contained** pour Kaggle (internet off) :
DICOM test → crop 130 mm + bande 20–80 % → volumes (24, 224, 224) → **blend par cible**
entre **KneeNet v20b distillé** (6 slots FS/T1, ResNet-50 RadImageNet, `{mix['v20b_run']}` epoch {v20b_ep}, gold macro-AUC {mix['v20b_lb']})
et **KneeNet ViT-e** (DINOv3 ViT-B/16, `vit_e_crop130_oofmask_full` epoch {vite_ep}, public LB {mix['vit_e_lb']})
→ TTA affine ×5 → `submission.csv`.

Poids (dataset `sultent/rsna-knee-ensemble-weights`) :
- `v20b_distil_lam03.pt` — `{mix['v20b_run']}/best.pt` (optimizer retiré, distil λ=0.3)
- `vit_e_ep7.pt` — `vit_e_crop130_oofmask_full/epoch7.pt` (optimizer retiré)

Le switch dur (ViT-e 100 % sur MCL / ménisque médial / épanchement) a fait **{mix.get('hard_switch_lb', 0.956)}** public. Ici on **lisse** : le modèle fort sur la cible garde l'avantage, l'autre apporte 10–20 %.

Gagnants (même assignment) :
- **v20b** : {", ".join(v20b_targets)}
- **ViT-e** : {", ".join(vite_targets)}

**Mix `{blend_name}`** (poids = part ViT-e ; v20b = 1 − w) :
{mix_lines}

TTA : identité + rotation ±5° + scale 0.95/1.05. **Pas de hflip**.
Le kernel **doit** écrire `submission.csv` (cwd / `/kaggle/working`).
"""

    v20b_tuple = repr(v20b_targets)
    w_repr = repr(w_tuple)
    config_code = f"""import os
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
        raise FileNotFoundError(f'{{name}} introuvable (dataset ensemble attache ?)')

    WEIGHTS_V20B = _find_weight('v20b_distil_lam03.pt')
    WEIGHTS_VITE = _find_weight('vit_e_ep7.pt')
else:
    COMP_DIR = Path('../data/csv')
    WEIGHTS_V20B = Path('../kaggle_dataset_ensemble/_upload_distil/v20b_distil_lam03.pt')
    WEIGHTS_VITE = Path('../kaggle_dataset_ensemble/_upload/vit_e_ep7.pt')

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
# True seulement si TOUS les bras du blend ont ete trains avec laterality: true.
# Ne pas mixer un ckpt flipped et un ckpt non-flipped sur le meme tenseur.
LATERALITY_CANON_L = False
# Blend par cible : poids ViT-e (v20b = 1 - w). Gagnant 90/10 ou mix adaptatif.
V20B_TARGETS = {v20b_tuple}
BLEND_NAME = {blend_name!r}
MIX_W_VITE = {w_repr}

print('competition dir :', COMP_DIR)
print('poids v20b      :', WEIGHTS_V20B, '| existe :', WEIGHTS_V20B.exists())
print('poids vit_e     :', WEIGHTS_VITE, '| existe :', WEIGHTS_VITE.exists())
print('device          :', DEVICE)
print('TTA variantes   :', TTA_VARIANTS)
print('blend           :', BLEND_NAME)
print('v20b cibles     :', V20B_TARGETS)
print('w_vite          :', MIX_W_VITE)
"""

    load_code = r"""TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
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
          '| gold macro-AUC :', round(float(ckpt.get('gold_macro_auc', float('nan'))), 4),
          '| n_series', model.n_series)
    return model, cfg


def load_vite(weights_path):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
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
    print('modele', cfg.get('model'), '| epoch', ckpt.get('epoch'),
          '| gold macro-AUC :', round(float(ckpt.get('gold_macro_auc', float('nan'))), 4),
          '| n_series', model.n_series)
    return model, cfg


model_v20b, cfg_v20b = load_v20b(WEIGHTS_V20B)
model_vite, cfg_vite = load_vite(WEIGHTS_VITE)
assert int(model_v20b.n_series) == 6, model_v20b.n_series
assert int(model_vite.n_series) == 6, model_vite.n_series
assert int(model_v20b.n_series) == int(model_vite.n_series)
"""

    infer_code = r"""# --- Inference : TTA puis blend lisse par cible (v20b vs ViT-e) -----------
import time
import torch.nn.functional as F

test = pd.read_csv(TEST_CSV)
test_series = pd.read_csv(TEST_SERIES_CSV)
assert len(MIX_W_VITE) == len(TARGETS), (len(MIX_W_VITE), len(TARGETS))
print(f'{len(test)} etudes a predire | TTA x{len(TTA_VARIANTS)} | blend {BLEND_NAME}')
for t, w in zip(TARGETS, MIX_W_VITE):
    print(f'  {t:<18} v20b {1.0 - w:.2f} | vit_e {w:.2f}')


def apply_affine(x, angle_deg=0.0, scale=1.0):
    '''x : (B, S, D, H, W) dans [0, 1]. Meme affine pour tous les plans.'''
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
    global DEVICE, model_v20b, model_vite
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
            model_v20b = model_v20b.to('cpu')
            model_vite = model_vite.to('cpu')
            return model.to('cpu')(x.cpu())


rows = []
t0 = time.time()
w_vite = torch.tensor(MIX_W_VITE, dtype=torch.float32)
n_series = int(model_v20b.n_series)
with torch.no_grad():
    for i, uid in enumerate(test.StudyInstanceUID):
        _SERIES_CACHE.clear()
        srows = test_series[test_series.StudyInstanceUID == uid]
        sdir = TEST_SERIES_DIR / uid
        vol = study_to_tensor(sdir, srows, n_series=n_series,
                              laterality=LATERALITY_CANON_L)
        x = torch.from_numpy(vol).unsqueeze(0).to(DEVICE)
        p20s, pes = [], []
        for angle_deg, scale in TTA_VARIANTS:
            xa = apply_affine(x, angle_deg, scale)
            p20s.append(torch.sigmoid(_forward(model_v20b, xa, use_amp=True).float()))
            pes.append(torch.sigmoid(_forward(model_vite, xa, use_amp=True).float()))
        p20 = torch.stack(p20s, dim=0).mean(dim=0)
        pe = torch.stack(pes, dim=0).mean(dim=0)
        w = w_vite.to(pe.device).view(1, -1)
        prob = ((1.0 - w) * p20 + w * pe).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(test)} | {(time.time()-t0)/60:.1f} min', flush=True)

pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
pred.head()
"""

    v20b_code = _patch_v20b(
        (ROOT / "src" / "models" / "kneenet_v20b.py").read_text(encoding="utf-8")
    )
    vite_code = _patch_vite(
        (ROOT / "src" / "models" / "kneenet_vit_e.py").read_text(encoding="utf-8")
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
            cell_md(md0),
            cell_code(config_code),
            cell_code(PREPROCESS_CODE),
            cell_code(
                "# --- KneeNet v20b (copie self-contained de src/models/kneenet_v20b.py) ---\n"
                + v20b_code
            ),
            cell_code(
                "# --- KneeNet ViT-e (copie self-contained de src/models/kneenet_vit_e.py) ---\n"
                + vite_code
                + "\n\n"
                + load_code
            ),
            cell_code(NPZ_CODE),
            cell_code(infer_code),
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
