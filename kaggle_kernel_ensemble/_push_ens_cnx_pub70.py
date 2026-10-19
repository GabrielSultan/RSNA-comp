# -*- coding: utf-8 -*-
"""4 SWA publies + CNX-r7 SWA 7-10, rang 70/30 contre degnonguidi. Sans TTA."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
OURS = KERNEL / "_push_ens_c145_soft" / "submit.ipynb"
PUBLIC = KERNEL / "_public_degnonguidi" / "rnsa-best-public-score.ipynb"
SLUG = "rsna-knee-ens-cnx-pub70"
PUSH = KERNEL / "_push_ens_cnx_pub70"
OUT = KERNEL / "_out_ens_cnx_pub70"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
TEMP = 0.05
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
HYBRIDS = [
    "subset_vit_k_llrd08_320_v3b_hybrid025",
    "subset_v20b_dino_ep11_v3b_hybrid025",
    "subset_v20h_dino_ep11_v3b_hybrid025_wd01_crop145",
    "subset_coat_a_llrd08_v3b_hybrid025",
    "subset_cnx_r7_in22k_v3b_hybrid025",
]
FROZEN = [
    (0.09, 0.32, 0.55, 0.04),
    (0.03, 0.44, 0.52, 0.01),
    (0.27, 0.29, 0.30, 0.14),
    (0.18, 0.36, 0.41, 0.05),
    (0.17, 0.32, 0.33, 0.18),
    (0.21, 0.29, 0.29, 0.21),
    (0.14, 0.32, 0.42, 0.12),
    (0.28, 0.23, 0.22, 0.27),
    (0.30, 0.26, 0.24, 0.20),
    (0.26, 0.27, 0.31, 0.16),
    (0.19, 0.31, 0.35, 0.15),
    (0.25, 0.20, 0.28, 0.27),
]
MSG = (
    "ens ViT-k320 v20h_full OAI3 SWA 1-3 + v20b OAI SWA + "
    "v20h distil crop145 SWA 7-9 + CoAt OAI SWA + CNX-r7 SWA 7-10, "
    "mix lineaire 70% + public degnonguidi 30%, rang, sans TTA, pas de flip, "
    "mix sur 2 T4, Raptor 384 et 224 en parallele"
)

MD = """# RSNA Knee — rsna-knee-ens-cnx-pub70

**Sans TTA.** Internet off, **deux T4**. Pas de flip.

1. Mix lineaire de 5 modeles, poids par finding
   (ViT-k320 / v20b / v20h distil / CoAt-a / CNX-r7).
   Les 4 premiers gardent les proportions du mix publie.
   CNX-r7 prend, par finding, la part du softmax hybrid T=0.05
   (`subset_cnx_r7_in22k_v3b_hybrid025`, epoch 7). Le reste est
   renormalise sur les 4.
   ViT-k320 : `vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3` SWA 1-3
   (gold 0.990). Entree 320, FOV 130 mm.
   v20b DINO : SWA OAI 1-3 (gold 0.970). Entree 224, FOV 130 mm.
   v20h distil : `v20h_crop145_dino_ep11_v3b_rsna_distil_main_full` SWA 7-9
   (gold 0.969). Entree 224, FOV 145 mm.
   CoAt-a : SWA OAI 1-3 (gold 0.988). Entree 224, FOV 130 mm.
   CNX-r7 : `cnx_r7_in22k_v3b_oofmask_full` SWA 7-10 (gold 0.976).
   Entree 224, FOV 130 mm. ConvNeXt-Small IN-22k.
   Les etudes sont coupees sur les deux T4 (une copie des 5 modeles par carte).
   Une seule etude est decodee a la fois. Budget 5 h, puis la memoire est rendue.
2. Notebook public `degnonguidi/rnsa-best-public-score`.
   Raptor-384 puis Raptor-224, l'un apres l'autre (les deux ensembles depassent la RAM).
   Echeance 8,9 h depuis le debut du notebook.
3. Fusion par rangs, **70 %** le mix des 5 modeles et **30 %** la solution publique.
"""

CLOCK = """import time
NB_T0 = time.time()
MIX5_BUDGET_S = 5.0 * 3600
print('horloge depart', flush=True)
print('budget mix5 h', MIX5_BUDGET_S / 3600, '| echeance publique h 8.9', flush=True)
"""

SAVE = """# Scores du mix 5, puis liberation du GPU avant Raptor.
from pathlib import Path as _Out
import gc

out_dir = _Out('/kaggle/working') if _Out('/kaggle/working').exists() else _Out('.')
mix_csv = out_dir / 'mix5_probs.csv'
pred.to_csv(mix_csv, index=False)
print('MIX5_SCORED', len(pred), '|', mix_csv, flush=True)
del model_vit, model_v20b, model_v20h, model_coat, model_cnx
del model_vit_b, model_v20b_b, model_v20h_b, model_coat_b, model_cnx_b
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
try:
    import ctypes
    ctypes.CDLL('libc.so.6').malloc_trim(0)
except Exception:
    pass
print('GPU mix5 libere', flush=True)
"""

BLEND = """# Rang 70/30 : 70 % mix5, 30 % submission publique.
import numpy as np
import pandas as pd
from pathlib import Path

W_MIX5, W_PUBLIC = 0.70, 0.30
work = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
pub = pd.read_csv(work / 'submission.csv', dtype={'StudyInstanceUID': str})
mix = pd.read_csv(work / 'mix5_probs.csv', dtype={'StudyInstanceUID': str})
labels = [c for c in pub.columns if c != 'StudyInstanceUID']
assert set(labels) <= set(mix.columns), 'colonnes mix5 incompletes'
pub = pub.set_index('StudyInstanceUID')
mix = mix.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
scored = [u for u in pub.index if u in mix.index]
print('BLEND70 mix5', len(scored), 'public', len(pub), 'poids mix5 0.70 public 0.30', flush=True)
out = pub[labels].copy()
for c in labels:
    pr = pub[c].rank(method='average', pct=True)
    mr = mix[c].reindex(pub.index)
    ok = mr.notna()
    rr = pd.Series(np.nan, index=pub.index, dtype=np.float64)
    if bool(ok.any()):
        rr.loc[ok] = mr.loc[ok].rank(method='average', pct=True).to_numpy()
    blended = W_MIX5 * rr + W_PUBLIC * pr
    blended = blended.fillna(pr)
    out[c] = blended.astype(np.float32)
out.insert(0, 'StudyInstanceUID', pub.index)
out = out.reset_index(drop=True)
dest = work / 'submission.csv'
out.to_csv(dest, index=False)
assert len(out) == len(pub) and np.isfinite(out[labels].to_numpy()).all()
print('BLEND70_OK', dest, 'lignes', len(out), flush=True)
out.head()
"""

LOAD_CNX = """
def load_cnx(weights_path):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    if str(cfg.get('run_name')) != 'cnx_r7_in22k_v3b_oofmask_full':
        raise RuntimeError(cfg.get('run_name'))
    if str(cfg.get('model')) != 'kneenet_cnx_r7':
        raise RuntimeError(cfg.get('model'))
    if str(cfg.get('backbone')) != 'convnext_small.fb_in22k_ft_in1k':
        raise RuntimeError(cfg.get('backbone'))
    if cfg.get('laterality'):
        raise RuntimeError('cnx ne doit pas etre flipped')
    if int(cfg.get('hidden', -1)) != 384:
        raise RuntimeError(cfg.get('hidden'))
    _expect_swa(ckpt, 'swa-7-8-9-10', [7, 8, 9, 10])
    chunk = _cap_chunk(cfg, 4)
    model = KneeNet(
        backbone=str(cfg.get('backbone')),
        n_series=int(cfg.get('n_series', 6)),
        n_targets=12,
        hidden=int(cfg.get('hidden', 384)),
        dropout=float(cfg.get('dropout', 0.1)),
        pretrained=False,
        encode_chunk=chunk,
        max_depth=int(cfg.get('max_depth', 64) or 64),
    )
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(DEVICE).eval()
    print('cnx', cfg.get('run_name'), 'epoch', ckpt.get('epoch'),
          'lat', cfg.get('laterality'),
          'gold', round(float(ckpt.get('gold_macro_auc')), 4))
    return model


model_cnx = load_cnx(WEIGHTS_CNX)
"""


def _best(name: str) -> dict:
    hist = json.loads((ROOT / "checkpoints" / name / "history.json").read_text(encoding="utf-8"))
    return max(hist, key=lambda r: r.get("val_macro_auc") or -1.0)


def mix_rows() -> list[tuple[float, ...]]:
    recs = [_best(name) for name in HYBRIDS]
    if int(recs[-1]["epoch"]) != 7:
        raise SystemExit(f"cnx hybrid epoch {recs[-1].get('epoch')}")
    frozen = np.array(FROZEN, dtype=np.float64)
    if frozen.shape != (12, 4) or np.any(np.abs(frozen.sum(axis=1) - 1.0) > 1e-6):
        raise SystemExit("poids 4 modeles invalides")
    shares = []
    for t in TARGETS:
        auc = np.array([float(r[f"auc_{t}"]) for r in recs], dtype=np.float64)
        z = auc / TEMP
        z = z - z.max()
        w = np.exp(z)
        w = w / w.sum()
        shares.append(float(w[4]))
    share = np.array(shares, dtype=np.float64)
    if np.any(share <= 0) or np.any(share >= 1):
        raise SystemExit(f"part cnx invalide {share}")
    raw = np.concatenate([frozen * (1.0 - share)[:, None], share[:, None]], axis=1)
    rnd = np.round(raw, 2)
    delta = np.round(1.0 - rnd.sum(axis=1), 2)
    for i, d in enumerate(delta):
        j = int(np.argmax(rnd[i]))
        rnd[i, j] = round(float(rnd[i, j] + d), 2)
    rows = []
    for t, w in zip(TARGETS, rnd):
        got = tuple(float(v) for v in w)
        if abs(sum(got) - 1.0) > 1e-6 or min(got) < -1e-9 or len(got) != 5:
            raise SystemExit(f"mix invalide {t} {got}")
        if got[4] + 1e-9 < share[len(rows)] - 0.02:
            raise SystemExit(f"cnx trop reduit {t}")
        rows.append(got)
    return rows


def _mix_src(rows: list[tuple[float, ...]]) -> str:
    lines = [
        "# Ordre : ViT-k320, v20b, v20h distil, CoAt-a, CNX-r7. Somme 1, plancher 0.",
        "N_MODELS = 5",
        "MIX = [",
    ]
    for t, w in zip(TARGETS, rows):
        nums = ", ".join(f"{v:.2f}" for v in w)
        lines.append(f"    ({nums}),  # {t}")
    lines.append("]")
    return "\n".join(lines) + "\n"


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _code(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


def _md(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


def _replace_nb(nb: dict, old: str, new: str, expect: int) -> None:
    found = 0
    for cell in nb["cells"]:
        src = _text(cell)
        n = src.count(old)
        if not n:
            continue
        found += n
        _set(cell, src.replace(old, new))
    if found != expect:
        raise SystemExit(f"remplacement x{found}, attendu {expect}: {old[:80]!r}")


def _patch_v20h(nb: dict) -> None:
    _replace_nb(nb, "v20h_crop145_swa_ep6_8.pt", "v20h_crop145_distil_swa_ep7_9.pt", 2)
    _replace_nb(
        nb,
        "kaggle_dataset_v20h_lat/_upload_crop145_swa678",
        "kaggle_dataset_v20h_lat/_upload_crop145_distil_swa789",
        1,
    )
    _replace_nb(
        nb,
        "v20h_crop145_dino_ep11_oof_v3b_full_oaiv2_wd01",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full",
        2,
    )
    _replace_nb(nb, "'swa-6-7-8', [6, 7, 8]", "'swa-7-8-9', [7, 8, 9]", 1)


_MIX2_LOOP = """import copy
import threading
from concurrent.futures import ThreadPoolExecutor

_TLS = threading.local()


def _center_crop(img, row_mm, col_mm):
    crop_mm = float(getattr(_TLS, 'crop_mm', CROP_MM))
    h, w = img.shape[:2]
    if row_mm is not None and col_mm is not None and row_mm > 0 and col_mm > 0:
        ch = min(h, max(8, int(round(crop_mm / row_mm))))
        cw = min(w, max(8, int(round(crop_mm / col_mm))))
    else:
        ch = min(h, max(8, int(round(h * CROP_REL))))
        cw = min(w, max(8, int(round(w * CROP_REL))))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def load_series_dicom(series_dir, img_size=224):
    cache = getattr(_TLS, 'series_cache', None)
    if cache is None:
        cache = {}
        _TLS.series_cache = cache
    key = str(Path(series_dir))
    if key not in cache:
        cache[key] = _series_band(series_dir)
    band = cache[key]
    size = int(img_size)
    out = np.empty((band.shape[0], size, size), dtype=np.uint8)
    for i in range(band.shape[0]):
        sl = band[i]
        if sl.shape != (size, size):
            sl = cv2.resize(sl, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = np.clip(np.asarray(sl, dtype=np.float32) * 255.0, 0, 255).round().astype(np.uint8)
    return out


def _split_gpu(model):
    model.cpu()
    replica = copy.deepcopy(model)
    return model.to('cuda:0').eval(), replica.to('cuda:1').eval()


n_gpu = torch.cuda.device_count() if DEVICE == 'cuda' else 0
if n_gpu < 2:
    raise RuntimeError(('attendu 2 GPU', n_gpu))
model_vit, model_vit_b = _split_gpu(model_vit)
model_v20b, model_v20b_b = _split_gpu(model_v20b)
model_v20h, model_v20h_b = _split_gpu(model_v20h)
model_coat, model_coat_b = _split_gpu(model_coat)
model_cnx, model_cnx_b = _split_gpu(model_cnx)
import gc as _gc
print('gpus mix 2 | cuda:0 cuda:1 | decode 1 etude a la fois', flush=True)
_gc.collect()
packs = {
    'cuda:0': (model_vit, model_v20b, model_v20h, model_coat, model_cnx),
    'cuda:1': (model_vit_b, model_v20b_b, model_v20h_b, model_coat_b, model_cnx_b),
}
_DECODE_LOCK = threading.Lock()


def _trim_ram():
    _gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def _forward_on(model, x):
    with torch.autocast('cuda', dtype=AMP_DTYPE):
        return model(x)


def _decode_study(uid):
    with _DECODE_LOCK:
        _TLS.series_cache = {}
        srows = test_series[test_series.StudyInstanceUID == uid]
        sdir = TEST_SERIES_DIR / uid
        _TLS.crop_mm = 130.0
        vol_224 = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)
        vol_vit = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=320)
        _TLS.series_cache = {}
        _TLS.crop_mm = 145.0
        vol_h = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)
        _TLS.series_cache = {}
        _TLS.crop_mm = 130.0
        return vol_224, vol_vit, vol_h


def _score_shard(dev, uids):
    torch.cuda.set_device(dev)
    m_vit, m_v20b, m_v20h, m_coat, m_cnx = packs[dev]
    local = {}
    t0 = time.time()
    for n, uid in enumerate(uids):
        if time.time() - NB_T0 > MIX5_BUDGET_S:
            print(f'MIX5_BUDGET {dev} {n}/{len(uids)}', flush=True)
            break
        vol_224, vol_vit, vol_h = _decode_study(uid)
        if vol_vit.shape[-1] != 320 or vol_224.shape[-1] != 224 or vol_h.shape[-1] != 224:
            raise RuntimeError((uid, vol_vit.shape, vol_224.shape, vol_h.shape))
        x_224 = torch.from_numpy(vol_224).unsqueeze(0).to(dev)
        x_vit = torch.from_numpy(vol_vit).unsqueeze(0).to(dev)
        x_h = torch.from_numpy(vol_h).unsqueeze(0).to(dev)
        del vol_224, vol_vit, vol_h
        with torch.no_grad():
            p_vit = torch.sigmoid(_forward_on(m_vit, x_vit).float())
            p_v20b = torch.sigmoid(_forward_on(m_v20b, x_224).float())
            p_v20h = torch.sigmoid(_forward_on(m_v20h, x_h).float())
            p_coat = torch.sigmoid(_forward_on(m_coat, x_224).float())
            with torch.autocast('cuda', enabled=False):
                p_cnx = torch.sigmoid(m_cnx(x_224.float()).float())
            w = torch.tensor(MIX, dtype=torch.float32, device=dev)
            prob = (
                w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h
                + w[:, 3] * p_coat + w[:, 4] * p_cnx
            ).float().cpu().numpy()[0]
            del p_vit, p_v20b, p_v20h, p_coat, p_cnx, w
        del x_224, x_vit, x_h
        local[uid] = prob
        if (n + 1) % 20 == 0:
            _trim_ram()
            print(f'{dev} {n + 1}/{len(uids)} | {(time.time() - t0) / 60:.1f} min', flush=True)
    _trim_ram()
    return local


uids = [str(u) for u in test.StudyInstanceUID]
with ThreadPoolExecutor(max_workers=2) as pool:
    fa = pool.submit(_score_shard, 'cuda:0', uids[0::2])
    fb = pool.submit(_score_shard, 'cuda:1', uids[1::2])
    scored = fa.result()
    scored.update(fb.result())
missing = [u for u in uids if u not in scored]
if missing and time.time() - NB_T0 < MIX5_BUDGET_S - 30:
    raise RuntimeError(('etudes manquantes', len(missing), len(uids)))
rows = [[u] + scored[u].tolist() for u in uids if u in scored]
pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
print(f'MIX5_COVERAGE {len(pred)}/{len(test)}', flush=True)
pred.head()
"""


def _patch_cnx(nb: dict, rows: list[tuple[float, ...]]) -> None:
    _replace_nb(
        nb,
        "    WEIGHTS_COAT = _find_weight('coat_a_oai3_swa_ep1_3.pt')\n",
        "    WEIGHTS_COAT = _find_weight('coat_a_oai3_swa_ep1_3.pt')\n"
        "    WEIGHTS_CNX = _find_weight('cnx_r7_swa_ep7_10.pt')\n",
        1,
    )
    _replace_nb(
        nb,
        "    WEIGHTS_COAT = Path('kaggle_dataset_coat/_upload_oai3_swa/coat_a_oai3_swa_ep1_3.pt')\n",
        "    WEIGHTS_COAT = Path('kaggle_dataset_coat/_upload_oai3_swa/coat_a_oai3_swa_ep1_3.pt')\n"
        "    WEIGHTS_CNX = Path('kaggle_dataset_cnx/_upload_swa78910/cnx_r7_swa_ep7_10.pt')\n",
        1,
    )
    old_mix = (
        "# Ordre : ViT-k320, v20b DINO, v20h DINO, CoAt-a. Somme 1, plancher 0.\n"
        "N_MODELS = 4\n"
        "MIX = [\n"
        "    (0.09, 0.32, 0.55, 0.04),  # ACL\n"
        "    (0.03, 0.44, 0.52, 0.01),  # MCL\n"
        "    (0.27, 0.29, 0.30, 0.14),  # Medial Meniscus\n"
        "    (0.18, 0.36, 0.41, 0.05),  # Lateral Meniscus\n"
        "    (0.17, 0.32, 0.33, 0.18),  # Medial OA\n"
        "    (0.21, 0.29, 0.29, 0.21),  # Lateral OA\n"
        "    (0.14, 0.32, 0.42, 0.12),  # PF OA\n"
        "    (0.28, 0.23, 0.22, 0.27),  # Effusion\n"
        "    (0.30, 0.26, 0.24, 0.20),  # Synovitis\n"
        "    (0.26, 0.27, 0.31, 0.16),  # Baker's\n"
        "    (0.19, 0.31, 0.35, 0.15),  # Contusion\n"
        "    (0.25, 0.20, 0.28, 0.27),  # Fracture\n"
        "]\n"
    )
    _replace_nb(nb, old_mix, _mix_src(rows), 1)
    _replace_nb(
        nb,
        "print('poids coat      :', WEIGHTS_COAT, '| existe :', WEIGHTS_COAT.exists())\n",
        "print('poids coat      :', WEIGHTS_COAT, '| existe :', WEIGHTS_COAT.exists())\n"
        "print('poids cnx       :', WEIGHTS_CNX, '| existe :', WEIGHTS_CNX.exists())\n",
        1,
    )
    _replace_nb(
        nb,
        "model_coat = load_coat(WEIGHTS_COAT)\n",
        "model_coat = load_coat(WEIGHTS_COAT)\n" + LOAD_CNX,
        1,
    )
    _replace_nb(
        nb,
        "print(len(test), 'etudes | sans TTA | pas de flip | FOV vit/v20b/coat 130 mm, v20h 145 mm')\n",
        "print(len(test), 'etudes | sans TTA | pas de flip | FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32 | 2 GPU')\n",
        1,
    )
    _replace_nb(nb, "or len(w) != 4:", "or len(w) != 5:", 1)
    _replace_nb(
        nb,
        "    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f}')\n",
        "    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f} cnx {w[4]:.2f}')\n",
        1,
    )
    _replace_nb(
        nb,
        "        p_coat = torch.sigmoid(_forward(model_coat, x_224).float())\n"
        "        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)\n"
        "        prob = (\n"
        "            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h + w[:, 3] * p_coat\n"
        "        ).cpu().numpy()[0]\n",
        "        p_coat = torch.sigmoid(_forward(model_coat, x_224).float())\n"
        "        with torch.autocast('cuda', enabled=False):\n"
        "            p_cnx = torch.sigmoid(model_cnx(x_224.float()).float())\n"
        "        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)\n"
        "        prob = (\n"
        "            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h"
        " + w[:, 3] * p_coat + w[:, 4] * p_cnx\n"
        "        ).cpu().numpy()[0]\n",
        1,
    )
    old_loop = """rows = []
t0 = time.time()
with torch.no_grad():
    for i, uid in enumerate(test.StudyInstanceUID):
        _SERIES_CACHE.clear()
        srows = test_series[test_series.StudyInstanceUID == uid]
        sdir = TEST_SERIES_DIR / uid
        CROP_MM = 130.0
        vol_224 = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)
        vol_vit = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=320)
        _SERIES_CACHE.clear()
        CROP_MM = 145.0
        vol_h = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)
        CROP_MM = 130.0
        if vol_vit.shape[-1] != 320 or vol_224.shape[-1] != 224 or vol_h.shape[-1] != 224:
            raise RuntimeError((vol_vit.shape, vol_224.shape, vol_h.shape))
        x_224 = torch.from_numpy(vol_224).unsqueeze(0).to(DEVICE)
        x_vit = torch.from_numpy(vol_vit).unsqueeze(0).to(DEVICE)
        x_h = torch.from_numpy(vol_h).unsqueeze(0).to(DEVICE)
        p_vit = torch.sigmoid(_forward(model_vit, x_vit).float())
        p_v20b = torch.sigmoid(_forward(model_v20b, x_224).float())
        p_v20h = torch.sigmoid(_forward(model_v20h, x_h).float())
        p_coat = torch.sigmoid(_forward(model_coat, x_224).float())
        with torch.autocast('cuda', enabled=False):
            p_cnx = torch.sigmoid(model_cnx(x_224.float()).float())
        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)
        prob = (
            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h + w[:, 3] * p_coat + w[:, 4] * p_cnx
        ).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(test)} | {(time.time()-t0)/60:.1f} min', flush=True)

pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
pred.head()
"""
    _replace_nb(nb, old_loop, _MIX2_LOOP, 1)


def _patch_public(cells: list[dict]) -> None:
    run = _text(cells[1])
    old = (
        "NB_T0 = time.time()\n"
        "rc = subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600).returncode\n"
    )
    new = (
        "NB_T0 = globals().get('NB_T0', time.time())\n"
        "_t384 = time.time()\n"
        "rc = subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600).returncode\n"
    )
    if run.count(old) != 1:
        raise SystemExit(f"NB_T0 x{run.count(old)}")
    run = run.replace(old, new, 1)
    old_sec = "RAPTOR_SECONDS = time.time() - NB_T0\n"
    if run.count(old_sec) != 1:
        raise SystemExit("RAPTOR_SECONDS introuvable")
    _set(cells[1], run.replace(old_sec, "RAPTOR_SECONDS = time.time() - _t384\n", 1))
    blend = _text(cells[6])
    old_raise = (
        "    raise RuntimeError('our stage did not score %d of %d studies "
        "(more than 50 %%)' % (our_fail.sum(), n))\n"
    )
    new_raise = "    print('PUBLIC_OURS_PARTIAL', int(our_fail.sum()), 'of', n, flush=True)\n"
    if blend.count(old_raise) != 1:
        raise SystemExit("garde 50% introuvable")
    _set(cells[6], blend.replace(old_raise, new_raise, 1))
    src4 = _text(cells[4])
    old_dead = "DEADLINE_HOURS = 8.3\n"
    if src4.count(old_dead) != 1:
        raise SystemExit(f"DEADLINE x{src4.count(old_dead)}")
    _set(cells[4], src4.replace(old_dead, "DEADLINE_HOURS = 8.9\n", 1))
    run = _text(cells[1])
    old_run = """_t384 = time.time()
rc = subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600).returncode
if rc != 0:
    raise RuntimeError('Raptor stage failed with exit code %d' % rc)
for f in ('raptor_sub.csv', 'raptor_raw.csv'):
    assert os.path.isfile('/kaggle/working/' + f), f + ' missing'
RAPTOR_SECONDS = time.time() - _t384
print('RAPTOR_STAGE_DONE seconds %.0f' % RAPTOR_SECONDS, flush=True)
subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], check=False)
"""
    new_run = """_t384 = time.time()
import gc as _gc
import ctypes as _ct
_gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
try:
    _ct.CDLL('libc.so.6').malloc_trim(0)
except Exception:
    pass
rc = subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600).returncode
if rc != 0:
    raise RuntimeError('Raptor stage failed with exit code %d' % rc)
for f in ('raptor_sub.csv', 'raptor_raw.csv'):
    assert os.path.isfile('/kaggle/working/' + f), f + ' missing'
RAPTOR_SECONDS = time.time() - _t384
print('RAPTOR_STAGE_DONE seconds %.0f' % RAPTOR_SECONDS, flush=True)
print('RAPTOR_SEQUENTIEL', flush=True)
subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], check=False)
"""
    if run.count(old_run) != 1:
        raise SystemExit("lancement raptor 384 introuvable")
    _set(cells[1], run.replace(old_run, new_run, 1))
    run224 = _text(cells[3])
    old224 = """_t224 = time.time()
rc = subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=2 * 3600).returncode
if rc != 0:
    raise RuntimeError('Raptor-224 stage failed with exit code %d' % rc)
for f in ('raptor224_sub.csv', 'raptor224_raw.csv'):
    assert os.path.isfile('/kaggle/working/' + f), f + ' missing'
RAPTOR224_SECONDS = time.time() - _t224
print('RAPTOR224_STAGE_DONE seconds %.0f' % RAPTOR224_SECONDS, flush=True)
subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], check=False)
"""
    new224 = """_t224 = time.time()
import gc as _gc
import ctypes as _ct
_gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
try:
    _ct.CDLL('libc.so.6').malloc_trim(0)
except Exception:
    pass
rc = subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=2 * 3600).returncode
if rc != 0:
    raise RuntimeError('Raptor-224 stage failed with exit code %d' % rc)
for f in ('raptor224_sub.csv', 'raptor224_raw.csv'):
    assert os.path.isfile('/kaggle/working/' + f), f + ' missing'
RAPTOR224_SECONDS = time.time() - _t224
print('RAPTOR224_STAGE_DONE seconds %.0f' % RAPTOR224_SECONDS, flush=True)
subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], check=False)
"""
    if run224.count(old224) != 1:
        raise SystemExit("lancement raptor 224 introuvable")
    _set(cells[3], run224.replace(old224, new224, 1))


def _cnx_cell() -> dict:
    text = (ROOT / "src" / "models" / "kneenet_cnx_r7.py").read_text(encoding="utf-8")
    needle = "import timm\n"
    if needle not in text:
        raise SystemExit("import timm introuvable")
    text = text.replace(needle, "import timm\nprint('timm', timm.__version__, flush=True)\n", 1)
    if "from src." in text:
        raise SystemExit("import src dans cnx")
    # Le notebook partage un seul namespace. TIMM_NAME du CNX ecrasait
    # celui du ViT-k320, qui instancie alors un ConvNeXt.
    code = (
        "# --- KneeNet CNX-r7, namespace isole ---\n"
        "import json as _json\n"
        f"_cnx_src = _json.loads({json.dumps(json.dumps(text))})\n"
        "_cnx_ns = {'__name__': 'kneenet_cnx_r7'}\n"
        "exec(compile(_cnx_src, 'kneenet_cnx_r7.py', 'exec'), _cnx_ns)\n"
        "KneeNet = _cnx_ns['KneeNet']\n"
        "print('KneeNet CNX-r7 isole', KneeNet.arch_id, flush=True)\n"
    )
    return _code(code)


def write_nb() -> str:
    rows = mix_rows()
    ours = json.loads(OURS.read_text(encoding="utf-8"))
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    if len(ours["cells"]) != 9 or len(public["cells"]) != 7:
        raise SystemExit("notebooks source inattendus")
    _patch_v20h(ours)
    _patch_cnx(ours, rows)
    ours["cells"].insert(7, _cnx_cell())
    cells = [_md(MD), _code(CLOCK)]
    cells.extend(ours["cells"][1:9])
    cells.append(_code(SAVE))
    _patch_public(public["cells"])
    cells.extend(public["cells"])
    cells.append(_code(BLEND))
    for cell in cells:
        src = _text(cell)
        if cell["cell_type"] != "code" or src.lstrip().startswith("%%"):
            continue
        ast.parse(src)
    blob = "\n".join(_text(c) for c in cells)
    acl = rows[0]
    acl_line = (
        f"vit {acl[0]:.2f} v20b {acl[1]:.2f} v20h {acl[2]:.2f} "
        f"coat {acl[3]:.2f} cnx {acl[4]:.2f}"
    )
    for needle in (
        "swa-7-8-9",
        "[7, 8, 9]",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full",
        "v20h_crop145_distil_swa_ep7_9.pt",
        "swa-1-2-3",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "cnx_r7_swa_ep7_10.pt",
        "cnx_r7_in22k_v3b_oofmask_full",
        "swa-7-8-9-10",
        "kneenet_cnx_r7",
        "N_MODELS = 5",
        f"({acl[0]:.2f}, {acl[1]:.2f}, {acl[2]:.2f}, {acl[3]:.2f}, {acl[4]:.2f})",
        "_TLS.crop_mm = 145.0",
        "laterality=False",
        "MIX5_BUDGET_S",
        "gpus mix 2 | cuda:0 cuda:1",
        "MIX5_COVERAGE",
        "RAPTOR_SEQUENTIEL",
        "DEADLINE_HOURS = 8.9",
        "model_cnx",
        "w[:, 4] * p_cnx",
        "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt",
        "A224 = 0.35",
        "BLEND70_OK",
        "W_MIX5, W_PUBLIC = 0.70, 0.30",
        "sans TTA",
        "PUBLIC_OURS_PARTIAL",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "v20h_crop145_swa_ep6_8.pt",
        "swa-6-7-8",
        "oaiv2_wd01",
        "TTA_VARIANTS",
        "laterality=True",
        "W_MIX4, W_PUBLIC = 0.60, 0.40",
        "N_MODELS = 4",
        "(0.09, 0.32, 0.55, 0.04)",
        "DEADLINE_HOURS = 8.3",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": KAGGLE_ID,
        "title": SLUG,
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": ACCEL,
        "competition_sources": [COMPETITION],
        "dataset_sources": [
            "heliosli/rsna-knee-r6-frozen-fe055406-w1",
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
            "sultent/rsna-knee-vit-k320-oai3-swa",
            "sultent/rsna-knee-v20b-dino-oai3-swa",
            "sultent/rsna-knee-v20h-crop145-distil-swa789",
            "sultent/rsna-knee-coat-a-oai3-swa",
            "sultent/rsna-knee-cnx-r7-swa78910",
        ],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    (PUSH / "_acl.txt").write_text(acl_line + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(cells), flush=True)
    for t, w in zip(TARGETS, rows):
        print(f"  {t:<18} {w}", flush=True)
    return acl_line


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        raise


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def _transient(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "expecting value", "remote end closed", "connection aborted",
        "temporarily unavailable", "timed out", "timeout", "502", "503", "504",
    ))


def push() -> str:
    while True:
        try:
            out = kaggle("kernels", "push", "-p", str(PUSH), "--accelerator", ACCEL, "-t", "32400")
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text):
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            if _transient(text):
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
                continue
            raise
        print(out, flush=True)
        if _quota(out):
            print("quota GPU, retry 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        version = m.group(1)
        (PUSH / "_version.txt").write_text(version + "\n", encoding="utf-8")
        print("kernel version", version, flush=True)
        return version


def wait() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KAGGLE_ID).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def _daily_limit(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "maximum number", "submission limit", "daily submission allowance",
        "try again tomorrow", "too many submission", "exceeded",
    ))


def submit(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KAGGLE_ID, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and SLUG in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    acl_line = (PUSH / "_acl.txt").read_text(encoding="utf-8").strip()
    for needle in (
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3",
        "gold 0.9903",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10 lat False gold 0.976",
        "cnx forward fp32",
        "lat False",
        "v20h 145 mm",
        "cnx 130 mm",
        acl_line,
        "MIX5_SCORED",
        "GPU mix5 libere",
        "RAPTOR224_STAGE_DONE",
        "NOTEBOOK_COMPLETE",
        "poids mix5 0.70 public 0.30",
        "BLEND70_OK",
        "sans TTA",
        "gpus mix 2 | cuda:0 cuda:1",
        "MIX5_COVERAGE 3/3",
        "RAPTOR_SEQUENTIEL",
        "lignes 3",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text or "lat True" in text or "oaiv2_wd01" in text or "public 0.40" in text:
        raise SystemExit("TTA, flip, ancien v20h ou ancien blend")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    waited = False
    while True:
        print("submit", KAGGLE_ID, "version", version, flush=True)
        try:
            resp = api.competition_submit_code(
                "submission.csv", MSG, COMPETITION, KAGGLE_ID, int(version), quiet=True,
            )
        except Exception as e:
            http = getattr(e, "response", None)
            body = http.text if http is not None else str(e)
            print(body, flush=True)
            if _daily_limit(body) and not waited:
                waited = True
                now = datetime.now(timezone.utc)
                target = now.replace(hour=0, minute=5, second=0, microsecond=0)
                if now >= target:
                    target = target + timedelta(days=1)
                delay = (target - now).total_seconds()
                print(f"ATTENTE_QUOTA {delay:.0f}s", flush=True)
                time.sleep(delay + 5)
                continue
            if _transient(body):
                time.sleep(20)
                continue
            raise
        print(resp, flush=True)
        print("SOUMIS", KAGGLE_ID, "version", version, flush=True)
        return


def main() -> None:
    write_nb()
    if "--write" in sys.argv:
        return
    version = push()
    wait()
    submit(version)


if __name__ == "__main__":
    main()
