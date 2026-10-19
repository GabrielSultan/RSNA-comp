# -*- coding: utf-8 -*-
"""Mix5, TTA 2 GPU, rang 70/30. Pas de soumission competition."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_cnx_pub60_tta" / "submit.ipynb"
SLUG = "rsna-knee-ens-cnx-pub70-tta"
PUSH = KERNEL / "_push_ens_cnx_pub70_tta"
OUT = KERNEL / "_out_ens_cnx_pub70_tta"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
MSG = (
    "ens CNX-r7 SWA 7-10 + ViT-k320 + v20b + v20h distil + CoAt, "
    "CNX reprend 30% v20b et 10% des trois autres, "
    "TTA rotation contraste scale 1.07, 2 T4, rang 70/30, pas soumis"
)
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
# ViT, v20b, v20h, CoAt, CNX. Mix pub60 avant la reprise CNX.
BASE = np.array([
    (0.07, 0.27, 0.46, 0.03, 0.17),
    (0.03, 0.43, 0.51, 0.01, 0.02),
    (0.20, 0.22, 0.23, 0.11, 0.24),
    (0.14, 0.29, 0.33, 0.04, 0.20),
    (0.13, 0.25, 0.26, 0.14, 0.22),
    (0.16, 0.22, 0.22, 0.16, 0.24),
    (0.12, 0.27, 0.37, 0.10, 0.14),
    (0.22, 0.18, 0.17, 0.21, 0.22),
    (0.26, 0.22, 0.21, 0.17, 0.14),
    (0.21, 0.22, 0.26, 0.13, 0.18),
    (0.15, 0.24, 0.28, 0.12, 0.21),
    (0.19, 0.16, 0.22, 0.21, 0.22),
], dtype=np.float64)


def mix_rows() -> list[tuple[float, ...]]:
    """CNX reprend 30 % de v20b et 10 % de ViT, v20h et CoAt. Somme 1."""
    new = BASE.copy()
    take_b = 0.30 * new[:, 1]
    take_o = 0.10 * new[:, [0, 2, 3]]
    new[:, 1] -= take_b
    new[:, 0] -= take_o[:, 0]
    new[:, 2] -= take_o[:, 1]
    new[:, 3] -= take_o[:, 2]
    new[:, 4] += take_b + take_o.sum(axis=1)
    if np.any(np.abs(new.sum(axis=1) - 1.0) > 1e-9):
        raise SystemExit("somme mix avant arrondi")
    rnd = np.round(new, 2)
    delta = np.round(1.0 - rnd.sum(axis=1), 2)
    rnd[:, 4] = np.round(rnd[:, 4] + delta, 2)
    rows = []
    for t, w, base in zip(TARGETS, rnd, BASE):
        got = tuple(float(v) for v in w)
        if abs(sum(got) - 1.0) > 1e-6 or min(got) < -1e-9:
            raise SystemExit(f"mix invalide {t} {got}")
        if got[4] + 1e-9 < base[4]:
            raise SystemExit(f"cnx pas augmente {t} {got} base {base}")
        if got[1] > base[1] + 1e-9:
            raise SystemExit(f"v20b pas baisse {t}")
        rows.append(got)
    return rows


def _mix_src(rows: list[tuple[float, ...]]) -> str:
    lines = [
        "# Ordre : ViT-k320, v20b, v20h distil, CoAt-a, CNX-r7. Somme 1.",
        "# Base pub60, puis CNX reprend 30 % du poids v20b et 10 % de ViT, v20h, CoAt.",
        "N_MODELS = 5",
        "MIX = [",
    ]
    for t, w in zip(TARGETS, rows):
        nums = ", ".join(f"{v:.2f}" for v in w)
        lines.append(f"    ({nums}),  # {t}")
    lines.append("]")
    return "\n".join(lines) + "\n"


INFER = r'''
if DEVICE == 'cuda' and torch.cuda.device_count() < 2:
    raise RuntimeError('attendu 2 GPU, vu %d' % torch.cuda.device_count())
DEV0 = 'cuda:0' if DEVICE == 'cuda' else DEVICE
DEV1 = 'cuda:1' if DEVICE == 'cuda' else DEVICE
print(
    'gpus visibles', torch.cuda.device_count() if DEVICE == 'cuda' else 0,
    '| cnx cuda:0 | vit v20b v20h coat cuda:1',
    flush=True,
)

_DEVICE_RESTORE = DEVICE
DEVICE = DEV1
model_vit = load_vit(WEIGHTS_VIT)
model_v20h = load_resnet(
    WEIGHTS_V20H, KneeNetV20h, 'kneenet_v20h',
    'v20h_crop145_dino_ep11_v3b_rsna_distil_main_full', 'swa-7-8-9', [7, 8, 9],
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
DEVICE = DEV0


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
for _m in (model_vit, model_v20b, model_v20h, model_coat):
    _m.to(DEV1)
model_cnx.to(DEV0)
DEVICE = _DEVICE_RESTORE
if str(DEV0).startswith('cuda'):
    torch.cuda.set_device(DEV0)
    torch.cuda.empty_cache()
    torch.cuda.set_device(DEV1)
    torch.cuda.empty_cache()
    print(
        'placement',
        'cnx', next(model_cnx.parameters()).device,
        'vit', next(model_vit.parameters()).device,
        'mem0', torch.cuda.memory_allocated(0) // (1024 * 1024),
        'mem1', torch.cuda.memory_allocated(1) // (1024 * 1024),
        flush=True,
    )

test = pd.read_csv(TEST_CSV)
test_series = pd.read_csv(TEST_SERIES_CSV)
print(
    len(test),
    'etudes | TTA v20h x3 v20b x2 cnx x3 vit x2 coat x2 | '
    'rotation +5 -5 contraste 1.08 scale 1.07 | 2 GPU | pas de flip | '
    'FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32',
)
for t, w in zip(TARGETS, MIX):
    if abs(sum(w) - 1.0) > 1e-6 or min(w) < -1e-9 or len(w) != 5:
        raise RuntimeError((t, w))
    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f} cnx {w[4]:.2f}')


def _forward(model, x):
    if x.is_cuda:
        with torch.autocast('cuda', dtype=AMP_DTYPE):
            return model(x)
    return model(x)


def apply_affine(x, angle_deg=0.0, scale=1.0):
    """Meme convention que src/data/transforms.py. Pas de flip."""
    if float(angle_deg) == 0.0 and float(scale) == 1.0:
        return x
    import torch.nn.functional as F
    if x.ndim != 5:
        raise RuntimeError(tuple(x.shape))
    b, s, d, h, w = x.shape
    xf = x.reshape(b * s * d, 1, h, w)
    angle = x.new_tensor(float(angle_deg) * 3.141592653589793 / 180.0)
    sc = x.new_tensor(float(scale))
    cos, sin = torch.cos(angle) / sc, torch.sin(angle) / sc
    theta = xf.new_zeros(b, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta = theta.repeat_interleave(s * d, dim=0)
    grid = F.affine_grid(theta, xf.shape, align_corners=False)
    xf = F.grid_sample(xf, grid, align_corners=False, padding_mode="zeros")
    return xf.reshape(b, s, d, h, w)


def _contrast(x):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    return ((x - mean) * 1.08 + mean).clamp(0.0, 1.0)


# (angle, scale, contraste). Pas de luminosite, pas de flip.
V20H_VIEWS = ((5.0, 1.0, False), (-5.0, 1.0, False), (0.0, 1.0, True))
V20B_VIEWS = ((5.0, 1.0, False), (0.0, 1.07, False))
CNX_VIEWS = ((5.0, 1.0, False), (-5.0, 1.0, False), (0.0, 1.0, True))
VIT_VIEWS = ((5.0, 1.0, False), (0.0, 1.0, True))
COAT_VIEWS = ((5.0, 1.0, False), (0.0, 1.07, False))


def _view(x, angle, scale, contrast):
    if contrast:
        if angle != 0.0 or scale != 1.0:
            raise RuntimeError((angle, scale))
        return _contrast(x)
    return apply_affine(x, angle, scale)


def _avg(model, x, views, fp32=False):
    if x.is_cuda:
        torch.cuda.set_device(x.device)
    ps = []
    for angle, scale, contrast in views:
        v = _view(x, angle, scale, contrast)
        if fp32:
            with torch.autocast('cuda', enabled=False):
                ps.append(torch.sigmoid(model(v.float()).float()))
        else:
            ps.append(torch.sigmoid(_forward(model, v).float()))
        del v
    return (sum(ps) / float(len(ps))).float().cpu()


def _gpu0(x224):
    with torch.no_grad():
        if str(DEV0).startswith('cuda'):
            torch.cuda.set_device(DEV0)
        x = x224.to(DEV0)
        out = _avg(model_cnx, x, CNX_VIEWS, fp32=True)
        del x
        return out


def _gpu1(x224, xvit, xh):
    with torch.no_grad():
        if str(DEV1).startswith('cuda'):
            torch.cuda.set_device(DEV1)
        a = x224.to(DEV1)
        b = xvit.to(DEV1)
        c = xh.to(DEV1)
        return (
            _avg(model_vit, b, VIT_VIEWS),
            _avg(model_v20b, a, V20B_VIEWS),
            _avg(model_v20h, c, V20H_VIEWS),
            _avg(model_coat, a, COAT_VIEWS),
        )


print(
    'TTA v20h rotation +5 -5 contraste 1.08 | v20b rotation +5 scale 1.07 | '
    'cnx rotation +5 -5 contraste 1.08 | vit rotation +5 contraste 1.08 | '
    'coat rotation +5 scale 1.07 | pas de flip pas de luminosite',
    flush=True,
)
from concurrent.futures import ThreadPoolExecutor

rows = []
t0 = time.time()
pool = ThreadPoolExecutor(max_workers=2)
try:
    with torch.no_grad():
        for i, uid in enumerate(test.StudyInstanceUID):
            if time.time() - NB_T0 > MIX5_BUDGET_S:
                print(f'MIX5_BUDGET {i}/{len(test)}', flush=True)
                break
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
            cpu224 = torch.from_numpy(vol_224).unsqueeze(0)
            cpuvit = torch.from_numpy(vol_vit).unsqueeze(0)
            cpuh = torch.from_numpy(vol_h).unsqueeze(0)
            f0 = pool.submit(_gpu0, cpu224)
            f1 = pool.submit(_gpu1, cpu224, cpuvit, cpuh)
            p_cnx = f0.result()
            p_vit, p_v20b, p_v20h, p_coat = f1.result()
            w = torch.tensor(MIX, dtype=torch.float32)
            prob = (
                w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h
                + w[:, 3] * p_coat + w[:, 4] * p_cnx
            ).numpy()[0]
            rows.append([uid] + prob.tolist())
            if (i + 1) % 50 == 0:
                print(f'{i+1}/{len(test)} | {(time.time()-t0)/60:.1f} min', flush=True)
finally:
    pool.shutdown(wait=True)

pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
pred.head()
'''


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _replace(nb: dict, old: str, new: str, expect: int) -> None:
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


def write_nb() -> None:
    rows = mix_rows()
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    old_infer = None
    for cell in nb["cells"]:
        src = _text(cell)
        if "model_vit = load_vit(WEIGHTS_VIT)" in src and "pred.head()" in src:
            old_infer = src[src.index("model_vit = load_vit(WEIGHTS_VIT)"):]
            break
    if old_infer is None:
        raise SystemExit("cellule inference introuvable")
    _replace(nb, old_infer, INFER.strip() + "\n", 1)
    _replace(nb, "rsna-knee-ens-cnx-pub60-tta", SLUG, 1)
    _replace(
        nb,
        "**TTA +5 deg seulement sur v20h distil** (identite et rotation +5, moyenne). "
        "Les 4 autres modeles : un seul passage. Internet off, **T4**. Pas de flip.",
        "**TTA sur les 5 modeles, 2 T4.** v20h : +5, -5, contraste 1.08. "
        "v20b : +5, scale 1.07. CNX-r7 : +5, -5, contraste 1.08. "
        "ViT : +5, contraste 1.08. CoAt : +5, scale 1.07. "
        "Pas de luminosite, pas de flip. Internet off.",
        1,
    )
    _replace(
        nb,
        "   Les 4 premiers gardent les proportions du mix publie.\n"
        "   CNX-r7 prend, par finding, la part du softmax hybrid T=0.05\n"
        "   (`subset_cnx_r7_in22k_v3b_hybrid025`, epoch 7). Le reste est\n"
        "   renormalise sur les 4.",
        "   Base : proportions du mix publie, CNX-r7 = part du softmax hybrid\n"
        "   T=0.05 (`subset_cnx_r7_in22k_v3b_hybrid025`, epoch 7), reste\n"
        "   renormalise sur les 4. Ensuite CNX-r7 reprend, par finding, 30 %\n"
        "   du poids v20b et 10 % de ViT, v20h et CoAt.",
        1,
    )
    _replace(
        nb,
        "3. Fusion par rangs, **60 %** le mix des 5 modeles et **40 %** la solution publique.",
        "3. Fusion par rangs, **70 %** le mix des 5 modeles et **30 %** la solution publique.",
        1,
    )
    old_mix_start = nb and "# Ordre : ViT-k320, v20b, v20h distil, CoAt-a, CNX-r7. Somme 1, plancher 0.\nN_MODELS = 5\nMIX = [\n"
    _replace(nb, "print('TTA             : v20h distil +5 deg, autres x1')", "print('TTA             : 5 modeles, rotation contraste scale 1.07, 2 GPU')", 1)
    # replace MIX block
    blob_now = "\n".join(_text(c) for c in nb["cells"])
    start = blob_now.find("# Ordre : ViT-k320, v20b, v20h distil, CoAt-a, CNX-r7. Somme 1, plancher 0.")
    end = blob_now.find("]\n", start)
    if start < 0 or end < 0:
        raise SystemExit("bloc MIX introuvable")
    old_mix = blob_now[start:end + 2]
    _replace(nb, old_mix, _mix_src(rows), 1)
    _replace(nb, "W_MIX5, W_PUBLIC = 0.60, 0.40", "W_MIX5, W_PUBLIC = 0.70, 0.30", 1)
    _replace(nb, "poids mix5 0.60 public 0.40", "poids mix5 0.70 public 0.30", 1)
    _replace(nb, "BLEND60", "BLEND70", 2)
    _replace(nb, "# Rang 60/40 : 60 % mix5, 40 % submission publique.", "# Rang 70/30 : 70 % mix5, 30 % submission publique.", 1)
    for cell in nb["cells"]:
        src = _text(cell)
        if cell["cell_type"] != "code" or src.lstrip().startswith("%%"):
            continue
        ast.parse(src)
    blob = "\n".join(_text(c) for c in nb["cells"])
    for needle in (
        SLUG,
        "attendu 2 GPU",
        "DEV1 = 'cuda:1'",
        "model_cnx.to(DEV0)" if False else         "model_cnx = load_cnx(WEIGHTS_CNX)",
        "DEVICE = DEV1",
        "V20B_VIEWS = ((5.0, 1.0, False), (0.0, 1.07, False))",
        "CNX_VIEWS = ((5.0, 1.0, False), (-5.0, 1.0, False), (0.0, 1.0, True))",
        "scale 1.07",
        "contraste 1.08",
        "W_MIX5, W_PUBLIC = 0.70, 0.30",
        "enabled=False",
        "cnx forward fp32",
        "laterality=False",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "rsna-knee-ens-cnx-pub60-tta",
        "W_MIX5, W_PUBLIC = 0.60, 0.40",
        "luminosite +",
        "_brightness",
        "p_v20h_id",
        "BLEND60",
        "autres x1",
        "laterality=True,",
        "laterality=True)",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    # CNX must be strictly above the old share on every finding after rounding
    for t, w, base in zip(TARGETS, rows, BASE):
        if w[4] < base[4] - 1e-9:
            raise SystemExit(f"cnx {t}")
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((KERNEL / "_push_ens_cnx_pub60_tta" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KAGGLE_ID
    meta["title"] = SLUG
    meta["machine_shape"] = ACCEL
    meta["enable_gpu"] = True
    meta["enable_internet"] = False
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(nb["cells"]), flush=True)
    for t, w in zip(TARGETS, rows):
        print(f"  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f} cnx {w[4]:.2f}", flush=True)


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
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
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

    meta = json.loads((PUSH / "kernel-metadata.json").read_text(encoding="utf-8"))
    nb = json.loads((PUSH / "submit.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
    req = ApiSaveKernelRequest()
    req.slug = meta["id"]
    req.new_title = meta["title"]
    req.text = json.dumps(nb)
    req.language = meta["language"]
    req.kernel_type = meta["kernel_type"]
    req.is_private = True
    req.enable_gpu = True
    req.enable_tpu = False
    req.enable_internet = False
    req.dataset_data_sources = meta["dataset_sources"]
    req.competition_data_sources = meta["competition_sources"]
    req.category_ids = meta.get("keywords") or []
    req.session_timeout_seconds = 32400
    req.machine_shape = ACCEL
    req.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
    t0 = time.time()
    while True:
        api = KaggleApi()
        api.authenticate()
        try:
            with api.build_kaggle_client() as client:
                result = client.kernels.kernels_api_client.save_kernel(req)
        except Exception as e:
            http = getattr(e, "response", None)
            text = http.text if http is not None else str(e)
            print(text[:800], flush=True)
            if _quota(text) and time.time() - t0 < 15 * 60:
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            if _transient(text):
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
                continue
            raise
        print("push", result.version_number, result.url, result.error, flush=True)
        if result.error:
            if _quota(result.error) and time.time() - t0 < 15 * 60:
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            raise RuntimeError(result.error)
        version = str(result.version_number)
        (PUSH / "_version.txt").write_text(version + "\n", encoding="utf-8")
        print("kernel version", version, flush=True)
        return version


def _raw_status() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        http = client._http_client
        http._init_session()
        req = ApiGetKernelSessionStatusRequest()
        req.user_name, req.kernel_slug = KAGGLE_ID.split("/", 1)
        prep = http._prepare_request("kernels.KernelsApiService", "GetKernelSessionStatus", req)
        resp = http._session.send(
            prep, **http._session.merge_environment_settings(prep.url, {}, None, None, None),
        )
        if resp.status_code == 429:
            return "RETRY"
        resp.raise_for_status()
        return resp.text


def wait() -> None:
    t0 = time.time()
    seen = False
    empty = 0
    while True:
        body = _raw_status()
        print("raw", body[:300], flush=True)
        if body == "RETRY":
            time.sleep(POLL_S)
            continue
        if body.strip() in ("", "{}"):
            empty += 1
            if empty >= 8:
                raise SystemExit("session non demarree")
            time.sleep(POLL_S)
            continue
        empty = 0
        up = body.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or '"QUEUED"' in up:
            seen = True
        if "COMPLETE" in up and "RUNNING" not in up and (seen or time.time() - t0 > 90):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def check_log() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KAGGLE_ID, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and SLUG in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "gold 0.9903",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10 lat False gold 0.976",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "gpus visibles 2",
        "cuda:1",
        "scale 1.07",
        "contraste 1.08",
        "rotation +5 -5",
        "pas de flip pas de luminosite",
        "poids mix5 0.70 public 0.30",
        "BLEND70_OK",
        "MIX5_SCORED",
        "lignes 3",
        "lat False",
        "cnx forward fp32",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "lat True" in text or "gpus visibles 1" in text or "luminosite +" in text:
        raise SystemExit("TTA ou GPU incomplet")
    print("LOG_OK", log.name, flush=True)
    print("PAS_DE_SOUMISSION", KAGGLE_ID, flush=True)


def main() -> None:
    write_nb()
    if "--write" in sys.argv:
        return
    push()
    wait()
    check_log()


if __name__ == "__main__":
    main()
