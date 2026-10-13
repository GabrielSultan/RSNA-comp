# -*- coding: utf-8 -*-
"""mix3 hybrid025 + dephasage demi-pas des 24 coupes. Pas de TTA geometrique."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_h025_mix3" / "submit.ipynb"
PUSH = KERNEL / "_push_ens_h025_mix3_phase"
SLUG = "rsna-knee-ens-oai3-h025-mix3-phase"
N_SLICES = 24

OLD_LOAD = """def load_series_dicom(series_dir, img_size=224):
    key = str(Path(series_dir))
    if key not in _SERIES_CACHE:
        _SERIES_CACHE[key] = _series_band(series_dir)
    band = _SERIES_CACHE[key]
    size = int(img_size)
    out = np.empty((band.shape[0], size, size), dtype=np.uint8)
    for i in range(band.shape[0]):
        sl = band[i]
        if sl.shape != (size, size):
            sl = cv2.resize(sl, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = np.clip(np.asarray(sl, dtype=np.float32) * 255.0, 0, 255).round().astype(np.uint8)
    return out
"""

NEW_LOAD = """def _phase_index(n_band, phase):
    '''phase 0 = linspace d'entrainement. phase 0.5 = demi-pas, borne dans la bande.'''
    last = float(n_band - 1)
    base = np.linspace(0.0, last, N_SLICES)
    if float(phase) == 0.0:
        return base.round().astype(int)
    step = last / (N_SLICES - 1)
    pos = np.clip(base + float(phase) * step, 0.0, last)
    return pos.round().astype(int)


def load_series_dicom(series_dir, img_size=224, phase=0.0):
    key = str(Path(series_dir))
    if key not in _SERIES_CACHE:
        _SERIES_CACHE[key] = _series_band(series_dir)
    band = _SERIES_CACHE[key]
    idx = _phase_index(band.shape[0], phase)
    picked = band[idx]
    size = int(img_size)
    out = np.empty((N_SLICES, size, size), dtype=np.uint8)
    for i in range(N_SLICES):
        sl = picked[i]
        if sl.shape != (size, size):
            sl = cv2.resize(sl, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = np.clip(np.asarray(sl, dtype=np.float32) * 255.0, 0, 255).round().astype(np.uint8)
    return out
"""

OLD_BAND = """    band = vol[a:b + 1]
    idx = np.linspace(0, band.shape[0] - 1, N_SLICES).round().astype(int)
    band = band[idx]
    return band
"""

NEW_BAND = """    return vol[a:b + 1]
"""

OLD_SIG = (
    "def study_to_tensor(study_dir, series_rows, n_series=3, "
    "laterality=False, img_size=224):"
)
NEW_SIG = (
    "def study_to_tensor(study_dir, series_rows, n_series=3, "
    "laterality=False, img_size=224, phase=0.0):"
)
OLD_CALL = "vol = load_series_dicom(sdir, img_size=size).astype(np.float32) / 255.0"
NEW_CALL = "vol = load_series_dicom(sdir, img_size=size, phase=phase).astype(np.float32) / 255.0"

OLD_LOOP = """print(len(test), 'etudes | sans TTA | pas de flip')
for t, w in zip(TARGETS, MIX):
    if abs(sum(w) - 1.0) > 1e-6 or min(w) < -1e-9 or len(w) != 3:
        raise RuntimeError((t, w))
    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f}')


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
        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)
        prob = (
            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h
        ).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(test)} | {(time.time()-t0)/60:.1f} min', flush=True)
"""

NEW_LOOP = """print(len(test), 'etudes | sans TTA | dephasage x2 | pas de flip')
print('dephasage       : phase 0 (train) + demi-pas 0.5, moyenne des sigmoides')
for t, w in zip(TARGETS, MIX):
    if abs(sum(w) - 1.0) > 1e-6 or min(w) < -1e-9 or len(w) != 3:
        raise RuntimeError((t, w))
    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f}')


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
        acc = []
        for phase in (0.0, 0.5):
            vol_224 = study_to_tensor(
                sdir, srows, n_series=6, laterality=False, img_size=224, phase=phase)
            vol_vit = study_to_tensor(
                sdir, srows, n_series=6, laterality=False, img_size=320, phase=phase)
            if vol_vit.shape[-1] != 320 or vol_224.shape[-1] != 224:
                raise RuntimeError((vol_vit.shape, vol_224.shape))
            if vol_vit.shape[1] != N_SLICES or vol_224.shape[1] != N_SLICES:
                raise RuntimeError((vol_vit.shape, vol_224.shape))
            x_224 = torch.from_numpy(vol_224).unsqueeze(0).to(DEVICE)
            x_vit = torch.from_numpy(vol_vit).unsqueeze(0).to(DEVICE)
            p_vit = torch.sigmoid(_forward(model_vit, x_vit).float())
            p_v20b = torch.sigmoid(_forward(model_v20b, x_224).float())
            p_v20h = torch.sigmoid(_forward(model_v20h, x_224).float())
            w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)
            acc.append(w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h)
        prob = torch.stack(acc, dim=0).mean(dim=0).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(test)} | {(time.time()-t0)/60:.1f} min', flush=True)
"""


def _phase_index(n_band: int, phase: float) -> np.ndarray:
    last = float(n_band - 1)
    base = np.linspace(0.0, last, N_SLICES)
    if float(phase) == 0.0:
        return base.round().astype(int)
    step = last / (N_SLICES - 1)
    pos = np.clip(base + float(phase) * step, 0.0, last)
    return pos.round().astype(int)


def _check_phase() -> None:
    for n in range(1, 80):
        old = np.linspace(0, n - 1, N_SLICES).round().astype(int)
        new = _phase_index(n, 0.0)
        if not np.array_equal(old, new):
            raise SystemExit(f"phase 0 diverge pour n={n}")
        shifted = _phase_index(n, 0.5)
        if shifted.min() < 0 or shifted.max() > n - 1 or shifted.shape != (N_SLICES,):
            raise SystemExit(f"phase 0.5 hors bande n={n}")
    if np.array_equal(_phase_index(30, 0.0), _phase_index(30, 0.5)):
        raise SystemExit("le demi-pas ne deplace aucune coupe sur n=30")


def _replace(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        raise SystemExit(f"marqueur absent : {label}")
    return src.replace(old, new, 1)


def build() -> Path:
    _check_phase()
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    md = "".join(nb["cells"][0]["source"])
    md = md.replace(
        "# RSNA Knee — rsna-knee-ens-oai3-h025-mix3\n",
        "# RSNA Knee — rsna-knee-ens-oai3-h025-mix3-phase\n",
        1,
    )
    old_head = "Notebook **self-contained** (internet off, **T4**). **Sans TTA.**\n"
    new_head = (
        "Notebook **self-contained** (internet off, **T4**). **Sans TTA** geometrique.\n"
        "\n"
        "Dephasage des 24 coupes, bande 20-80 % : phase 0 = linspace d'entrainement, "
        "phase 0.5 = demi-pas (indices bornes dans la bande). "
        "Moyenne des deux sigmoides, puis le mix. Pas de flip.\n"
    )
    if old_head not in md:
        raise SystemExit("entete markdown absent")
    nb["cells"][0]["source"] = [md.replace(old_head, new_head, 1)]

    pre = "".join(nb["cells"][2]["source"])
    pre = _replace(pre, OLD_LOAD, NEW_LOAD, "load")
    pre = _replace(pre, OLD_BAND, NEW_BAND, "band")
    pre = _replace(pre, OLD_SIG, NEW_SIG, "sig")
    pre = _replace(pre, OLD_CALL, NEW_CALL, "call")
    if "np.linspace(0, band.shape[0] - 1, N_SLICES)" in pre:
        raise SystemExit("linspace d'origine encore present")
    nb["cells"][2]["source"] = [pre]

    infer_i = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if "vol_224 = study_to_tensor" in text:
            infer_i = i
            break
    if infer_i is None:
        raise SystemExit("cellule inference absente")
    infer = _replace("".join(nb["cells"][infer_i]["source"]), OLD_LOOP, NEW_LOOP, "loop")
    nb["cells"][infer_i]["source"] = [infer]

    blob = "\n".join("".join(c["source"]) for c in nb["cells"])
    if "TTA_VARIANTS" in blob or "laterality=True" in blob:
        raise SystemExit("TTA ou flip")
    if "KneeNetCoat" in blob or "coat_a" in blob:
        raise SystemExit("coat present")
    for needle in (
        "dephasage x2",
        "phase=phase",
        "_phase_index",
        "swa-1-2-3",
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop130_dino_ep11_oof_v3b_full_oai3",
        "(0.11, 0.38, 0.51),  # ACL",
        "sans plancher, CoAt exclu",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))

    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
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
        "dataset_sources": [
            "sultent/rsna-knee-vit-k320-oai3-swa",
            "sultent/rsna-knee-v20b-dino-oai3-swa",
            "sultent/rsna-knee-v20h-dino-oai3-swa",
        ],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("ok", PUSH)
    return PUSH


if __name__ == "__main__":
    build()
