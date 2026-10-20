# -*- coding: utf-8 -*-
"""Deux ensembles a 6 modeles, 2 T4. Uniforme, puis par finding avec avantage CNX."""
from __future__ import annotations

import importlib.util
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
SPEC = importlib.util.spec_from_file_location(
    "pub70", KERNEL / "_push_ens_cnx_pub70.py",
)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

COMPETITION = base.COMPETITION
ACCEL = base.ACCEL
POLL_S = base.POLL_S
MAX_WAIT_S = base.MAX_WAIT_S
TARGETS = base.TARGETS

# Ordre : ViT, v20b, distil145, CoAt, CNX, v20h lat 130.
HYBRIDS = [
    "subset_vit_k_llrd08_320_v3b_hybrid025",
    "subset_v20b_dino_ep11_v3b_hybrid025",
    "subset_v20h_dino_ep11_v3b_hybrid025_wd01_crop145",
    "subset_coat_a_llrd08_v3b_hybrid025",
    "subset_cnx_r7_in22k_v3b_hybrid025",
    "subset_v20h_dino_ep11_v3b_hybrid025",
]
CNX_I = 4
TEMP = 0.05
CNX_BONUS = 0.20

LOAD_LAT = """
def load_lat(weights_path):
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    if str(cfg.get('run_name')) != 'v20h_crop130_oofmask_v20b_lat_full':
        raise RuntimeError(cfg.get('run_name'))
    if str(cfg.get('model')) != 'kneenet_v20h':
        raise RuntimeError(cfg.get('model'))
    if not cfg.get('laterality'):
        raise RuntimeError('v20h lat doit etre flipped')
    _expect_swa(ckpt, 'swa-7-8-9-10-11-12', [7, 8, 9, 10, 11, 12])
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
"""


def _recs() -> list[dict]:
    out = []
    for name in HYBRIDS:
        hist = json.loads((ROOT / "checkpoints" / name / "history.json").read_text(encoding="utf-8"))
        out.append(max(hist, key=lambda r: r.get("val_macro_auc") or -1.0))
    if int(out[CNX_I]["epoch"]) != 7:
        raise SystemExit(f"cnx hybrid epoch {out[CNX_I].get('epoch')}")
    return out


def finding_rows() -> list[tuple[float, ...]]:
    """Softmax T=0.05, puis le CNX prend 20 % de la masse des autres."""
    recs = _recs()
    raw = []
    for t in TARGETS:
        auc = np.array([float(r["auc_" + t]) for r in recs], dtype=np.float64)
        z = auc / TEMP
        z = z - z.max()
        w = np.exp(z)
        w = w / w.sum()
        cnx = float(w[CNX_I] + CNX_BONUS * (1.0 - w[CNX_I]))
        others = w.copy()
        others[CNX_I] = 0.0
        others = others / others.sum() * (1.0 - cnx)
        others[CNX_I] = cnx
        raw.append(others)
    mat = np.vstack(raw)
    rnd = np.round(mat, 2)
    delta = np.round(1.0 - rnd.sum(axis=1), 2)
    rnd[:, CNX_I] = np.round(rnd[:, CNX_I] + delta, 2)
    rows = []
    for t, w in zip(TARGETS, rnd):
        got = tuple(float(v) for v in w)
        if abs(sum(got) - 1.0) > 1e-6 or min(got) < -1e-9 or len(got) != 6:
            raise SystemExit(f"mix invalide {t} {got}")
        if got[CNX_I] + 1e-9 < 1.0 / 6.0:
            raise SystemExit(f"cnx sans avantage {t} {got}")
        rows.append(got)
    return rows


def uniform_rows() -> list[tuple[float, ...]]:
    u = 1.0 / 6.0
    return [tuple([u] * 6) for _ in TARGETS]


def _mix_src(rows: list[tuple[float, ...]], title: str) -> str:
    lines = [
        f"# {title}",
        "# Ordre : ViT-k320, v20b, v20h distil, CoAt, CNX-r7, v20h lat. Somme 1.",
        "N_MODELS = 6",
        "MIX = [",
    ]
    for t, w in zip(TARGETS, rows):
        nums = ", ".join(f"{v:.10g}" for v in w)
        lines.append(f"    ({nums}),  # {t}")
    lines.append("]")
    return "\n".join(lines) + "\n"


def _md(kind: str) -> str:
    tail = (
        "3. Moyenne uniforme, un sixieme chacun."
        if kind == "uniform"
        else (
            "3. Melange par finding. Softmax T=0.05 sur les AUC du subset hybrid, "
            "puis le CNX recupere 20 % de la masse des cinq autres."
        )
    )
    return f"""# RSNA Knee — rsna-knee-ens-mix6-{kind}

**Sans TTA.** Internet off, **deux T4**. Pas de flip, sauf le canon L du v20h lat.

1. Six modeles.
   ViT-k320 : `vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3` SWA 1-3.
   Entree 320, FOV 130 mm.
   v20b DINO : `v20b_crop130_dino_ep11_oof_v3b_full_oai3` SWA 1-3.
   Entree 224, FOV 130 mm.
   v20h distil : `v20h_crop145_dino_ep11_v3b_rsna_distil_main_full` SWA 7-9.
   Entree 224, FOV 145 mm.
   CoAt-a : `coat_a_llrd08_v3b_oofmask_full_oai3` SWA 1-3.
   Entree 224, FOV 130 mm.
   CNX-r7 : `cnx_r7_in22k_v3b_oofmask_full` SWA 7-10.
   Entree 224, FOV 130 mm. Forward float32.
   v20h lat : `v20h_crop130_oofmask_v20b_lat_full` SWA 7-12.
   Entree 224, FOV 130 mm. Canon L (flip des genoux droits).
2. Chaque T4 a une copie des six modeles. Etudes paires sur cuda:0,
   impaires sur cuda:1. Une seule etude est decodee a la fois.
   Plafond 8,5 h, pour couvrir les ~1 300 etudes avant les 9 h.
{tail}
"""


CLOCK = """import time
NB_T0 = time.time()
MIX6_BUDGET_S = 8.5 * 3600
MIX5_BUDGET_S = MIX6_BUDGET_S
print('horloge depart', flush=True)
print('budget mix6 h', MIX6_BUDGET_S / 3600, '| 2 GPU | decode 1 etude', flush=True)
"""


def _apply(nb: dict, rows: list[tuple[float, ...]], title: str) -> None:
    base._replace_nb(
        nb,
        "    WEIGHTS_CNX = _find_weight('cnx_r7_swa_ep7_10.pt')\n",
        "    WEIGHTS_CNX = _find_weight('cnx_r7_swa_ep7_10.pt')\n"
        "    WEIGHTS_LAT = _find_weight('v20h_v3b_lat_swa_ep7_12.pt')\n",
        1,
    )
    base._replace_nb(
        nb,
        "    WEIGHTS_CNX = Path('kaggle_dataset_cnx/_upload_swa78910/cnx_r7_swa_ep7_10.pt')\n",
        "    WEIGHTS_CNX = Path('kaggle_dataset_cnx/_upload_swa78910/cnx_r7_swa_ep7_10.pt')\n"
        "    WEIGHTS_LAT = Path('kaggle_dataset_v20h_lat/_upload_v3b_lat_swa/v20h_v3b_lat_swa_ep7_12.pt')\n",
        1,
    )
    base._replace_nb(
        nb,
        "print('poids cnx       :', WEIGHTS_CNX, '| existe :', WEIGHTS_CNX.exists())\n",
        "print('poids cnx       :', WEIGHTS_CNX, '| existe :', WEIGHTS_CNX.exists())\n"
        "print('poids v20h lat  :', WEIGHTS_LAT, '| existe :', WEIGHTS_LAT.exists())\n",
        1,
    )
    base._replace_nb(
        nb,
        "model_cnx = load_cnx(WEIGHTS_CNX)\n",
        "model_cnx = load_cnx(WEIGHTS_CNX)\n" + LOAD_LAT,
        1,
    )
    base._replace_nb(nb, "or len(w) != 5:", "or len(w) != 6:", 1)
    base._replace_nb(
        nb,
        "    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f} cnx {w[4]:.2f}')\n",
        "    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} "
        "coat {w[3]:.2f} cnx {w[4]:.2f} lat {w[5]:.2f}')\n",
        1,
    )
    old_mix_head = (
        "# Ordre : ViT-k320, v20b, v20h distil, CoAt-a, CNX-r7. Somme 1, plancher 0.\n"
        "N_MODELS = 5\n"
    )
    src_mix = _mix_src(rows, title)
    # Le bloc MIX a 5 colonnes est remplace entierement.
    found = 0
    for cell in nb["cells"]:
        src = base._text(cell)
        if old_mix_head not in src:
            continue
        start = src.index(old_mix_head)
        end = src.index("]\n", start)
        src = src[:start] + src_mix + src[end + 2:]
        base._set(cell, src)
        found += 1
    if found != 1:
        raise SystemExit(f"bloc MIX x{found}")
    base._replace_nb(
        nb,
        "model_cnx, model_cnx_b = _split_gpu(model_cnx)\n",
        "model_cnx, model_cnx_b = _split_gpu(model_cnx)\n"
        "model_lat, model_lat_b = _split_gpu(model_lat)\n",
        1,
    )
    base._replace_nb(
        nb,
        "'cuda:0': (model_vit, model_v20b, model_v20h, model_coat, model_cnx),\n"
        "    'cuda:1': (model_vit_b, model_v20b_b, model_v20h_b, model_coat_b, model_cnx_b),",
        "'cuda:0': (model_vit, model_v20b, model_v20h, model_coat, model_cnx, model_lat),\n"
        "    'cuda:1': (model_vit_b, model_v20b_b, model_v20h_b, model_coat_b, model_cnx_b, model_lat_b),",
        1,
    )
    base._replace_nb(
        nb,
        "        vol_224 = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)\n"
        "        vol_vit = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=320)\n",
        "        vol_224 = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)\n"
        "        vol_lat = study_to_tensor(sdir, srows, n_series=6, laterality=True, img_size=224)\n"
        "        vol_vit = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=320)\n",
        1,
    )
    base._replace_nb(
        nb,
        "        return vol_224, vol_vit, vol_h\n",
        "        return vol_224, vol_lat, vol_vit, vol_h\n",
        1,
    )
    base._replace_nb(
        nb,
        "    m_vit, m_v20b, m_v20h, m_coat, m_cnx = packs[dev]\n",
        "    m_vit, m_v20b, m_v20h, m_coat, m_cnx, m_lat = packs[dev]\n",
        1,
    )
    base._replace_nb(
        nb,
        "        vol_224, vol_vit, vol_h = _decode_study(uid)\n"
        "        if vol_vit.shape[-1] != 320 or vol_224.shape[-1] != 224 or vol_h.shape[-1] != 224:\n"
        "            raise RuntimeError((uid, vol_vit.shape, vol_224.shape, vol_h.shape))\n"
        "        x_224 = torch.from_numpy(vol_224).unsqueeze(0).to(dev)\n"
        "        x_vit = torch.from_numpy(vol_vit).unsqueeze(0).to(dev)\n"
        "        x_h = torch.from_numpy(vol_h).unsqueeze(0).to(dev)\n"
        "        del vol_224, vol_vit, vol_h\n",
        "        vol_224, vol_lat, vol_vit, vol_h = _decode_study(uid)\n"
        "        if vol_vit.shape[-1] != 320 or vol_224.shape[-1] != 224 or vol_lat.shape[-1] != 224 or vol_h.shape[-1] != 224:\n"
        "            raise RuntimeError((uid, vol_vit.shape, vol_224.shape, vol_lat.shape, vol_h.shape))\n"
        "        x_224 = torch.from_numpy(vol_224).unsqueeze(0).to(dev)\n"
        "        x_lat = torch.from_numpy(vol_lat).unsqueeze(0).to(dev)\n"
        "        x_vit = torch.from_numpy(vol_vit).unsqueeze(0).to(dev)\n"
        "        x_h = torch.from_numpy(vol_h).unsqueeze(0).to(dev)\n"
        "        del vol_224, vol_lat, vol_vit, vol_h\n",
        1,
    )
    base._replace_nb(
        nb,
        "            p_coat = torch.sigmoid(_forward_on(m_coat, x_224).float())\n"
        "            with torch.autocast('cuda', enabled=False):\n"
        "                p_cnx = torch.sigmoid(m_cnx(x_224.float()).float())\n"
        "            w = torch.tensor(MIX, dtype=torch.float32, device=dev)\n"
        "            prob = (\n"
        "                w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h\n"
        "                + w[:, 3] * p_coat + w[:, 4] * p_cnx\n"
        "            ).float().cpu().numpy()[0]\n"
        "            del p_vit, p_v20b, p_v20h, p_coat, p_cnx, w\n"
        "        del x_224, x_vit, x_h\n",
        "            p_coat = torch.sigmoid(_forward_on(m_coat, x_224).float())\n"
        "            p_lat = torch.sigmoid(_forward_on(m_lat, x_lat).float())\n"
        "            with torch.autocast('cuda', enabled=False):\n"
        "                p_cnx = torch.sigmoid(m_cnx(x_224.float()).float())\n"
        "            w = torch.tensor(MIX, dtype=torch.float32, device=dev)\n"
        "            prob = (\n"
        "                w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h\n"
        "                + w[:, 3] * p_coat + w[:, 4] * p_cnx + w[:, 5] * p_lat\n"
        "            ).float().cpu().numpy()[0]\n"
        "            del p_vit, p_v20b, p_v20h, p_coat, p_cnx, p_lat, w\n"
        "        del x_224, x_lat, x_vit, x_h\n",
        1,
    )
    base._replace_nb(
        nb,
        "del model_vit_b, model_v20b_b, model_v20h_b, model_coat_b, model_cnx_b\n",
        "del model_vit_b, model_v20b_b, model_v20h_b, model_coat_b, model_cnx_b, model_lat_b\n"
        "del model_lat\n",
        1,
    )
    base._replace_nb(
        nb,
        "print(len(test), 'etudes | sans TTA | pas de flip | FOV vit/v20b/coat/cnx 130 mm, v20h 145 mm | cnx forward fp32 | 2 GPU')\n",
        "print(len(test), 'etudes | sans TTA | FOV vit/v20b/coat/cnx/lat 130 mm, distil 145 mm | cnx fp32 | lat canon L | 2 GPU')\n",
        1,
    )


def _build(kind: str, rows: list[tuple[float, ...]]) -> tuple[Path, str]:
    slug = f"rsna-knee-ens-mix6-{kind}"
    push = KERNEL / f"_push_ens_mix6_{kind}"
    ours = json.loads(base.OURS.read_text(encoding="utf-8"))
    if len(ours["cells"]) != 9:
        raise SystemExit("notebook source inattendu")
    base._patch_v20h(ours)
    base._patch_cnx(ours, base.mix_rows())
    ours["cells"].insert(7, base._cnx_cell())
    title = "uniforme 1/6" if kind == "uniform" else "par finding, avantage CNX"
    cells = [base._md(_md(kind)), base._code(CLOCK)]
    cells.extend(ours["cells"][1:9])
    cells.append(base._code(base.SAVE))
    cells.append(base._code(
        "from pathlib import Path as _Sub\n"
        "out = pred.copy()\n"
        "_cands = list(_Sub('/kaggle/input').glob('**/sample_submission.csv')) "
        "if _Sub('/kaggle/input').exists() else []\n"
        "if _cands:\n"
        "    _sample = pd.read_csv(_cands[0])\n"
        "    out = out.set_index('StudyInstanceUID').reindex(_sample.StudyInstanceUID).reset_index()\n"
        "    out = out[list(_sample.columns)]\n"
        "    _prior = out[TARGETS].mean()\n"
        "    out[TARGETS] = out[TARGETS].fillna(_prior)\n"
        "_dest = _Sub('/kaggle/working/submission.csv') if _Sub('/kaggle/working').exists() else _Sub('submission.csv')\n"
        "out.to_csv(_dest, index=False)\n"
        "if _Sub('submission.csv').resolve() != _dest.resolve():\n"
        "    out.to_csv('submission.csv', index=False)\n"
        "print('SUBMISSION_OK', _dest, 'lignes', len(out), flush=True)\n"
        "assert not out[TARGETS].isna().any().any()\n"
    ))
    nb = {"cells": cells}
    _apply(nb, rows, title)
    for cell in cells:
        src = base._text(cell)
        if cell["cell_type"] != "code" or src.lstrip().startswith("%%"):
            continue
        ast_ok = True
        try:
            import ast
            ast.parse(src)
        except SyntaxError as e:
            ast_ok = False
            raise SystemExit(f"syntaxe {kind}: {e}") from e
        if not ast_ok:
            raise SystemExit(kind)
    blob = "\n".join(base._text(c) for c in cells)
    for needle in (
        "MIX6_BUDGET_S = 8.5 * 3600",
        "laterality=True",
        "v20h_crop130_oofmask_v20b_lat_full",
        "v20h_v3b_lat_swa_ep7_12.pt",
        "w[:, 5] * p_lat",
        "gpus mix 2 | cuda:0 cuda:1",
        "model_lat_b",
        "N_MODELS = 6",
    ):
        if needle not in blob:
            raise SystemExit(f"{kind} manque {needle}")
    if blob.count("N_MODELS = 5") or "w[:, 4] * p_cnx\n" in blob and "w[:, 5]" not in blob:
        raise SystemExit(f"{kind} residu 5 modeles")
    out = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }
    push.mkdir(parents=True, exist_ok=True)
    (push / "submit.ipynb").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": f"sultent/{slug}",
        "title": slug,
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
            "sultent/rsna-knee-vit-k320-oai3-swa",
            "sultent/rsna-knee-v20b-dino-oai3-swa",
            "sultent/rsna-knee-v20h-crop145-distil-swa789",
            "sultent/rsna-knee-coat-a-oai3-swa",
            "sultent/rsna-knee-cnx-r7-swa78910",
            "sultent/rsna-knee-v20h-v3b-lat-swa",
        ],
        "category_ids": [],
    }
    (push / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    msg = (
        "ens 6 modeles ViT-k320 OAI3 SWA 1-3 + v20b OAI SWA + v20h distil crop145 SWA 7-9 "
        "+ CoAt OAI SWA + CNX-r7 SWA 7-10 + v20h lat crop130 SWA 7-12, "
        + ("mix uniforme 1/6" if kind == "uniform" else "mix par finding avantage CNX")
        + ", sans TTA, 2 T4, canon L seulement sur v20h lat"
    )
    (push / "_message.txt").write_text(msg + "\n", encoding="utf-8")
    print("ecrit", push, "cells", len(cells), flush=True)
    for t, w in zip(TARGETS, rows):
        print(f"  {kind} {t:<18} " + " ".join(f"{v:.2f}" for v in w), flush=True)
    return push, msg


def kaggle(*args: str) -> str:
    return base.kaggle(*args)


def push_one(push: Path) -> str:
    while True:
        try:
            out = kaggle("kernels", "push", "-p", str(push), "--accelerator", ACCEL, "-t", "32400")
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if base._quota(text):
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            if base._transient(text):
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
                continue
            raise
        print(out, flush=True)
        if base._quota(out):
            print("quota GPU, retry 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        version = m.group(1)
        (push / "_version.txt").write_text(version + "\n", encoding="utf-8")
        print("kernel version", version, flush=True)
        return version


def wait_one(slug: str) -> None:
    kid = f"sultent/{slug}"
    t0 = time.time()
    seen = False
    while True:
        try:
            raw = kaggle("kernels", "status", kid).strip()
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if base._transient(text) or "timed out" in text.lower() or "retries exceeded" in text.lower():
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
                continue
            raise
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit(f"kernel en erreur {slug}")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout kernel {slug}")
        time.sleep(POLL_S)


def submit_one(slug: str, push: Path, msg: str, version: str) -> None:
    out_dir = KERNEL / f"_out_ens_mix6_{slug.split('-')[-1]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", f"sultent/{slug}", "-p", str(out_dir)), flush=True)
    log = next(p for p in out_dir.glob("*.log") if "cufile" not in p.name.lower() and slug in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10",
        "v20h lat v20h_crop130_oofmask_v20b_lat_full epoch swa-7-8-9-10-11-12",
        "lat True",
        "gpus mix 2 | cuda:0 cuda:1",
        "MIX5_COVERAGE 3/3",
        "SUBMISSION_OK",
        "lignes 3",
        "sans TTA",
    ):
        if needle not in text:
            raise SystemExit(f"{slug} log sans {needle}")
    if text.count("lat True") != 1:
        raise SystemExit(f"{slug} laterality inattendue")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    waited = False
    while True:
        print("submit", slug, "version", version, flush=True)
        try:
            resp = api.competition_submit_code(
                "submission.csv", msg, COMPETITION, f"sultent/{slug}", int(version), quiet=True,
            )
        except Exception as e:
            http = getattr(e, "response", None)
            body = http.text if http is not None else str(e)
            print(body, flush=True)
            quota = any(s in body.lower() for s in (
                "daily submission allowance", "submission allowance",
                "too many submission", "try again tomorrow",
            ))
            if quota and not waited:
                waited = True
                now = datetime.now(timezone.utc)
                target = now.replace(hour=0, minute=5, second=0, microsecond=0)
                if now >= target:
                    target = target + timedelta(days=1)
                delay = (target - now).total_seconds()
                print(f"ATTENTE_QUOTA {delay:.0f}s", flush=True)
                time.sleep(delay + 5)
                continue
            if base._transient(body):
                time.sleep(20)
                continue
            raise
        print(resp, flush=True)
        print("SOUMIS", slug, "version", version, flush=True)
        return


def main() -> None:
    specs = [
        ("uniform", uniform_rows()),
        ("cnx", finding_rows()),
    ]
    built = []
    for kind, rows in specs:
        built.append((kind, *_build(kind, rows)))
    if "--write" in sys.argv:
        return
    for kind, push, msg in built:
        slug = f"rsna-knee-ens-mix6-{kind}"
        version = push_one(push)
        wait_one(slug)
        submit_one(slug, push, msg, version)


if __name__ == "__main__":
    main()
