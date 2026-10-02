# -*- coding: utf-8 -*-
"""Poids + notebook T4 : v20h DINO ep11, OOF v3b fold 0, sans flip L/R."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_v20h import KneeNet as KneeNetV20h  # noqa: E402

KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
SRC = ROOT / "checkpoints" / "v20h_crop130_dino_ep11_oof_v3b_fold0" / "best.pt"
UPLOAD = ROOT / "kaggle_dataset_v20h_lat" / "_upload_dino_ep11_oof_v3b_f0"
WEIGHT = "v20h_dino_ep11_oof_v3b_f0.pt"
DATASET = "sultent/rsna-knee-v20h-dino-oof-v3b-f0"
SLUG = "rsna-knee-v20h-dino-oof-v3b-f0-solo"
RUN_NAME = "v20h_crop130_dino_ep11_oof_v3b_fold0"
KEEP = ("model", "config", "epoch", "val_macro_auc", "gold_macro_auc")


def _load_maker():
    p = KERNEL_DIR / "_make_nb_v20h.py"
    spec = importlib.util.spec_from_file_location("make_nb_v20h", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set_source(cell: dict, text: str) -> None:
    src = text if text.endswith("\n") else text + "\n"
    cell["source"] = [line + "\n" for line in src.rstrip("\n").split("\n")]


def strip_weights() -> tuple[Path, int, float, float]:
    if not SRC.is_file():
        raise FileNotFoundError(SRC)
    UPLOAD.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD / WEIGHT
    print("load", SRC, SRC.stat().st_size, flush=True)
    ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    if str(cfg.get("model")) != "kneenet_v20h":
        raise SystemExit(f"model={cfg.get('model')!r}")
    if cfg.get("run_name") != RUN_NAME:
        raise SystemExit(f"run inattendu {cfg.get('run_name')}")
    if int(cfg.get("oof_fold", -1)) != 0:
        raise SystemExit(f"oof_fold={cfg.get('oof_fold')!r}")
    if cfg.get("laterality"):
        raise SystemExit("ce checkpoint a laterality=true, notebook a revoir")
    hist = json.loads(
        (SRC.parent / "history.json").read_text(encoding="utf-8")
    )
    epoch = int(ckpt.get("epoch"))
    rec = next(r for r in hist if int(r["epoch"]) == epoch)
    gold = float(rec["gold_macro_auc"])
    val = float(ckpt.get("val_macro_auc"))
    cfg["laterality"] = False
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    out["config"] = cfg
    out["gold_macro_auc"] = gold
    torch.save(out, dest)
    model = KneeNetV20h(
        pretrained=False,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=int(cfg.get("encode_chunk", 8)),
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    )
    model.load_state_dict(out["model"], strict=True)
    print(
        "strip", dest.name, dest.stat().st_size,
        "epoch", epoch, "val", round(val, 4), "gold", round(gold, 4),
        "laterality", False, "oof_fold", 0, flush=True,
    )
    meta = {
        "title": "RSNA Knee v20h DINO OOF v3b f0",
        "id": DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (UPLOAD / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return dest, epoch, val, gold


def write_notebook(epoch: int, val: float, gold: float) -> Path:
    mh = _load_maker()
    local = "../kaggle_dataset_v20h_lat/_upload_dino_ep11_oof_v3b_f0"
    notebook = mh.build_notebook(
        f"epoch {epoch}", gold, WEIGHT, DATASET, local,
    )
    md = "".join(notebook["cells"][0]["source"])
    md = md.replace(
        "v20h oofmask + laterality L",
        "v20h DINO ep11 OOF v3b fold 0, sans flip L/R",
    ).replace(
        "DICOM test → crop 130 mm + bande 20–80 % → **canon laterality L** (flip des\n"
        "genoux droits : Coronal/Axial axe W, Sagittal axe D) → volumes (24, 224, 224)\n"
        "→ **KneeNet v20h** (`v20h_crop130_oofmask_v20b_lat_full`",
        "DICOM test → crop 130 mm + bande 20–80 % → volumes (24, 224, 224),\n"
        "**sans flip** gauche/droite → **KneeNet v20h** "
        f"(`{RUN_NAME}`, encodeur DINO epoch 11",
    ).replace(
        "Pas d'ensemble ViT-e : ce checkpoint a ete train **avec** flip L/R.",
        "OOF fold 0, labels Coolin targets_v3b, train des folds tenus hors fold 0. "
        f"Meilleure epoch supervisee {epoch} "
        f"(val fold {val:.3f}, gold {gold:.3f}). "
        "Pas de canon laterality : le run a ete train sans flip.",
    ).replace(
        "TTA : identite + rotation ±5° + scale 0.95/1.05. **Pas de hflip** (ce n'est\n"
        "pas la canon laterality).",
        "TTA : identite + rotation ±5° + scale 0.95/1.05. Pas de flip horizontal.",
    )
    _set_source(notebook["cells"][0], md)
    cfg = "".join(notebook["cells"][1]["source"])
    cfg = cfg.replace(
        "# Ce run a ete train avec laterality: true (canon L, flip R). Obligatoire.\n"
        "LATERALITY_CANON_L = True",
        "# Ce run a ete train SANS flip L/R. Ne pas canoniser.\n"
        "LATERALITY_CANON_L = False",
    ).replace(
        "print('laterality      :', LATERALITY_CANON_L, '(flip R -> canon L)')\n"
        "print('modele          : v20h oofmask lat",
        "print('laterality      :', LATERALITY_CANON_L, '(pas de flip)')\n"
        "print('modele          : v20h dino oof v3b f0",
    ).replace(
        "dataset v20h lat attache",
        "dataset v20h dino oof v3b f0 attache",
    )
    _set_source(notebook["cells"][1], cfg)
    infer = "".join(notebook["cells"][5]["source"])
    infer = infer.replace(
        "    if not cfg.get('laterality'):\n"
        "        raise RuntimeError(f'checkpoint sans laterality=true : {weights_path}')\n",
        "    if cfg.get('laterality'):\n"
        "        raise RuntimeError(f'checkpoint avec laterality=true : {weights_path}')\n",
    ).replace(
        "assert LATERALITY_CANON_L is True\n",
        "assert LATERALITY_CANON_L is False\n",
    ).replace(
        "print(f'{len(test)} etudes a predire | TTA x{len(TTA_VARIANTS)} | v20h lat solo')\n",
        "print(f'{len(test)} etudes a predire | TTA x{len(TTA_VARIANTS)} | v20h dino oof v3b f0')\n",
    )
    _set_source(notebook["cells"][5], infer)
    push = KERNEL_DIR / "_push_v20h_dino_oof_v3b_f0"
    push.mkdir(parents=True, exist_ok=True)
    text = json.dumps(notebook, indent=1, ensure_ascii=False)
    (push / "submit.ipynb").write_text(text, encoding="utf-8")
    (KERNEL_DIR / "submit_v20h_dino_oof_v3b_f0.ipynb").write_text(
        text, encoding="utf-8"
    )
    (push / "kernel-metadata.json").write_text(
        json.dumps(mh._metadata(SLUG, DATASET), indent=2) + "\n",
        encoding="utf-8",
    )
    print("ecrit", push / "submit.ipynb", flush=True)
    return push


def main() -> None:
    _dest, epoch, val, gold = strip_weights()
    write_notebook(epoch, val, gold)


if __name__ == "__main__":
    main()
