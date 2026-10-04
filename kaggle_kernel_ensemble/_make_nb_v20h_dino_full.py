# -*- coding: utf-8 -*-
"""Poids + notebook T4 : v20h DINO ep11 full oofmask v3b, sans flip L/R."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_v20h import KneeNet as KneeNetV20h  # noqa: E402

KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
RUN = ROOT / "checkpoints" / "v20h_crop130_dino_ep11_oof_v3b_full"
RUN_NAME = "v20h_crop130_dino_ep11_oof_v3b_full"
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


def _ids(epoch: int) -> tuple[Path, str, str, str]:
    upload = ROOT / "kaggle_dataset_v20h_lat" / f"_upload_dino_full_ep{epoch}"
    weight = f"v20h_dino_full_ep{epoch}.pt"
    dataset = f"sultent/rsna-knee-v20h-dino-full-ep{epoch}"
    slug = f"rsna-knee-v20h-dino-full-ep{epoch}-solo"
    return upload, weight, dataset, slug


def strip_epoch(epoch: int) -> tuple[float, float | None]:
    src = RUN / f"epoch{epoch}.pt"
    if not src.is_file():
        raise FileNotFoundError(src)
    upload, weight, dataset, _slug = _ids(epoch)
    upload.mkdir(parents=True, exist_ok=True)
    dest = upload / weight
    print("load", src.name, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    if str(cfg.get("model")) != "kneenet_v20h":
        raise SystemExit(f"model={cfg.get('model')!r}")
    if cfg.get("run_name") != RUN_NAME:
        raise SystemExit(f"run inattendu {cfg.get('run_name')}")
    if cfg.get("laterality"):
        raise SystemExit("ce checkpoint a laterality=true, notebook a revoir")
    got = int(ckpt.get("epoch"))
    if got != epoch:
        raise SystemExit(f"{src.name} epoch {got}, attendu {epoch}")
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    rec = next(r for r in hist if int(r["epoch"]) == epoch)
    gold = float(rec["gold_macro_auc"])
    raw_val = ckpt.get("val_macro_auc")
    val = float(raw_val) if isinstance(raw_val, (int, float)) and raw_val > 0 else None
    cfg["laterality"] = False
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    out["config"] = cfg
    out["gold_macro_auc"] = gold
    if val is None:
        out.pop("val_macro_auc", None)
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
        "epoch", epoch, "gold", round(gold, 4),
        "laterality", False, flush=True,
    )
    meta = {
        "title": f"RSNA Knee v20h DINO full ep{epoch}",
        "id": dataset,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (upload / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return gold, val


def write_notebook(epoch: int, gold: float) -> Path:
    mh = _load_maker()
    upload, weight, dataset, slug = _ids(epoch)
    local = f"../kaggle_dataset_v20h_lat/_upload_dino_full_ep{epoch}"
    notebook = mh.build_notebook(f"epoch {epoch}", gold, weight, dataset, local)
    md = "".join(notebook["cells"][0]["source"])
    md = md.replace(
        "v20h oofmask + laterality L",
        "v20h DINO ep11 full oofmask v3b, sans flip L/R",
    ).replace(
        "DICOM test → crop 130 mm + bande 20–80 % → **canon laterality L** (flip des\n"
        "genoux droits : Coronal/Axial axe W, Sagittal axe D) → volumes (24, 224, 224)\n"
        "→ **KneeNet v20h** (`v20h_crop130_oofmask_v20b_lat_full`",
        "DICOM test → crop 130 mm + bande 20–80 % → volumes (24, 224, 224),\n"
        "**sans flip** gauche/droite → **KneeNet v20h** "
        f"(`{RUN_NAME}`, encodeur DINO",
    ).replace(
        "Pas d'ensemble ViT-e : ce checkpoint a ete train **avec** flip L/R.",
        "Full train apres masque OOF, labels Coolin targets_v3b_oofmask, "
        "gold dans le train (poids x4). "
        f"Snapshot epoch {epoch} (gold in-sample {gold:.3f}). "
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
        f"print('laterality      :', LATERALITY_CANON_L, '(pas de flip)')\n"
        f"print('modele          : v20h dino full ep{epoch}",
    ).replace(
        "dataset v20h lat attache",
        f"dataset v20h dino full ep{epoch} attache",
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
        f"print(f'{{len(test)}} etudes a predire | TTA x{{len(TTA_VARIANTS)}} | v20h dino full ep{epoch}')\n",
    )
    _set_source(notebook["cells"][5], infer)
    push = KERNEL_DIR / f"_push_v20h_dino_full_ep{epoch}"
    push.mkdir(parents=True, exist_ok=True)
    text = json.dumps(notebook, indent=1, ensure_ascii=False)
    (push / "submit.ipynb").write_text(text, encoding="utf-8")
    (KERNEL_DIR / f"submit_v20h_dino_full_ep{epoch}.ipynb").write_text(
        text, encoding="utf-8"
    )
    (push / "kernel-metadata.json").write_text(
        json.dumps(mh._metadata(slug, dataset), indent=2) + "\n",
        encoding="utf-8",
    )
    print("ecrit", push / "submit.ipynb", flush=True)
    return push


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, nargs="+", required=True)
    args = p.parse_args()
    for epoch in args.epoch:
        gold, _val = strip_epoch(epoch)
        write_notebook(epoch, gold)


if __name__ == "__main__":
    main()
