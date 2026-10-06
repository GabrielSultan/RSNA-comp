# -*- coding: utf-8 -*-
"""Retire l'optimizer d'un snapshot ViT-k v3b oofmask full."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_vit_e import KneeNet as KneeNetVitE  # noqa: E402

RUN = ROOT / "checkpoints" / "vit_k_llrd08_v3b_oofmask_full"
RUN_NAME = "vit_k_llrd08_v3b_oofmask_full"
KEEP = ("model", "config", "epoch", "val_macro_auc", "gold_macro_auc")


def gold_of(epoch: int) -> float:
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    for rec in hist:
        if int(rec["epoch"]) == epoch and rec.get("gold_macro_auc") is not None:
            return float(rec["gold_macro_auc"])
    raise SystemExit(f"gold absent pour epoch {epoch}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, required=True)
    args = p.parse_args()
    epoch = args.epoch
    src = RUN / f"epoch{epoch}.pt"
    dst = (
        ROOT / "kaggle_dataset_vit_k" / f"_upload_v3b_ep{epoch}" / f"vit_k_v3b_ep{epoch}.pt"
    )
    gold = gold_of(epoch)
    print("load", src, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(
        "epoch", ckpt.get("epoch"),
        "run", cfg.get("run_name"),
        "model", cfg.get("model"),
        "lat", cfg.get("laterality"),
        "gold hist", round(gold, 4),
        "gold ckpt", ckpt.get("gold_macro_auc"),
        flush=True,
    )
    if cfg.get("run_name") != RUN_NAME:
        raise SystemExit(cfg.get("run_name"))
    if int(ckpt.get("epoch", -1)) != epoch:
        raise SystemExit(ckpt.get("epoch"))
    if cfg.get("laterality"):
        raise SystemExit("vit-k ne doit pas avoir laterality")
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    out["gold_macro_auc"] = gold
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print("wrote", dst, dst.stat().st_size, flush=True)
    del ckpt
    model = KneeNetVitE(
        pretrained=False,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=int(cfg.get("encode_chunk", 4)),
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    )
    model.load_state_dict(out["model"], strict=True)
    print("load strict OK", flush=True)
    meta = {
        "title": f"RSNA Knee ViT-k v3b oofmask ep{epoch}",
        "id": f"sultent/rsna-knee-vit-k-v3b-ep{epoch}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dst.parent / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
