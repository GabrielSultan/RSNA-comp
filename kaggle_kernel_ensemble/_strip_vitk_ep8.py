# -*- coding: utf-8 -*-
"""Strip vit_k_llrd08_det_oofmask_full epoch 8."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_vit_e import KneeNet as KneeNetVitE  # noqa: E402

SRC = ROOT / "checkpoints" / "vit_k_llrd08_det_oofmask_full" / "epoch8.pt"
DST = ROOT / "kaggle_dataset_vit_k" / "_upload_ep8" / "vit_k_ep8.pt"
KEEP = ("model", "config", "epoch", "val_macro_auc", "gold_macro_auc")
GOLD = 0.9395190056117061


def main() -> None:
    print("load", SRC, SRC.stat().st_size, flush=True)
    ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(
        "epoch", ckpt.get("epoch"),
        "run", cfg.get("run_name"),
        "model", cfg.get("model"),
        "lat", cfg.get("laterality"),
        flush=True,
    )
    if cfg.get("run_name") != "vit_k_llrd08_det_oofmask_full":
        raise SystemExit(cfg.get("run_name"))
    if int(ckpt.get("epoch", -1)) != 8:
        raise SystemExit(ckpt.get("epoch"))
    if cfg.get("laterality"):
        raise SystemExit("vit-k ne doit pas avoir laterality")
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    out["gold_macro_auc"] = GOLD
    DST.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, DST)
    print("wrote", DST, DST.stat().st_size, flush=True)
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


if __name__ == "__main__":
    main()
