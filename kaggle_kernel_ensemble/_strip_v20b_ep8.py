# -*- coding: utf-8 -*-
"""Strip v20b oofmask full epoch 8 (pas le distil qui a ecrase v20b_ep8.pt)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_v20b import KneeNet as KneeNetV20b  # noqa: E402

SRC = ROOT / "checkpoints" / "v20b_crop130_oofmask_full" / "epoch8.pt"
DST = ROOT / "kaggle_dataset_v20b" / "_upload_ep8" / "v20b_oofmask_ep8.pt"
KEEP = ("model", "config", "epoch", "val_macro_auc", "gold_macro_auc")
GOLD = 0.9593223634133367


def main() -> None:
    print("load", SRC, SRC.stat().st_size, flush=True)
    ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(
        "epoch", ckpt.get("epoch"),
        "run", cfg.get("run_name"),
        "model", cfg.get("model"),
        "lat", cfg.get("laterality"),
        "gold", ckpt.get("gold_macro_auc"),
        flush=True,
    )
    if cfg.get("run_name") != "v20b_crop130_oofmask_full":
        raise SystemExit(f"run inattendu {cfg.get('run_name')}")
    if int(ckpt.get("epoch", -1)) != 8:
        raise SystemExit(f"epoch {ckpt.get('epoch')}")
    if cfg.get("laterality"):
        raise SystemExit("v20b full ne doit pas avoir laterality")
    if str(cfg.get("model")) != "kneenet_v20b":
        raise SystemExit(cfg.get("model"))
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    out["gold_macro_auc"] = GOLD
    DST.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, DST)
    print("wrote", DST, DST.stat().st_size, flush=True)
    model = KneeNetV20b(
        pretrained=False,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=int(cfg.get("encode_chunk", 8)),
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    )
    model.load_state_dict(out["model"], strict=True)
    print("load strict OK n_series", model.n_series, flush=True)


if __name__ == "__main__":
    main()
