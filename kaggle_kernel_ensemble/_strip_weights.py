# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_v20b import KneeNet as KneeNetV20b  # noqa: E402
from src.models.kneenet_vit_e import KneeNet as KneeNetVitE  # noqa: E402

UPLOAD = ROOT / "kaggle_dataset_ensemble" / "_upload"
KEEP = ("model", "config", "epoch", "val_macro_auc", "gold_macro_auc")


def strip_ckpt(src: Path, dst: Path) -> dict:
    print("load", src, "size", src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    print(
        "  keys", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt),
        "epoch", ckpt.get("epoch"),
        "gold", ckpt.get("gold_macro_auc"),
        "val", ckpt.get("val_macro_auc"),
        flush=True,
    )
    cfg = ckpt.get("config", {})
    print(
        "  model", cfg.get("model"),
        "n_series", cfg.get("n_series"),
        "dropout", cfg.get("dropout"),
        "encode_chunk", cfg.get("encode_chunk"),
        flush=True,
    )
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    torch.save(out, dst)
    print("  wrote", dst, dst.stat().st_size, flush=True)
    return out


def main() -> None:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    v20_src = (
        ROOT / "checkpoints" / "subset_v20b_distil_vite_lam03_frac025" / "best.pt"
    )
    vite_src = ROOT / "checkpoints" / "vit_e_crop130_oofmask_full" / "epoch7.pt"
    v20_dst = UPLOAD / "v20b_ep8.pt"
    vite_dst = UPLOAD / "vit_e_ep7.pt"

    tmp_v20 = UPLOAD / "_tmp_v20b_distil_full.pt"
    shutil.copy2(v20_src, tmp_v20)
    print("copied tmp v20b", tmp_v20.stat().st_size, flush=True)

    sd20 = strip_ckpt(tmp_v20, v20_dst)
    tmp_v20.unlink(missing_ok=True)

    if vite_dst.exists() and vite_dst.stat().st_size > 0:
        print("reuse stripped vit_e", vite_dst, vite_dst.stat().st_size, flush=True)
        sde = torch.load(vite_dst, map_location="cpu", weights_only=False)
    else:
        tmp_vite = UPLOAD / "_tmp_vit_e_ep7_full.pt"
        shutil.copy2(vite_src, tmp_vite)
        print("copied tmp vite", tmp_vite.stat().st_size, flush=True)
        sde = strip_ckpt(tmp_vite, vite_dst)
        tmp_vite.unlink(missing_ok=True)

    print("--- verify load ---", flush=True)
    cfg20 = sd20["config"]
    m20 = KneeNetV20b(
        pretrained=False,
        n_series=int(cfg20.get("n_series", 6)),
        dropout=float(cfg20.get("dropout", 0.4)),
        encode_chunk=int(cfg20.get("encode_chunk", 8)),
        head_dim=int(cfg20.get("head_dim", 512)),
        lstm_hidden=int(cfg20.get("lstm_hidden", 256)),
    )
    m20.load_state_dict(sd20["model"], strict=True)
    print("v20b load strict OK n_series", m20.n_series, flush=True)
    del m20

    cfge = sde["config"]
    me = KneeNetVitE(
        pretrained=False,
        n_series=int(cfge.get("n_series", 6)),
        dropout=float(cfge.get("dropout", 0.4)),
        encode_chunk=int(cfge.get("encode_chunk", 4)),
        head_dim=int(cfge.get("head_dim", 512)),
        lstm_hidden=int(cfge.get("lstm_hidden", 256)),
    )
    me.load_state_dict(sde["model"], strict=True)
    print("vit_e load strict OK n_series", me.n_series, flush=True)
    del me
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
