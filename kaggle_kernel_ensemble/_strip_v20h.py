# -*- coding: utf-8 -*-
"""Retire l'optimizer des snapshots v20h oofmask+lat."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_v20h import KneeNet as KneeNetV20h  # noqa: E402

RUN = ROOT / "checkpoints" / "v20h_crop130_oofmask_v20b_lat_full"
UPLOAD = ROOT / "kaggle_dataset_v20h_lat" / "_upload"
KEEP = ("model", "config", "epoch", "val_macro_auc", "gold_macro_auc")
EPOCHS = (7, 10)


def _gold_by_epoch(run: Path) -> dict[int, float]:
    hist = json.loads((run / "history.json").read_text(encoding="utf-8"))
    out = {}
    for rec in hist:
        ep = int(rec["epoch"])
        g = rec.get("gold_macro_auc")
        if g is not None:
            out[ep] = float(g)
    return out


def strip_ckpt(src: Path, dst: Path, gold: float | None) -> dict:
    print("load", src, "size", src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(
        "  keys", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt),
        "epoch", ckpt.get("epoch"),
        "gold ckpt", ckpt.get("gold_macro_auc"),
        "gold hist", gold,
        "val", ckpt.get("val_macro_auc"),
        "laterality", cfg.get("laterality"),
        "model", cfg.get("model"),
        "n_series", cfg.get("n_series"),
        "dropout", cfg.get("dropout"),
        flush=True,
    )
    if not cfg.get("laterality"):
        raise SystemExit(f"{src} sans laterality=true dans config")
    if str(cfg.get("model")) != "kneenet_v20h":
        raise SystemExit(f"{src} model={cfg.get('model')!r}, attendu kneenet_v20h")
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    if gold is not None:
        out["gold_macro_auc"] = gold
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print("  wrote", dst, dst.stat().st_size, flush=True)
    return out


def verify(sd: dict) -> None:
    cfg = sd["config"]
    model = KneeNetV20h(
        pretrained=False,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=int(cfg.get("encode_chunk", 8)),
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    )
    model.load_state_dict(sd["model"], strict=True)
    print(
        "  load strict OK n_series", model.n_series,
        "epoch", sd.get("epoch"),
        "gold", sd.get("gold_macro_auc"),
        flush=True,
    )
    del model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, nargs="*", default=list(EPOCHS))
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--out", type=Path, default=UPLOAD)
    args = p.parse_args()
    run = args.run if args.run.is_absolute() else (ROOT / args.run)
    out_dir = args.out if args.out.is_absolute() else (ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    golds = _gold_by_epoch(run)
    for ep in args.epoch:
        src = run / f"epoch{ep}.pt"
        dst = out_dir / f"v20h_ep{ep}.pt"
        if not src.is_file():
            raise FileNotFoundError(src)
        sd = strip_ckpt(src, dst, golds.get(ep))
        verify(sd)
        del sd
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
