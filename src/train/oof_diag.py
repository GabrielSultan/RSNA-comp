# -*- coding: utf-8 -*-
"""Etape 0 OOF : desaccord |label LLM - proba| sur un split deja hors train.

Usage :
  python -X utf8 -m src.train.oof_diag --config configs/archive/kneenet_v17_crop130.yaml
  python -X utf8 -m src.train.oof_diag --config configs/archive/kneenet_v17_crop130.yaml --split gold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.infer.inspect_attention import build_loader, load_ckpt_model
from src.train.train import evaluate, load_config


THRESHOLDS = (0.7, 0.8, 0.9)


def _summarize(y: np.ndarray, p: np.ndarray, targets: list[str],
               thresh: float = 0.8) -> dict:
    """y : labels (-1 masque, 0/1 ou mou). p : probabilites."""
    valid = y >= 0
    n_valid = int(valid.sum())
    delta = np.abs(y - p)
    disagree = valid & (delta >= thresh)
    n_dis = int(disagree.sum())
    rows = []
    for i, name in enumerate(targets):
        m = valid[:, i]
        nv = int(m.sum())
        d = disagree[:, i]
        nd = int(d.sum())
        llm1_model0 = int((d & (y[:, i] >= 0.5) & (p[:, i] < 0.5)).sum())
        llm0_model1 = int((d & (y[:, i] < 0.5) & (p[:, i] >= 0.5)).sum())
        mean_abs = float(delta[m, i].mean()) if nv else float("nan")
        rows.append({
            "target": name,
            "n_valid": nv,
            "n_disagree": nd,
            "pct": (100.0 * nd / nv) if nv else 0.0,
            "llm1_model_low": llm1_model0,
            "llm0_model_high": llm0_model1,
            "mean_abs_delta": mean_abs,
        })
    return {
        "threshold": thresh,
        "n_studies": int(y.shape[0]),
        "n_cells_valid": n_valid,
        "n_disagree": n_dis,
        "pct_cells": (100.0 * n_dis / n_valid) if n_valid else 0.0,
        "per_target": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/archive/kneenet_v17_crop130.yaml")
    parser.add_argument("--ckpt", default=None,
                        help="defaut : checkpoints/<run_name>/best.pt")
    parser.add_argument("--split", default="val", choices=("val", "gold"))
    parser.add_argument("--thresh", type=float, default=0.8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path("checkpoints") / cfg["run_name"]
    ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / "best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint introuvable : {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, epoch, val_auc = load_ckpt_model(cfg, ckpt_path, device)
    print(f"ckpt {ckpt_path} | epoch {epoch} | val_macro_auc ckpt {val_auc}",
          flush=True)

    loader, targets = build_loader(
        cfg, args.split, batch_size=int(cfg.get("batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 0) or 0),
        max_studies=None,
    )
    print(f"split {args.split} : {len(loader.dataset)} etudes", flush=True)

    probs, labels, uids, _loss = evaluate(model, loader, device)
    print(f"inference ok : {len(uids)} x {len(targets)}", flush=True)

    out_dir = run_dir / f"oof_diag_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_thresh = {str(t): _summarize(labels, probs, targets, t) for t in THRESHOLDS}
    main = _summarize(labels, probs, targets, args.thresh)

    payload = {
        "ckpt": str(ckpt_path),
        "epoch": epoch,
        "val_macro_auc_ckpt": val_auc,
        "split": args.split,
        "n_studies": len(uids),
        "targets": targets,
        "summary": main,
        "by_threshold": by_thresh,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    df = pd.DataFrame(main["per_target"])
    df.to_csv(out_dir / "per_target.csv", index=False)

    valid = labels >= 0
    delta = np.abs(labels - probs)
    hit = valid & (delta >= args.thresh)
    recs = []
    for i, uid in enumerate(uids):
        for j, name in enumerate(targets):
            if not hit[i, j]:
                continue
            recs.append({
                "StudyInstanceUID": uid,
                "target": name,
                "label": float(labels[i, j]),
                "proba": float(probs[i, j]),
                "abs_delta": float(delta[i, j]),
            })
    pd.DataFrame(recs).to_csv(out_dir / "disagreements.csv", index=False)

    print(f"\n=== desaccord |y-p| >= {args.thresh:g} ({args.split}) ===")
    print(f"etudes {main['n_studies']} | cellules valides {main['n_cells_valid']} | "
          f"desaccords {main['n_disagree']} ({main['pct_cells']:.2f} %)")
    print(df.to_string(index=False, formatters={
        "pct": "{:.2f}".format,
        "mean_abs_delta": "{:.3f}".format,
    }))
    print("\nautres seuils :")
    for t in THRESHOLDS:
        s = by_thresh[str(t)]
        print(f"  |d| >= {t:g} : {s['n_disagree']:5d} / {s['n_cells_valid']} "
              f"({s['pct_cells']:.2f} %)")
    print(f"\necrit -> {out_dir}")


if __name__ == "__main__":
    main()
