# -*- coding: utf-8 -*-
"""Dump logits + proba d'un checkpoint (teacher) sur un split.

Pas de TTA, pas d'aug, series_drop off — les soft labels du KD.

  python -X utf8 -m src.train.dump_teacher_preds \\
      --config configs/kneenet_vit_e_crop130_subset.yaml
  python -X utf8 -m src.train.dump_teacher_preds \\
      --config configs/kneenet_vit_e_crop130_subset.yaml \\
      --ckpt checkpoints/subset_vit_dinov3_e_crop130_frac025/best.pt \\
      --splits train --out data/labels/teacher_vite_subset_e.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.infer.inspect_attention import build_loader, load_ckpt_model
from src.train.train import load_config


@torch.no_grad()
def collect_logits(model, loader, device: str):
    """Forward eval fp32 -> logits (N, T), uids."""
    model.eval()
    chunks, uids = [], []
    n_done = 0
    n_total = len(loader.dataset)
    for x, _y, uid in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).float()
        chunks.append(logits.cpu())
        uids.extend(uid)
        n_done += x.size(0)
        if n_done % 80 == 0 or n_done >= n_total:
            print(f"  {n_done}/{n_total}", flush=True)
    return torch.cat(chunks, dim=0), uids


def _resolve_ckpt(cfg: dict, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"checkpoint introuvable : {p}")
        return p
    run_dir = Path("checkpoints") / cfg["run_name"]
    for name in ("best.pt", "last.pt"):
        cand = run_dir / name
        if cand.is_file():
            return cand
    raise SystemExit(f"aucun ckpt dans {run_dir} (best/last)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument(
        "--splits", default="train",
        help="virgules : train, val, gold (defaut train)",
    )
    parser.add_argument("--out", default=None,
                        help="CSV ; defaut data/labels/teacher_<run_name>.csv")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = _resolve_ckpt(cfg, args.ckpt)
    out = Path(args.out) if args.out else Path(
        "data/labels") / f"teacher_{cfg['run_name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, epoch, val_auc = load_ckpt_model(cfg, ckpt_path, device)
    print(f"ckpt {ckpt_path} | epoch {epoch} | val_macro_auc ckpt {val_auc}",
          flush=True)
    print(f"device {device} | arch {getattr(model, 'arch_id', '?')}",
          flush=True)

    bs = int(args.batch_size or cfg.get("batch_size", 2))
    nw = int(cfg.get("num_workers", 0) or 0)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    frames = []
    meta_splits = {}
    for split in splits:
        loader, targets = build_loader(
            cfg, split, batch_size=bs, num_workers=nw, max_studies=None,
        )
        print(f"split {split} : {len(loader.dataset)} etudes", flush=True)
        logits, uids = collect_logits(model, loader, device)
        probs = torch.sigmoid(logits)
        recs = []
        for i, uid in enumerate(uids):
            row = {"StudyInstanceUID": uid, "split": split}
            for j, t in enumerate(targets):
                row[f"logit_{t}"] = float(logits[i, j])
                row[f"proba_{t}"] = float(probs[i, j])
            recs.append(row)
        frames.append(pd.DataFrame(recs))
        meta_splits[split] = len(recs)
        print(f"  dump {len(recs)} x {len(targets)}", flush=True)

    df = pd.concat(frames, ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates("StudyInstanceUID", keep="first")
    if len(df) < n_before:
        print(f"doublons UID retires : {n_before - len(df)}", flush=True)
    df.to_csv(out, index=False)
    meta = {
        "csv": str(out),
        "teacher_config": args.config,
        "ckpt": str(ckpt_path),
        "epoch": epoch,
        "val_macro_auc_ckpt": val_auc,
        "run_name": cfg.get("run_name"),
        "model": cfg.get("model"),
        "splits": meta_splits,
        "n": int(len(df)),
        "tta": False,
    }
    meta_p = out.with_suffix(".json")
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"ecrit {out} ({len(df)} etudes) + {meta_p}", flush=True)


if __name__ == "__main__":
    main()
