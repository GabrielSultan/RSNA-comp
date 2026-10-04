# -*- coding: utf-8 -*-
"""Dump OOF (best.pt) des 3 folds v20h DINO ep11, puis masque targets_v3b.

Regle (pas les seuils gpt56sol de build_oof_labels) :
  masquer seulement si le label v3b est deja tranche (<=0.15 ou >=0.85)
  ET l'OOF, qui n'a pas vu l'etude, est sur de l'autre sens (<=0.15 ou >=0.85).
  ACL, sens « label haut / OOF bas » : on ne masque que si ce sens est rare
  (<=1 % des etudes) et que le gold (moyenne des 3 folds) ne sous-appelle pas.
  Si une cible depasse 5 % de cellules masquees, on resserre a 0.10 / 0.90.
  Les gold ne sont pas injectes dans le CSV.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset_npz import competition_train_csv, make_loaders
from src.models.registry import build_model
from src.nlp.lexicon import TARGETS
from src.train.train import evaluate, load_config, make_gold_loader

CONFIG = "configs/kneenet_v20h_crop130_dino_ep11_oof_v3b.yaml"
RUN = "v20h_crop130_dino_ep11_oof_v3b"
LABELS = Path("data/labels/coolin/targets_v3b.csv")
OOF_OUT = Path("data/labels/oof_preds_v20h_dino_ep11_v3b.csv")
MASK_OUT = Path("data/labels/coolin/targets_v3b_oofmask.csv")
LO, HI = 0.15, 0.85
LO_TIGHT, HI_TIGHT = 0.10, 0.90
PCT_CAP = 5.0
ACL_RARE_MAX = 43  # ~1 % de 4348 etudes
GOLD_RECALL_MIN = 0.90
GOLD_MEAN_POS_MIN = 0.75


def _dump_fold(cfg: dict, fold: int, device: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    run_dir = Path("checkpoints") / f"{RUN}_fold{fold}"
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"manque {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = int(ckpt.get("epoch", -1))
    print(f"fold {fold} : {ckpt_path} epoch {epoch} "
          f"val AUC {float(ckpt.get('val_macro_auc', float('nan'))):.4f}",
          flush=True)
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    train_ld, val_ld, targets = make_loaders(
        labels_csv=cfg["labels_csv"], series_csv=cfg["series_csv"],
        zip_path=cfg["zip_path"], batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        n_series=int(cfg.get("n_series", 6)),
        train_frac=1.0, train_frac_seed=int(cfg.get("train_frac_seed", 42)),
        img_size=int(cfg.get("img_size", 224)),
        oof_fold=fold, folds_csv=cfg.get("folds_csv"),
        n_folds=int(cfg.get("n_folds", 3)),
        fold_seed=int(cfg.get("fold_seed", 42)),
        include_gold=False, val_frac=float(cfg.get("val_frac", 0.15)),
        seed=int(cfg.get("seed", 42)),
    )
    if list(targets) != list(TARGETS):
        raise SystemExit(f"cibles inattendues : {targets}")
    probs, _labels, uids, _loss = evaluate(model, val_ld, device)
    rows = []
    for i, uid in enumerate(uids):
        row = {"StudyInstanceUID": uid, "fold": fold, "epoch": epoch}
        for j, t in enumerate(TARGETS):
            row[f"proba_{t}"] = float(probs[i, j])
        rows.append(row)
    out_p = run_dir / "oof_preds.csv"
    pd.DataFrame(rows).to_csv(out_p, index=False)
    print(f"dump fold {fold} : {len(rows)} etudes -> {out_p}", flush=True)

    gold_ld = make_gold_loader(cfg, int(cfg["batch_size"]), num_workers=0)
    gp, _gy, guids, _ = evaluate(model, gold_ld, device)
    gold_map = {uid: gp[i] for i, uid in enumerate(guids)}

    del train_ld, val_ld, gold_ld, model, ckpt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), gold_map


def _gold_acl(gold_by_fold: list[dict[str, np.ndarray]]) -> dict:
    train_csv = competition_train_csv(LABELS)
    gold = pd.read_csv(train_csv, usecols=["StudyInstanceUID", "ACL"])
    gold = gold[gold.ACL.notna()].copy()
    uids = gold.StudyInstanceUID.tolist()
    missing = [u for u in uids if any(u not in g for g in gold_by_fold)]
    if missing:
        raise SystemExit(f"gold sans proba : {len(missing)}")
    stacked = np.stack(
        [np.stack([g[u] for u in uids], axis=0) for g in gold_by_fold],
        axis=0,
    )
    avg = stacked.mean(axis=0)
    acl_i = TARGETS.index("ACL")
    p = avg[:, acl_i]
    y = gold.ACL.to_numpy(dtype=float)
    pos = y >= 0.5
    neg = ~pos
    n_pos = int(pos.sum())
    recall = float((p[pos] >= 0.5).mean()) if n_pos else float("nan")
    mean_pos = float(p[pos].mean()) if n_pos else float("nan")
    fpr = float((p[neg] >= 0.5).mean()) if int(neg.sum()) else float("nan")
    stats = {
        "n_gold": int(len(uids)),
        "n_acl_pos": n_pos,
        "acl_recall_at_0.5": recall,
        "acl_mean_proba_pos": mean_pos,
        "acl_fpr_at_0.5": fpr,
        "under_calls": bool(
            n_pos < 10 or recall < GOLD_RECALL_MIN or mean_pos < GOLD_MEAN_POS_MIN
        ),
    }
    print(
        f"gold ACL (moyenne 3 folds) : n_pos {n_pos}/{len(uids)} | "
        f"recall@0.5 {recall:.3f} | mean p|y=1 {mean_pos:.3f} | "
        f"fpr {fpr:.3f} | under_calls {stats['under_calls']}",
        flush=True,
    )
    return stats


def _directions(y: np.ndarray, p: np.ndarray, lo: float, hi: float):
    high_low = (y >= hi) & (p <= lo)
    low_high = (y <= lo) & (p >= hi)
    return high_low, low_high


def apply_mask(oof: pd.DataFrame, gold_acl: dict) -> None:
    src = pd.read_csv(LABELS)
    if src[TARGETS].isna().any().any():
        raise SystemExit("targets_v3b a des NaN inattendus")
    merged = src.merge(
        oof[["StudyInstanceUID", "fold", "epoch"] + [f"proba_{t}" for t in TARGETS]],
        on="StudyInstanceUID", how="left",
    )
    if merged[f"proba_{TARGETS[0]}"].isna().any():
        n = int(merged[f"proba_{TARGETS[0]}"].isna().sum())
        raise SystemExit(f"{n} etudes de targets_v3b sans proba OOF")
    if merged["StudyInstanceUID"].duplicated().any():
        raise SystemExit("UID dupliques apres jointure")

    n_acl_high_low = int(_directions(
        merged["ACL"].to_numpy(float),
        merged["proba_ACL"].to_numpy(float), LO, HI,
    )[0].sum())
    rare = n_acl_high_low <= ACL_RARE_MAX
    mask_acl_high_low = bool(rare and not gold_acl["under_calls"])
    print(
        f"ACL label haut / OOF bas : {n_acl_high_low} "
        f"(rare<={ACL_RARE_MAX} : {rare}) | "
        f"masquer ce sens : {mask_acl_high_low}",
        flush=True,
    )

    out = src[["StudyInstanceUID"] + TARGETS].copy()
    per_target = []
    n_mask_total = 0
    for t in TARGETS:
        y = merged[t].to_numpy(dtype=float)
        p = merged[f"proba_{t}"].to_numpy(dtype=float)
        lo, hi = LO, HI
        tightened = False
        high_low, low_high = _directions(y, p, lo, hi)
        if t == "ACL" and not mask_acl_high_low:
            high_low = np.zeros_like(high_low)
        hit = high_low | low_high
        pct = 100.0 * float(hit.sum()) / len(merged)
        if pct > PCT_CAP:
            lo, hi = LO_TIGHT, HI_TIGHT
            tightened = True
            high_low, low_high = _directions(y, p, lo, hi)
            if t == "ACL" and not mask_acl_high_low:
                high_low = np.zeros_like(high_low)
            hit = high_low | low_high
            pct = 100.0 * float(hit.sum()) / len(merged)
        n_mask_total += int(hit.sum())
        out.loc[hit, t] = np.nan
        per_target.append({
            "target": t,
            "lo": lo,
            "hi": hi,
            "tightened": tightened,
            "n_label_high_model_low": int(high_low.sum()),
            "n_label_low_model_high": int(low_high.sum()),
            "n_masked": int(hit.sum()),
            "pct": pct,
        })

    MASK_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MASK_OUT, index=False)
    n_cells = len(out) * len(TARGETS)
    n_nan = int(out[TARGETS].isna().sum().sum())
    print(f"\n=== masque -> {MASK_OUT} ===", flush=True)
    print(f"etudes {len(out)} | NaN {n_nan} / {n_cells} "
          f"({100.0 * n_nan / n_cells:.2f} %)", flush=True)
    print(f"{'cible':20s} {'lo':>5} {'hi':>5} {'haut->bas':>9} {'bas->haut':>9} "
          f"{'masque':>7} {'pct':>6} tight", flush=True)
    for row in per_target:
        print(f"{row['target']:20s} {row['lo']:5.2f} {row['hi']:5.2f} "
              f"{row['n_label_high_model_low']:9d} {row['n_label_low_model_high']:9d} "
              f"{row['n_masked']:7d} {row['pct']:5.2f}% {int(row['tightened'])}",
              flush=True)

    meta = {
        "out_csv": str(MASK_OUT),
        "oof_csv": str(OOF_OUT),
        "n_studies": int(len(out)),
        "n_masked": n_mask_total,
        "n_cells": n_cells,
        "rule": {
            "lo": LO, "hi": HI,
            "tight_lo": LO_TIGHT, "tight_hi": HI_TIGHT,
            "pct_cap": PCT_CAP,
            "acl_mask_label_high_model_low": mask_acl_high_low,
            "acl_label_high_model_low_n": n_acl_high_low,
        },
        "gold_acl": gold_acl,
        "per_target": per_target,
        "epochs_by_fold": {
            str(int(k)): int(v) for k, v in
            oof.groupby("fold")["epoch"].first().items()
        },
    }
    meta_p = MASK_OUT.with_suffix(".oof_filter.json")
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"meta -> {meta_p}", flush=True)


def main() -> None:
    cfg = load_config(CONFIG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parts = []
    gold_by_fold = []
    for fold in range(3):
        part, gold_map = _dump_fold(cfg, fold, device)
        parts.append(part)
        gold_by_fold.append(gold_map)
    oof = pd.concat(parts, ignore_index=True)
    if oof["StudyInstanceUID"].duplicated().any():
        raise SystemExit("UID dupliques dans l'OOF concatene")
    OOF_OUT.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(OOF_OUT, index=False)
    print(f"OOF concat {len(oof)} -> {OOF_OUT}", flush=True)
    vc = oof["fold"].value_counts().sort_index()
    print("par fold :", {int(k): int(v) for k, v in vc.items()}, flush=True)
    gold_acl = _gold_acl(gold_by_fold)
    apply_mask(oof, gold_acl)
    print("filtre termine.", flush=True)


if __name__ == "__main__":
    main()
