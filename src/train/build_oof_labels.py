# -*- coding: utf-8 -*-
"""Filtre OOF des labels LLM + injection des 58 gold (labels radiologue).

Masque une cellule (NaN / hors loss) si |label - proba OOF| >= seuil.
Les gold ne sont jamais masques : on ecrase par train.csv.

Usage :
  python -X utf8 -m src.train.build_oof_labels
  python -X utf8 -m src.train.build_oof_labels --out data/labels/labels_llm_gpt56sol_oofmask.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.dataset_npz import gold_uids_from_train_csv
from src.nlp.lexicon import TARGETS

# ACL : le modele OOF est plus conservateur que le LLM ; 0.8 effacerait
# trop de vrais 1 (cible qui a fait +0.07 LB). Fracture / synovite : rares
# ou OOF moins fiable que le LLM vs gold -> seuil plus strict.
DEFAULT_THRESH = {
    "ACL": 0.95,
    "Fracture": 0.90,
    "Synovitis": 0.90,
}
DEFAULT_THRESH_OTHER = 0.80


def _thresh_map(overrides: dict[str, float] | None = None) -> dict[str, float]:
    out = {t: float(DEFAULT_THRESH.get(t, DEFAULT_THRESH_OTHER)) for t in TARGETS}
    if overrides:
        out.update({k: float(v) for k, v in overrides.items() if k in out})
    return out


def build_oof_labels(
    labels_csv: str,
    oof_csv: str,
    train_csv: str,
    out_csv: str,
    thresh: dict[str, float],
) -> pd.DataFrame:
    src = pd.read_csv(labels_csv)
    oof = pd.read_csv(oof_csv)
    gold_uids = gold_uids_from_train_csv(labels_csv)
    train_df = pd.read_csv(train_csv, usecols=["StudyInstanceUID"] + TARGETS)
    gold = train_df[train_df.StudyInstanceUID.isin(gold_uids)].copy()
    if len(gold) == 0:
        raise SystemExit(f"aucun gold dans {train_csv}")

    missing_oof = set(src.StudyInstanceUID) - set(oof.StudyInstanceUID) - gold_uids
    print(f"labels {len(src)} | oof {len(oof)} | gold {len(gold)} | "
          f"hors oof (non-gold) {len(missing_oof)}", flush=True)

    merged = src.merge(oof[["StudyInstanceUID"] +
                           [f"proba_{t}" for t in TARGETS]],
                       on="StudyInstanceUID", how="left")

    stats = []
    n_mask_total = 0
    n_valid_total = 0
    for t in TARGETS:
        th = thresh[t]
        y = pd.to_numeric(merged[t], errors="coerce")
        p = pd.to_numeric(merged[f"proba_{t}"], errors="coerce")
        is_gold = merged.StudyInstanceUID.isin(gold_uids)
        valid = y.notna() & p.notna() & ~is_gold
        hit = valid & ((y - p).abs() >= th)
        n_mask_total += int(hit.sum())
        n_valid_total += int(valid.sum())
        stats.append({
            "target": t,
            "threshold": th,
            "n_valid": int(valid.sum()),
            "n_masked": int(hit.sum()),
            "pct": (100.0 * float(hit.sum()) / float(valid.sum())) if valid.sum() else 0.0,
        })
        merged.loc[hit, t] = float("nan")

    # Gold : labels radiologue, jamais masques. Ecrase le LLM s'il etait la.
    gold_idx = merged.StudyInstanceUID.isin(gold_uids)
    n_gold_in_src = int(gold_idx.sum())
    gold_map = gold.set_index("StudyInstanceUID")
    for t in TARGETS:
        merged.loc[gold_idx, t] = merged.loc[gold_idx, "StudyInstanceUID"].map(
            gold_map[t])

    extra = gold[~gold.StudyInstanceUID.isin(merged.StudyInstanceUID)]
    if len(extra):
        extra = extra.copy()
        merged = pd.concat([merged, extra], ignore_index=True)
        print(f"gold ajoutes (absents du CSV LLM) : {len(extra)}", flush=True)

    out = merged[["StudyInstanceUID"] + TARGETS].copy()
    out["split"] = "train"
    out["is_gold"] = out.StudyInstanceUID.isin(gold_uids).astype(int)
    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_p, index=False)

    n_studies = len(out)
    n_gold = int(out.is_gold.sum())
    n_cells = n_studies * len(TARGETS)
    n_nan = int(out[TARGETS].isna().sum().sum())
    print(f"\n=== filtre OOF -> {out_p} ===", flush=True)
    print(f"etudes {n_studies} (gold {n_gold}) | cellules {n_cells} | "
          f"NaN {n_nan} ({100.0 * n_nan / n_cells:.2f} %)", flush=True)
    print(f"masque OOF (hors gold) : {n_mask_total} / {n_valid_total} "
          f"({100.0 * n_mask_total / max(n_valid_total, 1):.2f} %)", flush=True)
    print(f"{'cible':20s} {'seuil':>5} {'valid':>7} {'masque':>7} {'pct':>6}",
          flush=True)
    for row in stats:
        print(f"{row['target']:20s} {row['threshold']:5.2f} {row['n_valid']:7d} "
              f"{row['n_masked']:7d} {row['pct']:5.2f}%", flush=True)
    print(f"gold ecrases dans le CSV LLM : {n_gold_in_src}", flush=True)

    meta = {
        "out_csv": str(out_p),
        "n_studies": n_studies,
        "n_gold": n_gold,
        "n_masked_oof": n_mask_total,
        "n_valid_oof": n_valid_total,
        "thresholds": thresh,
        "per_target": stats,
        "gold_in_src": n_gold_in_src,
        "missing_oof_non_gold": len(missing_oof),
    }
    (out_p.with_suffix(".oof_filter.json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return out


def restore_acl_for_v20b(
    oofmask_csv: str,
    labels_csv: str,
    out_csv: str,
) -> pd.DataFrame:
    """Recopie ACL gpt56sol hors gold. Le masque OOF v17 reste pour les 11 autres."""
    masked = pd.read_csv(oofmask_csv)
    src = pd.read_csv(labels_csv, usecols=["StudyInstanceUID", "ACL"])
    if "is_gold" not in masked.columns:
        raise SystemExit(f"{oofmask_csv} sans colonne is_gold")
    before_nan = int(masked.ACL.isna().sum())
    gold = masked.is_gold == 1
    acl_map = src.set_index("StudyInstanceUID")["ACL"]
    restored = masked.loc[~gold, "StudyInstanceUID"].map(acl_map)
    n_miss = int(restored.isna().sum())
    masked = masked.copy()
    masked.loc[~gold, "ACL"] = restored.values
    after_nan = int(masked.ACL.isna().sum())
    n_put = before_nan - after_nan
    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    masked.to_csv(out_p, index=False)
    print(f"ACL restauree hors gold : {n_put} cellules "
          f"(NaN {before_nan} -> {after_nan}, "
          f"gold inchanges {int(gold.sum())}, src manquants {n_miss})",
          flush=True)
    print(f"ecrit -> {out_p} ({len(masked)} etudes)", flush=True)
    meta = {
        "out_csv": str(out_p),
        "source_oofmask": oofmask_csv,
        "acl_restored": n_put,
        "acl_nan_before": before_nan,
        "acl_nan_after": after_nan,
        "n_gold": int(gold.sum()),
        "src_acl_missing_non_gold": n_miss,
    }
    (out_p.with_suffix(".oof_filter.json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return masked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/labels/labels_llm_gpt56sol.csv")
    parser.add_argument("--oof", default="data/labels/oof_preds_v17.csv")
    parser.add_argument("--train-csv", default="data/csv/train.csv")
    parser.add_argument("--out", default="data/labels/labels_llm_gpt56sol_oofmask.csv")
    parser.add_argument(
        "--restore-acl", action="store_true",
        help="depuis un CSV oofmask : remet ACL gpt56sol hors gold (recette v20b)")
    parser.add_argument(
        "--oofmask", default="data/labels/labels_llm_gpt56sol_oofmask.csv",
        help="CSV oofmask source (avec --restore-acl)")
    args = parser.parse_args()
    if args.restore_acl:
        restore_acl_for_v20b(args.oofmask, args.labels, args.out)
        return
    thresh = _thresh_map()
    print("seuils :", thresh, flush=True)
    build_oof_labels(args.labels, args.oof, args.train_csv, args.out, thresh)


if __name__ == "__main__":
    main()
