# -*- coding: utf-8 -*-
"""Enchaine les 3 folds OOF puis concatene oof_preds.csv.

La concat ecrit cfg['oof_preds_csv'] (sinon data/labels/oof_preds_<run_name>.csv).
Toujours fixer oof_preds_csv sur un nouveau run pour ne pas ecraser v17.

Usage :
  python -X utf8 -m src.train.run_oof_folds --config configs/kneenet_v17_crop130_oof.yaml
  python -X utf8 -m src.train.run_oof_folds --config configs/kneenet_v20h_crop130_oof.yaml
  python -X utf8 -m src.train.run_oof_folds --config configs/kneenet_v17_crop130_oof.yaml --from-fold 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.data.dataset_npz import write_folds_csv
from src.train.train import load_config


def _fold_dir(run_name: str, fold: int) -> Path:
    return Path("checkpoints") / f"{run_name}_fold{fold}"


def _oof_out(cfg: dict) -> Path:
    if cfg.get("oof_preds_csv"):
        return Path(str(cfg["oof_preds_csv"]))
    return Path("data/labels") / f"oof_preds_{cfg['run_name']}.csv"


def _merge_oof(cfg: dict, n_folds: int) -> Path:
    run_name = str(cfg["run_name"])
    parts = []
    for k in range(n_folds):
        p = _fold_dir(run_name, k) / "oof_preds.csv"
        if not p.is_file():
            raise SystemExit(f"manque {p}")
        parts.append(pd.read_csv(p))
    all_df = pd.concat(parts, ignore_index=True)
    dups = all_df["StudyInstanceUID"].duplicated().sum()
    if dups:
        raise SystemExit(f"UID dupliques dans l'OOF : {dups}")
    out = _oof_out(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out, index=False)
    vc = all_df["fold"].value_counts().sort_index()
    print(f"OOF concat {len(all_df)} etudes -> {out}", flush=True)
    print("par fold :", {int(k): int(v) for k, v in vc.items()}, flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kneenet_v17_crop130_oof.yaml")
    parser.add_argument("--from-fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true",
                        help="ne fait que concatener les oof_preds.csv deja la")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n_folds = int(args.n_folds or cfg.get("n_folds", 3))
    run_name = str(cfg["run_name"])

    folds_csv = cfg.get("folds_csv") or "data/labels/folds3_gpt56sol.csv"
    write_folds_csv(
        labels_csv=cfg["labels_csv"], series_csv=cfg["series_csv"],
        zip_path=cfg["zip_path"], out_csv=folds_csv,
        n_folds=n_folds, seed=int(cfg.get("fold_seed", 42)),
    )

    if not args.skip_train:
        py = sys.executable
        for k in range(int(args.from_fold), n_folds):
            done = _fold_dir(run_name, k) / "oof_preds.csv"
            if done.is_file():
                print(f"fold {k} deja dump {done} : skip", flush=True)
                continue
            cmd = [
                py, "-X", "utf8", "-m", "src.train.train",
                "--config", args.config, "--fold", str(k),
            ]
            last_p = _fold_dir(run_name, k) / "last.pt"
            if last_p.is_file():
                cmd.append("--resume")
                print(f"fold {k} : resume {last_p}", flush=True)
            else:
                print(f"fold {k} : train {cmd}", flush=True)
            subprocess.run(cmd, check=True)

    _merge_oof(cfg, n_folds)
    print("etape 1 terminee.", flush=True)


if __name__ == "__main__":
    main()
