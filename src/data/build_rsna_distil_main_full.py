# -*- coding: utf-8 -*-
"""Train full RSNA distille sur l'OOF Original_main. Pas d'OAI, pas de val.

Les 4348 etudes de Original_main_LB0954_OOF_4348.csv, labels v3b.
Le hold-out hybrid (652) rentre dans le train : un full n'a pas de val.
Les 58 gold restent hors de ce CSV (include_gold les ajoute au lancement,
poids x4, sans teacher). Le teacher est l'OOF complet, pas le sous-ensemble
1687.

    python -X utf8 -m src.data.build_rsna_distil_main_full
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset_npz import gold_uids_from_train_csv, make_series_index
from src.data.hybrid_subset import TARGETS
from src.nlp.lexicon import TARGETS as LEXICON_TARGETS

ROOT = Path(__file__).resolve().parents[2]
V3B = ROOT / "data/labels/coolin/targets_v3b.csv"
SUBSET_CSV = ROOT / "data/labels/coolin/targets_v3b_rsna1687_distil_main.csv"
SUBSET_TEACHER = ROOT / "data/labels/teacher_original_main_oof_rsna1687.csv"
OOF_ZIP = ROOT / "Original_main_predictions_20261014.zip"
OOF_NAME = "Original_main_LB0954_OOF_4348.csv"
OUT = ROOT / "data/labels/coolin/targets_v3b_rsna_distil_main_full.csv"
META = ROOT / "data/labels/coolin/targets_v3b_rsna_distil_main_full.json"
TEACHER = ROOT / "data/labels/teacher_original_main_oof_rsna_full.csv"
N_OOF = 4348
SERIES_CSV = "data/oai/derived/train_series_oai.csv"
ZIP_PATH = ["data/crop145", "data/crop130"]


def main() -> None:
    if list(LEXICON_TARGETS) != list(TARGETS):
        raise SystemExit("TARGETS hybrid_subset != lexicon")

    with zipfile.ZipFile(OOF_ZIP) as z:
        oof = pd.read_csv(z.open(OOF_NAME))
    oof["StudyInstanceUID"] = oof.StudyInstanceUID.astype(str)
    if len(oof) != N_OOF:
        raise SystemExit(f"OOF {len(oof)} != {N_OOF}")
    if oof.StudyInstanceUID.duplicated().any():
        raise SystemExit("OOF : UID en double")
    missing_cols = [t for t in TARGETS if t not in oof.columns]
    if missing_cols:
        raise SystemExit(f"OOF colonnes manquantes : {missing_cols}")
    if any(str(u).startswith("OAI_") for u in oof.StudyInstanceUID):
        raise SystemExit("OAI dans l'OOF Original_main")
    oof_ids = set(oof.StudyInstanceUID)

    gold = gold_uids_from_train_csv(str(V3B))
    if not gold:
        raise SystemExit("gold introuvable dans train.csv")
    gold_in_oof = oof_ids & gold
    if gold_in_oof:
        raise SystemExit(f"gold dans l'OOF : {len(gold_in_oof)}")

    v3 = pd.read_csv(V3B)
    v3["StudyInstanceUID"] = v3.StudyInstanceUID.astype(str)
    if v3.StudyInstanceUID.duplicated().any():
        raise SystemExit("v3b : UID en double")
    train = v3[v3.StudyInstanceUID.isin(oof_ids)].copy()
    if len(train) != N_OOF:
        raise SystemExit(
            f"v3b couvre {len(train)} / {N_OOF} UID OOF"
        )
    if train[TARGETS].isna().all(axis=1).any():
        raise SystemExit("etude OOF sans aucun label v3b")

    print("index series (crop145 + crop130)...", flush=True)
    index, _vol = make_series_index(SERIES_CSV, ZIP_PATH)
    indexed = set(index.by_study)
    missing_vol = oof_ids - indexed
    if missing_vol:
        raise SystemExit(f"OOF sans volume : {len(missing_vol)}")
    gold_indexed = gold & indexed

    subset = pd.read_csv(SUBSET_CSV)
    subset["StudyInstanceUID"] = subset.StudyInstanceUID.astype(str)
    subset_ids = set(subset.StudyInstanceUID)
    if not subset_ids <= oof_ids:
        raise SystemExit(
            f"subset hors OOF : {len(subset_ids - oof_ids)}"
        )
    merged = train.merge(
        subset[["StudyInstanceUID", *TARGETS]],
        on="StudyInstanceUID", how="inner", suffixes=("_full", "_sub"),
    )
    for t in TARGETS:
        a = pd.to_numeric(merged[f"{t}_full"], errors="coerce")
        b = pd.to_numeric(merged[f"{t}_sub"], errors="coerce")
        both = a.notna() & b.notna()
        if not np.allclose(
            a[both].to_numpy(), b[both].to_numpy(), equal_nan=True,
        ):
            raise SystemExit(f"labels subset differentes de v3b sur {t}")

    sub_teacher = pd.read_csv(SUBSET_TEACHER)
    sub_teacher["StudyInstanceUID"] = sub_teacher.StudyInstanceUID.astype(str)
    if not set(sub_teacher.StudyInstanceUID) <= oof_ids:
        raise SystemExit("teacher 1687 hors OOF")

    out = train[["StudyInstanceUID", *TARGETS]].copy()
    out["split"] = "train"
    out = out[["StudyInstanceUID", "split", *TARGETS]]
    if set(out.StudyInstanceUID) != oof_ids:
        raise SystemExit("UID perdus a l'ecriture")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    ren = {t: f"proba_{t}" for t in TARGETS}
    teacher = oof[["StudyInstanceUID", *TARGETS]].rename(columns=ren)
    for col in ren.values():
        vals = pd.to_numeric(teacher[col], errors="coerce")
        if vals.isna().all():
            raise SystemExit(f"teacher {col} vide")
        if ((vals < 0) | (vals > 1)).fillna(False).any():
            raise SystemExit(f"teacher {col} hors [0, 1]")
    TEACHER.parent.mkdir(parents=True, exist_ok=True)
    teacher.to_csv(TEACHER, index=False)

    chk = teacher.merge(sub_teacher, on="StudyInstanceUID", suffixes=("_f", "_s"))
    if len(chk) != len(sub_teacher):
        raise SystemExit("teacher 1687 incomplet dans le full")
    for t in TARGETS:
        a = pd.to_numeric(chk[f"proba_{t}_f"], errors="coerce")
        b = pd.to_numeric(chk[f"proba_{t}_s"], errors="coerce")
        if not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True, atol=1e-6):
            raise SystemExit(f"teacher full != teacher 1687 sur {t}")

    n_train_launch = N_OOF + len(gold_indexed)
    coverage = N_OOF / n_train_launch
    if coverage < 0.95:
        raise SystemExit(
            f"couverture teacher {coverage:.1%} < 95 % "
            f"({N_OOF} / {n_train_launch})"
        )

    meta = {
        "n_train_csv": N_OOF,
        "n_val": 0,
        "n_oai": 0,
        "n_gold_held_in_csv": 0,
        "n_gold_total": len(gold),
        "n_gold_with_volume": len(gold_indexed),
        "teacher_coverage_at_launch": round(coverage, 4),
        "train_source": "targets_v3b.csv, les 4348 UID de l'OOF Original_main",
        "teacher": str(TEACHER.relative_to(ROOT)).replace("\\", "/"),
        "teacher_rows": int(len(teacher)),
        "csv": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "subset_csv_contained": int(len(subset_ids)),
        "former_val_now_train": int((subset.split == "val").sum()),
        "gold_in_oof": 0,
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
