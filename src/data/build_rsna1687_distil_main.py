# -*- coding: utf-8 -*-
"""Train RSNA de taille hybrid025, val identique, teacher OOF main.

1687 etudes tirees dans l'OOF Original_main, hors val hybrid et hors gold.
La val est copiee telle quelle depuis targets_v3b_hybrid025.csv.
Pas d'OAI : ces etudes n'ont pas de prediction teacher.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset_npz import gold_uids_from_train_csv, make_series_index
from src.data.hybrid_subset import DET, TARGETS, _take
from src.nlp.lexicon import TARGETS as LEXICON_TARGETS

ROOT = Path(__file__).resolve().parents[2]
HYBRID = ROOT / "data/labels/coolin/targets_v3b_hybrid025.csv"
V3B = ROOT / "data/labels/coolin/targets_v3b.csv"
OOF_ZIP = ROOT / "Original_main_predictions_20261014.zip"
OOF_NAME = "Original_main_LB0954_OOF_4348.csv"
OUT = ROOT / "data/labels/coolin/targets_v3b_rsna1687_distil_main.csv"
META = ROOT / "data/labels/coolin/targets_v3b_rsna1687_distil_main.json"
TEACHER = ROOT / "data/labels/teacher_original_main_oof_rsna1687.csv"
N_TRAIN = 1687
SEED = 42
SERIES_CSV = "data/oai/derived/train_series_oai.csv"
ZIP_PATH = ["data/crop145", "data/crop130"]


def main() -> None:
    if list(LEXICON_TARGETS) != list(TARGETS):
        raise SystemExit("TARGETS hybrid_subset != lexicon")
    hybrid = pd.read_csv(HYBRID)
    hybrid["StudyInstanceUID"] = hybrid.StudyInstanceUID.astype(str)
    val = hybrid.loc[hybrid.split == "val", ["StudyInstanceUID", *TARGETS]].copy()
    val_ids = set(val.StudyInstanceUID)
    det = pd.read_csv(DET)
    det_val = set(det.loc[det.split == "val", "StudyInstanceUID"].astype(str))
    if val_ids != det_val:
        raise SystemExit(
            f"val hybrid != val det : {len(val_ids)} vs {len(det_val)} "
            f"symdiff {len(val_ids ^ det_val)}"
        )
    if len(val) != 652:
        raise SystemExit(f"val {len(val)} != 652")

    with zipfile.ZipFile(OOF_ZIP) as z:
        oof = pd.read_csv(z.open(OOF_NAME))
    oof["StudyInstanceUID"] = oof.StudyInstanceUID.astype(str)
    if oof.StudyInstanceUID.duplicated().any():
        raise SystemExit("OOF : UID en double")
    missing_cols = [t for t in TARGETS if t not in oof.columns]
    if missing_cols:
        raise SystemExit(f"OOF colonnes manquantes : {missing_cols}")
    oof_ids = set(oof.StudyInstanceUID)
    if not val_ids <= oof_ids:
        raise SystemExit(f"val hors OOF : {len(val_ids - oof_ids)}")

    gold = gold_uids_from_train_csv(str(HYBRID))
    if val_ids & gold:
        raise SystemExit(f"gold dans la val : {len(val_ids & gold)}")
    if any(u.startswith("OAI_") for u in val_ids):
        raise SystemExit("OAI dans la val")

    v3 = pd.read_csv(V3B)
    v3["StudyInstanceUID"] = v3.StudyInstanceUID.astype(str)
    # Memes valeurs que le CSV hybrid pour la val (pas un re-export).
    merged = val.merge(
        v3[["StudyInstanceUID", *TARGETS]], on="StudyInstanceUID",
        suffixes=("_h", "_v"),
    )
    if len(merged) != len(val):
        raise SystemExit("val absente de targets_v3b")
    for t in TARGETS:
        a = pd.to_numeric(merged[f"{t}_h"], errors="coerce")
        b = pd.to_numeric(merged[f"{t}_v"], errors="coerce")
        if not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True):
            raise SystemExit(f"labels val differentes de v3b sur {t}")

    print("index series (crop145 + crop130)...", flush=True)
    index, _vol = make_series_index(SERIES_CSV, ZIP_PATH)
    indexed = set(index.by_study)
    val_missing = val_ids - indexed
    if val_missing:
        raise SystemExit(f"val sans volume : {len(val_missing)}")

    pool = v3[v3.StudyInstanceUID.isin(oof_ids - val_ids - gold)].copy()
    pool = pool[pool.StudyInstanceUID.isin(indexed)]
    pool = pool.dropna(subset=TARGETS, how="all")
    if pool.StudyInstanceUID.duplicated().any():
        raise SystemExit("pool : UID en double")
    if any(str(u).startswith("OAI_") for u in pool.StudyInstanceUID):
        raise SystemExit("OAI dans le pool RSNA")
    print(f"pool eligible {len(pool)} (OOF hors val/gold, volume ok)", flush=True)

    train = _take(pool, N_TRAIN, SEED)
    train_ids = set(train.StudyInstanceUID.astype(str))
    if train_ids & val_ids:
        raise SystemExit("fuite : train intersecte la val")
    if train_ids & gold:
        raise SystemExit("fuite : gold dans le train")
    if len(train_ids) != N_TRAIN:
        raise SystemExit(f"train {len(train_ids)} != {N_TRAIN}")

    hybrid_train = set(hybrid.loc[hybrid.split == "train", "StudyInstanceUID"])
    hybrid_oai = {u for u in hybrid_train if str(u).startswith("OAI_")}
    if train_ids & hybrid_oai:
        raise SystemExit("OAI hybrid dans le train")

    train_out = train[["StudyInstanceUID", *TARGETS]].copy()
    train_out["split"] = "train"
    val_out = val.copy()
    val_out["split"] = "val"
    out = pd.concat([train_out, val_out], ignore_index=True)
    out = out[["StudyInstanceUID", "split", *TARGETS]]
    if set(out.loc[out.split == "val", "StudyInstanceUID"]) != val_ids:
        raise SystemExit("la val a bouge a l'ecriture")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    teacher = oof[oof.StudyInstanceUID.isin(train_ids)].copy()
    if len(teacher) != N_TRAIN:
        raise SystemExit("teacher incomplet")
    ren = {t: f"proba_{t}" for t in TARGETS}
    teacher = teacher[["StudyInstanceUID", *TARGETS]].rename(columns=ren)
    TEACHER.parent.mkdir(parents=True, exist_ok=True)
    teacher.to_csv(TEACHER, index=False)

    meta = {
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_val": len(val_ids),
        "n_pool": int(len(pool)),
        "n_oai": 0,
        "val_source": "data/labels/coolin/targets_v3b_hybrid025.csv",
        "val_identity": "targets_v3b_det_frac025.csv split=val",
        "train_source": "targets_v3b.csv, OOF Original_main hors val et hors gold",
        "teacher": str(TEACHER.relative_to(ROOT)).replace("\\", "/"),
        "teacher_rows": int(len(teacher)),
        "csv": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "overlap_hybrid_train_rsna": int(len(train_ids & (hybrid_train - hybrid_oai))),
        "hybrid_train_rsna": int(len(hybrid_train - hybrid_oai)),
        "train_val_intersection": 0,
        "gold_in_train": 0,
        "gold_in_val": 0,
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
