# -*- coding: utf-8 -*-
"""Quels slots protocole etaient vides, pour le stack.

Meme regle que le forward val (train=False, series_drop_p=0) : un slot est
vide si pick_protocol ne trouve aucune serie de ce plan et de ce contraste
dans les volumes. Le drop aleatoire du train des modeles de base n'entre
pas ici : ce n'est pas une propriete de l'etude.

Index de reference : crop130 (v20b, v20h, CoAt, CNX). ViT-k (crop130_320)
donne le meme masque. crop145 puis crop130 aussi, sauf une etude de val
(AX_FS present seulement dans crop145) : elle est signalee, pas dupliquee
en colonnes.

Liste et ordre = data/hybrid_pred/hybrid_preds_unseen_rsna.csv.
split : fit (ajustement), val (652), gold (58). Meme six colonnes pour les trois.

    python -X utf8 runpod/_dump_slot_empty.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data.dataset_npz import (  # noqa: E402
    PROTOCOL_SLOT_NAMES,
    PROTOCOL_SLOTS,
    make_series_index,
)
from src.nlp.lexicon import TARGETS  # noqa: E402

VIT_CSV = ROOT / "data" / "hybrid_pred" / "hybrid_preds_unseen_rsna.csv"
HYBRID_CSV = ROOT / "data" / "labels" / "coolin" / "targets_v3b_hybrid025.csv"
TRAIN_CSV = ROOT / "data" / "csv" / "train.csv"
SERIES_CSV = "data/oai/derived/train_series_oai.csv"
OUT = ROOT / "data" / "hybrid_pred" / "slot_empty_unseen_rsna.csv"

SOURCES = (
    ("crop130", "data/crop130"),
    ("crop320", "data/crop130_320"),
    ("crop145", ["data/crop145", "data/crop130"]),
)


def _empty_frame(uids: list[str], zip_path: str | list[str]) -> pd.DataFrame:
    index, _vol = make_series_index(SERIES_CSV, zip_path)
    cols = {f"empty_{name}": [] for name in PROTOCOL_SLOT_NAMES}
    for uid in uids:
        for name, (plane, fluid) in zip(PROTOCOL_SLOT_NAMES, PROTOCOL_SLOTS):
            sid = index.pick_protocol(uid, plane, fluid, None)
            cols[f"empty_{name}"].append(int(sid is None))
    return pd.DataFrame(cols)


def _split(uids: list[str]) -> pd.Series:
    hybrid = pd.read_csv(HYBRID_CSV, usecols=["StudyInstanceUID", "split"])
    val_ids = set(hybrid.loc[hybrid.split == "val", "StudyInstanceUID"].astype(str))
    gold_df = pd.read_csv(TRAIN_CSV, usecols=["StudyInstanceUID"] + TARGETS)
    gold_ids = set(
        gold_df.loc[gold_df[TARGETS].notna().all(axis=1), "StudyInstanceUID"].astype(str)
    )
    roles = []
    for uid in uids:
        if uid in gold_ids:
            roles.append("gold")
        elif uid in val_ids:
            roles.append("val")
        else:
            roles.append("fit")
    return pd.Series(roles, name="split")


def main() -> None:
    uids = pd.read_csv(VIT_CSV, usecols=["StudyInstanceUID"]).StudyInstanceUID.astype(str).tolist()
    frames = {prefix: _empty_frame(uids, zip_path) for prefix, zip_path in SOURCES}
    ref = frames["crop130"]
    n_320 = int((ref.to_numpy() != frames["crop320"].to_numpy()).sum())
    diff_145 = ref.to_numpy() != frames["crop145"].to_numpy()
    n_145 = int(diff_145.sum())
    print(f"cellules differentes crop130 vs crop320 : {n_320}", flush=True)
    print(f"cellules differentes crop130 vs crop145 : {n_145}", flush=True)
    if n_320:
        raise SystemExit("crop320 ne copie plus crop130 : le masque unique est faux")
    out = pd.concat(
        [pd.DataFrame({"StudyInstanceUID": uids, "split": _split(uids)}), ref],
        axis=1,
    )
    if out.StudyInstanceUID.duplicated().any():
        raise SystemExit("UID en double")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"-> {OUT} ({len(out)} etudes)", flush=True)
    print(out["split"].value_counts().to_string(), flush=True)
    slot_cols = [c for c in out.columns if c.startswith("empty_")]
    print(out.groupby("split")[slot_cols].mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
