# -*- coding: utf-8 -*-
"""Subset hybride fige, partage par les 4 runs hybrid025.

Train = 25 % de (etudes RSNA v3b hors gold + etudes OAI avec volume),
reparti 3/4 RSNA et 1/4 OAI. La val est le hold-out officiel deja fige
dans targets_v3b_det_frac025.csv (652 etudes RSNA, aucune OAI). Les RSNA
du train sont tires dans le complement de cette val, pas dedans.

    python -m src.data.hybrid_subset
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp.lexicon import TARGETS

ROOT = Path(__file__).resolve().parents[2]
DET = ROOT / "data/labels/coolin/targets_v3b_det_frac025.csv"
V3B = ROOT / "data/labels/coolin/targets_v3b.csv"
OAI = ROOT / "data/oai/derived/targets_oai.csv"
CROP = ROOT / "data/crop130"
OUT = ROOT / "data/labels/coolin/targets_v3b_hybrid025.csv"
META = ROOT / "data/labels/coolin/targets_v3b_hybrid025.json"
SEED = 42
FRAC = 0.25
OAI_SHARE = 0.25
BINS = [-1, 0, 2, 4, 20]
BIN_LABELS = ["0", "1-2", "3-4", "5+"]


def _studies_ok(manifests: list[Path], pred) -> set[str]:
    if not manifests:
        raise SystemExit(f"aucun manifeste dans {CROP}")
    df = pd.concat([pd.read_csv(p) for p in manifests], ignore_index=True)
    df = df[pred(df)]
    return set(df.StudyInstanceUID.astype(str))


def _take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n lignes, stratifiees sur le nombre de cibles >= 0.5 (meme bins que le subset)."""
    if n <= 0 or n > len(df):
        raise SystemExit(f"effectif {n} hors 1..{len(df)}")
    rng = np.random.default_rng(seed)
    n_pos = (df[TARGETS] >= 0.5).sum(axis=1)
    bins = pd.cut(n_pos, bins=BINS, labels=BIN_LABELS)
    groups = [(label, g) for label, g in df.groupby(bins, observed=False)]
    sizes = np.array([len(g) for _, g in groups], dtype=float)
    raw = sizes / sizes.sum() * n
    alloc = np.floor(raw).astype(int)
    left = int(n - alloc.sum())
    for i in np.argsort(-(raw - alloc)):
        if left <= 0:
            break
        room = int(sizes[i]) - int(alloc[i])
        if room <= 0:
            continue
        alloc[i] += 1
        left -= 1
    if int(alloc.sum()) != n:
        raise SystemExit(f"allocation {int(alloc.sum())} != {n}")
    keep: list = []
    for (_, g), k in zip(groups, alloc):
        if k <= 0 or len(g) == 0:
            continue
        chosen = rng.choice(g.index.to_numpy(), size=int(k), replace=False)
        keep.extend(chosen.tolist())
    out = df.loc[keep]
    if len(out) != n:
        raise SystemExit(f"tirage {len(out)} != {n}")
    return out


def build() -> pd.DataFrame:
    det = pd.read_csv(DET)
    val_ids = set(det.loc[det.split == "val", "StudyInstanceUID"].astype(str))
    v3 = pd.read_csv(V3B)
    v3["StudyInstanceUID"] = v3.StudyInstanceUID.astype(str)
    oai = pd.read_csv(OAI)
    oai["StudyInstanceUID"] = oai.StudyInstanceUID.astype(str)

    rsna_vol = _studies_ok(
        sorted(CROP.glob("manifest*.csv")),
        lambda df: df.ok == 1,
    )
    oai_vol = _studies_ok(
        sorted(CROP.glob("oai_manifest_b*.csv")),
        lambda df: (df.ok == 1) & (df.nbytes224 > 0) & (df.nbytes320 > 0),
    )
    rsna = v3[v3.StudyInstanceUID.isin(rsna_vol)].copy()
    oai = oai[oai.StudyInstanceUID.isin(oai_vol)].copy()
    missing_val = val_ids - set(rsna.StudyInstanceUID)
    if missing_val:
        raise SystemExit(f"{len(missing_val)} etudes val sans volume")
    pool = rsna[~rsna.StudyInstanceUID.isin(val_ids)].copy()
    n_total = len(rsna) + len(oai)
    n_train = int(round(FRAC * n_total))
    n_oai = int(round(OAI_SHARE * n_train))
    n_rsna = n_train - n_oai
    rsna_tr = _take(pool, n_rsna, SEED)
    oai_tr = _take(oai, n_oai, SEED)
    val = rsna[rsna.StudyInstanceUID.isin(val_ids)].copy()

    rsna_tr["split"] = "train"
    oai_tr["split"] = "train"
    val["split"] = "val"
    out = pd.concat([rsna_tr, oai_tr, val], ignore_index=True)
    out = out[["StudyInstanceUID", "split", *TARGETS]]
    if set(out.loc[out.split == "val", "StudyInstanceUID"]) != val_ids:
        raise SystemExit("la val a bouge")
    if out.StudyInstanceUID.duplicated().any():
        raise SystemExit("UID en double")
    n_oai_val = int(out.loc[out.split == "val", "StudyInstanceUID"].str.startswith("OAI_").sum())
    if n_oai_val:
        raise SystemExit("OAI dans la val")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    meta = {
        "seed": SEED,
        "frac": FRAC,
        "oai_share": OAI_SHARE,
        "n_rsna_eligible": int(len(rsna)),
        "n_oai_eligible": int(len(oai)),
        "n_total": int(n_total),
        "n_train": int(n_train),
        "n_train_rsna": int(n_rsna),
        "n_train_oai": int(n_oai),
        "n_val_rsna": int(len(val_ids)),
        "val_source": str(DET.relative_to(ROOT)).replace("\\", "/"),
        "csv": str(OUT.relative_to(ROOT)).replace("\\", "/"),
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return out


if __name__ == "__main__":
    build()
