# -*- coding: utf-8 -*-
"""Assemble train_val_labels_llm.csv a partir des predictions Gemini.

- Filet lexique (optionnel) : desaccord 0/1 -> NaN, pas un second vote.
- Plafond de prevalence (optionnel).
- Split etude 85/15 stratifie langue x n_pos (meme logique que le lexique).
Les 58 gold ne sont jamais inclus.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_labels import cap_positive_prevalence, make_split
from src.nlp.lexicon import TARGETS


def apply_lexicon_net(
    llm: pd.DataFrame,
    lexicon: pd.DataFrame,
    conf_thr: float = 0.75,
) -> tuple[pd.DataFrame, int]:
    """Masque les cellules ou lexique (confiant) et LLM sont en desaccord 0/1."""
    out = llm.copy()
    lex = lexicon.set_index("StudyInstanceUID")
    n_mask = 0
    for idx, r in out.iterrows():
        uid = r["StudyInstanceUID"]
        if uid not in lex.index:
            continue
        lx = lex.loc[uid]
        if isinstance(lx, pd.DataFrame):
            lx = lx.iloc[0]
        for t in TARGETS:
            lv = lx[t] if t in lx.index else np.nan
            lc = lx[t + "_conf"] if (t + "_conf") in lx.index else 0.0
            mv = r[t]
            if pd.isna(lv) or pd.isna(mv):
                continue
            if float(lc) < conf_thr:
                continue
            if int(lv) != int(mv):
                out.at[idx, t] = np.nan
                out.at[idx, t + "_conf"] = 0.0
                n_mask += 1
    return out, n_mask


def assemble_train_val(
    llm_df: pd.DataFrame,
    out_csv: str | Path,
    lexicon_csv: str | Path | None = None,
    lexicon_conf: float = 0.75,
    prevalence_cap: float | None = 0.65,
    val_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    df = llm_df.copy()
    # unknown deja en NaN ; on masque aussi conf == 0
    for t in TARGETS:
        if t + "_conf" in df.columns:
            df.loc[df[t + "_conf"] <= 0, t] = np.nan

    if lexicon_csv is not None and Path(lexicon_csv).exists():
        lex = pd.read_csv(lexicon_csv)
        df, n_mask = apply_lexicon_net(df, lex, conf_thr=lexicon_conf)
        print(f"filet lexique : {n_mask} cellules masquees (desaccord, conf>={lexicon_conf})")
    elif lexicon_csv is not None:
        print(f"filet lexique ignore (fichier absent) : {lexicon_csv}")

    print("\n=== Couverture brute LLM (avant plafond) ===")
    print(df[TARGETS].notna().mean().round(3).to_string())
    print("\n=== Prevalences brutes ===")
    print(df[TARGETS].mean().round(3).sort_values(ascending=False).to_string())

    if prevalence_cap is not None:
        df = cap_positive_prevalence(df, cap=prevalence_cap, seed=seed)
        print(f"\nplafond prevalence {prevalence_cap} applique")

    split = make_split(df, val_frac=val_frac, seed=seed)

    out = df[["StudyInstanceUID", "lang"] + TARGETS].copy()
    out["split"] = split.values
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print("\n=== Couverture finale ===")
    print(out[TARGETS].notna().mean().round(3).to_string())
    kept = int(out[TARGETS].notna().sum().sum())
    total = len(out) * len(TARGETS)
    print(f"cellules gardees : {kept}/{total} ({100 * kept / total:.1f}%)")
    full_na = int(out[TARGETS].isna().all(axis=1).sum())
    print(f"etudes sans aucun label : {full_na} ({100 * full_na / len(out):.1f}%)")
    print("\n=== Prevalences finales ===")
    print(out[TARGETS].mean().round(3).sort_values(ascending=False).to_string())
    print("\n=== Split ===")
    print(out["split"].value_counts().to_string())
    print("sauvegarde:", out_csv)
    return out
