# -*- coding: utf-8 -*-
"""Pseudo-labels + split study-level.

Sorties (data/labels/) :
- pseudo_labels.csv : une ligne par etude non-gold, 12 labels (0/1/NaN=masque)
  + 12 confiances + langue.
- train_val_labels.csv : labels utilisables pour le train vision
  (pseudo-labels masques par confiance ; les 58 gold NE sont PAS inclus dans
  le train — reserve de sanity-check).
- split.csv : StudyInstanceUID -> 'train' | 'val' (85/15 stratifie).

Politique de confiance (v1) :
- conf >= CONF_KEEP (0.70) -> label garde
- Effusion : CONF_KEEP_EFFUSION (0.80) — label connu bruite
- sinon -> NaN (masque dans la loss)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nlp.extractor import extract_text
from src.nlp.lang_detect import detect_lang
from src.nlp.lexicon import TARGETS

CONF_KEEP = 0.70
CONF_KEEP_EFFUSION = 0.80
VAL_FRAC = 0.15
SEED = 42

# Plafond de prevalence parmi les labels gardes (au-dela, on masque des
# positifs au hasard pour rendre la cible apprenable ; l'AUC est insensible
# au reequilibrage, la BCE non).
PREVALENCE_CAP = 0.65

# Rapport globalement normal -> 0 fiable sur les cibles non mentionnees.
GLOBAL_NORMAL = {
    "en": [r"no significant abnormality", r"unremarkable (mri|mr|examination|study|knee)",
           r"normal (mri|mr|examination|study|knee|appearance)", r"within normal limits"],
    "es": [r"estudio normal", r"sin alteraciones significativas",
           r"dentro de la normalidad", r"examen normal", r"rm normal"],
    "tr": [r"normal s[ıi]n[ıi]rlar", r"patoloji saptanmad[ıi]",
           r"normal bulgular", r"patolojik bulgu yok"],
    "hr": [r"uredan nalaz", r"normalan nalaz", r"bez patologij",
           r"redovit nalaz"],
    "el": [r"φυσιολογικη εικονα", r"χωρισ παθολογικα ευρηματα",
           r"εντοσ φυσιολογικων οριων"],
    "de": [r"unauffalliger befund", r"normalbefund",
           r"kein pathologischer befund", r"o\.?b\.?\b"],
    "bg": [r"нормална находка", r"без патологични", r"без патология"],
    "nl": [r"normaal beeld", r"geen afwijkingen", r"normale mri"],
    "fr": [r"examen normal", r"pas d'anomalie", r"irm normale",
           r"sans anomalie"],
}

DATA = Path("data")
OUT = DATA / "labels"


def build_pseudo_labels() -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(DATA / "csv" / "train.csv")
    unlabeled = train[train[TARGETS].isna().any(axis=1)].copy()
    print(f"etudes non-gold a etiqueter : {len(unlabeled)}")

    unlabeled["lang"] = unlabeled["Report"].map(detect_lang)

    rows = []
    for _, r in unlabeled.iterrows():
        pred = extract_text(str(r["Report"]), r["lang"])
        row = {"StudyInstanceUID": r["StudyInstanceUID"], "lang": r["lang"]}
        for t in TARGETS:
            label, conf = pred[t]
            row[t] = np.nan if label is None else float(label)
            row[t + "_conf"] = conf
        rows.append(row)
    df = pd.DataFrame(rows)
    reports = unlabeled.set_index(df.index)["Report"] if len(df) == len(unlabeled) else unlabeled["Report"]
    df.to_csv(OUT / "pseudo_labels.csv", index=False)
    print("sauvegarde:", OUT / "pseudo_labels.csv")
    reports = unlabeled["Report"].reset_index(drop=True)
    return df, reports


def apply_global_normal(df: pd.DataFrame, reports: pd.Series) -> pd.DataFrame:
    """Rapport globalement normal -> 0 (conf 0.80) sur les cibles sans mention.

    Garde-fou : on n'applique QUE si l'extracteur n'a trouve AUCUN positif sur
    les 12 cibles (sinon le rapport n'est pas "globalement normal" — les
    phrases locales type "within normal limits" ne doivent pas injecter de
    faux 0 dans des etudes pathologiques).
    """
    import re

    from src.nlp.extractor import normalize

    out = df.copy()
    compiled = {lang: [re.compile(p) for p in pats] for lang, pats in GLOBAL_NORMAL.items()}
    n_applied = 0
    for idx, r in out.iterrows():
        lang = r["lang"]
        if lang not in compiled:
            continue
        has_positive = (out.loc[idx, TARGETS] == 1.0).any()
        if has_positive:
            continue
        norm = normalize(str(reports.loc[idx]), lang)
        if any(p.search(norm) for p in compiled[lang]):
            for t in TARGETS:
                if pd.isna(out.loc[idx, t]):
                    out.loc[idx, t] = 0.0
                    out.loc[idx, t + "_conf"] = 0.80
            n_applied += 1
    print(f"regle 'globalement normal' appliquee a {n_applied} etudes")
    return out


def apply_confidence_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Masque les labels sous le seuil de confiance (NaN dans les colonnes)."""
    out = df.copy()
    for t in TARGETS:
        thr = CONF_KEEP_EFFUSION if t == "Effusion" else CONF_KEEP
        keep = out[t + "_conf"] >= thr
        out.loc[~keep, t] = np.nan
    return out


def cap_positive_prevalence(df: pd.DataFrame, cap: float = PREVALENCE_CAP,
                            seed: int = SEED) -> pd.DataFrame:
    """Masque des positifs au hasard si la prevalence des labels gardes > cap."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for t in TARGETS:
        pos_idx = out.index[out[t] == 1.0]
        neg_idx = out.index[out[t] == 0.0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_pos == 0 or n_neg == 0:
            continue
        prev = n_pos / (n_pos + n_neg)
        if prev > cap:
            n_pos_keep = int(round(cap * n_neg / (1 - cap)))
            drop = rng.choice(pos_idx, size=n_pos - min(n_pos_keep, n_pos),
                              replace=False)
            out.loc[drop, t] = np.nan
    return out


def make_split(df: pd.DataFrame, val_frac: float | None = None,
               seed: int | None = None) -> pd.Series:
    """Split 85/15 au niveau etude, stratifie sur (langue, nb de positifs).

    On n'utilise PAS les 58 gold : pseudo-labels uniquement.
    """
    val_frac = VAL_FRAC if val_frac is None else val_frac
    seed = SEED if seed is None else seed
    y = df[TARGETS].fillna(0).astype(int)
    n_pos = y.sum(axis=1)
    pos_bin = pd.cut(n_pos, bins=[-1, 0, 2, 4, 12], labels=["0", "1-2", "3-4", "5+"])
    strat_key = df["lang"].astype(str) + "_" + pos_bin.astype(str)

    rng = np.random.default_rng(seed)
    val_idx = []
    for _, group in df.groupby(strat_key):
        n_val = max(1, int(round(len(group) * val_frac))) if len(group) >= 4 else (
            1 if rng.random() < val_frac * len(group) else 0
        )
        if n_val > 0:
            val_idx.extend(
                rng.choice(group.index, size=min(n_val, len(group)), replace=False)
            )
    split = pd.Series("train", index=df.index, name="split")
    split.loc[val_idx] = "val"
    return split


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, reports = build_pseudo_labels()
    df = apply_global_normal(df, reports)
    masked = apply_confidence_mask(df)
    masked = cap_positive_prevalence(masked)
    split = make_split(masked)

    out = masked[["StudyInstanceUID", "lang"] + TARGETS].copy()
    out["split"] = split.values
    out.to_csv(OUT / "train_val_labels.csv", index=False)

    print("\n=== Couverture apres masquage (fraction de labels gardes) ===")
    cov = out[TARGETS].notna().mean().round(3)
    print(cov.to_string())
    kept_cells = out[TARGETS].notna().sum().sum()
    total_cells = len(out) * len(TARGETS)
    print(f"\ncellules gardees : {kept_cells}/{total_cells} ({100*kept_cells/total_cells:.1f}%)")
    full_na = out[TARGETS].isna().all(axis=1).sum()
    print(f"etudes sans aucun label garde : {full_na} ({100*full_na/len(out):.1f}%)")

    print("\n=== Prevalences (labels gardes) ===")
    prev = out[TARGETS].mean().round(3).sort_values(ascending=False)
    print(prev.to_string())

    print("\n=== Split ===")
    print(out["split"].value_counts().to_string())
    tr = out[out.split == "train"][TARGETS].mean()
    va = out[out.split == "val"][TARGETS].mean()
    comp = pd.DataFrame({"prev_train": tr.round(3), "prev_val": va.round(3)})
    print(comp.to_string())
    print("\nsauvegarde:", OUT / "train_val_labels.csv")


if __name__ == "__main__":
    main()
