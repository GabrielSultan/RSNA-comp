# -*- coding: utf-8 -*-
"""Metriques gold (meme formule que l'extracteur lexique v1).

AUC : score = conf si pred=1, 1-conf si pred=0, 0.5 si masque.
F1  : masques imputes a 0 (conservateur).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.nlp.lexicon import TARGETS


def _is_filled(v: Any) -> bool:
    if v is None:
        return False
    try:
        return not bool(pd.isna(v))
    except (TypeError, ValueError):
        return True


def gold_metrics(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """gold et pred indexes alignes (meme ordre d'etudes), colonnes = cibles.

    pred peut avoir des colonnes {cible}_conf ; sinon conf=1.0 sur les remplis.
    """
    targets = targets or TARGETS
    gold = gold.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    rows = []
    for t in targets:
        y_true = gold[t].astype(float).values
        yp = pred[t].values
        mask = np.array([_is_filled(v) for v in yp])
        conf_col = t + "_conf"
        if conf_col in pred.columns:
            conf = pred[conf_col].fillna(0).astype(float).values
        else:
            conf = np.where(mask, 1.0, 0.0)
        yp_num = pd.to_numeric(pd.Series(yp), errors="coerce").values
        score = np.where(mask, np.where(yp_num == 1, conf, 1.0 - conf), 0.5)
        try:
            auc = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        yp_bin = np.where(mask, np.nan_to_num(yp_num, nan=0.0), 0)
        f1 = float(f1_score(y_true, yp_bin, zero_division=0))
        covered = mask
        if covered.any():
            acc = float((yp_num[covered] == y_true[covered]).mean())
        else:
            acc = float("nan")
        rows.append({
            "cible": t,
            "AUC": auc,
            "F1": f1,
            "fidelite_couverte": acc,
            "couverture": float(mask.mean()),
            "n_couvert": int(mask.sum()),
            "prev_gold": float(y_true.mean()),
            "prev_pred": float(yp_bin.mean()),
        })
    return pd.DataFrame(rows)


def summarize(stats: pd.DataFrame) -> dict[str, float]:
    return {
        "macro_AUC": float(stats["AUC"].mean()),
        "macro_F1": float(stats["F1"].mean()),
        "macro_fidelite": float(stats["fidelite_couverte"].mean()),
        "couverture_moy": float(stats["couverture"].mean()),
    }


def recommend_production(stats_by_model: dict[str, pd.DataFrame],
                         production: str,
                         preview: str,
                         auc_margin: float = 0.03,
                         f1_margin: float = 0.03) -> str:
    """Ne bascule sur le preview que s'il gagne franchement."""
    if production not in stats_by_model or preview not in stats_by_model:
        return production
    a = summarize(stats_by_model[production])
    b = summarize(stats_by_model[preview])
    if (b["macro_AUC"] >= a["macro_AUC"] + auc_margin
            and b["macro_F1"] >= a["macro_F1"] + f1_margin):
        return preview
    return production
