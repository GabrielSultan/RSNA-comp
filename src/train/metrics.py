# -*- coding: utf-8 -*-
"""Macro-AUC avec labels partiellement masques (-1 = masque)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(y_true: np.ndarray, y_score: np.ndarray,
              target_names: list[str] | None = None) -> tuple[float, dict[str, float]]:
    """y_true (N, C) avec -1 = masque ; y_score (N, C) probabilites.

    Retourne (macro-AUC, {cible: AUC}) en ignorant les cellules masquees et
    les cibles sans les deux classes. Labels mous (hors {0, 1}) : binarises
    a 0.5 pour l'AUC (la loss, elle, garde les cibles molles).
    """
    aucs: dict[str, float] = {}
    for c in range(y_true.shape[1]):
        mask = y_true[:, c] >= 0
        yt, ys = y_true[mask, c], y_score[mask, c]
        if yt.size == 0:
            continue
        # labels mous (blend) : AUC binaire au seuil 0.5
        uniq = np.unique(yt)
        if not np.isin(uniq, (0.0, 1.0)).all():
            yt = (yt >= 0.5).astype(np.float64)
        if len(np.unique(yt)) < 2:
            continue
        name = target_names[c] if target_names else f"c{c}"
        aucs[name] = float(roc_auc_score(yt, ys))
    macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return macro, aucs
