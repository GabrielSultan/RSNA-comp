# -*- coding: utf-8 -*-
"""Detection de langue des rapports (seed fixe, reproductible)."""
from __future__ import annotations

from langdetect import DetectorFactory, detect

from .lexicon import LANGS

DetectorFactory.seed = 0


def detect_lang(text: str) -> str:
    """Retourne un code langue du lexique, ou 'other' si hors couverture."""
    t = (text or "").strip()
    if len(t) < 20:
        return "other"
    try:
        lang = detect(t)
    except Exception:
        return "other"
    return lang if lang in LANGS else "other"
