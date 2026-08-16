# -*- coding: utf-8 -*-
"""Alias de kneenet_v8 (ne pas y mettre une nouvelle archi).

L'architecture figee vit dans `kneenet_v8.py`. Pour experimenter :
copier ce fichier-la vers `kneenet_v9.py` (voir `registry.py`).
"""
from src.models.kneenet_v8 import KneeNet  # noqa: F401
from src.models.registry import count_params  # noqa: F401
