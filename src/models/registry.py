# -*- coding: utf-8 -*-
"""Catalogue des architectures KneeNet.

Chaque version = un fichier Python **fige** (ex. kneenet_v8.py). On ne
modifie pas une version deja soumise : on copie vers kneenet_v9.py.

Entrainer n'importe quelle version :
  python -X utf8 -m src.train.train --config configs/kneenet_v8.yaml
  python -X utf8 -m src.train.train --list-models

Nouvelle architecture :
  1. copier src/models/kneenet_v8.py -> src/models/kneenet_v9.py
  2. changer arch_id = "kneenet_v9" dans la classe
  3. ajouter "kneenet_v9": "src.models.kneenet_v9" dans MODELS ci-dessous
  4. copier configs/kneenet_v8.yaml -> configs/kneenet_v9.yaml
     (model: kneenet_v9, run_name: ... unique)
  5. python -X utf8 -m src.train.train --config configs/kneenet_v9.yaml
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any

import torch.nn as nn

# nom CLI / YAML  ->  module Python qui expose `class KneeNet`
MODELS: dict[str, str] = {
    "kneenet_v8": "src.models.kneenet_v8",
    "kneenet_v8bis": "src.models.kneenet_v8bis",
    "kneenet_v9": "src.models.kneenet_v9",
    "kneenet_v10": "src.models.kneenet_v10",
    "kneenet_v11": "src.models.kneenet_v11",
    "kneenet_v12": "src.models.kneenet_v12",
    "kneenet_v13": "src.models.kneenet_v13",
    "kneenet_v14": "src.models.kneenet_v14",
    "kneenet_v14bis": "src.models.kneenet_v14bis",
    "kneenet_v15": "src.models.kneenet_v15",
    "kneenet_v16": "src.models.kneenet_v16",
    "kneenet_v17": "src.models.kneenet_v17",
    "kneenet_v17bis": "src.models.kneenet_v17bis",
    "kneenet_v17bis2": "src.models.kneenet_v17bis2",
    "kneenet_v18": "src.models.kneenet_v18",
    "kneenet_v18bis": "src.models.kneenet_v18bis",
    "kneenet_v18ter": "src.models.kneenet_v18ter",
    "kneenet_v19": "src.models.kneenet_v19",
    "kneenet_v20a": "src.models.kneenet_v20a",
    "kneenet_v20b": "src.models.kneenet_v20b",
    "kneenet_v20b320": "src.models.kneenet_v20b320",
    "kneenet_v20c": "src.models.kneenet_v20c",
    "kneenet_v20d": "src.models.kneenet_v20d",
    "kneenet_v20e": "src.models.kneenet_v20e",
    "kneenet_v20f": "src.models.kneenet_v20f",
    "kneenet_v20g": "src.models.kneenet_v20g",
    "kneenet_v20h": "src.models.kneenet_v20h",
    "kneenet_v20i": "src.models.kneenet_v20i",
    "kneenet_v20j": "src.models.kneenet_v20j",
    "kneenet_vit": "src.models.kneenet_vit",
    "kneenet_vit_b": "src.models.kneenet_vit_b",
    "kneenet_vit_c": "src.models.kneenet_vit_c",
    "kneenet_vit_d": "src.models.kneenet_vit_d",
    "kneenet_vit_e": "src.models.kneenet_vit_e",
    "kneenet_vit_e_d": "src.models.kneenet_vit_e_d",
    "kneenet_vit_f": "src.models.kneenet_vit_f",
    "kneenet_vit_g": "src.models.kneenet_vit_g",
    "kneenet_vit_h": "src.models.kneenet_vit_h",
    "kneenet_vit_i": "src.models.kneenet_vit_i",
    "kneenet_vit_j": "src.models.kneenet_vit_j",
    "kneenet_vit_k320": "src.models.kneenet_vit_k320",
    "kneenet_coat_a": "src.models.kneenet_coat_a",
    "kneenet_coat_a320": "src.models.kneenet_coat_a320",
    "kneenet_coat_d": "src.models.kneenet_coat_d",
    "kneenet_cnx_a": "src.models.kneenet_cnx_a",
    "kneenet_cnx_r7": "src.models.kneenet_cnx_r7",
    "kneenet_mvlqm": "src.models.kneenet_mvlqm",
}

DEFAULT_MODEL = "kneenet_v8"


def list_models() -> list[str]:
    return sorted(MODELS)


def default_n_series(name: str) -> int:
    cls = get_model_class(name)
    p = inspect.signature(cls.__init__).parameters.get("n_series")
    if p is None or p.default is inspect.Parameter.empty:
        return 3
    return int(p.default)


def get_model_class(name: str) -> type[nn.Module]:
    if name not in MODELS:
        connus = ", ".join(list_models()) or "(aucun)"
        raise SystemExit(
            f"modele inconnu : {name!r}. Connus : {connus}.\n"
            "Pour en ajouter un : copier un kneenet_vN.py et l'inscrire "
            "dans src/models/registry.py (dict MODELS)."
        )
    mod = importlib.import_module(MODELS[name])
    cls = getattr(mod, "KneeNet", None)
    if cls is None:
        raise SystemExit(f"{MODELS[name]} n'expose pas class KneeNet")
    return cls


def build_model(cfg: dict, **overrides: Any) -> nn.Module:
    """Instancie l'archi demandee par cfg['model'] (ou DEFAULT_MODEL).

    Les cles YAML dont le nom matche un argument de KneeNet.__init__
    sont passees automatiquement (backbone, dropout, encode_chunk, ...).
    `overrides` gagne sur le YAML (ex. pretrained=False a l'inference).
    """
    name = overrides.pop("model", None) or cfg.get("model") or DEFAULT_MODEL
    cls = get_model_class(str(name))
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    kwargs: dict[str, Any] = {}
    for key in accepted:
        if key in overrides:
            kwargs[key] = overrides[key]
        elif key in cfg:
            kwargs[key] = cfg[key]
    return cls(**kwargs)


def count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{n/1e6:.1f}M"
