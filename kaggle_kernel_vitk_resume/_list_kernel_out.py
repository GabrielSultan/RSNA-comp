# -*- coding: utf-8 -*-
"""Liste les fichiers de sortie des kernels rgb25d proto, sans les telecharger."""
from __future__ import annotations

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
for i in range(6):
    slug = f"sultent/rsna-knee-convert-rgb-2-5d-proto-part{i}"
    result = api.kernels_list_files(slug, page_size=50)
    files = list(result.files) if result is not None and result.files else []
    print(slug, "n", len(files))
    for item in files:
        data = {k: getattr(item, k) for k in dir(item) if not k.startswith("_")}
        print(" ", data)
