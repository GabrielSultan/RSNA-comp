# -*- coding: utf-8 -*-
"""Taille reelle du zip de sortie du kernel part0, sans le telecharger."""
from __future__ import annotations

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as client:
    request = ApiListKernelSessionOutputRequest()
    request.user_name = "sultent"
    request.kernel_slug = "rsna-knee-convert-rgb-2-5d-proto-part0"
    api._set_paging(request, 20, None)
    response = client.kernels.kernels_api_client.list_kernel_session_output(request)

for item in response.files or []:
    resp = requests.get(item.url, stream=True, timeout=60)
    length = resp.headers.get("Content-Length")
    print(item.file_name, "status", resp.status_code, "content-length", length)
    resp.close()
