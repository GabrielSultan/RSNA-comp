# -*- coding: utf-8 -*-
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()
status = api.kernels_status("sultent/rsna-knee-vit-k-v3b-resume-ep9")
print("status fields", {k: getattr(status, k) for k in dir(status) if not k.startswith("_") and not callable(getattr(status, k))})
with api.build_kaggle_client() as client:
    request = ApiListKernelSessionOutputRequest()
    request.user_name = "sultent"
    request.kernel_slug = "rsna-knee-vit-k-v3b-resume-ep9"
    api._set_paging(request, 20, None)
    response = client.kernels.kernels_api_client.list_kernel_session_output(request)
log = response.log or ""
Path = __import__("pathlib").Path
out = Path("kaggle_kernel_vitk_resume/_kernel_log_raw.txt")
out.write_text(log, encoding="utf-8")
print("wrote", out, "chars", len(log))
