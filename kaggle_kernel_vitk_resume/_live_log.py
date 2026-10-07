# -*- coding: utf-8 -*-
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as client:
    http = client._http_client
    req = ApiGetKernelSessionLogsStreamRequest()
    req.user_name = "sultent"
    req.kernel_slug = "rsna-knee-vit-k-v3b-resume-ep9"
    req.wait_for_logs_url_seconds = 5
    http._init_session()
    prepared = http._prepare_request(
        "kernels.KernelsApiService", "GetKernelSessionLogsStream", req,
    )
    resp = http._session.send(prepared, stream=True, timeout=(15, 20))
    print("status", resp.status_code, resp.headers.get("Content-Type"), flush=True)
    chunks = []
    try:
        for chunk in resp.iter_content(chunk_size=16384):
            if not chunk:
                break
            chunks.append(chunk)
            if sum(len(c) for c in chunks) > 400_000:
                break
    except Exception as exc:
        print("read stop", type(exc).__name__, exc, flush=True)
text = b"".join(chunks).decode("utf-8", errors="replace")
print("chars", len(text), flush=True)
# garder les lignes utiles
keep = []
for line in text.splitlines():
    low = line.lower()
    if any(s in low for s in ("epoch", "resume", "zips", "error", "traceback", "dataset", "gold", "cuda", "oom")):
        keep.append(line[:300])
print("hits", len(keep), flush=True)
print("\n".join(keep[-80:]))
print("--- tail ---")
print(text[-2500:])
