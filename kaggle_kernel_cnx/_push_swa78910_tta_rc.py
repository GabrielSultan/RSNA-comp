# -*- coding: utf-8 -*-
"""CNX-r7 SWA ep7-10 seul. TTA x3 : rotation +5, rotation -5, contraste 1.08. Deux T4."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PY = ROOT / "kaggle_kernel_cnx" / "_push_swa78910.py"
PUSH = ROOT / "kaggle_kernel_cnx" / "_push_swa78910_tta_rc"
OUT = ROOT / "kaggle_kernel_cnx" / "_out_swa78910_tta_rc"
KERNEL = "sultent/rsna-knee-cnx-r7-swa78910-tta-rc"
TITLE = "rsna-knee-cnx-r7-swa78910-tta-rc"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
MSG = (
    "CNX-r7 IN22k v3b oofmask SWA ep7-10 seul, TTA x3 "
    "(rotation +5 deg, rotation -5 deg, contraste 1.08), "
    "sans hflip, sans zoom, pas d'ensemble, 2 T4, forward fp32"
)

TAIL_OLD = '''model, _cfg = load_model(WEIGHTS_PATH)
n_series = int(getattr(model, "n_series", 6))

rows = []
print(f"{len(test)} etudes a predire | sans TTA | crop {CROP_MM} mm | cnx forward fp32", flush=True)
for i, uid in enumerate(test.StudyInstanceUID):
    study_dir = test_series_dir / uid
    srows = series_by_uid.get(uid, series.iloc[0:0])

    vol = study_to_tensor(
        study_dir,
        srows,
        n_series=n_series,
        rgb_neighbors=False,
        img_size=224,
    )
    if vol.shape != (n_series, 24, 224, 224):
        raise RuntimeError(f"volume inattendu {vol.shape}")
    x0 = torch.from_numpy(vol).unsqueeze(0).to(device=DEVICE, dtype=torch.float32)
    logits = _predict_logits(model, x0)
    prob = torch.sigmoid(logits.float()).cpu().numpy()[0]
    rows.append([uid] + prob.tolist())
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(test)} ok", flush=True)
'''

TAIL_NEW = r'''import copy
import math
from concurrent.futures import ThreadPoolExecutor

def _rotate(x, degrees):
    """Rotation dans le plan de chaque coupe. Pas de hflip, pas de zoom."""
    if x.ndim != 5:
        raise RuntimeError(tuple(x.shape))
    deg = float(degrees)
    if abs(deg) != 5.0:
        raise RuntimeError(deg)
    b, s, d, h, w = x.shape
    ang = deg * math.pi / 180.0
    c, sn = math.cos(ang), math.sin(ang)
    theta = x.new_tensor([[c, -sn, 0.0], [sn, c, 0.0]]).unsqueeze(0)
    theta = theta.repeat(b * s * d, 1, 1)
    flat = x.reshape(b * s * d, 1, h, w)
    grid = torch.nn.functional.affine_grid(theta, flat.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(
        flat, grid, mode="bilinear", padding_mode="border", align_corners=False,
    )
    return out.reshape(b, s, d, h, w)


def _contrast(x, factor):
    """Contraste leger autour de la moyenne de chaque coupe. Pas de luminosite ajoutee."""
    fac = float(factor)
    if abs(fac - 1.08) > 1e-6:
        raise RuntimeError(fac)
    mean = x.mean(dim=(-2, -1), keepdim=True)
    return ((x - mean) * fac + mean).clamp(0.0, 1.0)


def _tta(model, device, x_cpu):
    with torch.no_grad():
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        x = x_cpu.to(device).float()
        with torch.autocast("cuda", enabled=False):
            p_pos = torch.sigmoid(model(_rotate(x, 5)).float())
            p_neg = torch.sigmoid(model(_rotate(x, -5)).float())
            p_con = torch.sigmoid(model(_contrast(x, 1.08)).float())
        return ((p_pos + p_neg + p_con) / 3.0).float().cpu()


N_GPU = torch.cuda.device_count() if DEVICE == "cuda" else 1
if DEVICE == "cuda" and N_GPU < 2:
    raise RuntimeError(("attendu 2 GPU", N_GPU))
devices = ["cuda:0", "cuda:1"] if DEVICE == "cuda" else ["cpu"]
base_model, _cfg = load_model(WEIGHTS_PATH)
models = [copy.deepcopy(base_model).to(dev).eval() for dev in devices]
del base_model
n_series = int(getattr(models[0], "n_series", 6))
print(
    f"{len(test)} etudes | TTA x3 rotation +5, rotation -5, contraste 1.08, "
    "sans hflip, sans zoom, pas d'ensemble | "
    f"gpus visibles {N_GPU} | {' '.join(devices)} | cnx forward fp32",
    flush=True,
)

pool = ThreadPoolExecutor(max_workers=len(devices))
inflight = []
rows_map = {}
uids = list(test.StudyInstanceUID)
for i, uid in enumerate(uids):
    study_dir = test_series_dir / uid
    srows = series_by_uid.get(uid, series.iloc[0:0])
    vol = study_to_tensor(
        study_dir,
        srows,
        n_series=n_series,
        rgb_neighbors=False,
        img_size=224,
    )
    if vol.shape != (n_series, 24, 224, 224):
        raise RuntimeError(f"volume inattendu {vol.shape}")
    x0 = torch.from_numpy(vol.copy()).unsqueeze(0)
    slot = i % len(devices)
    inflight.append((uid, devices[slot], pool.submit(_tta, models[slot], devices[slot], x0)))
    if len(inflight) >= len(devices):
        uid_d, dev_d, fut_d = inflight.pop(0)
        rows_map[uid_d] = fut_d.result().numpy()[0]
        if (i + 1) % 25 == 0 or len(uids) <= 20:
            print(f"{i+1}/{len(uids)} {dev_d}", flush=True)
while inflight:
    uid_d, dev_d, fut_d = inflight.pop(0)
    rows_map[uid_d] = fut_d.result().numpy()[0]
    print(f"fini {uid_d} {dev_d}", flush=True)
pool.shutdown(wait=True)
if set(map(str, rows_map)) != set(map(str, uids)):
    raise RuntimeError(("etudes manquantes", len(rows_map), len(uids)))
rows = [[uid] + rows_map[uid].tolist() for uid in uids]
'''


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _load_base():
    spec = importlib.util.spec_from_file_location("push_swa78910", BASE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_nb(base) -> float:
    base.build_swa()
    gold = base.package()
    spec = importlib.util.spec_from_file_location(
        "make_nb_cnx", ROOT / "kaggle_kernel_cnx" / "_make_nb.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    local = str(base.UPLOAD.relative_to(ROOT)).replace("\\", "/")
    mod.build(
        gold=gold,
        weight=base.WEIGHT,
        dataset=base.DATASET,
        local_dir=local,
        out_dir=PUSH,
        kernel_id=KERNEL,
        title=TITLE,
    )
    nb_path = PUSH / "submit.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    md = _text(nb["cells"][0])
    md = md.replace("T4 — sans TTA", "T4 x2 — TTA rotation contraste")
    md = md.replace(
        "Notebook **self-contained** (internet off, GPU **T4**). **Sans TTA.**",
        "Notebook **self-contained** (internet off, GPU **T4 x2**). **CNX seul, TTA x3.**",
    )
    md = md.replace(
        "- un seul forward\n",
        "- **TTA x3** : rotation +5 deg, rotation -5 deg, contraste 1.08\n"
        "- sans hflip, sans zoom, pas d'echange de labels. Pas d'ensemble\n"
        "- etudes reparties sur cuda:0 et cuda:1\n",
    )
    _set(nb["cells"][0], md)
    found = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = _text(cell)
        if "mm | sans TTA" not in src:
            continue
        found += 1
        src = src.replace(
            '"| n_series", n_series, "| img 224 | crop", CROP_MM, "mm | sans TTA",',
            '"| n_series", n_series, "| img 224 | crop", CROP_MM, "mm | TTA x3 rotation contraste | 2 GPU",',
        )
        src = src.replace(
            "    model.to(DEVICE).eval()\n    return model, cfg\n",
            "    model.eval()\n    return model, cfg\n",
        )
        if TAIL_OLD not in src:
            raise SystemExit("boucle sans TTA introuvable")
        src = src.replace(TAIL_OLD, TAIL_NEW)
        _set(cell, src)
        ast.parse(src)
    if found != 1:
        raise SystemExit(f"cellules TTA {found}")
    blob = "\n".join(_text(c) for c in nb["cells"])
    for needle in (
        "attendu 2 GPU",
        "_rotate(x, 5)",
        "_rotate(x, -5)",
        "_contrast(x, 1.08)",
        "contraste 1.08",
        "sans hflip",
        "sans zoom",
        "gpus visibles",
        "cuda:0",
        "cuda:1",
        "enabled=False",
        "cnx forward fp32",
        "submission.csv",
        "swa-7-8-9-10",
        "[7, 8, 9, 10]",
        "cnx_r7_in22k_v3b_oofmask_full",
        "convnext_small.fb_in22k_ft_in1k",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "sans TTA",
        "un seul forward",
        "model_vit",
        "model_v20h",
        "_hflip",
        "_center_zoom",
        "_swap_medlat",
        "zoom 1.05",
        "zoom 1.10",
        "TTA_VARIANTS",
        "apply_affine",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    if "laterality=True," in blob or "laterality=True)" in blob:
        raise SystemExit("residu laterality=True")
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((PUSH / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("id") != KERNEL or meta.get("machine_shape") != ACCEL or meta.get("enable_internet"):
        raise SystemExit(f"metadata {meta}")
    if meta.get("dataset_sources") != ["sultent/rsna-knee-cnx-r7-swa78910"]:
        raise SystemExit(meta.get("dataset_sources"))
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    print("ecrit", PUSH, "gold", round(gold, 4), flush=True)
    return gold


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        raise


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def _transient(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "expecting value", "remote end closed", "connection aborted",
        "temporarily unavailable", "timed out", "timeout", "502", "503", "504",
    ))


def push() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

    meta = json.loads((PUSH / "kernel-metadata.json").read_text(encoding="utf-8"))
    nb = json.loads((PUSH / "submit.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
    req = ApiSaveKernelRequest()
    req.slug = meta["id"]
    req.new_title = meta["title"]
    req.text = json.dumps(nb)
    req.language = meta["language"]
    req.kernel_type = meta["kernel_type"]
    req.is_private = True
    req.enable_gpu = True
    req.enable_tpu = False
    req.enable_internet = False
    req.dataset_data_sources = meta["dataset_sources"]
    req.competition_data_sources = meta["competition_sources"]
    req.category_ids = meta.get("keywords") or []
    req.session_timeout_seconds = 32400
    req.machine_shape = ACCEL
    req.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
    t0 = time.time()
    while True:
        api = KaggleApi()
        api.authenticate()
        try:
            with api.build_kaggle_client() as client:
                result = client.kernels.kernels_api_client.save_kernel(req)
        except Exception as e:
            http = getattr(e, "response", None)
            text = http.text if http is not None else str(e)
            print(text[:800], flush=True)
            if _quota(text) and time.time() - t0 < 30 * 60:
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            if _transient(text):
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
                continue
            raise
        print("push", result.version_number, result.url, result.error, flush=True)
        if result.error:
            if _quota(result.error) and time.time() - t0 < 30 * 60:
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            raise RuntimeError(result.error)
        version = str(result.version_number)
        (PUSH / "_version.txt").write_text(version + "\n", encoding="utf-8")
        print("kernel version", version, flush=True)
        return version


def _raw_status() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        http = client._http_client
        http._init_session()
        req = ApiGetKernelSessionStatusRequest()
        req.user_name, req.kernel_slug = KERNEL.split("/", 1)
        prep = http._prepare_request("kernels.KernelsApiService", "GetKernelSessionStatus", req)
        resp = http._session.send(
            prep, **http._session.merge_environment_settings(prep.url, {}, None, None, None),
        )
        resp.raise_for_status()
        return resp.text


def wait() -> None:
    t0 = time.time()
    seen = False
    empty = 0
    while True:
        body = _raw_status()
        print("raw", body[:300], flush=True)
        if body.strip() in ("", "{}"):
            empty += 1
            if empty >= 6:
                raise SystemExit("session non demarree")
            time.sleep(POLL_S)
            continue
        empty = 0
        raw = kaggle("kernels", "status", KERNEL).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def _daily_limit(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "maximum number", "submission limit", "daily submission allowance",
        "try again tomorrow", "too many submission", "exceeded",
    ))


def submit(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and "tta-rc" in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "cnx_r7_in22k_v3b_oofmask_full",
        "epoch swa-7-8-9-10",
        "[7, 8, 9, 10]",
        "load strict OK",
        "TTA x3",
        "rotation +5",
        "rotation -5",
        "contraste 1.08",
        "sans hflip",
        "sans zoom",
        "gpus visibles 2",
        "cuda:0",
        "cuda:1",
        "cnx forward fp32",
        "lignes 3",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "zoom 1." in text or "gpus visibles 1" in text or "lat True" in text:
        raise SystemExit("TTA ou GPU incomplet")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("submit", KERNEL, "version", version, flush=True)
    try:
        resp = api.competition_submit_code(
            "submission.csv", MSG, COMPETITION, KERNEL, int(version), quiet=True,
        )
    except Exception as e:
        http = getattr(e, "response", None)
        body = http.text if http is not None else str(e)
        print(body, flush=True)
        if _daily_limit(body):
            raise SystemExit("limite journaliere, kernel pret mais non soumis")
        raise
    print(resp, flush=True)
    print("SOUMIS", KERNEL, "version", version, flush=True)


def main() -> None:
    base = _load_base()
    write_nb(base)
    if "--write" in sys.argv:
        return
    base.ensure_dataset()
    version = push()
    wait()
    submit(version)


if __name__ == "__main__":
    main()
