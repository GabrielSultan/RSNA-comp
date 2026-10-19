# -*- coding: utf-8 -*-
"""CNX-r7 SWA ep7-10 seul, meme TTA x3 que l'ensemble, deux T4, puis soumission."""
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
PUSH = ROOT / "kaggle_kernel_cnx" / "_push_swa78910_tta3"
OUT = ROOT / "kaggle_kernel_cnx" / "_out_swa78910_tta3"
KERNEL = "sultent/rsna-knee-cnx-r7-swa78910-tta3"
TITLE = "rsna-knee-cnx-r7-swa78910-tta3"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
MSG = (
    "CNX-r7 IN22k v3b oofmask SWA ep7-10 seul, TTA x3 "
    "(hflip, center-crop zoom 1.05, center-crop zoom 1.10), "
    "swap seulement apres hflip, pas de rotation, pas d'ensemble, 2 T4, forward fp32"
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
from concurrent.futures import ThreadPoolExecutor

def _hflip(x):
    """Retourner l'image, sans rotation. Sagittal : axe D. Coronal/Axial : axe W."""
    if x.ndim != 5 or int(x.shape[1]) != 6:
        raise RuntimeError(tuple(x.shape))
    y = x.clone()
    for s in (0, 3):
        y[:, s] = torch.flip(y[:, s], dims=(1,))
    for s in (1, 2, 4, 5):
        y[:, s] = torch.flip(y[:, s], dims=(3,))
    return y


def _swap_medlat(p):
    q = p.clone()
    q[:, 2] = p[:, 3]
    q[:, 3] = p[:, 2]
    q[:, 4] = p[:, 5]
    q[:, 5] = p[:, 4]
    return q


def _center_zoom(x, scale):
    """Zoom avant par center-crop puis resize. Pas de rotation."""
    if x.ndim != 5 or float(scale) <= 1.0:
        raise RuntimeError((tuple(x.shape), scale))
    b, s, d, h, w = x.shape
    ch = int(round(h / float(scale)))
    cw = int(round(w / float(scale)))
    if ch < 8 or cw < 8 or ch >= h or cw >= w:
        raise RuntimeError((ch, cw, h, w, scale))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = x[:, :, :, y0:y0 + ch, x0:x0 + cw]
    flat = crop.reshape(b * s * d, 1, ch, cw)
    up = torch.nn.functional.interpolate(flat, size=(h, w), mode="bilinear", align_corners=False)
    return up.reshape(b, s, d, h, w)


def _tta(model, device, x_cpu):
    with torch.no_grad():
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        x = x_cpu.to(device).float()
        with torch.autocast("cuda", enabled=False):
            p_flip = torch.sigmoid(model(_hflip(x)).float())
            p_z1 = torch.sigmoid(model(_center_zoom(x, 1.05)).float())
            p_z2 = torch.sigmoid(model(_center_zoom(x, 1.10)).float())
        return ((_swap_medlat(p_flip) + p_z1 + p_z2) / 3.0).float().cpu()


N_GPU = torch.cuda.device_count() if DEVICE == "cuda" else 1
if DEVICE == "cuda" and N_GPU < 2:
    raise RuntimeError(("attendu 2 GPU", N_GPU))
devices = ["cuda:0", "cuda:1"] if DEVICE == "cuda" else ["cpu"]
base_model, _cfg = load_model(WEIGHTS_PATH)
models = [copy.deepcopy(base_model).to(dev).eval() for dev in devices]
del base_model
n_series = int(getattr(models[0], "n_series", 6))
print(
    f"{len(test)} etudes | TTA x3 hflip + zoom 1.05 + zoom 1.10, "
    "swap seulement apres hflip, sans identite, pas de rotation | "
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
    md = md.replace("T4 — sans TTA", "T4 x2 — TTA x3")
    md = md.replace(
        "Notebook **self-contained** (internet off, GPU **T4**). **Sans TTA.**",
        "Notebook **self-contained** (internet off, GPU **T4 x2**). **CNX seul, TTA x3.**",
    )
    md = md.replace(
        "- un seul forward\n",
        "- **TTA x3**, sans l'identite : hflip, center-crop zoom 1.05, center-crop zoom 1.10\n"
        "- echange med/lat seulement apres le hflip. Pas de rotation. Pas d'ensemble\n"
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
            '"| n_series", n_series, "| img 224 | crop", CROP_MM, "mm | TTA x3 | 2 GPU",',
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
        "_center_zoom(x, 1.05)",
        "_center_zoom(x, 1.10)",
        "_swap_medlat(p_flip)",
        "gpus visibles",
        "cuda:0",
        "cuda:1",
        "enabled=False",
        "cnx forward fp32",
        "sans identite",
        "pas de rotation",
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
    t0 = time.time()
    while True:
        try:
            out = kaggle("kernels", "push", "-p", str(PUSH), "--accelerator", ACCEL, "-t", "32400")
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text) and time.time() - t0 < 30 * 60:
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            if _transient(text):
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
                continue
            raise
        print(out, flush=True)
        if _quota(out) and time.time() - t0 < 30 * 60:
            print("quota GPU, retry 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        version = m.group(1)
        (PUSH / "_version.txt").write_text(version + "\n", encoding="utf-8")
        print("kernel version", version, flush=True)
        return version


def wait() -> None:
    t0 = time.time()
    seen = False
    while True:
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
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and "tta3" in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "cnx_r7_in22k_v3b_oofmask_full",
        "epoch swa-7-8-9-10",
        "[7, 8, 9, 10]",
        "load strict OK",
        "TTA x3",
        "zoom 1.05",
        "zoom 1.10",
        "swap seulement apres hflip",
        "sans identite",
        "gpus visibles 2",
        "cuda:0",
        "cuda:1",
        "cnx forward fp32",
        "lignes 3",
        "pas de rotation",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "sans TTA" in text or "lat True" in text or "gpus visibles 1" in text:
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
