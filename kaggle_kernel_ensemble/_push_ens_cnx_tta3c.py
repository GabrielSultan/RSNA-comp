# -*- coding: utf-8 -*-
"""Ensemble tta3b, TTA remplace par rotation +/-6, contraste et luminosite. Deux T4. Pas de soumission."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_cnx_tta3b" / "submit.ipynb"
SLUG = "rsna-knee-ens-cnx-tta3c"
PUSH = KERNEL / "_push_ens_cnx_tta3c"
OUT = KERNEL / "_out_ens_cnx_tta3c"
KAGGLE_ID = f"sultent/{SLUG}"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 10 * 3600
MSG = (
    "ens CNX-r7 SWA 7-10 + ViT-k320 OAI3 SWA 1-3 + v20h distil crop145 SWA 7-9, "
    "TTA x4 rotation +6, rotation -6, contraste 1.08, luminosite +0.05, "
    "sans hflip, sans zoom, 2 T4, pas soumis"
)

OLD_FUNCS = '''def _hflip(x):
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
    up = torch.nn.functional.interpolate(flat, size=(h, w), mode='bilinear', align_corners=False)
    return up.reshape(b, s, d, h, w)


print(
    'TTA cnx x3 hflip + zoom 1.05 + zoom 1.10, swap seulement apres hflip, '
    'vit identite + hflip + zoom 1.05, distil identite + hflip, pas de rotation',
    flush=True,
)
'''

NEW_FUNCS = '''import math


def _rotate(x, degrees):
    """Rotation dans le plan de chaque coupe. Pas de hflip, pas de zoom."""
    if x.ndim != 5:
        raise RuntimeError(tuple(x.shape))
    deg = float(degrees)
    if abs(deg) != 6.0:
        raise RuntimeError(deg)
    b, s, d, h, w = x.shape
    ang = deg * math.pi / 180.0
    c, sn = math.cos(ang), math.sin(ang)
    theta = x.new_tensor([[c, -sn, 0.0], [sn, c, 0.0]]).unsqueeze(0)
    theta = theta.repeat(b * s * d, 1, 1)
    flat = x.reshape(b * s * d, 1, h, w)
    grid = torch.nn.functional.affine_grid(theta, flat.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(
        flat, grid, mode='bilinear', padding_mode='border', align_corners=False,
    )
    return out.reshape(b, s, d, h, w)


def _contrast(x):
    """Contraste 1.08 autour de la moyenne de chaque coupe."""
    mean = x.mean(dim=(-2, -1), keepdim=True)
    return ((x - mean) * 1.08 + mean).clamp(0.0, 1.0)


def _brightness(x):
    """Luminosite +0.05, bornes [0, 1]."""
    return (x + 0.05).clamp(0.0, 1.0)


def _views(x):
    return (_rotate(x, 6), _rotate(x, -6), _contrast(x), _brightness(x))


print(
    'TTA x4 rotation +6, rotation -6, contraste 1.08, luminosite +0.05, '
    'cnx vit distil, sans hflip, sans zoom, sans echange de labels',
    flush=True,
)
'''

OLD_CNX = '''        with torch.autocast('cuda', enabled=False):
            p_flip = torch.sigmoid(model_cnx(_hflip(x_cnx)).float())
            p_z1 = torch.sigmoid(model_cnx(_center_zoom(x_cnx, 1.05)).float())
            p_z2 = torch.sigmoid(model_cnx(_center_zoom(x_cnx, 1.10)).float())
        return ((_swap_medlat(p_flip) + p_z1 + p_z2) / 3.0).float().cpu()
'''

NEW_CNX = '''        with torch.autocast('cuda', enabled=False):
            ps = [torch.sigmoid(model_cnx(v).float()) for v in _views(x_cnx)]
        return (sum(ps) / 4.0).float().cpu()
'''

OLD_REST = '''        p_vit_id = torch.sigmoid(_forward(model_vit, x_vit).float())
        p_vit_flip = torch.sigmoid(_forward(model_vit, _hflip(x_vit)).float())
        p_vit_zoom = torch.sigmoid(_forward(model_vit, _center_zoom(x_vit, 1.05)).float())
        p_vit = (p_vit_id + _swap_medlat(p_vit_flip) + p_vit_zoom) / 3.0
        p_h_id = torch.sigmoid(_forward(model_v20h, x_h).float())
        p_h_flip = torch.sigmoid(_forward(model_v20h, _hflip(x_h)).float())
        p_v20h = 0.5 * (p_h_id + _swap_medlat(p_h_flip))
        return p_vit.float().cpu(), p_v20h.float().cpu()
'''

NEW_REST = '''        def _avg(model, x):
            ps = [torch.sigmoid(_forward(model, v).float()) for v in _views(x)]
            return sum(ps) / 4.0
        p_vit = _avg(model_vit, x_vit)
        p_v20h = _avg(model_v20h, x_h)
        return p_vit.float().cpu(), p_v20h.float().cpu()
'''


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _replace(nb: dict, old: str, new: str, expect: int) -> None:
    found = 0
    for cell in nb["cells"]:
        src = _text(cell)
        n = src.count(old)
        if not n:
            continue
        found += n
        _set(cell, src.replace(old, new))
    if found != expect:
        raise SystemExit(f"remplacement x{found}, attendu {expect}: {old[:80]!r}")


def write_nb() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    _replace(nb, "rsna-knee-ens-cnx-tta3b", SLUG, 1)
    _replace(
        nb,
        "Pas de notebook public. Pas de rotation.\n",
        "Pas de notebook public. Pas de hflip, pas de zoom.\n",
        1,
    )
    _replace(
        nb,
        "**TTA x3, trois passages, sans l'identite** : hflip, center-crop zoom 1.05,\n"
        "center-crop zoom 1.10. Le hflip retourne l'image (sagittal sur l'axe des\n"
        "coupes, coronal/axial sur la largeur) puis echange menisque et OA\n"
        "medial/lateral. Les zooms recadrent le centre et redimensionnent, sans\n"
        "echange de labels. Moyenne des trois sigmoides.\n",
        "**TTA x4, quatre passages, sans l'identite** : rotation +6 deg, rotation -6 deg,\n"
        "contraste 1.08, luminosite +0.05. Meme recette sur le CNX, le ViT et le distil.\n"
        "Pas de hflip, pas de zoom, pas d'echange de labels. Moyenne des quatre sigmoides.\n",
        1,
    )
    _replace(
        nb,
        "  Entree 320, FOV 130 mm. **Identite + hflip + center-crop zoom 1.05**.\n"
        "  Echange de labels seulement apres le hflip. Pas de rotation.\n",
        "  Entree 320, FOV 130 mm. **Rotation +/-6, contraste 1.08, luminosite +0.05.**\n",
        1,
    )
    _replace(
        nb,
        "  Entree 224, FOV 145 mm. **Identite + hflip**, meme echange de labels.\n",
        "  Entree 224, FOV 145 mm. **Meme TTA x4**, sans hflip.\n",
        1,
    )
    _replace(
        nb,
        "print('TTA             : cnx x3, vit hflip + zoom 1.05, distil hflip, 2 GPU')",
        "print('TTA             : cnx vit distil x4 rotation contraste luminosite, 2 GPU')",
        1,
    )
    _replace(
        nb,
        "print(len(test), 'etudes | TTA cnx x3 | vit hflip + zoom 1.05 | distil hflip | "
        "2 GPU | pas de rotation | FOV vit/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        "print(len(test), 'etudes | TTA x4 rotation +6 rotation -6 contraste 1.08 "
        "luminosite +0.05 | cnx vit distil | 2 GPU | sans hflip sans zoom | "
        "FOV vit/cnx 130 mm, v20h 145 mm | cnx forward fp32')",
        1,
    )
    _replace(nb, OLD_FUNCS, NEW_FUNCS, 1)
    _replace(nb, OLD_CNX, NEW_CNX, 1)
    _replace(nb, OLD_REST, NEW_REST, 1)
    for cell in nb["cells"]:
        src = _text(cell)
        if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
            ast.parse(src)
    blob = "\n".join(_text(c) for c in nb["cells"])
    for needle in (
        SLUG,
        "attendu 2 GPU",
        "DEV_REST = 'cuda:1'",
        "model_cnx.to(DEV_CNX)",
        "model_vit.to(DEV_REST)",
        "model_v20h.to(DEV_REST)",
        "_rotate(x, 6)",
        "_rotate(x, -6)",
        "_contrast(x)",
        "_brightness(x)",
        "contraste 1.08",
        "luminosite +0.05",
        "sans hflip",
        "sans zoom",
        "enabled=False",
        "cnx forward fp32",
        "(0.00, 0.40, 0.60),  # ACL",
        "submission.csv",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "_hflip",
        "_center_zoom",
        "_swap_medlat",
        "zoom 1.05",
        "zoom 1.10",
        "pas de rotation",
        "apply_affine",
        "TTA_VARIANTS",
        "model_v20b",
        "model_coat",
        "degnonguidi",
        "laterality=True",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((KERNEL / "_push_ens_cnx_tta3b" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KAGGLE_ID
    meta["title"] = SLUG
    meta["machine_shape"] = ACCEL
    meta["enable_gpu"] = True
    meta["enable_internet"] = False
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(nb["cells"]), flush=True)


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
        req.user_name, req.kernel_slug = KAGGLE_ID.split("/", 1)
        prep = http._prepare_request("kernels.KernelsApiService", "GetKernelSessionStatus", req)
        resp = http._session.send(
            prep, **http._session.merge_environment_settings(prep.url, {}, None, None, None),
        )
        if resp.status_code == 429:
            return "RETRY"
        resp.raise_for_status()
        return resp.text


def wait() -> None:
    t0 = time.time()
    seen = False
    empty = 0
    while True:
        body = _raw_status()
        print("raw", body[:300], flush=True)
        if body == "RETRY":
            time.sleep(POLL_S)
            continue
        if body.strip() in ("", "{}"):
            empty += 1
            if empty >= 6:
                raise SystemExit("session non demarree")
            time.sleep(POLL_S)
            continue
        empty = 0
        raw = kaggle("kernels", "status", KAGGLE_ID).strip()
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


def check_log() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KAGGLE_ID, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and SLUG in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3 gold 0.9903",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10 lat False gold 0.976",
        "cnx forward fp32",
        "gpus visibles 2",
        "cuda:1",
        "rotation +6",
        "rotation -6",
        "contraste 1.08",
        "luminosite +0.05",
        "sans hflip",
        "sans zoom",
        "vit 0.00 v20h 0.40 cnx 0.60",
        "lignes 3",
        "lat False",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "lat True" in text or "gpus visibles 1" in text or "zoom 1." in text or "hflip" in text.split("sans hflip")[-1]:
        raise SystemExit("TTA ou GPU incomplet")
    print("LOG_OK", log.name, flush=True)
    print("PAS_DE_SOUMISSION", KAGGLE_ID, flush=True)


def main() -> None:
    write_nb()
    if "--write" in sys.argv:
        return
    push()
    wait()
    check_log()


if __name__ == "__main__":
    main()
