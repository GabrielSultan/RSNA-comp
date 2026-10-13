# -*- coding: utf-8 -*-
"""Mix3 hybrid025 en INT4 + TTA x3. Notebook Kaggle, pas de soumission competition."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC_NB = KERNEL / "_push_ens_h025_mix3" / "submit.ipynb"
PUSH = KERNEL / "_push_ens_h025_mix3_int4"
OUT = KERNEL / "_out_ens_h025_mix3_int4"
UPLOAD = ROOT / "kaggle_dataset_int4" / "_upload_mix3_oai3"
DATASET = "sultent/rsna-knee-ens-oai3-mix3-int4"
SLUG = "rsna-knee-ens-oai3-mix3-int4-tta-x3"
KAGGLE_KERNEL = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
UPLOAD_PY = ROOT / "kaggle_kernel_vit_e_dino_last" / "_upload_chunked.py"
POLL_S = 30
MAX_WAIT_S = 8 * 3600

SOURCES = (
    (
        "vit",
        ROOT / "kaggle_dataset_vit_k" / "_upload_k320_oai3_swa" / "vit_k320_oai3_swa_ep1_3.pt",
        "vit_k320_oai3_swa_int4.pt",
        "kneenet_vit_k320",
    ),
    (
        "v20b",
        ROOT / "kaggle_dataset_v20b" / "_upload_dino_oai3_swa" / "v20b_dino_oai3_swa_ep1_3.pt",
        "v20b_oai3_swa_int4.pt",
        "kneenet_v20b",
    ),
    (
        "v20h",
        ROOT / "kaggle_dataset_v20h_lat" / "_upload_dino_oai3_swa" / "v20h_dino_oai3_swa_ep1_3.pt",
        "v20h_oai3_swa_int4.pt",
        "kneenet_v20h",
    ),
)


def _quant():
    spec = importlib.util.spec_from_file_location(
        "qint4", ROOT / "kaggle_kernel_int4" / "_quantize_v20b_int4.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(name: str, cfg: dict):
    common = dict(
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
        pretrained=False,
    )
    if name == "kneenet_vit_k320":
        from src.models.kneenet_vit_k320 import KneeNet
        return KneeNet(img_size=320, encode_chunk=2, grad_checkpoint=False, **common)
    if name == "kneenet_v20b":
        from src.models.kneenet_v20b import KneeNet
        return KneeNet(encode_chunk=8, **common)
    if name == "kneenet_v20h":
        from src.models.kneenet_v20h import KneeNet
        return KneeNet(encode_chunk=8, **common)
    raise SystemExit(name)


def quantize_all(q) -> None:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    for tag, src, dst_name, model_name in SOURCES:
        if not src.is_file():
            raise SystemExit(f"manquant {src}")
        print("quantize", tag, src.name, src.stat().st_size, flush=True)
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        cfg = dict(ckpt.get("config") or {})
        if str(cfg.get("model")) != model_name:
            raise SystemExit(cfg.get("model"))
        if str(ckpt.get("epoch")) != "swa-1-2-3":
            raise SystemExit(ckpt.get("epoch"))
        sd = ckpt["model"]
        packed = {}
        n_q = n_keep = 0
        for name, t in sd.items():
            if q._should_quant(name, t):
                packed[name] = q.quantize_per_channel(t)
                n_q += 1
            elif torch.is_tensor(t) and t.is_floating_point():
                packed[name] = t.detach().half().cpu().contiguous()
                n_keep += 1
            else:
                packed[name] = t.detach().cpu() if torch.is_tensor(t) else t
                n_keep += 1
        recon = q.dequantize_state(packed)
        model = _build(model_name, cfg)
        model.load_state_dict(recon, strict=True)
        print(f"  load strict OK {model_name} Q={n_q} keep={n_keep}", flush=True)
        del model, recon
        out = {
            "format": "kneenet_int4_v1",
            "model_int4": packed,
            "config": cfg,
            "epoch": ckpt.get("epoch"),
            "swa_epochs": ckpt.get("swa_epochs"),
            "gold_macro_auc": ckpt.get("gold_macro_auc"),
            "gold_by_epoch": ckpt.get("gold_by_epoch"),
            "val_macro_auc": ckpt.get("val_macro_auc", -1.0),
            "quant": {
                "bits": 4,
                "scheme": "groupwise_symmetric",
                "group_size": q.GROUP_SIZE,
                "n_quantized": n_q,
                "n_kept": n_keep,
            },
        }
        dst = UPLOAD / dst_name
        torch.save(out, dst)
        print(f"  wrote {dst.name} {dst.stat().st_size}", flush=True)
        del ckpt, packed, out
    meta = {
        "title": "RSNA Knee ens OAI3 mix3 INT4",
        "id": DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (UPLOAD / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )


DEQUANT = r'''
def _unpack_int4(packed, numel):
    packed = packed.reshape(-1).to(torch.uint8)
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    q = torch.empty(packed.numel() * 2, dtype=torch.int8)
    q[0::2] = lo.to(torch.int8)
    q[1::2] = hi.to(torch.int8)
    return q[:numel]


def _dequant_entry(entry):
    shape = tuple(entry["shape"])
    out_ch = int(shape[0])
    rest = int(torch.tensor(shape[1:]).prod().item())
    q = _unpack_int4(entry["packed"], int(entry["numel"]))
    scale = entry["scale"].float()
    g = int(entry.get("group_size", 64))
    rest = int(entry.get("rest", rest))
    n_g = int(scale.shape[1])
    grouped = q.to(torch.float32).reshape(out_ch, n_g, g) * scale.unsqueeze(-1)
    rows = grouped.reshape(out_ch, n_g * g)[:, :rest]
    return rows.reshape(shape)


def _sd(ckpt):
    if ckpt.get("format") != "kneenet_int4_v1":
        raise RuntimeError(f"format {ckpt.get('format')}")
    blob = ckpt["model_int4"]
    out = {}
    n_q = 0
    for k, v in blob.items():
        if isinstance(v, dict) and "packed" in v:
            out[k] = _dequant_entry(v)
            n_q += 1
        elif torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.float()
        else:
            out[k] = v
    print("INT4", n_q, "tenseurs quantizes")
    return out


def apply_affine(x, angle_deg=0.0, scale=1.0):
    if angle_deg == 0.0 and scale == 1.0:
        return x
    b, s, d, h, w = x.shape
    xf = x.reshape(b * s * d, 1, h, w)
    angle = x.new_tensor(angle_deg * 3.141592653589793 / 180.0)
    sc = x.new_tensor(scale)
    cos, sin = torch.cos(angle) / sc, torch.sin(angle) / sc
    theta = xf.new_zeros(b, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta = theta.repeat_interleave(s * d, dim=0)
    grid = F.affine_grid(theta, xf.shape, align_corners=False)
    xf = F.grid_sample(xf, grid, align_corners=False, padding_mode="zeros")
    return xf.reshape(b, s, d, h, w)


TTA_VARIANTS = (
    (0.0, 1.00),
    (5.0, 1.00),
    (-5.0, 1.00),
)
'''

LOOP = r'''rows = []
t0 = time.time()
mix_w = torch.tensor(MIX, dtype=torch.float32)
with torch.no_grad():
    for i, uid in enumerate(test.StudyInstanceUID):
        _SERIES_CACHE.clear()
        srows = test_series[test_series.StudyInstanceUID == uid]
        sdir = TEST_SERIES_DIR / uid
        vol_224 = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)
        vol_vit = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=320)
        if vol_vit.shape[-1] != 320 or vol_224.shape[-1] != 224:
            raise RuntimeError((vol_vit.shape, vol_224.shape))
        x_224 = torch.from_numpy(vol_224).unsqueeze(0).to(DEVICE)
        x_vit = torch.from_numpy(vol_vit).unsqueeze(0).to(DEVICE)
        w = mix_w.to(device=x_vit.device)
        acc = None
        for angle_deg, scale in TTA_VARIANTS:
            xv = apply_affine(x_vit, angle_deg, scale)
            xr = apply_affine(x_224, angle_deg, scale)
            p_vit = torch.sigmoid(_forward(model_vit, xv).float())
            p_v20b = torch.sigmoid(_forward(model_v20b, xr).float())
            p_v20h = torch.sigmoid(_forward(model_v20h, xr).float())
            blended = w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h
            acc = blended if acc is None else acc + blended
        prob = (acc / len(TTA_VARIANTS)).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(test)} | {(time.time()-t0)/60:.1f} min', flush=True)

pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
pred.head()
'''


def _set(cell: dict, text: str) -> None:
    src = text if text.endswith("\n") else text + "\n"
    cell["source"] = [line + "\n" for line in src.rstrip("\n").split("\n")]


def write_notebook() -> None:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    local = str(UPLOAD.relative_to(ROOT)).replace("\\", "/")
    md = "".join(nb["cells"][0]["source"])
    md = md.replace("**Sans TTA.**", "**INT4** + **TTA x3** (identite, rotation +5°, rotation -5°).")
    md = md.replace(
        "Poids = SWA OAI epochs 1-3.",
        "Poids = SWA OAI epochs 1-3, quantizes INT4 group-wise (groupes de 64). "
        "LSTM, tete et embeds restent fp16.",
    )
    _set(nb["cells"][0], md)
    cfg = "".join(nb["cells"][1]["source"])
    repl = {
        "vit_k320_oai3_swa_ep1_3.pt": "vit_k320_oai3_swa_int4.pt",
        "v20h_dino_oai3_swa_ep1_3.pt": "v20h_oai3_swa_int4.pt",
        "v20b_dino_oai3_swa_ep1_3.pt": "v20b_oai3_swa_int4.pt",
        "kaggle_dataset_vit_k/_upload_k320_oai3_swa/vit_k320_oai3_swa_int4.pt":
            f"{local}/vit_k320_oai3_swa_int4.pt",
        "kaggle_dataset_v20h_lat/_upload_dino_oai3_swa/v20h_oai3_swa_int4.pt":
            f"{local}/v20h_oai3_swa_int4.pt",
        "kaggle_dataset_v20b/_upload_dino_oai3_swa/v20b_oai3_swa_int4.pt":
            f"{local}/v20b_oai3_swa_int4.pt",
        "print('TTA             : non')": "print('TTA             : x3 rotation ±5')",
    }
    # The path replacements above assume the filename replace already happened.
    cfg = cfg.replace("vit_k320_oai3_swa_ep1_3.pt", "vit_k320_oai3_swa_int4.pt")
    cfg = cfg.replace("v20h_dino_oai3_swa_ep1_3.pt", "v20h_oai3_swa_int4.pt")
    cfg = cfg.replace("v20b_dino_oai3_swa_ep1_3.pt", "v20b_oai3_swa_int4.pt")
    cfg = cfg.replace(
        "kaggle_dataset_vit_k/_upload_k320_oai3_swa/",
        local + "/",
    )
    cfg = cfg.replace(
        "kaggle_dataset_v20h_lat/_upload_dino_oai3_swa/",
        local + "/",
    )
    cfg = cfg.replace(
        "kaggle_dataset_v20b/_upload_dino_oai3_swa/",
        local + "/",
    )
    cfg = cfg.replace(
        "print('TTA             : non')",
        "print('TTA             : x3 rotation ±5')",
    )
    _set(nb["cells"][1], cfg)
    infer = "".join(nb["cells"][-2]["source"])
    infer = infer.replace(
        "import pandas as pd\nimport time\n",
        "import pandas as pd\nimport time\nimport torch.nn.functional as F\n" + DEQUANT,
    )
    old_load = "model.load_state_dict(ckpt['model'], strict=True)"
    if infer.count(old_load) != 2:
        raise SystemExit(f"load_state_dict x{infer.count(old_load)}")
    infer = infer.replace(old_load, "model.load_state_dict(_sd(ckpt), strict=True)")
    infer = infer.replace(
        "print(len(test), 'etudes | sans TTA | pas de flip')",
        "print(len(test), 'etudes | TTA x3 | pas de flip')",
    )
    start = infer.find("rows = []\n")
    if start < 0:
        raise SystemExit("boucle introuvable")
    infer = infer[:start] + LOOP
    if "TTA_VARIANTS" not in infer or "kneenet_int4_v1" not in infer:
        raise SystemExit("infer INT4/TTA incomplet")
    if "sans TTA" in infer:
        raise SystemExit("sans TTA encore present")
    _set(nb["cells"][-2], infer)
    blob = "\n".join("".join(c["source"]) for c in nb["cells"])
    if "KneeNetCoat" in blob or "coat_a_oai3" in blob:
        raise SystemExit("coat encore present")
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
    )
    meta = {
        "id": KAGGLE_KERNEL,
        "title": SLUG,
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": ACCEL,
        "competition_sources": [COMPETITION],
        "dataset_sources": [DATASET],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("ecrit", PUSH / "submit.ipynb", flush=True)


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        raise


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def ensure_dataset() -> None:
    try:
        st = kaggle("datasets", "status", DATASET).strip().lower()
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        if any(s in text.lower() for s in ("404", "403", "not found", "forbidden")):
            st = ""
        else:
            raise
    print("[ds]", st or "(absent)", flush=True)
    version = "ready" in st and "pending" not in st
    cmd = [
        sys.executable, "-X", "utf8", str(UPLOAD_PY),
        "--folder", str(UPLOAD),
        "--message", "mix3 OAI3 SWA INT4",
    ]
    if version:
        cmd.append("--version")
    if "pending" not in st:
        print("+", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
    t0 = time.time()
    while True:
        try:
            st = kaggle("datasets", "status", DATASET).strip().lower()
        except subprocess.CalledProcessError:
            st = ""
        print("[ds]", st or "(pending)", flush=True)
        if "ready" in st and "pending" not in st:
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout dataset")
        time.sleep(20)


def push_kernel() -> str:
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(PUSH),
                "--accelerator", ACCEL, "-t", "32400",
            )
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text):
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if _quota(out):
            print("quota GPU, retry 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        print("kernel version", m.group(1), flush=True)
        return m.group(1)


def main() -> None:
    quantize_all(_quant())
    write_notebook()
    ensure_dataset()
    version = push_kernel()
    print("KERNEL_POUSSE", KAGGLE_KERNEL, "version", version, flush=True)
    print("PAS_DE_SOUMISSION_COMPETITION", flush=True)


if __name__ == "__main__":
    main()
