# -*- coding: utf-8 -*-
"""Duplique v20b_ep8.pt et quantize les poids 2D en INT4 (per-channel, symmetric)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_dataset_ensemble" / "_upload" / "v20b_ep8.pt"
DST_ENSEMBLE = ROOT / "kaggle_dataset_ensemble" / "_upload" / "v20b_ep8_int4.pt"
DST_UPLOAD = ROOT / "kaggle_dataset_int4" / "_upload" / "v20b_ep8_int4.pt"
COPY_FP32 = ROOT / "kaggle_dataset_ensemble" / "_upload" / "v20b_ep8_fp32_copy.pt"

QMIN, QMAX = -8, 7
MIN_NUMEL = 256
GROUP_SIZE = 64
# Tetes / LSTM / embeds : peu de params, sensibles. Le backbone reste INT4.
KEEP_FP16_SUBSTR = (
    "slice_lstm", "lstm_proj", "head.", "stem_mix",
    "attn_score", "spatial_score", "plane_embed", "slice_embed", "norm.",
)


def _should_quant(name: str, t: torch.Tensor) -> bool:
    if not t.is_floating_point():
        return False
    if t.ndim < 2 or t.numel() < MIN_NUMEL:
        return False
    low = name.lower()
    if "running_" in low:
        return False
    if any(s in low for s in KEEP_FP16_SUBSTR):
        return False
    return True


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """q int8 in [-8, 7] -> uint8 packed (2 nibbles / byte)."""
    flat = q.reshape(-1).to(torch.int8)
    n = int(flat.numel())
    pad = n & 1
    if pad:
        flat = torch.cat([flat, flat.new_zeros(1)])
    lo = (flat[0::2].to(torch.int16) & 0x0F).to(torch.uint8)
    hi = (flat[1::2].to(torch.int16) & 0x0F).to(torch.uint8)
    return lo | (hi << 4)


def unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    packed = packed.reshape(-1).to(torch.uint8)
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    q = torch.empty(packed.numel() * 2, dtype=torch.int8)
    q[0::2] = lo.to(torch.int8)
    q[1::2] = hi.to(torch.int8)
    return q[:numel]


def quantize_per_channel(w: torch.Tensor, group_size: int = GROUP_SIZE) -> dict:
    """INT4 symmetric, groupes le long de l'axe aplati (hors dim 0)."""
    w32 = w.detach().float().contiguous()
    shape = tuple(w32.shape)
    rows = w32.reshape(w32.shape[0], -1)
    rest = int(rows.shape[1])
    g = int(group_size)
    pad = (g - rest % g) % g
    if pad:
        rows = torch.nn.functional.pad(rows, (0, pad))
    n_g = rows.shape[1] // g
    grouped = rows.reshape(w32.shape[0], n_g, g)
    absmax = grouped.abs().amax(dim=2).clamp_min(1e-8)
    scale = absmax / float(QMAX)
    q = torch.round(grouped / scale.unsqueeze(-1)).clamp(QMIN, QMAX).to(torch.int8)
    packed = pack_int4(q)
    return {
        "packed": packed.cpu().contiguous(),
        "scale": scale.half().cpu().contiguous(),
        "shape": shape,
        "numel": int(q.numel()),
        "rest": rest,
        "group_size": g,
    }


def dequantize_entry(entry: dict, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    shape = tuple(entry["shape"])
    out_ch = int(shape[0])
    rest = int(torch.tensor(shape[1:]).prod().item())
    q = unpack_int4(entry["packed"], int(entry["numel"]))
    scale = entry["scale"].float()
    if scale.ndim == 1:
        rows = q.to(torch.float32).reshape(out_ch, -1)[:, :rest] * scale.reshape(-1, 1)
        return rows.reshape(shape).to(dtype)
    g = int(entry.get("group_size", GROUP_SIZE))
    rest = int(entry.get("rest", rest))
    n_g = int(scale.shape[1])
    grouped = q.to(torch.float32).reshape(out_ch, n_g, g) * scale.unsqueeze(-1)
    rows = grouped.reshape(out_ch, n_g * g)[:, :rest]
    return rows.reshape(shape).to(dtype)


def dequantize_state(blob: dict, dtype: torch.dtype = torch.float32) -> dict:
    out = {}
    for k, v in blob.items():
        if isinstance(v, dict) and "packed" in v:
            out[k] = dequantize_entry(v, dtype=dtype)
        else:
            t = v
            if torch.is_tensor(t) and t.is_floating_point():
                out[k] = t.to(dtype)
            else:
                out[k] = t
    return out


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"introuvable : {SRC}")
    DST_UPLOAD.parent.mkdir(parents=True, exist_ok=True)
    if not COPY_FP32.exists():
        shutil.copy2(SRC, COPY_FP32)
        print(f"copie fp32 : {COPY_FP32} ({COPY_FP32.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"copie fp32 deja la : {COPY_FP32}")

    ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    packed = {}
    n_q = n_keep = 0
    q_params = keep_params = 0
    for name, t in sd.items():
        if _should_quant(name, t):
            packed[name] = quantize_per_channel(t)
            n_q += 1
            q_params += t.numel()
        else:
            if torch.is_tensor(t) and t.is_floating_point():
                packed[name] = t.detach().half().cpu().contiguous()
            else:
                packed[name] = t.detach().cpu() if torch.is_tensor(t) else t
            n_keep += 1
            keep_params += t.numel() if torch.is_tensor(t) else 0

    out = {
        "format": "kneenet_int4_v1",
        "model_int4": packed,
        "config": ckpt.get("config", {}),
        "epoch": ckpt.get("epoch"),
        "val_macro_auc": ckpt.get("val_macro_auc"),
        "quant": {
            "bits": 4,
            "scheme": "groupwise_symmetric",
            "group_size": GROUP_SIZE,
            "qmin": QMIN,
            "qmax": QMAX,
            "min_numel": MIN_NUMEL,
            "n_quantized": n_q,
            "n_kept": n_keep,
            "quantized_params": q_params,
            "kept_params": keep_params,
        },
    }
    torch.save(out, DST_UPLOAD)
    shutil.copy2(DST_UPLOAD, DST_ENSEMBLE)
    print(
        f"int4 : {DST_UPLOAD} ({DST_UPLOAD.stat().st_size / 1e6:.2f} MB) "
        f"| tensors Q={n_q} keep={n_keep} | params Q={q_params} keep={keep_params}"
    )

    recon = dequantize_state(packed, dtype=torch.float32)
    max_err = 0.0
    mean_err = 0.0
    n_cmp = 0
    for k, t in sd.items():
        if not torch.is_tensor(t) or not t.is_floating_point():
            continue
        d = (recon[k].float() - t.float()).abs()
        max_err = max(max_err, float(d.max()))
        mean_err += float(d.mean())
        n_cmp += 1
    print(f"poids | max-abs {max_err:.5f} | mean-abs {mean_err / max(n_cmp, 1):.6f}")

    sys.path.insert(0, str(ROOT))
    from src.models.kneenet_v20b import KneeNet

    cfg = ckpt.get("config", {})
    kwargs = dict(
        backbone=cfg.get("backbone", "resnet50"),
        n_series=int(cfg.get("n_series", 6)),
        head_dim=int(cfg.get("head_dim", 512)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=int(cfg.get("encode_chunk", 8)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
        pretrained=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = KneeNet(**kwargs).eval()
    ref.load_state_dict(sd, strict=True)
    qmodel = KneeNet(**kwargs).eval()
    qmodel.load_state_dict(recon, strict=True)
    ref.to(device)
    qmodel.to(device)
    torch.manual_seed(0)
    x = torch.rand(1, int(cfg.get("n_series", 6)), 24, 224, 224, device=device)
    x[:, 4:] = 0
    with torch.inference_mode():
        a = ref(x).float()
        b = qmodel(x).float()
    diff = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    print(
        f"logits {device} | max-abs {float(diff.max()):.5f} | "
        f"mean-abs {float(diff.mean()):.5f} | cosine {float(cos):.6f}"
    )
    print("probs ref ", torch.sigmoid(a).cpu().numpy().round(4))
    print("probs int4", torch.sigmoid(b).cpu().numpy().round(4))


if __name__ == "__main__":
    main()
