# -*- coding: utf-8 -*-
"""SWA uniforme des snapshots v20h lat ep7-10. BN gelée : pas de re-estimation."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_v20h import KneeNet as KneeNetV20h  # noqa: E402

UPLOAD_SRC = ROOT / "kaggle_dataset_v20h_lat" / "_upload"
UPLOAD_EP7 = ROOT / "kaggle_dataset_v20h_lat" / "_upload_ep7"
OUT_DIR = ROOT / "kaggle_dataset_v20h_lat" / "_upload_swa"
EPOCHS = (7, 8, 9, 10)
OUT_NAME = "v20h_swa_ep7_10.pt"


def _src(ep: int) -> Path:
    if ep == 7:
        p = UPLOAD_EP7 / "v20h_ep7.pt"
        if p.is_file():
            return p
    p = UPLOAD_SRC / f"v20h_ep{ep}.pt"
    if p.is_file():
        return p
    raise FileNotFoundError(p)


def average_state_dicts(sds: list[dict]) -> dict:
    keys = list(sds[0].keys())
    for i, sd in enumerate(sds[1:], start=1):
        if list(sd.keys()) != keys:
            raise SystemExit(f"cles state_dict differentes a l'index {i}")
    n = float(len(sds))
    out = {}
    n_float = n_other = n_diff = 0
    for k in keys:
        tensors = [sd[k].cpu() for sd in sds]
        t0 = tensors[0]
        if t0.is_floating_point():
            acc = t0.float().clone()
            for t in tensors[1:]:
                if t.shape != t0.shape:
                    raise SystemExit(f"{k}: shape {tuple(t.shape)} vs {tuple(t0.shape)}")
                acc += t.float()
            acc /= n
            out[k] = acc.to(dtype=t0.dtype)
            n_float += 1
        else:
            for t in tensors[1:]:
                if not torch.equal(t, t0):
                    n_diff += 1
                    print(f"  warn buffer {k} non identique, on garde le dernier", flush=True)
                    break
            out[k] = tensors[-1].clone()
            n_other += 1
    print(f"moyenne {n:.0f} : {n_float} tenseurs float, {n_other} buffers, "
          f"{n_diff} buffers divergents", flush=True)
    return out


def main() -> None:
    sds = []
    cfg = None
    for ep in EPOCHS:
        src = _src(ep)
        print("load", src, src.stat().st_size, flush=True)
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        if not ckpt.get("config", {}).get("laterality"):
            raise SystemExit(f"{src} sans laterality=true")
        if str(ckpt.get("config", {}).get("model")) != "kneenet_v20h":
            raise SystemExit(f"{src} model={ckpt.get('config', {}).get('model')!r}")
        sds.append(ckpt["model"])
        cfg = ckpt["config"]
        del ckpt
    avg = average_state_dicts(sds)
    del sds
    model = KneeNetV20h(
        pretrained=False,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=int(cfg.get("encode_chunk", 8)),
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    )
    model.load_state_dict(avg, strict=True)
    print("load strict OK n_series", model.n_series, flush=True)
    del model
    out = {
        "model": avg,
        "config": cfg,
        "epoch": "swa-7-10",
        "swa_epochs": list(EPOCHS),
        "val_macro_auc": -1.0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / OUT_NAME
    torch.save(out, dst)
    print("wrote", dst, dst.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
