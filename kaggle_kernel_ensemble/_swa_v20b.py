# -*- coding: utf-8 -*-
"""SWA uniforme des snapshots v20b oofmask ep8/9/11. BN gelee : pas de re-estimation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_v20b import KneeNet as KneeNetV20b  # noqa: E402

RUN = ROOT / "checkpoints" / "v20b_crop130_oofmask_full"
OUT_DIR = ROOT / "kaggle_dataset_v20b" / "_upload_swa"
EPOCHS = (8, 9, 11)
OUT_NAME = "v20b_swa_ep8_9_11.pt"
KEEP = ("model", "config", "epoch", "val_macro_auc", "gold_macro_auc")


def _gold_by_epoch() -> dict[int, float]:
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    out = {}
    for rec in hist:
        ep = int(rec["epoch"])
        g = rec.get("gold_macro_auc")
        if g is not None:
            out[ep] = float(g)
    return out


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


def _load_model_sd(src: Path) -> tuple[dict, dict, object]:
    print("load", src, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(
        "  keys", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt),
        "epoch", ckpt.get("epoch"),
        "gold ckpt", ckpt.get("gold_macro_auc"),
        "laterality", cfg.get("laterality"),
        "model", cfg.get("model"),
        "n_series", cfg.get("n_series"),
        "dropout", cfg.get("dropout"),
        flush=True,
    )
    if cfg.get("laterality"):
        raise SystemExit(f"{src} laterality={cfg.get('laterality')!r}, attendu false/absent")
    if str(cfg.get("model")) != "kneenet_v20b":
        raise SystemExit(f"{src} model={cfg.get('model')!r}, attendu kneenet_v20b")
    sd = ckpt["model"]
    ep = ckpt.get("epoch")
    del ckpt
    return sd, cfg, ep


def main() -> None:
    golds = _gold_by_epoch()
    sds = []
    cfg = None
    for ep in EPOCHS:
        src = RUN / f"epoch{ep}.pt"
        if not src.is_file():
            raise FileNotFoundError(src)
        sd, cfg, ckpt_ep = _load_model_sd(src)
        if ckpt_ep is not None and int(ckpt_ep) != ep:
            raise SystemExit(f"{src} epoch ckpt={ckpt_ep} != {ep}")
        print(f"  gold hist ep{ep} {golds.get(ep)}", flush=True)
        sds.append(sd)
    avg = average_state_dicts(sds)
    del sds
    model = KneeNetV20b(
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
        "epoch": "swa-8-9-11",
        "swa_epochs": list(EPOCHS),
        "gold_by_epoch": {str(ep): golds.get(ep) for ep in EPOCHS},
        "val_macro_auc": -1.0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / OUT_NAME
    torch.save(out, dst)
    print("wrote", dst, dst.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
