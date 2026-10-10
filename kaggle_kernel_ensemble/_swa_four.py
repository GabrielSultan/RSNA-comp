# -*- coding: utf-8 -*-
"""SWA uniforme de 4 runs. Sortie dans le dossier de checkpoints du run.

BN gelée sur ResNet / CoAtNet, LayerNorm sur le ViT : pas de
re-estimation des statistiques.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JOBS = (
    {
        "run": "vit_k_llrd08_320_v3b_oofmask_v20h_full",
        "epochs": (7, 8, 9, 10),
        "expect_model": "kneenet_vit_k320",
        "out": "swa_ep7_10.pt",
    },
    {
        "run": "v20h_crop130_dino_ep11_oof_v3b_full",
        "epochs": (6, 7),
        "expect_model": "kneenet_v20h",
        "out": "swa_ep6_7.pt",
    },
    {
        "run": "v20b_crop130_dino_ep11_oof_v3b_full",
        "epochs": (6, 7, 8, 9),
        "expect_model": "kneenet_v20b",
        "out": "swa_ep6_9.pt",
    },
    {
        "run": "coat_a_llrd08_v3b_oofmask_full",
        "epochs": (7, 8, 9, 10),
        "expect_model": "kneenet_coat_a",
        "out": "swa_ep7_10.pt",
    },
)


def _gold_by_epoch(run: Path) -> dict[int, float | None]:
    hist_p = run / "history.json"
    if not hist_p.is_file():
        return {}
    hist = json.loads(hist_p.read_text(encoding="utf-8"))
    out: dict[int, float | None] = {}
    for rec in hist:
        ep = int(rec["epoch"])
        g = rec.get("gold_macro_auc")
        out[ep] = None if g is None else float(g)
    return out


def _add(acc: dict, sd: dict, first: bool) -> tuple[int, int]:
    n_float = n_other = 0
    if first:
        if not sd:
            raise SystemExit("state_dict vide")
        for k, t in sd.items():
            t = t.detach().cpu()
            if t.is_floating_point():
                acc[k] = t.float().clone()
                n_float += 1
            else:
                acc[k] = t.clone()
                n_other += 1
        return n_float, n_other
    if list(sd.keys()) != list(acc.keys()):
        raise SystemExit("cles state_dict differentes entre epochs")
    for k, t in sd.items():
        t = t.detach().cpu()
        a = acc[k]
        if a.is_floating_point():
            if not t.is_floating_point() or t.shape != a.shape:
                raise SystemExit(f"{k}: type/shape incompatible")
            a += t.float()
            n_float += 1
        else:
            if not torch.equal(a, t):
                print(f"  warn buffer {k} non identique, on garde le dernier",
                      flush=True)
                acc[k] = t.clone()
            n_other += 1
    return n_float, n_other


def _divide(acc: dict, sd_ref: dict, n: int) -> dict:
    out = {}
    for k, a in acc.items():
        ref = sd_ref[k]
        if a.is_floating_point() and ref.is_floating_point():
            out[k] = (a / float(n)).to(dtype=ref.dtype)
        else:
            out[k] = a
    return out


def _build(name: str, cfg: dict):
    common = dict(
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.2)),
        encode_chunk=int(cfg.get("encode_chunk", 8)),
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
        pretrained=False,
    )
    if name == "kneenet_vit_k320":
        from src.models.kneenet_vit_k320 import KneeNet
        return KneeNet(
            img_size=int(cfg.get("img_size", 320)),
            drop_path_rate=float(cfg.get("drop_path_rate") or 0.0),
            **common,
        )
    if name == "kneenet_v20h":
        from src.models.kneenet_v20h import KneeNet
        return KneeNet(**common)
    if name == "kneenet_v20b":
        from src.models.kneenet_v20b import KneeNet
        return KneeNet(**common)
    if name == "kneenet_coat_a":
        from src.models.kneenet_coat_a import KneeNet
        return KneeNet(
            backbone=str(cfg.get("backbone", "coatnet_0_rw_224.sw_in1k")),
            img_size=int(cfg.get("img_size", 224)),
            **common,
        )
    raise SystemExit(f"archi inconnue {name}")


def _one(job: dict) -> None:
    run = ROOT / "checkpoints" / job["run"]
    epochs = tuple(int(e) for e in job["epochs"])
    print(f"\n=== {job['run']} epochs {epochs}", flush=True)
    golds = _gold_by_epoch(run)
    acc: dict = {}
    cfg = None
    ref_sd = None
    n_float = n_other = 0
    for i, ep in enumerate(epochs):
        src = run / f"epoch{ep}.pt"
        if not src.is_file():
            raise FileNotFoundError(src)
        print(f"load {src.name} {src.stat().st_size}", flush=True)
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        c = ckpt.get("config") or {}
        model_name = str(c.get("model"))
        if model_name != job["expect_model"]:
            raise SystemExit(
                f"{src} model={model_name!r}, attendu {job['expect_model']}"
            )
        ckpt_ep = ckpt.get("epoch")
        if ckpt_ep is not None and int(ckpt_ep) != ep:
            raise SystemExit(f"{src} epoch ckpt={ckpt_ep} != {ep}")
        if cfg is None:
            cfg = c
        elif c != cfg:
            print("  warn config differente, on garde celle du premier epoch",
                  flush=True)
        sd = ckpt["model"]
        n_float, n_other = _add(acc, sd, first=i == 0)
        if ref_sd is None:
            ref_sd = {k: v.detach().cpu() for k, v in sd.items()}
        del sd, ckpt
        gc.collect()
        g = golds.get(ep)
        print(f"  epoch {ep} gold hist {g}", flush=True)
    avg = _divide(acc, ref_sd, len(epochs))
    del acc, ref_sd
    gc.collect()
    print(f"moyenne {len(epochs)} : {n_float} tenseurs float, "
          f"{n_other} buffers", flush=True)
    model = _build(job["expect_model"], cfg)
    incompat = model.load_state_dict(avg, strict=True)
    print(f"load strict OK {incompat}", flush=True)
    del model
    gc.collect()
    tag = "-".join(str(e) for e in epochs)
    out = {
        "model": avg,
        "config": cfg,
        "epoch": f"swa-{tag}",
        "swa_epochs": list(epochs),
        "gold_by_epoch": {str(ep): golds.get(ep) for ep in epochs},
        "val_macro_auc": -1.0,
    }
    dst = run / job["out"]
    torch.save(out, dst)
    print(f"wrote {dst} {dst.stat().st_size}", flush=True)
    del out, avg
    gc.collect()


def main() -> None:
    for job in JOBS:
        _one(job)


if __name__ == "__main__":
    main()
