# -*- coding: utf-8 -*-
"""SWA manquants (v20h lat, ViT-k) puis paquets Kaggle des 5 modeles."""
from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "swa_four", ROOT / "kaggle_kernel_ensemble" / "_swa_four.py",
)
swa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(swa)

VITK_EP7 = ROOT / "kaggle_dataset_vit_e_dino_last" / "_upload" / "last.pt"


def _gold_mean(run: str, epochs: tuple[int, ...]) -> tuple[float, dict[str, float]]:
    hist = json.loads(
        (ROOT / "checkpoints" / run / "history.json").read_text(encoding="utf-8")
    )
    by = {}
    for ep in epochs:
        rec = next(r for r in hist if int(r["epoch"]) == ep)
        if rec.get("gold_macro_auc") is None:
            raise SystemExit(f"gold absent {run} epoch {ep}")
        by[str(ep)] = float(rec["gold_macro_auc"])
    return sum(by.values()) / len(by), by


def _build(name: str, cfg: dict):
    if name == "kneenet_vit_e":
        from src.models.kneenet_vit_e import KneeNet
        return KneeNet(
            n_series=int(cfg.get("n_series", 6)),
            dropout=float(cfg.get("dropout", 0.4)),
            encode_chunk=int(cfg.get("encode_chunk", 4)),
            head_dim=int(cfg.get("head_dim", 512)),
            lstm_hidden=int(cfg.get("lstm_hidden", 256)),
            drop_path_rate=float(cfg.get("drop_path_rate") or 0.0),
            pretrained=False,
            grad_checkpoint=False,
        )
    return swa._build(name, cfg)


def _source(run: Path, epoch: int) -> Path:
    if run.name == "vit_k_llrd08_det_oofmask_full" and epoch == 7:
        if not VITK_EP7.is_file():
            raise FileNotFoundError(VITK_EP7)
        return VITK_EP7
    path = run / f"epoch{epoch}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_missing(run_name: str, epochs: tuple[int, ...], expect_model: str, out_name: str) -> None:
    run = ROOT / "checkpoints" / run_name
    dst = run / out_name
    if dst.is_file() and dst.stat().st_size > 80_000_000:
        print("SWA deja present", dst.name, dst.stat().st_size, flush=True)
        return
    print(f"\n=== build {run_name} {epochs}", flush=True)
    gold, by = _gold_mean(run_name, epochs)
    acc: dict = {}
    cfg = None
    ref_sd = None
    for i, ep in enumerate(epochs):
        src = _source(run, ep)
        print("load", src, src.stat().st_size, flush=True)
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        c = dict(ckpt.get("config") or {})
        if str(c.get("model")) != expect_model:
            raise SystemExit(f"{src} model={c.get('model')!r}")
        if str(c.get("run_name")) != run_name:
            raise SystemExit(f"{src} run={c.get('run_name')!r}")
        got = ckpt.get("epoch")
        if got is not None and int(got) != ep:
            raise SystemExit(f"{src} epoch {got} != {ep}")
        if cfg is None or len(c) > len(cfg):
            cfg = c
        sd = ckpt["model"]
        swa._add(acc, sd, first=i == 0)
        if ref_sd is None:
            ref_sd = {k: v.detach().cpu() for k, v in sd.items()}
        del sd, ckpt
        gc.collect()
        print(f"  epoch {ep} gold {by[str(ep)]:.4f}", flush=True)
    avg = swa._divide(acc, ref_sd, len(epochs))
    del acc, ref_sd
    gc.collect()
    model = _build(expect_model, cfg)
    model.load_state_dict(avg, strict=True)
    print("load strict OK", expect_model, flush=True)
    del model
    gc.collect()
    tag = "-".join(str(e) for e in epochs)
    torch.save(
        {
            "model": avg,
            "config": cfg,
            "epoch": f"swa-{tag}",
            "swa_epochs": list(epochs),
            "gold_by_epoch": by,
            "gold_macro_auc": gold,
            "val_macro_auc": -1.0,
        },
        dst,
    )
    print("wrote", dst, dst.stat().st_size, "gold", round(gold, 4), flush=True)
    del avg
    gc.collect()


PACKS = (
    {
        "src_run": "vit_k_llrd08_320_v3b_oofmask_v20h_full",
        "src_name": "swa_ep7_10.pt",
        "epochs": (7, 8, 9, 10),
        "label": "swa-7-8-9-10",
        "model": "kneenet_vit_k320",
        "laterality": False,
        "img_size": 320,
        "upload": ROOT / "kaggle_dataset_vit_k" / "_upload_k320_swa",
        "weight": "vit_k320_v3b_swa_ep7_10.pt",
        "dataset": "sultent/rsna-knee-vit-k320-v3b-swa",
        "title": "RSNA Knee ViT-k320 v3b SWA ep7-10",
    },
    {
        "src_run": "v20h_crop130_dino_ep11_oof_v3b_full",
        "src_name": "swa_ep6_7.pt",
        "epochs": (6, 7),
        "label": "swa-6-7",
        "model": "kneenet_v20h",
        "laterality": False,
        "img_size": None,
        "upload": ROOT / "kaggle_dataset_v20h_lat" / "_upload_dino_full_swa",
        "weight": "v20h_dino_full_swa_ep6_7.pt",
        "dataset": "sultent/rsna-knee-v20h-dino-full-swa",
        "title": "RSNA Knee v20h DINO full SWA ep6-7",
    },
    {
        "src_run": "v20b_crop130_dino_ep11_oof_v3b_full",
        "src_name": "swa_ep6_9.pt",
        "epochs": (6, 7, 8, 9),
        "label": "swa-6-7-8-9",
        "model": "kneenet_v20b",
        "laterality": False,
        "img_size": None,
        "upload": ROOT / "kaggle_dataset_v20b" / "_upload_dino_full_swa",
        "weight": "v20b_dino_full_swa_ep6_9.pt",
        "dataset": "sultent/rsna-knee-v20b-dino-full-swa",
        "title": "RSNA Knee v20b DINO full SWA ep6-9",
    },
    {
        "src_run": "v20h_crop130_oofmask_v20b_lat_full",
        "src_name": "swa_ep7_12.pt",
        "epochs": (7, 8, 9, 10, 11, 12),
        "label": "swa-7-8-9-10-11-12",
        "model": "kneenet_v20h",
        "laterality": True,
        "img_size": None,
        "upload": ROOT / "kaggle_dataset_v20h_lat" / "_upload_v3b_lat_swa",
        "weight": "v20h_v3b_lat_swa_ep7_12.pt",
        "dataset": "sultent/rsna-knee-v20h-v3b-lat-swa",
        "title": "RSNA Knee v20h v3b lat SWA ep7-12",
    },
    {
        "src_run": "vit_k_llrd08_det_oofmask_full",
        "src_name": "swa_ep7_10.pt",
        "epochs": (7, 8, 9, 10),
        "label": "swa-7-8-9-10",
        "model": "kneenet_vit_e",
        "laterality": False,
        "img_size": 224,
        "upload": ROOT / "kaggle_dataset_vit_e_dino_last" / "_upload_det_swa",
        "weight": "vit_k_det_swa_ep7_10.pt",
        "dataset": "sultent/rsna-knee-vit-k-det-swa",
        "title": "RSNA Knee ViT-k det SWA ep7-10",
    },
)


def package(job: dict) -> float:
    src = ROOT / "checkpoints" / job["src_run"] / job["src_name"]
    if not src.is_file():
        raise SystemExit(f"SWA manquant: {src}")
    gold, by = _gold_mean(job["src_run"], job["epochs"])
    print("package", src.name, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    if str(cfg.get("run_name")) != job["src_run"]:
        raise SystemExit(cfg.get("run_name"))
    if str(cfg.get("model")) != job["model"]:
        raise SystemExit(cfg.get("model"))
    if str(ckpt.get("epoch")) != job["label"]:
        raise SystemExit(ckpt.get("epoch"))
    got = tuple(int(e) for e in (ckpt.get("swa_epochs") or []))
    if got != tuple(job["epochs"]):
        raise SystemExit(f"swa_epochs {got}")
    if job["laterality"]:
        if not cfg.get("laterality"):
            raise SystemExit(f"{job['src_run']} sans laterality")
    else:
        if cfg.get("laterality"):
            raise SystemExit(f"{job['src_run']} laterality inattendue")
        cfg["laterality"] = False
    if job["img_size"] is not None:
        cfg["img_size"] = int(job["img_size"])
    stored = ckpt.get("gold_by_epoch") or {}
    for ep in job["epochs"]:
        prev = stored.get(str(ep))
        if prev is not None and abs(float(prev) - by[str(ep)]) > 1e-6:
            raise SystemExit(f"gold incoherent {job['src_run']} ep {ep}")
    out = {
        "model": ckpt["model"],
        "config": cfg,
        "epoch": job["label"],
        "swa_epochs": list(job["epochs"]),
        "gold_by_epoch": by,
        "gold_macro_auc": gold,
        "val_macro_auc": -1.0,
    }
    model = _build(job["model"], cfg)
    model.load_state_dict(out["model"], strict=True)
    print("load strict OK", job["model"], "gold", round(gold, 4), flush=True)
    del model, ckpt
    gc.collect()
    dest_dir = job["upload"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / job["weight"]
    torch.save(out, dest)
    if dest.stat().st_size < 80_000_000:
        raise SystemExit(f"poids trop petit {dest}")
    meta = {
        "title": job["title"],
        "id": job["dataset"],
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dest_dir / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", dest, dest.stat().st_size, flush=True)
    del out
    gc.collect()
    return gold


def main() -> None:
    build_missing(
        "v20h_crop130_oofmask_v20b_lat_full",
        (7, 8, 9, 10, 11, 12),
        "kneenet_v20h",
        "swa_ep7_12.pt",
    )
    build_missing(
        "vit_k_llrd08_det_oofmask_full",
        (7, 8, 9, 10),
        "kneenet_vit_e",
        "swa_ep7_10.pt",
    )
    for job in PACKS:
        package(job)
    print("PREP_OK", flush=True)


if __name__ == "__main__":
    main()
