# -*- coding: utf-8 -*-
"""Poids + notebooks T4 : v20h v3b lat full, epochs 6-10."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_v20h import KneeNet as KneeNetV20h  # noqa: E402

RUN = ROOT / "checkpoints" / "v20h_crop130_v3b_lat_full"
KERNEL_DIR = ROOT / "kaggle_kernel_ensemble"
EPOCHS = (6, 7, 8, 9, 10)
KEEP = ("model", "config", "epoch", "val_macro_auc", "gold_macro_auc")


def _load_maker():
    p = KERNEL_DIR / "_make_nb_v20h.py"
    spec = importlib.util.spec_from_file_location("make_nb_v20h", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mh = _load_maker()


def _golds() -> dict[int, float]:
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    out = {}
    for rec in hist:
        g = rec.get("gold_macro_auc")
        if g is not None:
            out[int(rec["epoch"])] = float(g)
    return out


def _set_source(cell: dict, text: str) -> None:
    src = text if text.endswith("\n") else text + "\n"
    cell["source"] = [line + "\n" for line in src.rstrip("\n").split("\n")]


def strip_epoch(epoch: int, gold: float) -> Path:
    src = RUN / f"epoch{epoch}.pt"
    if not src.is_file():
        raise FileNotFoundError(src)
    upload = ROOT / "kaggle_dataset_v20h_lat" / f"_upload_v3b_ep{epoch}"
    upload.mkdir(parents=True, exist_ok=True)
    dest = upload / f"v20h_v3b_lat_ep{epoch}.pt"
    if (
        dest.is_file()
        and dest.stat().st_size > 50_000_000
        and dest.stat().st_mtime >= src.stat().st_mtime
    ):
        print("deja strip", dest.name, dest.stat().st_size, flush=True)
        _write_meta(upload, epoch)
        return dest
    print("load", src.name, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    got = int(ckpt.get("epoch", -1))
    if got != epoch:
        raise SystemExit(f"{src.name} epoch {got}, attendu {epoch}")
    if str(cfg.get("model")) != "kneenet_v20h":
        raise SystemExit(f"model={cfg.get('model')!r}")
    if cfg.get("run_name") != "v20h_crop130_v3b_lat_full":
        raise SystemExit(f"run inattendu {cfg.get('run_name')}")
    if not cfg.get("laterality"):
        raise SystemExit(f"{src.name} sans laterality=true")
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    out["gold_macro_auc"] = gold
    torch.save(out, dest)
    model = KneeNetV20h(
        pretrained=False,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=int(cfg.get("encode_chunk", 8)),
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    )
    model.load_state_dict(out["model"], strict=True)
    print(
        "strip", dest.name, dest.stat().st_size,
        "epoch", epoch, "gold", round(gold, 4),
        "n_series", model.n_series, "laterality", True,
        flush=True,
    )
    del model, ckpt
    _write_meta(upload, epoch)
    return dest


def _write_meta(upload: Path, epoch: int) -> None:
    meta = {
        "title": f"RSNA Knee v20h v3b lat ep{epoch}",
        "id": f"sultent/rsna-knee-v20h-v3b-lat-ep{epoch}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (upload / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


def write_epoch(epoch: int, gold: float) -> Path:
    weight = f"v20h_v3b_lat_ep{epoch}.pt"
    dataset = f"sultent/rsna-knee-v20h-v3b-lat-ep{epoch}"
    local = f"../kaggle_dataset_v20h_lat/_upload_v3b_ep{epoch}"
    notebook = mh.build_notebook(f"epoch {epoch}", gold, weight, dataset, local)
    md = "".join(notebook["cells"][0]["source"])
    md = md.replace(
        "v20h oofmask + laterality L",
        "v20h v3b lat full + laterality L",
    ).replace(
        "v20h_crop130_oofmask_v20b_lat_full",
        "v20h_crop130_v3b_lat_full",
    ).replace(
        "Pas d'ensemble ViT-e : ce checkpoint a ete train **avec** flip L/R.",
        "Soft labels Coolin targets_v3b, train 100 %, gold x4. "
        "Canon laterality L (flip R).",
    )
    _set_source(notebook["cells"][0], md)
    cfg = "".join(notebook["cells"][1]["source"])
    cfg = cfg.replace("v20h oofmask lat", "v20h v3b lat")
    cfg = cfg.replace("dataset v20h lat attache", "dataset v20h v3b lat attache")
    _set_source(notebook["cells"][1], cfg)
    slug = f"rsna-knee-v20h-v3b-lat-solo-ep{epoch}"
    push = KERNEL_DIR / f"_push_v20h_v3b_ep{epoch}"
    push.mkdir(parents=True, exist_ok=True)
    text = json.dumps(notebook, indent=1, ensure_ascii=False)
    (push / "submit.ipynb").write_text(text, encoding="utf-8")
    (KERNEL_DIR / f"submit_v20h_v3b_ep{epoch}.ipynb").write_text(text, encoding="utf-8")
    (push / "kernel-metadata.json").write_text(
        json.dumps(mh._metadata(slug, dataset), indent=2) + "\n",
        encoding="utf-8",
    )
    print("ecrit", push / "submit.ipynb", flush=True)
    return push


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, nargs="*", default=list(EPOCHS))
    args = p.parse_args()
    golds = _golds()
    for ep in args.epoch:
        if ep not in golds:
            raise SystemExit(f"gold absent pour epoch {ep}")
        strip_epoch(ep, golds[ep])
        write_epoch(ep, golds[ep])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
