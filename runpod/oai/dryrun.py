# -*- coding: utf-8 -*-
"""Controle a blanc des 4 configs configs/oai3 (local puis pod), sans train.

Par config :
1. init_*.pt : ecrit depuis le SWA source si absent (state_dict + config en
   types Python, --write-init), puis charge en strict dans build_model
   (pretrained=False, comme train.py avec init_ckpt).
2. Ecarts YAML / config du SWA sur les arguments du modele (attendu :
   encode_chunk, grad_checkpoint, pretrained).
3. make_loaders : comptes train et par slot, series RSNA indexees (21 594),
   aucune serie RSNA vers un zip oai_*, series OAI indexees.
4. 4 etudes RSNA et 4 OAI : slots non nuls, labels ; forward GPU (--forward).

    python runpod/oai/dryrun.py --write-init [--forward] [configs/oai3/x.yaml ...]
    python runpod/oai/dryrun.py --model-only        (pod, avant les donnees)
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data.dataset_npz import (  # noqa: E402
    PROTOCOL_SLOT_NAMES, gold_uids_from_train_csv, make_loaders, resolve_volume_source)
from src.models.registry import build_model, get_model_class  # noqa: E402
from src.train.train import OAI_PREFIX, load_config, load_init_weights  # noqa: E402

SOURCES = {
    "vit_k_llrd08_320_v3b_oofmask_v20h_full": "swa_ep7_10.pt",
    "v20h_crop130_dino_ep11_oof_v3b_full": "swa_ep6_7.pt",
    "v20b_crop130_dino_ep11_oof_v3b_full": "swa_ep6_9.pt",
    "coat_a_llrd08_v3b_oofmask_full": "swa_ep7_10.pt",
}
N_RSNA_SERIES = 21594
N_GOLD = 58
EXPECTED_DIFF = {"encode_chunk", "grad_checkpoint", "pretrained"}


def write_init(init_p: Path) -> None:
    src = ROOT / "checkpoints" / init_p.stem / SOURCES[init_p.stem]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = {k: v.detach().cpu().contiguous() for k, v in ckpt["model"].items()}
    cfg = json.loads(json.dumps(ckpt["config"], default=str))
    init_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "config": cfg, "epoch": str(ckpt.get("epoch")),
                "source": str(src.relative_to(ROOT))}, init_p)
    print(f"  init ecrit : {init_p.relative_to(ROOT)} <- {src.relative_to(ROOT)} "
          f"({init_p.stat().st_size / 1e6:.0f} Mo)")


def check(cfg_path: Path, args) -> list[str]:
    errors: list[str] = []
    cfg = load_config(str(cfg_path))
    print(f"\n=== {cfg_path.name} -> {cfg['run_name']}")
    if cfg.get("init_ckpt"):
        init_p = ROOT / cfg["init_ckpt"]
        if not init_p.is_file():
            if not args.write_init:
                return [f"{init_p} absent (relancer avec --write-init)"]
            write_init(init_p)

        init = torch.load(init_p, map_location="cpu", weights_only=False)
        accepted = set(inspect.signature(get_model_class(cfg["model"]).__init__).parameters)
        diff = {k: (init["config"].get(k), cfg.get(k)) for k in sorted(accepted)
                if k in cfg and init["config"].get(k) != cfg.get(k)}
        print(f"  ecarts vs SWA ({init['epoch']}) : {diff or 'aucun'}")
        for k in set(diff) - EXPECTED_DIFF:
            errors.append(f"argument modele {k} different du SWA : {diff[k]}")

        model = build_model(cfg, pretrained=False)
        load_init_weights(model, str(init_p), cfg["model"])
    else:
        print("  from scratch (pas d'init_ckpt)")
        model = build_model(cfg, pretrained=False)
    attrs = {a: getattr(model, a) for a in ("encode_chunk", "grad_checkpoint", "img_size")
             if hasattr(model, a)}
    print(f"  modele {cfg['model']} | {attrs}")
    if attrs.get("encode_chunk") not in (None, cfg.get("encode_chunk")):
        errors.append(f"encode_chunk modele {attrs['encode_chunk']} != YAML {cfg.get('encode_chunk')}")
    n_gold = len(gold_uids_from_train_csv(cfg["labels_csv"]))
    if n_gold != N_GOLD:
        errors.append(f"{n_gold} uids gold, attendu {N_GOLD}")
    if args.model_only:
        return errors

    vol = resolve_volume_source(str(ROOT / cfg["zip_path"]))
    rsna = {s: z for s, z in vol.items() if not s.startswith(OAI_PREFIX)}
    oai = {s: z for s, z in vol.items() if s.startswith(OAI_PREFIX)}
    bad = [s for s, z in rsna.items() if Path(z).name.startswith("oai_")]
    print(f"  volumes : RSNA {len(rsna)} | OAI {len(oai)} | RSNA vers oai_* : {len(bad)}")
    if len(rsna) != N_RSNA_SERIES:
        errors.append(f"{len(rsna)} series RSNA indexees, attendu {N_RSNA_SERIES}")
    if bad:
        errors.append(f"{len(bad)} series RSNA indexees vers un zip oai_*")
    if not oai:
        errors.append("aucune serie OAI indexee")

    train_ld, val_ld, targets = make_loaders(
        labels_csv=cfg["labels_csv"], series_csv=cfg["series_csv"],
        zip_path=cfg["zip_path"], batch_size=cfg["batch_size"], num_workers=0,
        n_series=int(cfg["n_series"]), img_size=int(cfg.get("img_size", 224)),
        train_frac=float(cfg.get("train_frac", 1.0)),
        train_frac_seed=int(cfg.get("train_frac_seed", 42)),
        include_gold=bool(cfg.get("include_gold")), val_frac=float(cfg.get("val_frac", 0.15)),
        seed=int(cfg.get("seed", 42)))
    ds = train_ld.dataset
    ds.series_drop_p = 0.0
    items = ds.items
    n_oai = sum(u.startswith(OAI_PREFIX) for u in items)
    val_items = val_ld.dataset.items
    n_oai_val = sum(u.startswith(OAI_PREFIX) for u in val_items)
    print(f"  train {len(items)} etudes (OAI {n_oai}) | val {len(val_items)} "
          f"(OAI {n_oai_val}) | {len(train_ld)} steps/epoch")
    if n_oai_val:
        errors.append(f"{n_oai_val} etudes OAI dans la val")
    if float(cfg.get("val_frac", 0.15)) > 0:
        lab = pd.read_csv(ROOT / cfg["labels_csv"])
        if "split" in lab.columns:
            expect_tr = set(lab.loc[lab.split == "train", "StudyInstanceUID"].astype(str))
            expect_va = set(lab.loc[lab.split == "val", "StudyInstanceUID"].astype(str))
            if set(items) != expect_tr:
                errors.append(
                    f"train {len(items)} != csv {len(expect_tr)} (volumes manquants)")
            if set(val_items) != expect_va:
                errors.append(
                    f"val {len(val_items)} != csv {len(expect_va)} (volumes manquants)")

    rng = np.random.default_rng(0)
    pick = {"RSNA": [u for u in items if not u.startswith(OAI_PREFIX)],
            "OAI": [u for u in items if u.startswith(OAI_PREFIX)]}
    samples = []
    for kind, uids in pick.items():
        for u in rng.choice(uids, size=min(4, len(uids)), replace=False):
            x, y, uid = ds[items.index(u)]
            filled = [n for n, v in zip(PROTOCOL_SLOT_NAMES, x) if float(v.abs().sum()) > 0]
            n_lab = int((y >= 0).sum())
            print(f"  {kind:4s} {uid[-16:]:>16s} | slots {filled} | labels {n_lab}/12 "
                  f"| x {tuple(x.shape)}")
            if kind == "OAI" and not {"SAG_FS", "COR_FS", "AX_FS"} & set(filled):
                errors.append(f"{uid} : aucun slot fluide charge")
            if kind == "RSNA" and not filled:
                errors.append(f"{uid} : volumes vides")
            samples.append((kind, x))

    if args.forward and torch.cuda.is_available():
        model = model.cuda().eval()
        for kind, x in samples[-1:] + samples[:1]:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(model(x.unsqueeze(0).cuda()).float())[0]
            print(f"  forward {kind} : probas {' '.join(f'{v:.2f}' for v in p.tolist())}")
            if not torch.isfinite(p).all():
                errors.append(f"forward {kind} non fini")
        model.cpu()
        torch.cuda.empty_cache()
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--write-init", action="store_true")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--model-only", action="store_true",
                    help="init + modele seulement (avant l'arrivee des donnees)")
    args = ap.parse_args()
    paths = [Path(p).resolve() for p in args.configs] or sorted((ROOT / "configs" / "oai3").glob("*.yaml"))
    os.chdir(ROOT)
    failed = {}
    for p in paths:
        errs = check(p, args)
        if errs:
            failed[p.name] = errs
    print()
    for name, errs in failed.items():
        for e in errs:
            print(f"ECHEC {name} : {e}")
    if failed:
        raise SystemExit(1)
    print(f"OK : {len(paths)} config(s)")


if __name__ == "__main__":
    main()
