# -*- coding: utf-8 -*-
"""Forwards des runs hybrid025 (hors ViT-k, deja fait) sur les etudes RSNA
non vues en train hybrid025.

La liste d'etudes est celle de
data/hybrid_pred/hybrid_preds_unseen_rsna.csv (ViT-k, best.pt).
Meme protocole que l'eval : train=False, pas d'augmentation, sigmoid des
logits fp32. Checkpoint = best.pt. Un CSV par run dans data/hybrid_pred/.

    python -X utf8 runpod/_dump_hybrid_unseen.py
    python -X utf8 runpod/_dump_hybrid_unseen.py --only subset_v20h_dino_ep11_v3b_hybrid025 --limit 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.dataset_npz import (  # noqa: E402
    StudyDataset, _dataloader_worker_init, make_series_index)
from src.data.laterality import laterality_map_from_cfg  # noqa: E402
from src.models.registry import build_model  # noqa: E402
from src.nlp.lexicon import TARGETS  # noqa: E402
from src.train.train import load_config  # noqa: E402

VIT_CANDIDATES = (
    ROOT / "data" / "hybrid_pred" / "hybrid_preds_unseen_rsna.csv",
    ROOT / "data" / "hybrid_pred" / "hybrid_preds_unseen_rsna_vitkllrd08.csv",
)
OUT_DIR = ROOT / "data" / "hybrid_pred"

# (config, run). ViT-k est exclu : CSV deja ecrit.
JOBS = [
    (
        "configs/hybrid025/kneenet_v20b_crop130_dino_ep11_v3b_hybrid025.yaml",
        "subset_v20b_dino_ep11_v3b_hybrid025",
    ),
    (
        "configs/hybrid025/kneenet_v20h_crop130_dino_ep11_v3b_hybrid025.yaml",
        "subset_v20h_dino_ep11_v3b_hybrid025",
    ),
    (
        "configs/hybrid025/kneenet_v20h_crop145_dino_ep11_v3b_hybrid025_wd01.yaml",
        "subset_v20h_dino_ep11_v3b_hybrid025_wd01_crop145",
    ),
    (
        "configs/hybrid025/kneenet_coat_a_llrd08_v3b_hybrid025.yaml",
        "subset_coat_a_llrd08_v3b_hybrid025",
    ),
    (
        "configs/hybrid025/kneenet_cnx_r7_in22k_v3b_hybrid025.yaml",
        "subset_cnx_r7_in22k_v3b_hybrid025",
    ),
    (
        "configs/archive/kneenet_v20h_crop145_dino_ep11_v3b_rsna1687_distil_main.yaml",
        "subset_v20h_dino_ep11_v3b_rsna1687_distil_main_crop145",
    ),
]


def _out_path(run: str) -> Path:
    return OUT_DIR / f"hybrid_preds_unseen_rsna__{run}.csv"


def _frame(uids: list[str], epoch: int, probs: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"StudyInstanceUID": uids, "epoch": epoch})
    for j, t in enumerate(TARGETS):
        out[f"proba_{t}"] = probs[:, j].astype(np.float32)
    return out


def _load_partial(path: Path, uids: list[str]) -> tuple[list[str], np.ndarray | None]:
    if not path.is_file():
        return [], None
    df = pd.read_csv(path)
    got = df.StudyInstanceUID.astype(str).tolist()
    if got == uids:
        return got, None
    if got != uids[: len(got)]:
        raise SystemExit(f"{path.name} : ordre des UID incompatible, supprimer le partiel")
    cols = [f"proba_{t}" for t in TARGETS]
    return got, df[cols].to_numpy(dtype=np.float32)


def predict_one(cfg_rel: str, run: str, uids: list[str], batch_size: int,
                num_workers: int, flush_every: int) -> None:
    final = _out_path(run)
    partial = final.with_suffix(".partial.csv")
    if final.is_file():
        done = pd.read_csv(final, usecols=["StudyInstanceUID"])
        if done.StudyInstanceUID.astype(str).tolist() == uids:
            print(f"{run} : deja complet ({len(uids)} etudes) -> {final.name}",
                  flush=True)
            return
        raise SystemExit(f"{final.name} existe mais ne couvre pas la liste ViT")

    cfg = load_config(cfg_rel)
    ckpt_p = ROOT / "checkpoints" / run / "best.pt"
    if not ckpt_p.is_file():
        raise SystemExit(f"{run} : best.pt absent ({ckpt_p})")
    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    epoch = int(ckpt.get("epoch", -1))
    print(f"\n{run} | best.pt epoch {epoch} | "
          f"val AUC {float(ckpt.get('val_macro_auc', float('nan'))):.4f}",
          flush=True)

    done_uids, done_probs = _load_partial(partial, uids)
    todo = uids[len(done_uids):]
    print(f"a predire {len(todo)} | deja dans le partiel {len(done_uids)}", flush=True)
    if not todo:
        _frame(done_uids, epoch, done_probs).to_csv(final, index=False)
        partial.unlink(missing_ok=True)
        print(f"-> {final.name}", flush=True)
        return

    img_size = int(cfg.get("img_size", 224))
    index, volume_index = make_series_index(cfg["series_csv"], cfg["zip_path"])
    n_idx = sum(1 for u in todo if u in index.by_study)
    print(f"img_size {img_size} | zip {cfg['zip_path']} | "
          f"series indexees {n_idx}/{len(todo)}", flush=True)
    labels_df = pd.DataFrame({
        "StudyInstanceUID": todo,
        **{t: np.zeros(len(todo), dtype=np.float32) for t in TARGETS},
    })
    ds = StudyDataset(
        todo, labels_df[["StudyInstanceUID"] + TARGETS],
        index, cfg["zip_path"], train=False, series_drop_p=0.0,
        n_series=int(cfg.get("n_series", 6)), volume_index=volume_index,
        rgb_neighbors=bool(cfg.get("rgb_neighbors", False)),
        img_size=img_size,
        rgb_as_gray=bool(cfg.get("rgb_as_gray", False)),
        laterality_map=laterality_map_from_cfg(cfg),
    )
    ld = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=_dataloader_worker_init if num_workers > 0 else None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    del ckpt
    model.eval()
    n_batch = (len(todo) + batch_size - 1) // batch_size
    print(f"modele charge (strict) | {device} | batch {batch_size} | "
          f"{n_batch} batches", flush=True)

    chunks: list[np.ndarray] = []
    got: list[str] = []
    t0 = time.time()
    with torch.no_grad():
        for i, (x, _y, uid) in enumerate(ld, 1):
            logits = model(x.to(device, non_blocking=True))
            chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
            got.extend(uid)
            if i % 50 == 0 or i == n_batch:
                dt = time.time() - t0
                rate = len(got) / max(dt, 1e-6)
                left = (len(todo) - len(got)) / max(rate, 1e-6)
                print(f"  {i}/{n_batch} | {len(got)}/{len(todo)} | "
                      f"{rate:.1f} et/s | reste {left/60:.1f} min", flush=True)
            if flush_every and i % flush_every == 0:
                probs_now = np.concatenate(chunks, axis=0)
                all_uids = done_uids + got
                all_probs = probs_now if done_probs is None else np.concatenate(
                    [done_probs, probs_now], axis=0)
                _frame(all_uids, epoch, all_probs).to_csv(partial, index=False)
    probs = np.concatenate(chunks, axis=0)
    if got != todo:
        raise SystemExit(f"{run} : {len(got)} UID rendus, {len(todo)} attendus")
    all_uids = done_uids + got
    all_probs = probs if done_probs is None else np.concatenate([done_probs, probs], axis=0)
    if all_uids != uids:
        raise SystemExit(f"{run} : liste finale != liste ViT")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _frame(all_uids, epoch, all_probs).to_csv(final, index=False)
    partial.unlink(missing_ok=True)
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
    print(f"-> {final} ({len(all_uids)} etudes, {final.stat().st_size / 1e6:.1f} Mo, "
          f"{time.time() - t0:.0f} s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="un run_name, sinon tous")
    ap.add_argument("--limit", type=int, default=0, help="N premieres etudes (smoke)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--flush-every", type=int, default=100)
    args = ap.parse_args()

    vit_csv = next((p for p in VIT_CANDIDATES if p.is_file()), None)
    if vit_csv is None:
        raise SystemExit("liste ViT introuvable dans data/hybrid_pred")
    uids = pd.read_csv(vit_csv, usecols=["StudyInstanceUID"]).StudyInstanceUID.astype(str).tolist()
    if args.limit:
        uids = uids[: args.limit]
    print(f"etudes {len(uids)} (source {vit_csv.name})", flush=True)
    distil_uids = ROOT / "checkpoints" / (
        "subset_v20h_dino_ep11_v3b_rsna1687_distil_main_crop145"
    ) / "train_uids.txt"
    if distil_uids.is_file() and not args.only:
        seen = {l.strip() for l in distil_uids.read_text(encoding="utf-8").splitlines() if l.strip()}
        n_over = len(set(uids) & seen)
        print(f"distil a vu {n_over}/{len(uids)} de ces etudes en train "
              f"(le forward est quand meme fait, comme demande)", flush=True)
    jobs = [(c, r) for c, r in JOBS if not args.only or r == args.only]
    if not jobs:
        raise SystemExit(f"run inconnu : {args.only}")
    missing = [r for _c, r in jobs if not (ROOT / "checkpoints" / r / "best.pt").is_file()]
    if missing and not args.limit:
        print("best.pt absent, ignores : " + ", ".join(missing), flush=True)
    for cfg_rel, run in jobs:
        if run in missing:
            continue
        predict_one(cfg_rel, run, uids, args.batch_size, args.num_workers, args.flush_every)


if __name__ == "__main__":
    main()
