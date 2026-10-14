# -*- coding: utf-8 -*-
"""Pod : predictions du hybrid ViT-k 320 (best.pt) sur les etudes RSNA non
vues en train.

Univers = les etudes RSNA de data/csv/train.csv (4348 + 58 gold = 4406).
Vu en train = les 1265 RSNA de checkpoints/<run>/train_uids.txt (les 422 OAI
sont ignores, on ne predit que du RSNA). A predire = ~3141 etudes, dont les
652 de val (jamais entrainees) et les 58 gold (include_gold false au train).
Images : data/crop130_320 (meme pretraitement que le train, grayscale 320).
Eval en mode val (train=False : aucune augmentation), batch 4 (chunk 288 =
2 forwards, memoire forward-only). CSV -> data/hybrid_pred/.

    python runpod/_dump_hybrid_preds.py [--limit N] [--batch-size 4]
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
from src.models.registry import build_model  # noqa: E402
from src.nlp.lexicon import TARGETS  # noqa: E402
from src.train.train import load_config  # noqa: E402

CFG = "configs/hybrid025/kneenet_vit_k_llrd08_320_v3b_hybrid025.yaml"
RUN = "subset_vit_k_llrd08_320_v3b_hybrid025"
OUT = ROOT / "data" / "hybrid_pred" / "hybrid_preds_unseen_rsna.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test sur N etudes")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(CFG)
    ckpt_p = ROOT / "checkpoints" / RUN / "best.pt"
    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    epoch = int(ckpt.get("epoch", -1))
    print(f"best.pt : epoch {epoch} | "
          f"val AUC {float(ckpt.get('val_macro_auc', float('nan'))):.4f}", flush=True)

    seen = {l.strip() for l in
            (ROOT / "checkpoints" / RUN / "train_uids.txt")
            .read_text(encoding="utf-8").splitlines() if l.strip()}
    seen_rsna = {u for u in seen if not u.startswith("OAI_")}
    all_rsna = set(pd.read_csv(ROOT / "data/csv/train.csv",
                               usecols=["StudyInstanceUID"]).StudyInstanceUID)
    unseen = sorted(all_rsna - seen_rsna)
    print(f"RSNA total {len(all_rsna)} | vues en train {len(seen_rsna & all_rsna)} "
          f"| a predire {len(unseen)}", flush=True)
    if len(seen_rsna & all_rsna) != 1265:
        raise SystemExit("train_uids RSNA != 1265 : verifier train_uids.txt")
    if args.limit:
        unseen = unseen[: args.limit]

    # labels factices (la loss n'est pas utilisee ; seules les probas comptent)
    labels_df = pd.DataFrame({
        "StudyInstanceUID": unseen,
        **{t: np.zeros(len(unseen), dtype=np.float32) for t in TARGETS},
    })
    index, volume_index = make_series_index("data/csv/train_series.csv", cfg["zip_path"])
    n_idx = sum(1 for u in unseen if u in index.by_study)
    print(f"etudes avec serie indexee : {n_idx}/{len(unseen)}", flush=True)
    ds = StudyDataset(
        unseen, labels_df[["StudyInstanceUID"] + TARGETS],
        index, cfg["zip_path"], train=False, series_drop_p=0.0,
        n_series=int(cfg.get("n_series", 6)), volume_index=volume_index,
        rgb_neighbors=False, img_size=int(cfg.get("img_size", 320)),
        rgb_as_gray=False, laterality_map=None,
    )
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True,
                    worker_init_fn=_dataloader_worker_init if args.num_workers > 0 else None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"modele charge (strict) | batch {args.batch_size} | "
          f"{(len(unseen) + args.batch_size - 1) // args.batch_size} batches", flush=True)

    probs, uids = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (x, _y, uid) in enumerate(ld, 1):
            logits = model(x.to(device, non_blocking=True))
            probs.append(torch.sigmoid(logits.float()).cpu())
            uids.extend(uid)
            if i % 100 == 0:
                print(f"{i} batches | {len(uids)} etudes | {time.time() - t0:.0f} s",
                      flush=True)
    probs = torch.cat(probs).numpy()
    assert len(uids) == len(unseen), f"{len(uids)} != {len(unseen)}"

    out = pd.DataFrame({"StudyInstanceUID": uids, "epoch": epoch})
    for j, t in enumerate(TARGETS):
        out[f"proba_{t}"] = probs[:, j].astype(np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"-> {OUT} ({len(out)} etudes, {OUT.stat().st_size / 1e6:.1f} Mo)", flush=True)


if __name__ == "__main__":
    main()
