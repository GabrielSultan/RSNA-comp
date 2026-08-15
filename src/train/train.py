# -*- coding: utf-8 -*-
"""Boucle d'entrainement KneeNet sur pseudo-labels masques.

- Loss : BCEWithLogits masquee (y == -1 ignore) + pos_weight par cible
  (n_neg/n_pos sur le train garde, clampe) + label smoothing.
- Optim : AdamW, lr backbone < lr tete, cosine + warmup, AMP bf16.
- Augmentation GPU dans la boucle ; normalisation ImageNet dans KneeNet.
- Checkpoint du meilleur macro-AUC val -> checkpoints/<run>/best.pt
- Early stopping sur la val AUC (patience dans le YAML).
- Val loss (BCE, pas d'aug) loggee chaque epoch pour comparer a la train loss.
- Macro-AUC gold (58) loggee chaque epoch, jamais utilisee pour piloter.

Usage :
  python -X utf8 -m src.train.train --config configs/baseline.yaml
  python -X utf8 -m src.train.train --config configs/baseline.yaml --resume
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset_npz import StudyDataset, StudySeriesIndex, make_loaders
from src.data.transforms import augment_batch
from src.models.kneenet import KneeNet, count_params
from src.nlp.lexicon import TARGETS
from src.train.metrics import macro_auc


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_pos_weight(labels_csv: str, targets: list[str],
                       lo: float = 0.5, hi: float = 5.0,
                       max_weight: float | None = None) -> torch.Tensor:
    df = pd.read_csv(labels_csv)
    df = df[df.split == "train"]
    w = []
    for t in targets:
        col = df[t].dropna()
        n_pos, n_neg = (col == 1).sum(), (col == 0).sum()
        w.append(float(np.clip(n_neg / max(n_pos, 1), lo, hi)))
    if max_weight is not None:
        w = [min(x, max_weight) for x in w]
    return torch.tensor(w, dtype=torch.float32)


def masked_bce(logits: torch.Tensor, y: torch.Tensor,
               pos_weight: torch.Tensor, smoothing: float = 0.02) -> torch.Tensor:
    mask = y >= 0
    if mask.sum() == 0:
        return logits.sum() * 0.0
    yc = y.clamp(0, 1)
    yc = yc * (1 - smoothing) + 0.5 * smoothing
    loss_mat = F.binary_cross_entropy_with_logits(
        logits, yc, reduction="none",
        pos_weight=pos_weight.to(logits.device).unsqueeze(0),
    )
    return (loss_mat * mask).sum() / mask.sum().clamp(min=1)


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, device, pos_weight: torch.Tensor | None = None,
             smoothing: float = 0.0) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Forward eval : probabilites, labels, uids, BCE masquee moyenne.

    Pas d'augmentation. `pos_weight` / `smoothing` alignes sur la train loss
    pour que train vs val soient comparables (l'aug reste le seul ecart).
    """
    model.eval()
    probs, labels, uids = [], [], []
    loss_sum, n_mask = 0.0, 0.0
    pw = None if pos_weight is None else pos_weight.to(device)
    for x, y, uid in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        logits_f = logits.float()
        probs.append(torch.sigmoid(logits_f).cpu())
        labels.append(y)
        uids.extend(uid)
        if pw is not None:
            mask = y_dev >= 0
            if mask.any():
                yc = y_dev.clamp(0, 1)
                yc = yc * (1 - smoothing) + 0.5 * smoothing
                loss_mat = F.binary_cross_entropy_with_logits(
                    logits_f, yc, reduction="none",
                    pos_weight=pw.unsqueeze(0),
                )
                loss_sum += (loss_mat * mask).sum().item()
                n_mask += mask.sum().item()
    val_loss = loss_sum / n_mask if n_mask > 0 else float("nan")
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), uids, val_loss


def make_gold_loader(cfg: dict, batch_size: int, num_workers: int):
    """58 etudes gold depuis train.csv, ou None si indisponible."""
    train_csv = Path(cfg["labels_csv"]).parent.parent / "csv" / "train.csv"
    if not train_csv.exists():
        return None
    train_df = pd.read_csv(train_csv)
    gold = train_df[train_df[TARGETS].notna().all(axis=1)]
    if len(gold) == 0:
        return None
    index = StudySeriesIndex(cfg["series_csv"])
    gold = gold[gold.StudyInstanceUID.isin(index.by_study.keys())]
    if len(gold) == 0:
        return None
    ds = StudyDataset(
        gold.StudyInstanceUID.tolist(),
        gold[["StudyInstanceUID"] + TARGETS], index,
        cfg["zip_path"], train=False, series_drop_p=0.0,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def _ckpt_path(run_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    last_p, best_p = run_dir / "last.pt", run_dir / "best.pt"
    if last_p.exists():
        return last_p
    if best_p.exists():
        return best_p
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="reprend last.pt (sinon best.pt) du run_name")
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint explicite (avec --resume)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    run_dir = Path("checkpoints") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ld, val_ld, targets = make_loaders(
        labels_csv=cfg["labels_csv"], series_csv=cfg["series_csv"],
        zip_path=cfg["zip_path"], batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print(f"train {len(train_ld.dataset)} | val {len(val_ld.dataset)}", flush=True)

    resume_path = _ckpt_path(run_dir, args.ckpt) if args.resume else None
    if args.resume and resume_path is None:
        raise SystemExit("aucun checkpoint (last.pt / best.pt) pour --resume")

    model = KneeNet(
        backbone=cfg["backbone"], n_series=3, head_dim=cfg["head_dim"],
        dropout=cfg["dropout"],
        pretrained=bool(cfg["pretrained"]) and resume_path is None,
        encode_chunk=int(cfg.get("encode_chunk", 16)),
    ).to(device)
    print(f"KneeNet 2.5D-RGB attn({cfg['backbone']}) : {count_params(model)} params",
          flush=True)

    max_pw = cfg.get("pos_weight_max", None)  # None -> desactive (BCE simple)
    if max_pw is None:
        pos_weight = torch.ones(len(targets))
        print("pos_weight desactive (BCE simple)")
    else:
        pos_weight = compute_pos_weight(cfg["labels_csv"], targets,
                                        max_weight=max_pw)
        print(f"pos_weight (max {max_pw}):",
              [round(w, 2) for w in pos_weight.tolist()], flush=True)

    backbone_params = list(model.encoder_parameters())
    head_params = list(model.head_parameters())
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["lr_backbone"]},
        {"params": head_params, "lr": cfg["lr_head"]},
    ], weight_decay=cfg["weight_decay"])

    epochs = cfg["epochs"]
    steps_per_epoch = len(train_ld)
    total_steps = epochs * steps_per_epoch
    warmup = int(0.05 * total_steps)

    gold_ld = make_gold_loader(cfg, cfg["batch_size"], cfg["num_workers"])
    if gold_ld is not None:
        print(f"gold sanity loader : {len(gold_ld.dataset)} etudes "
              "(loggee, jamais pour le checkpoint)", flush=True)
    else:
        print("gold sanity loader : indisponible", flush=True)

    best_auc, history = -1.0, []
    step = 0
    stale = 0
    start_epoch = 0
    patience = int(cfg.get("early_stop_patience", 0) or 0)
    smoothing = float(cfg.get("label_smoothing", 0.0) or 0.0)
    if patience > 0:
        print(f"early stopping : patience {patience} (val macro-AUC)", flush=True)

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt.get("epoch", 0))
        step = int(ckpt.get("step", start_epoch * steps_per_epoch))
        best_auc = float(ckpt.get("val_macro_auc", -1.0))
        stale = int(ckpt.get("stale", 0))
        hist_p = run_dir / "history.json"
        if hist_p.exists():
            history = json.loads(hist_p.read_text(encoding="utf-8"))
            history = [h for h in history if int(h.get("epoch", 0)) <= start_epoch]
        print(f"resume {resume_path} | epoch {start_epoch} | "
              f"best val AUC {best_auc:.4f} | step {step}"
              f"{'' if 'optim' in ckpt else ' (AdamW neuf)'}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        for x, y, _uid in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = augment_batch(x)

            for g in optim.param_groups:
                g["lr"] = lr_at(step, total_steps,
                                cfg["lr_backbone"] if g is optim.param_groups[0]
                                else cfg["lr_head"], warmup)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
                loss = masked_bce(logits, y, pos_weight, smoothing)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running += loss.item()
            nb += 1
            step += 1

        train_loss = running / max(nb, 1)
        vp, vy, _, val_loss = evaluate(
            model, val_ld, device, pos_weight=pos_weight, smoothing=smoothing)
        val_auc, per_target = macro_auc(vy, vp, targets)
        gold_auc = None
        if gold_ld is not None:
            gp, gy, _, _ = evaluate(
                model, gold_ld, device, pos_weight=pos_weight, smoothing=smoothing)
            gold_auc, _ = macro_auc(gy, gp, TARGETS)
        pstd = float(vp.std())
        dt = time.time() - t0
        gold_txt = f" | gold AUC {gold_auc:.4f}" if gold_auc is not None else ""
        print(f"epoch {epoch+1}/{epochs} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | val AUC {val_auc:.4f}{gold_txt} | "
              f"proba std {pstd:.4f} | {dt:.0f}s",
              flush=True)
        rec = {"epoch": epoch + 1, "loss": train_loss, "val_loss": val_loss,
               "val_macro_auc": val_auc,
               **{f"auc_{k}": v for k, v in per_target.items()}}
        if gold_auc is not None:
            rec["gold_macro_auc"] = gold_auc
        history.append(rec)

        if val_auc > best_auc:
            best_auc = val_auc
            stale = 0
            improved = True
        else:
            stale += 1
            improved = False

        payload = {
            "model": model.state_dict(), "optim": optim.state_dict(),
            "config": cfg, "val_macro_auc": best_auc, "epoch": epoch + 1,
            "step": step, "stale": stale, "targets": targets,
        }
        torch.save(payload, run_dir / "last.pt")
        if improved:
            torch.save(payload, run_dir / "best.pt")
            print(f"  -> checkpoint (best {best_auc:.4f})", flush=True)
        elif patience > 0:
            print(f"  -> pas de gain val AUC ({stale}/{patience})", flush=True)
            if stale >= patience:
                print(f"early stop a l'epoch {epoch+1} "
                      f"(meilleur val AUC {best_auc:.4f})", flush=True)
                break
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Sanity gold du meilleur checkpoint (jamais utilise pour piloter)
    if gold_ld is not None:
        ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        gp, gy, _, _ = evaluate(
            model, gold_ld, device, pos_weight=pos_weight, smoothing=smoothing)
        g_auc, g_per = macro_auc(gy, gp, TARGETS)
        print(f"\n=== SANITY GOLD (58 etudes, jamais vues, best epoch {ckpt['epoch']}) ===")
        print(f"macro-AUC gold : {g_auc:.4f}")
        for k, v in sorted(g_per.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<18} {v:.3f}")
        (run_dir / "gold_sanity.json").write_text(json.dumps(
            {"macro_auc": g_auc, "per_target": g_per, "epoch": ckpt["epoch"]},
            indent=2))

    print(f"\ntermine. best val macro-AUC = {best_auc:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
