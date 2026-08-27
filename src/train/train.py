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
  python -X utf8 -m src.train.train --config configs/kneenet_v8.yaml
  python -X utf8 -m src.train.train --config configs/kneenet_v8_subset.yaml
  python -X utf8 -m src.train.train --config configs/kneenet_v9_t1_subset.yaml
  python -X utf8 -m src.train.train --list-models
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset_npz import (
    StudyDataset, gold_uids_from_train_csv, make_loaders, make_series_index,
)
from src.data.transforms import augment_batch
from src.models.registry import (
    build_model, count_params, default_n_series, list_models,
)
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


def compute_prior_logits(labels_csv: str, targets: list[str],
                         eps: float = 1e-4) -> list[float]:
    """Logit de la frequence positive de chaque cible (init du biais final).

    Demarrer le head exactement au prior empeche la phase "course au prior"
    de la BCE, qui ecrase la variance des logits avant que le head apprenne
    (collapse observe sur v8bis / convnext_small).
    """
    df = pd.read_csv(labels_csv)
    if "split" in df.columns:  # certains CSV n'ont pas encore de split
        df = df[df.split == "train"]
    priors = []
    for t in targets:
        p = float(df[t].dropna().mean())
        p = min(max(p, eps), 1 - eps)
        priors.append(float(np.log(p / (1 - p))))
    return priors


def masked_bce(logits: torch.Tensor, y: torch.Tensor,
               pos_weight: torch.Tensor, smoothing: float = 0.02,
               sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    mask = y >= 0
    if mask.sum() == 0:
        return logits.sum() * 0.0
    yc = y.clamp(0, 1)
    yc = yc * (1 - smoothing) + 0.5 * smoothing
    loss_mat = F.binary_cross_entropy_with_logits(
        logits, yc, reduction="none",
        pos_weight=pos_weight.to(logits.device).unsqueeze(0),
    )
    w = mask.float()
    if sample_weight is not None:
        w = w * sample_weight.to(logits.device).float().unsqueeze(1)
    return (loss_mat * w).sum() / w.sum().clamp(min=1)


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
        # eval en fp32 : en bf16, des logits peu disperses sont arrondis a des
        # probas identiques -> AUC = 0.5 exact (artefact de ties, vu sur v9)
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
    if cfg.get("skip_gold"):
        return None
    train_csv = Path(cfg["labels_csv"]).parent.parent / "csv" / "train.csv"
    if not train_csv.exists():
        return None
    train_df = pd.read_csv(train_csv)
    gold = train_df[train_df[TARGETS].notna().all(axis=1)]
    if len(gold) == 0:
        return None
    index, volume_index = make_series_index(cfg["series_csv"], cfg["zip_path"])
    gold = gold[gold.StudyInstanceUID.isin(index.by_study.keys())]
    if len(gold) == 0:
        return None
    ds = StudyDataset(
        gold.StudyInstanceUID.tolist(),
        gold[["StudyInstanceUID"] + TARGETS], index,
        cfg["zip_path"], train=False, series_drop_p=0.0,
        n_series=int(cfg.get("n_series", 3)),
        volume_index=volume_index,
        rgb_neighbors=bool(cfg.get("rgb_neighbors", False)),
        img_size=int(cfg.get("img_size", 224)),
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


def _publish_ckpt(src: Path, dest: Path, retries: int = 8) -> None:
    """Copie src -> dest via un .tmp unique (OneDrive verrouille souvent
    le meme nom .tmp d'un essai a l'autre, WinError 32)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: BaseException | None = None
    for attempt in range(retries):
        tmp = dest.with_name(
            f"{dest.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            shutil.copy2(src, tmp)
            try:
                os.replace(tmp, dest)
            except OSError:
                shutil.copy2(tmp, dest)
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            return
        except OSError as e:
            last_err = e
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(
        f"echec copie vers {dest} ({retries} essais) : {last_err}\n"
        "Souvent OneDrive/antivirus qui verrouille le .pt dans Documents."
    ) from last_err


def _disk_free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return float("nan")


def _torch_save_robust(obj, staging: Path) -> None:
    """torch.save avec retry. Le zip PyTorch plante souvent disque plein
    (`unexpected pos`) : 2e essai en pickle legacy."""
    last: BaseException | None = None
    for attempt, use_zip in enumerate((True, True, False)):
        try:
            staging.unlink(missing_ok=True)
            torch.save(obj, staging, _use_new_zipfile_serialization=use_zip)
            if staging.stat().st_size < 1_000_000:
                raise RuntimeError(
                    f"checkpoint trop petit : {staging.stat().st_size} o")
            return
        except (RuntimeError, OSError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def _atomic_torch_save(obj, path: Path, also: Path | None = None,
                       retries: int = 8) -> None:
    """torch.save dans %TEMP% (hors sync OneDrive), puis copie vers path.

    Evite un .pt tronque (`iostream stream error`) et WinError 32 sur un
    .tmp deja pris par OneDrive pendant l'ecriture lente (~300 Mo)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    free = _disk_free_gb(path)
    if free == free and free < 1.0:
        raise RuntimeError(
            f"disque presque plein ({free:.2f} Go libres) : "
            f"impossible d'ecrire {path}. Liberer de l'espace "
            "(ex. checkpoints/*/last.pt des runs termines)."
        )
    fd, raw = tempfile.mkstemp(suffix=".pt", prefix=f"{path.stem}_")
    os.close(fd)
    staging = Path(raw)
    try:
        _torch_save_robust(obj, staging)
        _publish_ckpt(staging, path, retries=retries)
        if also is not None:
            _publish_ckpt(staging, also, retries=retries)
    except (RuntimeError, OSError) as e:
        raise RuntimeError(
            f"echec sauvegarde {path} : {e}\n"
            "Souvent disque plein, ou OneDrive/antivirus qui verrouille "
            "le fichier (Documents est souvent synchronise)."
        ) from e
    finally:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass


def _copy_ckpt(src: Path, dst: Path) -> None:
    _publish_ckpt(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--model", default=None,
                        help="surcharge cfg['model'] (ex. kneenet_v8)")
    parser.add_argument("--list-models", action="store_true",
                        help="affiche les architectures et quitte")
    parser.add_argument("--resume", action="store_true",
                        help="reprend last.pt (sinon best.pt) du run_name")
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint explicite (avec --resume)")
    parser.add_argument("--fold", type=int, default=None,
                        help="OOF : tenir ce fold a l'ecart (val + dump oof_preds.csv)")
    args = parser.parse_args()
    if args.list_models:
        for name in list_models():
            print(name)
        return
    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    cfg.setdefault("model", "kneenet_v8")
    cfg.setdefault("n_series", default_n_series(cfg["model"]))
    if args.fold is not None:
        cfg["oof_fold"] = int(args.fold)
        cfg["dump_oof"] = True
        base = str(cfg.get("run_name", "run"))
        suffix = f"_fold{args.fold}"
        if not base.endswith(suffix):
            cfg["run_name"] = base + suffix
        print(f"OOF fold {args.fold} -> run_name {cfg['run_name']}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    run_dir = Path("checkpoints") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ld, val_ld, targets = make_loaders(
        labels_csv=cfg["labels_csv"], series_csv=cfg["series_csv"],
        zip_path=cfg["zip_path"], batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        n_series=int(cfg.get("n_series", 3)),
        train_frac=float(cfg.get("train_frac", 1.0)),
        train_frac_seed=int(cfg.get("train_frac_seed", 42)),
        rgb_neighbors=bool(cfg.get("rgb_neighbors", False)),
        img_size=int(cfg.get("img_size", 224)),
        oof_fold=None if cfg.get("oof_fold") is None else int(cfg["oof_fold"]),
        folds_csv=cfg.get("folds_csv"),
        n_folds=int(cfg.get("n_folds", 3)),
        fold_seed=int(cfg.get("fold_seed", 42)),
        include_gold=bool(cfg.get("include_gold", False)),
        val_frac=float(cfg.get("val_frac", 0.15)),
    )
    n_train, n_val = len(train_ld.dataset), len(val_ld.dataset)
    print(f"train {n_train} | val {n_val} "
          f"| n_series {int(cfg.get('n_series', 3))}", flush=True)
    (run_dir / "train_uids.txt").write_text(
        "\n".join(train_ld.dataset.items), encoding="utf-8")

    resume_path = _ckpt_path(run_dir, args.ckpt) if args.resume else None
    if args.resume and resume_path is None:
        raise SystemExit("aucun checkpoint (last.pt / best.pt) pour --resume")

    arch = cfg.get("model", "kneenet_v8")
    prior_logits = compute_prior_logits(cfg["labels_csv"], targets)
    model = build_model(
        cfg, pretrained=bool(cfg.get("pretrained", True)) and resume_path is None,
        bias_init=prior_logits,
    ).to(device)
    print(f"{arch} ({cfg.get('backbone', '?')}) : {count_params(model)} params",
          flush=True)

    max_pw = cfg.get("pos_weight_max", None)  # None -> desactive (BCE simple)
    if max_pw is None:
        pos_weight = torch.ones(len(targets))
        print("pos_weight desactive (BCE simple)", flush=True)
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

    # Linear probing : encodeur gele au debut (features stables pour le
    # head/attention), degel a l'epoch `freeze_encoder_epochs`.
    freeze_epochs = int(cfg.get("freeze_encoder_epochs", 0) or 0)
    encoder_frozen = False
    # Rampe courte au degel : les moments AdamW du backbone sont vierges
    # (aucun gradient pendant le gel) -> sans re-warmup, les premiers pas
    # sont ~lr par coordonnee et detruisent l'adaptation du head.
    rewarm_steps = 0
    unfreeze_step = None

    epochs = int(cfg["epochs"])
    steps_per_epoch = len(train_ld)
    cosine_epochs = int(cfg.get("cosine_epochs") or epochs)
    total_steps = cosine_epochs * steps_per_epoch
    warmup = int(0.05 * total_steps)
    if cosine_epochs != epochs:
        print(f"cosine calcule sur {cosine_epochs} epochs "
              f"(train stop a {epochs})", flush=True)

    gold_w = float(cfg.get("gold_sample_weight") or 1.0)
    gold_uids: set[str] = set()
    if gold_w != 1.0:
        gold_uids = gold_uids_from_train_csv(cfg["labels_csv"])
        print(f"gold sample_weight x{gold_w:g} "
              f"({len(gold_uids)} uids)", flush=True)

    save_epochs = {int(e) for e in (cfg.get("save_epochs") or [])}
    if save_epochs:
        print("snapshots epochs : " +
              ", ".join(str(e) for e in sorted(save_epochs)), flush=True)

    # Gold = 58 etudes : 0 worker. Sinon, sous Windows, train+val+gold
    # ouvrent chacun 3 processus qui mmap les zip -> OOM CPU (~14 Mo
    # refusés alors que le tenseur est petit).
    gold_in_train = bool(cfg.get("include_gold", False))
    gold_ld = make_gold_loader(cfg, cfg["batch_size"], num_workers=0)
    if gold_ld is not None:
        hold = ("DANS le train, AUC n'est plus un hold-out"
                if gold_in_train else "loggee, jamais pour le checkpoint")
        print(f"gold sanity loader : {len(gold_ld.dataset)} etudes "
              f"({hold})", flush=True)
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

    # Recap apres le spawn DataLoader : sous Windows, les prints du
    # make_loaders sont souvent avales par les workers (stdout partage).
    recap = (
        f"dataset effectif : train {n_train} | val {n_val} "
        f"| n_series {int(cfg.get('n_series', 3))} "
        f"| train_frac {float(cfg.get('train_frac', 1.0)):g} "
        f"| val_frac {float(cfg.get('val_frac', 0.15)):g} "
        f"| img_size {int(cfg.get('img_size', 224))} "
        f"| oof_fold {cfg.get('oof_fold', '-')} "
        f"| include_gold {int(bool(cfg.get('include_gold', False)))} "
        f"| steps/epoch {steps_per_epoch}"
    )
    print(recap, flush=True)
    print(recap, file=sys.stderr, flush=True)

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

    if (freeze_epochs > 0 and start_epoch < freeze_epochs
            and hasattr(model, "set_encoder_frozen")):
        model.set_encoder_frozen(True)
        encoder_frozen = True
        print(f"encodeur gele jusqu'a l'epoch {freeze_epochs} "
              "(linear probing)", flush=True)

    rewarm_steps = int(0.25 * steps_per_epoch)

    for epoch in range(start_epoch, epochs):
        if encoder_frozen and epoch >= freeze_epochs:
            model.set_encoder_frozen(False)
            encoder_frozen = False
            unfreeze_step = step
            print(f"epoch {epoch + 1} : encodeur degele "
                  f"(re-warmup {rewarm_steps} steps)", flush=True)
        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        for x, y, _uid in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = augment_batch(x)

            for g in optim.param_groups:
                if g is optim.param_groups[0]:
                    base = cfg["lr_backbone"]
                    if (unfreeze_step is not None
                            and step - unfreeze_step < rewarm_steps):
                        g["lr"] = base * (step - unfreeze_step + 1) / rewarm_steps
                    else:
                        g["lr"] = lr_at(step, total_steps, base, warmup)
                else:
                    g["lr"] = lr_at(step, total_steps, cfg["lr_head"], warmup)

            sw = None
            if gold_uids:
                sw = torch.tensor(
                    [gold_w if u in gold_uids else 1.0 for u in _uid],
                    device=device, dtype=torch.float32)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
                loss = masked_bce(logits, y, pos_weight, smoothing,
                                  sample_weight=sw)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running += loss.item()
            nb += 1
            step += 1

        train_loss = running / max(nb, 1)
        per_target: dict[str, float] = {}
        gold_auc = None
        if n_val > 0:
            vp, vy, _, val_loss = evaluate(
                model, val_ld, device, pos_weight=pos_weight, smoothing=smoothing)
            val_auc, per_target = macro_auc(vy, vp, targets)
            pstd = float(vp.std())
        else:
            val_loss, val_auc, pstd = float("nan"), float("nan"), float("nan")
        if gold_ld is not None:
            gp, gy, _, _ = evaluate(
                model, gold_ld, device, pos_weight=pos_weight, smoothing=smoothing)
            gold_auc, _ = macro_auc(gy, gp, TARGETS)
        dt = time.time() - t0
        gold_txt = f" | gold AUC {gold_auc:.4f}" if gold_auc is not None else ""
        val_txt = (f"val {val_loss:.4f} | val AUC {val_auc:.4f}"
                   if n_val > 0 else "val -")
        pstd_txt = f"{pstd:.4f}" if pstd == pstd else "-"
        print(f"epoch {epoch+1}/{epochs} | train {train_loss:.4f} | "
              f"{val_txt}{gold_txt} | proba std {pstd_txt} | {dt:.0f}s",
              flush=True)
        if pstd == pstd and pstd < 0.005:
            print(f"  !! proba std {pstd:.2e} : probas presque constantes, "
                  "le modele s'effondre (graphe casse)", flush=True)
        rec = {"epoch": epoch + 1, "loss": train_loss, "val_loss": val_loss,
               "val_macro_auc": val_auc, "proba_std": pstd,
               **{f"auc_{k}": v for k, v in per_target.items()}}
        if gold_auc is not None:
            rec["gold_macro_auc"] = gold_auc
        history.append(rec)

        if n_val > 0 and val_auc > best_auc:
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
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        last_p = run_dir / "last.pt"
        saved = False
        try:
            _atomic_torch_save(
                payload, last_p,
                also=(run_dir / "best.pt") if improved else None)
            saved = True
        except RuntimeError as e:
            print(f"  !! {e}", flush=True)
        ep_num = epoch + 1
        if saved and ep_num in save_epochs:
            snap = run_dir / f"epoch{ep_num}.pt"
            try:
                _publish_ckpt(last_p, snap)
                print(f"  -> snapshot {snap.name}", flush=True)
            except RuntimeError as e:
                print(f"  !! snapshot {snap.name} : {e}", flush=True)
        if improved:
            if saved:
                print(f"  -> checkpoint (best {best_auc:.4f})", flush=True)
            else:
                print(f"  -> best AUC {best_auc:.4f} en RAM, "
                      "PAS ecrit sur disque", flush=True)
        elif patience > 0 and n_val > 0:
            print(f"  -> pas de gain val AUC ({stale}/{patience})", flush=True)
            if stale >= patience:
                print(f"early stop a l'epoch {epoch+1} "
                      f"(meilleur val AUC {best_auc:.4f})", flush=True)
                break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Sanity gold du meilleur checkpoint (jamais utilise pour piloter)
    if gold_ld is not None:
        ckpt_p = run_dir / "best.pt"
        if not ckpt_p.is_file():
            ckpt_p = run_dir / "last.pt"
        if ckpt_p.is_file():
            ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            gp, gy, _, _ = evaluate(
                model, gold_ld, device, pos_weight=pos_weight, smoothing=smoothing)
            g_auc, g_per = macro_auc(gy, gp, TARGETS)
            seen = "dans le train" if gold_in_train else "jamais vues"
            print(f"\n=== SANITY GOLD (58 etudes, {seen}, "
                  f"epoch {ckpt['epoch']}, {ckpt_p.name}) ===")
            print(f"macro-AUC gold : {g_auc:.4f}")
            for k, v in sorted(g_per.items(), key=lambda kv: -kv[1]):
                print(f"  {k:<18} {v:.3f}")
            (run_dir / "gold_sanity.json").write_text(json.dumps(
                {"macro_auc": g_auc, "per_target": g_per,
                 "epoch": ckpt["epoch"], "ckpt": str(ckpt_p),
                 "gold_in_train": gold_in_train},
                indent=2))

    if cfg.get("dump_oof") and n_val > 0:
        best_p = run_dir / "best.pt"
        if not best_p.is_file():
            print("dump OOF ignore : pas de best.pt", flush=True)
        else:
            ckpt = torch.load(best_p, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            vp, vy, vuids, _ = evaluate(model, val_ld, device)
            fold = int(cfg.get("oof_fold", -1))
            recs = []
            for i, uid in enumerate(vuids):
                row = {"StudyInstanceUID": uid, "fold": fold,
                       "epoch": int(ckpt.get("epoch", -1))}
                for j, t in enumerate(targets):
                    row[f"label_{t}"] = float(vy[i, j])
                    row[f"proba_{t}"] = float(vp[i, j])
                recs.append(row)
            out_p = run_dir / "oof_preds.csv"
            pd.DataFrame(recs).to_csv(out_p, index=False)
            print(f"OOF dump {len(recs)} etudes (best epoch {ckpt.get('epoch')}) "
                  f"-> {out_p}", flush=True)

    print(f"\ntermine. best val macro-AUC = {best_auc:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
