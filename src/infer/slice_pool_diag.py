# -*- coding: utf-8 -*-
"""Pooling coupe max / top-2 / mean a l'inference (sans retrain).

Le forward v20b reduit les tokens S*D par un softmax unique, identique
pour les 12 cibles. Ici on reutilise interpret() + attn_score et on
compare d'autres reductions, a priori focale vs diffus :

  max   : Fracture, Contusion, menisques, Baker's
  top-2 : ACL, MCL
  softmax (actuel) : OA, Effusion, Synovitis

Usage :
  python -X utf8 -m src.infer.slice_pool_diag --config configs/kneenet_v20b_crop130_subset.yaml
  python -X utf8 -m src.infer.slice_pool_diag --config configs/kneenet_v20b_crop130_oofmask_full.yaml --ckpt checkpoints/v20b_crop130_oofmask_full/epoch8.pt --splits gold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.infer.inspect_attention import build_loader, load_ckpt_model
from src.nlp.lexicon import TARGETS
from src.train.metrics import macro_auc
from src.train.train import load_config

# Recette DINOsaur, figee avant de voir les AUC (pas de gridsearch gold).
MAX_TARGETS = (
    "Fracture", "Contusion", "Medial Meniscus", "Lateral Meniscus", "Baker's",
)
TOP2_TARGETS = ("ACL", "MCL")
SOFTMAX_TARGETS = (
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
)


def _resolve_ckpt(cfg: dict, explicit: str | None) -> Path:
    run_dir = Path("checkpoints") / cfg["run_name"]
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"checkpoint introuvable : {p}")
        return p
    for name in ("best.pt", "epoch8.pt", "last.pt"):
        cand = run_dir / name
        if cand.is_file():
            return cand
    raise SystemExit(f"aucun ckpt dans {run_dir} (best/epoch8/last)")


def _hook_interpret(model, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def _hook(_mod, _inp, out):
        captured["scores"] = out

    handle = model.attn_score.register_forward_hook(_hook)
    try:
        out = model.interpret(x)
    finally:
        handle.remove()
    if "scores" not in captured:
        raise RuntimeError("attn_score n'a pas ete appele")
    return out, captured["scores"]


def _gather_tokens(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """tokens (B, N, T, C), index (B, K, T) -> (B, T, K, C)."""
    b, _n, t, c = tokens.shape
    k = index.size(1)
    src = tokens.permute(0, 2, 1, 3).contiguous()
    idx = index.permute(0, 2, 1).unsqueeze(-1).expand(b, t, k, c)
    return torch.gather(src, 2, idx)


def _head_logits(model, pooled: torch.Tensor) -> torch.Tensor:
    """pooled (B, T, C) -> logits (B, T)."""
    return model.head(model.norm(pooled.float())).squeeze(-1)


def reductions_from_batch(model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Un batch -> probabilites (B, T) par methode."""
    out, scores = _hook_interpret(model, x)
    tokens = out["cam_tokens"].float()
    valid = out["valid"]
    softmax_logits = out["logits"].float()
    b, s, d, t, c = tokens.shape
    n = s * d
    flat_tok = tokens.reshape(b, n, t, c)
    scores = scores.float().masked_fill(~valid.unsqueeze(-1), -1.0e4)
    flat_sc = scores.reshape(b, n, t)
    valid_n = valid.reshape(b, n)

    k2 = min(2, n)
    _max_v, max_idx = flat_sc.max(dim=1)
    top2_idx = flat_sc.topk(k2, dim=1).indices
    token_max = _head_logits(model, _gather_tokens(
        flat_tok, max_idx.unsqueeze(1)).squeeze(2))
    token_top2 = _head_logits(
        model, _gather_tokens(flat_tok, top2_idx).mean(dim=2))

    slice_logits = model.head(model.norm(flat_tok)).squeeze(-1)
    slice_logits = slice_logits.masked_fill(~valid_n.unsqueeze(-1), -1.0e4)
    logit_max = slice_logits.max(dim=1).values
    logit_top2 = slice_logits.topk(k2, dim=1).values.mean(dim=1)

    slice_prob = torch.sigmoid(slice_logits)
    slice_prob = slice_prob.masked_fill(~valid_n.unsqueeze(-1), 0.0)
    denom = valid_n.sum(dim=1, keepdim=True).clamp(min=1).float()
    logit_mean = slice_prob.sum(dim=1) / denom

    w = torch.softmax(8.0 * slice_prob.masked_fill(
        ~valid_n.unsqueeze(-1), -1.0e4), dim=1)
    logit_soft8 = (w * slice_prob).sum(dim=1)

    parts = {
        "softmax": torch.sigmoid(softmax_logits),
        "token_max": torch.sigmoid(token_max),
        "token_top2": torch.sigmoid(token_top2),
        "logit_max": torch.sigmoid(logit_max),
        "logit_top2": torch.sigmoid(logit_top2),
        "logit_mean": logit_mean,
        "logit_soft8": logit_soft8,
    }
    mix = parts["softmax"].clone()
    names = list(TARGETS)
    for i, name in enumerate(names):
        if name in MAX_TARGETS:
            mix[:, i] = parts["logit_max"][:, i]
        elif name in TOP2_TARGETS:
            mix[:, i] = parts["logit_top2"][:, i]
    parts["mix_dino"] = mix
    return parts


@torch.no_grad()
def collect(model, loader, device: str) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    model.eval()
    buckets: dict[str, list[torch.Tensor]] = {}
    labels: list[torch.Tensor] = []
    uids: list[str] = []
    n_done = 0
    for x, y, uid in loader:
        x = x.to(device, non_blocking=True)
        parts = reductions_from_batch(model, x)
        for k, v in parts.items():
            buckets.setdefault(k, []).append(v.cpu())
        labels.append(y)
        uids.extend(uid)
        n_done += x.size(0)
        print(f"  {n_done} etudes", flush=True)
    probs = {k: torch.cat(v).numpy() for k, v in buckets.items()}
    y_np = torch.cat(labels).numpy()
    return probs, y_np, uids


def _score_all(probs: dict[str, np.ndarray], y: np.ndarray) -> dict:
    rows = []
    per_method = {}
    for name, p in probs.items():
        macro, aucs = macro_auc(y, p, list(TARGETS))
        per_method[name] = {"macro": macro, "aucs": aucs}
        rec = {"method": name, "macro": macro}
        rec.update({t: aucs.get(t, float("nan")) for t in TARGETS})
        rows.append(rec)
    oracle_p = probs["softmax"].copy()
    oracle_src = {}
    base = per_method["softmax"]["aucs"]
    for t in TARGETS:
        best_name, best_auc = "softmax", base.get(t, float("nan"))
        for name, info in per_method.items():
            if name in ("mix_dino",):
                continue
            auc = info["aucs"].get(t, float("nan"))
            if auc == auc and (best_auc != best_auc or auc > best_auc):
                best_name, best_auc = name, auc
        oracle_src[t] = best_name
        j = TARGETS.index(t)
        oracle_p[:, j] = probs[best_name][:, j]
    o_macro, o_aucs = macro_auc(y, oracle_p, list(TARGETS))
    rec = {"method": "oracle_per_target", "macro": o_macro}
    rec.update({t: o_aucs.get(t, float("nan")) for t in TARGETS})
    rows.append(rec)
    per_method["oracle_per_target"] = {
        "macro": o_macro, "aucs": o_aucs, "source": oracle_src,
    }
    return {"table": rows, "per_method": per_method}


def _print_table(split: str, scored: dict, n: int) -> None:
    df = pd.DataFrame(scored["table"]).set_index("method")
    base = df.loc["softmax", "macro"]
    df["d_macro"] = df["macro"] - base
    cols = ["macro", "d_macro"] + list(TARGETS)
    print(f"\n=== {split}  n={n}  (delta vs softmax) ===", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[cols].round(4).to_string(), flush=True)
    src = scored["per_method"]["oracle_per_target"].get("source") or {}
    if src:
        print("oracle (sur ce split, plafond) :",
              ", ".join(f"{t}={src[t]}" for t in TARGETS), flush=True)


def run_split(cfg: dict, model, split: str, device: str,
              batch_size: int, workers: int, max_studies: int | None,
              out_dir: Path) -> dict | None:
    try:
        loader, _targets = build_loader(
            cfg, split, batch_size, workers, max_studies)
    except SystemExit as exc:
        print(f"split {split} ignore : {exc}", flush=True)
        return None
    n = len(loader.dataset)
    if n == 0:
        print(f"split {split} vide, ignore", flush=True)
        return None
    print(f"split {split} : {n} etudes", flush=True)
    probs, y, uids = collect(model, loader, device)
    scored = _score_all(probs, y)
    _print_table(split, scored, n)
    payload = {
        "split": split,
        "n_studies": n,
        "uids": uids,
        "mapping": {
            "max": list(MAX_TARGETS),
            "top2": list(TOP2_TARGETS),
            "softmax": list(SOFTMAX_TARGETS),
        },
        "per_method": {
            k: {
                "macro": v["macro"],
                "aucs": v["aucs"],
                **({"source": v["source"]} if "source" in v else {}),
            }
            for k, v in scored["per_method"].items()
        },
        "delta_vs_softmax": {
            k: float(v["macro"] - scored["per_method"]["softmax"]["macro"])
            for k, v in scored["per_method"].items()
        },
    }
    out_path = out_dir / f"slice_pool_{split}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(scored["table"]).to_csv(
        out_dir / f"slice_pool_{split}.csv", index=False)
    print(f"ecrit -> {out_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kneenet_v20b_crop130_subset.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--splits", default="val,gold")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-studies", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("model", "kneenet_v20b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _resolve_ckpt(cfg, args.ckpt)
    out_dir = Path(args.out) if args.out else (
        Path("checkpoints") / cfg["run_name"] / "slice_pool_diag"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(args.batch_size or cfg.get("batch_size") or 4)

    model, epoch, val_auc = load_ckpt_model(cfg, ckpt, device)
    if not hasattr(model, "interpret"):
        raise SystemExit(f"{cfg.get('model')} n'a pas interpret()")
    print(f"ckpt {ckpt} | epoch {epoch} | val_macro_auc ckpt {val_auc} | {device}",
          flush=True)
    print(f"mix_dino : max={list(MAX_TARGETS)} | top2={list(TOP2_TARGETS)} | "
          f"softmax={list(SOFTMAX_TARGETS)}", flush=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    summary = {"ckpt": str(ckpt), "epoch": epoch, "device": device, "splits": {}}
    for split in splits:
        payload = run_split(
            cfg, model, split, device, batch_size, args.workers,
            args.max_studies, out_dir,
        )
        if payload is not None:
            summary["splits"][split] = {
                "n_studies": payload["n_studies"],
                "delta_vs_softmax": payload["delta_vs_softmax"],
                "macros": {k: v["macro"] for k, v in payload["per_method"].items()},
            }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nresume -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
