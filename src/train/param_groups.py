# -*- coding: utf-8 -*-
"""Groupes AdamW : LLRD ViT + no_decay (LN, biais, embeds).

LLRD : lr_bloc = lr_backbone * (llrd ** (n_blocks - layer_id)).
layer 0 = patch-embed / tokens / stem_mix, layer 1..12 = blocks,
layer 12 aussi pour le LN final. Pic (bloc 12) = `lr_backbone`.
Hybrides timm (CoAtNet : `stages.S.blocks.B`) : blocs numerotes
globalement dans l'ordre (stage, bloc), stem = 0.

Sans `llrd` : un groupe backbone + un groupe head (recette actuelle),
mais chaque groupe porte `base_lr` + `role` pour le re-warmup.
"""
from __future__ import annotations

import re
from collections import defaultdict

import torch
from torch.optim import AdamW

_BLOCK_RE = re.compile(r"\.blocks\.(\d+)\.")
_STAGE_BLOCK_RE = re.compile(r"\.stages\.(\d+)\.blocks\.(\d+)\.")


def is_no_decay(name: str, param: torch.Tensor) -> bool:
    if param.ndim <= 1:
        return True
    n = name.lower()
    if name.endswith(".bias"):
        return True
    if "gamma" in n.rsplit(".", 1)[-1] or n.endswith(".gamma"):
        return True
    if any(k in n for k in ("cls_token", "reg_token", "pos_embed",
                             "pos_bias", "mask_token")):
        return True
    return False


def _n_blocks(named: list[tuple[str, torch.nn.Parameter]]) -> int:
    ids = []
    for name, _ in named:
        m = _BLOCK_RE.search(name)
        if m:
            ids.append(int(m.group(1)))
    return (max(ids) + 1) if ids else 0


def _stage_block_ids(names: list[str]) -> dict[tuple[int, int], int]:
    """(stage, bloc) -> index global 1..N, vide si pas d'encodeur par stages."""
    keys = set()
    for name in names:
        m = _STAGE_BLOCK_RE.search(name)
        if m:
            keys.add((int(m.group(1)), int(m.group(2))))
    return {k: i + 1 for i, k in enumerate(sorted(keys))}


def vit_layer_id(name: str, n_blocks: int,
                 stage_ids: dict[tuple[int, int], int] | None = None) -> int:
    """0 = embed, 1..n_blocks = bloc Transformer, n_blocks = LN final."""
    if stage_ids:
        m = _STAGE_BLOCK_RE.search(name)
        if m:
            return stage_ids[(int(m.group(1)), int(m.group(2)))]
    else:
        m = _BLOCK_RE.search(name)
        if m:
            return int(m.group(1)) + 1
    if n_blocks and (".backbone.norm." in name or name.endswith("backbone.norm.weight")
                      or name.endswith("backbone.norm.bias")):
        return n_blocks
    return 0


def _is_backbone(name: str) -> bool:
    return name.startswith("encoder.") or name.startswith("stem_mix.")


def iter_named_trainable(model: torch.nn.Module):
    for name, p in model.named_parameters():
        if p.requires_grad:
            yield name, p


def build_param_groups(model: torch.nn.Module, cfg: dict) -> list[dict]:
    wd = float(cfg.get("weight_decay", 0.05) or 0.0)
    lr_bb = float(cfg["lr_backbone"])
    lr_hd = float(cfg["lr_head"])
    llrd = cfg.get("llrd")
    llrd = None if llrd is None else float(llrd)
    split_nd = bool(cfg.get("no_decay", False)) or (llrd is not None)

    named = list(iter_named_trainable(model))
    bb = [(n, p) for n, p in named if _is_backbone(n)]
    hd = [(n, p) for n, p in named if not _is_backbone(n)]
    stage_ids = _stage_block_ids([n for n, _ in bb])
    n_blocks = len(stage_ids) if stage_ids else _n_blocks(bb)

    groups: list[dict] = []
    if llrd is not None and n_blocks == 0:
        print("llrd ignore : pas de .blocks.* dans l'encodeur", flush=True)
        llrd = None

    if llrd is not None:
        buckets: dict[tuple[int, bool], list[torch.nn.Parameter]] = defaultdict(list)
        for name, p in bb:
            lid = vit_layer_id(name, n_blocks, stage_ids)
            buckets[(lid, is_no_decay(name, p) if split_nd else False)].append(p)
        for (lid, nd) in sorted(buckets):
            scale = llrd ** (n_blocks - lid)
            base = lr_bb * scale
            params = buckets[(lid, nd)]
            if not params:
                continue
            label = "embed" if lid == 0 else f"block{lid:02d}"
            groups.append({
                "params": params,
                "lr": base,
                "base_lr": base,
                "weight_decay": 0.0 if nd else wd,
                "role": "backbone",
                "llrd_layer": lid,
                "label": f"{label}_{'nodecay' if nd else 'decay'}",
            })
        print(
            f"LLRD {llrd:g} | {n_blocks} blocs | pic (bloc {n_blocks}) "
            f"{lr_bb:.2e} | bloc1 {lr_bb * (llrd ** (n_blocks - 1)):.2e} "
            f"| embed {lr_bb * (llrd ** n_blocks):.2e}",
            flush=True,
        )
        for g in groups:
            if g["role"] != "backbone":
                continue
            print(
                f"  {g['label']:<18} lr={g['base_lr']:.3e} "
                f"n={len(g['params'])} wd={g['weight_decay']:g}",
                flush=True,
            )
    elif split_nd:
        decay = [p for n, p in bb if not is_no_decay(n, p)]
        nod = [p for n, p in bb if is_no_decay(n, p)]
        if decay:
            groups.append({
                "params": decay, "lr": lr_bb, "base_lr": lr_bb,
                "weight_decay": wd, "role": "backbone", "label": "bb_decay",
            })
        if nod:
            groups.append({
                "params": nod, "lr": lr_bb, "base_lr": lr_bb,
                "weight_decay": 0.0, "role": "backbone", "label": "bb_nodecay",
            })
    else:
        if bb:
            groups.append({
                "params": [p for _, p in bb],
                "lr": lr_bb, "base_lr": lr_bb, "weight_decay": wd,
                "role": "backbone", "label": "backbone",
            })

    if split_nd:
        hd_d = [p for n, p in hd if not is_no_decay(n, p)]
        hd_n = [p for n, p in hd if is_no_decay(n, p)]
        if hd_d:
            groups.append({
                "params": hd_d, "lr": lr_hd, "base_lr": lr_hd,
                "weight_decay": wd, "role": "head", "label": "head_decay",
            })
        if hd_n:
            groups.append({
                "params": hd_n, "lr": lr_hd, "base_lr": lr_hd,
                "weight_decay": 0.0, "role": "head", "label": "head_nodecay",
            })
    elif hd:
        groups.append({
            "params": [p for _, p in hd],
            "lr": lr_hd, "base_lr": lr_hd, "weight_decay": wd,
            "role": "head", "label": "head",
        })
    if not groups:
        raise SystemExit("aucun parametre trainable pour AdamW")
    return groups


def build_optimizer(model: torch.nn.Module, cfg: dict) -> AdamW:
    groups = build_param_groups(model, cfg)
    return AdamW(groups, weight_decay=0.0)


def set_group_lrs(optim: torch.optim.Optimizer, step: int, total_steps: int,
                  warmup: int, unfreeze_step: int | None,
                  rewarm_steps: int, lr_at) -> None:
    """Cosine par `base_lr`. Re-warmup **tous** les groupes backbone au degel."""
    for g in optim.param_groups:
        base = float(g.get("base_lr", g["lr"]))
        if g.get("role") == "backbone" and unfreeze_step is not None:
            dt = step - unfreeze_step
            if 0 <= dt < rewarm_steps:
                g["lr"] = base * (dt + 1) / max(1, rewarm_steps)
                continue
        g["lr"] = lr_at(step, total_steps, base, warmup)


def backbone_lr_range(optim: torch.optim.Optimizer) -> tuple[float, float]:
    lrs = [float(g["lr"]) for g in optim.param_groups
           if g.get("role") == "backbone"]
    if not lrs:
        return float("nan"), float("nan")
    return min(lrs), max(lrs)
