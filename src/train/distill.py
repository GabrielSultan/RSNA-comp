# -*- coding: utf-8 -*-
"""Distillation binaire (KL / BCE molle) depuis un CSV de logits teacher.

Loss = BCE(LLM masquee) + λ * T² * BCE(z_s/T, σ(z_t/T))
sur un sous-ensemble de cibles. Equivalent (gradient) a KL(teacher || student)
a l'entropie du teacher pres.

v20b <- ViT-e : MCL, menisque medial, epanchement (trou v20b).
ViT-e <- v20b : Contusion, Fracture, 3 OA (trou 3D / osseux).

Usage train : cles YAML `distil_csv`, `distil_targets`, `distil_lambda`,
`distil_temp` (defaut 1). Dumper le teacher avant :

  python -X utf8 -m src.train.dump_teacher_preds \\
      --config configs/archive/kneenet_vit_e_crop130_subset.yaml
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Cibles ou ViT-e gagne (enseigner a v20b).
DISTIL_V20B_FROM_VITE = ("MCL", "Medial Meniscus", "Effusion")
# Cibles ou v20b gagne (enseigner a ViT-e).
DISTIL_VITE_FROM_V20B = (
    "Contusion", "Fracture", "Medial OA", "Lateral OA", "PF OA",
)


@dataclass
class DistilSpec:
    table: dict[str, np.ndarray]
    idx: torch.Tensor
    names: list[str]
    lam: float
    temp: float
    n: int
    csv: Path


def _logit_col(name: str) -> str:
    return f"logit_{name}"


def _proba_col(name: str) -> str:
    return f"proba_{name}"


def logits_from_frame(df: pd.DataFrame, targets: list[str],
                      eps: float = 1e-4) -> dict[str, np.ndarray]:
    """uid -> logits (T,) float32. Prefere `logit_*`, sinon logit(proba)."""
    if "StudyInstanceUID" not in df.columns:
        raise ValueError("CSV teacher : colonne StudyInstanceUID manquante")
    have_logit = all(_logit_col(t) in df.columns for t in targets)
    have_proba = all(_proba_col(t) in df.columns for t in targets)
    if not have_logit and not have_proba:
        raise ValueError(
            "CSV teacher : ni logit_<cible> ni proba_<cible> "
            f"pour {targets}"
        )
    out: dict[str, np.ndarray] = {}
    uids = df["StudyInstanceUID"].astype(str).tolist()
    if have_logit:
        mat = np.stack(
            [pd.to_numeric(df[_logit_col(t)], errors="coerce").to_numpy(
                dtype=np.float32) for t in targets],
            axis=1,
        )
    else:
        p = np.stack(
            [pd.to_numeric(df[_proba_col(t)], errors="coerce").to_numpy(
                dtype=np.float64) for t in targets],
            axis=1,
        )
        p = np.clip(p, eps, 1.0 - eps)
        mat = np.log(p / (1.0 - p)).astype(np.float32)
    for uid, row in zip(uids, mat):
        out[uid] = np.where(np.isfinite(row), row, np.nan).astype(np.float32)
    return out


def load_distil(cfg: dict, targets: list[str]) -> DistilSpec | None:
    raw = cfg.get("distil_csv")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_file():
        raise SystemExit(
            f"distil_csv introuvable : {path}\n"
            "Dumper le teacher, par ex. :\n"
            "  python -X utf8 -m src.train.dump_teacher_preds "
            "--config configs/archive/kneenet_vit_e_crop130_subset.yaml"
        )
    names = list(cfg.get("distil_targets") or DISTIL_V20B_FROM_VITE)
    unknown = [n for n in names if n not in targets]
    if unknown:
        raise SystemExit(f"distil_targets inconnues : {unknown}")
    if not names:
        return None
    lam = float(cfg.get("distil_lambda", 0.3) or 0.0)
    if lam <= 0.0:
        print("distil_lambda <= 0 : distillation desactivee", flush=True)
        return None
    temp = float(cfg.get("distil_temp", 1.0) or 1.0)
    if temp <= 0.0:
        raise SystemExit(f"distil_temp={temp} : doit etre > 0")
    df = pd.read_csv(path)
    table = logits_from_frame(df, targets)
    idx = torch.tensor([targets.index(n) for n in names], dtype=torch.long)
    return DistilSpec(
        table=table, idx=idx, names=names, lam=lam, temp=temp,
        n=len(table), csv=path,
    )


def teacher_batch(uids: list[str] | tuple[str, ...], spec: DistilSpec,
                  n_targets: int, device: torch.device | str) -> torch.Tensor:
    """(B, T) logits teacher, NaN si uid absent."""
    rows = np.full((len(uids), n_targets), np.nan, dtype=np.float32)
    for i, uid in enumerate(uids):
        row = spec.table.get(str(uid))
        if row is not None and row.shape[0] == n_targets:
            rows[i] = row
    return torch.from_numpy(rows).to(device)


def coverage(uids: list[str], spec: DistilSpec) -> tuple[int, int]:
    hit = sum(1 for u in uids if str(u) in spec.table)
    return hit, len(uids)


def masked_kd(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
              target_idx: torch.Tensor, temp: float = 1.0,
              sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """T² BCE(z_s/T, σ(z_t/T)) sur `target_idx`. NaN teacher = ignore.

    student_logits / teacher_logits : (B, T). Retour scalaire fp32.
    """
    idx = target_idx.to(device=student_logits.device)
    s = student_logits.float().index_select(1, idx)
    t = teacher_logits.float().index_select(1, idx)
    valid = torch.isfinite(t)
    if not valid.any():
        return s.sum() * 0.0
    tau = float(temp)
    t_clean = torch.where(valid, t, torch.zeros_like(t))
    t_soft = torch.sigmoid(t_clean / tau).clamp(1e-4, 1.0 - 1e-4)
    s_scaled = s / tau
    loss_mat = F.binary_cross_entropy_with_logits(
        s_scaled, t_soft, reduction="none",
    ) * (tau * tau)
    w = valid.float()
    if sample_weight is not None:
        w = w * sample_weight.float().unsqueeze(1)
    loss_mat = loss_mat.masked_fill(~valid, 0.0)
    return (loss_mat * w).sum() / w.sum().clamp(min=1)


def default_targets_for_student(model_name: str) -> tuple[str, ...]:
    name = str(model_name)
    if name.startswith("kneenet_vit"):
        return DISTIL_VITE_FROM_V20B
    return DISTIL_V20B_FROM_VITE


# reexport pour les YAML / docs
__all__ = (
    "DISTIL_V20B_FROM_VITE", "DISTIL_VITE_FROM_V20B", "DistilSpec",
    "load_distil", "teacher_batch", "coverage", "masked_kd",
    "logits_from_frame", "default_targets_for_student",
)
