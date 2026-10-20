# -*- coding: utf-8 -*-
"""Gate de mélange. Il renvoie des poids, et les 12 probabilités sont le mélange.

Pour chaque cible, un softmax sur les modèles. La probabilité rendue est
la somme des poids fois la proba du modèle (`blend="prob"`), ou le
sigmoïde de la somme des poids fois les logits (`blend="logit"`).

Le biais (T, M) est le mélange constant, le même pour chaque étude.
Trois résidus optionnels, nuls à l'initialisation :

- `pred_mode="full"` : matrice logits, comme le notebook 13 dans le simplexe
- `pred_mode="diagonal"` : un coefficient par modèle, multiplié par son logit
- `empty_mode="full"` ou `"shared"` : décalage quand un slot est vide
- `use_image` : axes PCA du sagittal fat-sat

La loss est une BCE équilibrée : chaque cible pèse autant, et dans chaque
cible les positifs pèsent autant que les négatifs. Le checkpoint retenu
est l'epoch de meilleure macro-AUC train.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nlp.lexicon import TARGETS
from src.train.metrics import macro_auc

SLOTS = (
    "empty_SAG_FS", "empty_COR_FS", "empty_AX_FS",
    "empty_SAG_T1", "empty_COR_T1", "empty_AX_T1",
)
CLIP = 1e-4


class MixGate(nn.Module):
    """Score (B, T, M) puis softmax sur M. Tous les résidus sont nuls à l'init."""

    def __init__(
        self,
        n_models: int,
        n_targets: int,
        n_empty: int,
        gate_dim: int,
        use_pred: bool,
        use_empty: bool,
        use_image: bool,
        pred_mode: str = "full",
        empty_mode: str = "full",
        blend: str = "prob",
    ) -> None:
        super().__init__()
        self.n_models = int(n_models)
        self.n_targets = int(n_targets)
        self.pred_mode = "none" if not use_pred else pred_mode
        self.empty_mode = "none" if not use_empty else empty_mode
        self.use_image = bool(use_image)
        self.blend = blend
        if self.pred_mode not in ("none", "full", "diagonal"):
            raise ValueError(f"pred_mode={pred_mode}")
        if self.empty_mode not in ("none", "full", "shared"):
            raise ValueError(f"empty_mode={empty_mode}")
        if blend not in ("prob", "logit"):
            raise ValueError(f"blend={blend}")
        self.use_pred = self.pred_mode != "none"
        self.use_empty = self.empty_mode != "none"
        self.bias = nn.Parameter(torch.zeros(n_targets, n_models))
        if self.pred_mode == "full":
            self.pred_w = nn.Parameter(torch.zeros(n_targets, n_models, n_models))
        elif self.pred_mode == "diagonal":
            self.pred_diag = nn.Parameter(torch.zeros(n_targets, n_models))
        if self.empty_mode == "full":
            self.empty_w = nn.Parameter(torch.zeros(n_targets, n_empty, n_models))
        elif self.empty_mode == "shared":
            self.empty_shared = nn.Parameter(torch.zeros(n_empty, n_models))
        if self.use_image:
            if gate_dim < 1:
                raise ValueError("use_image exige gate_dim >= 1")
            self.img_w = nn.Parameter(torch.zeros(n_targets, gate_dim, n_models))

    def scores(
        self,
        logits: torch.Tensor,
        empty: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """logits (B, T, M), empty (B, E), gates (B, G) -> score (B, T, M)."""
        b = logits.shape[0]
        score = self.bias.unsqueeze(0).expand(b, -1, -1)
        if self.pred_mode == "full":
            score = score + torch.einsum("btm,tmk->btk", logits, self.pred_w)
        elif self.pred_mode == "diagonal":
            score = score + logits * self.pred_diag.unsqueeze(0)
        if self.empty_mode == "full":
            score = score + torch.einsum("be,tek->btk", empty, self.empty_w)
        elif self.empty_mode == "shared":
            shift = torch.einsum("be,em->bm", empty, self.empty_shared)
            score = score + shift.unsqueeze(1)
        if self.use_image:
            score = score + torch.einsum("bg,tgk->btk", gates, self.img_w)
        return score

    def forward(
        self,
        probs: torch.Tensor,
        logits: torch.Tensor,
        empty: torch.Tensor,
        gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """probs/logits (B, T, M). Retourne poids (B, T, M) et mélange (B, T)."""
        weight = torch.softmax(self.scores(logits, empty, gates), dim=-1)
        if self.blend == "logit":
            mixed = torch.sigmoid((weight * logits).sum(-1))
        else:
            mixed = (weight * probs).sum(-1)
        return weight, mixed


def _macro_bce(mixed: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BCE moyenne sur les cibles. Dans chaque cible, positifs = négatifs."""
    terms = []
    for t in range(y.shape[1]):
        yt = y[:, t]
        pt = mixed[:, t].clamp(CLIP, 1.0 - CLIP)
        pos = yt >= 0.5
        neg = ~pos
        parts = []
        if bool(pos.any()):
            parts.append(F.binary_cross_entropy(pt[pos], yt[pos]))
        if bool(neg.any()):
            parts.append(F.binary_cross_entropy(pt[neg], yt[neg]))
        if parts:
            terms.append(torch.stack(parts).mean())
    if not terms:
        raise RuntimeError("aucune cible exploitable dans le fit")
    return torch.stack(terms).mean()


def _logits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, CLIP, 1.0 - CLIP)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _pca_project(
    embed: np.ndarray,
    present: np.ndarray,
    fit_mask: np.ndarray,
    pca_dim: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """PCA sur les études fit qui ont la série. Les absentes restent à 0."""
    m = fit_mask & present.astype(bool)
    x = embed[m].astype(np.float64)
    if x.shape[0] < 2:
        raise RuntimeError("moins de 2 séries SAG_FS dans le fit")
    mu = x.mean(axis=0)
    xc = x - mu
    # SVD économique. vt[k] est l'axe k.
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = int(min(pca_dim, vt.shape[0], x.shape[0] - 1))
    if k < 1:
        raise RuntimeError("PCA vide")
    components = vt[:k].astype(np.float64)
    scores = (embed.astype(np.float64) - mu) @ components.T
    sd = scores[m].std(axis=0)
    sd = np.where(sd > 1e-4, sd, 1.0)
    center = scores[m].mean(axis=0)
    gates = ((scores - center) / sd).astype(np.float32)
    gates[~present.astype(bool)] = 0.0
    pca = {
        "mean": mu.astype(np.float32),
        "components": components.astype(np.float32),
        "score_mean": center.astype(np.float32),
        "score_scale": sd.astype(np.float32),
    }
    return gates, pca


def project_embed(
    embed: np.ndarray,
    present: np.ndarray,
    pca: dict[str, np.ndarray],
) -> np.ndarray:
    """Même projection que l'ajustement. present=0 -> vecteur nul."""
    scores = (embed.astype(np.float64) - pca["mean"].astype(np.float64)) @ pca["components"].astype(np.float64).T
    scale = pca["score_scale"].astype(np.float64)
    gates = (scores - pca["score_mean"].astype(np.float64)) / scale
    out = gates.astype(np.float32)
    out[~present.astype(bool)] = 0.0
    return out


def _auc(y: np.ndarray, score: np.ndarray, mask: np.ndarray, names: list[str]) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(macro_auc(y[mask], score[mask], names)[0])


def fit_mix_gate(
    probs: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    gold_mask: np.ndarray,
    empty: np.ndarray,
    embed: np.ndarray | None = None,
    present: np.ndarray | None = None,
    model_names: list[str] | None = None,
    target_names: list[str] | None = None,
    use_pred: bool = True,
    use_empty: bool = True,
    use_image: bool = False,
    pred_mode: str = "full",
    empty_mode: str = "full",
    blend: str = "prob",
    epochs: int = 400,
    lr: float = 0.05,
    weight_decay: float = 1e-2,
    residual_decay: float | None = None,
    seed: int = 42,
    pca_dim: int = 16,
) -> dict:
    """Ajuste un gate. `probs` est (N, M, T), dans [0, 1].

    Le checkpoint retenu est l'epoch de meilleure macro-AUC train.
    Val et gold sont lues, elles ne choisissent pas l'epoch.
    """
    names = list(target_names or TARGETS)
    probs = np.clip(np.asarray(probs, np.float32), 0.0, 1.0)
    if probs.ndim != 3:
        raise ValueError(f"probs {probs.shape} : attendu (N, M, T)")
    n, n_models, n_targets = probs.shape
    if n_targets != len(names):
        raise ValueError(f"{n_targets} cibles pour {len(names)} noms")
    y = np.asarray(y, np.float32)
    empty = np.asarray(empty, np.float32)
    if empty.shape != (n, len(SLOTS)):
        raise ValueError(f"empty {empty.shape} : attendu {(n, len(SLOTS))}")
    fit_mask = np.asarray(fit_mask, bool)
    val_mask = np.asarray(val_mask, bool)
    gold_mask = np.asarray(gold_mask, bool)

    gates_np = np.zeros((n, 1), np.float32)
    pca = None
    if use_image:
        if embed is None or present is None:
            raise ValueError("use_image exige embed et present")
        embed = np.asarray(embed, np.float32)
        present = np.asarray(present, bool)
        if embed.shape[0] != n or present.shape != (n,):
            raise ValueError("embed/present mal alignés sur les études")
        gates_np, pca = _pca_project(embed, present, fit_mask, pca_dim)

    torch.manual_seed(seed)
    mode_pred = "none" if not use_pred else pred_mode
    mode_empty = "none" if not use_empty else empty_mode
    model = MixGate(
        n_models, n_targets, empty.shape[1], gates_np.shape[1],
        mode_pred != "none", mode_empty != "none", use_image,
        pred_mode=mode_pred, empty_mode=mode_empty, blend=blend,
    )
    decay_r = weight_decay if residual_decay is None else float(residual_decay)
    groups = [{"params": [model.bias], "weight_decay": weight_decay}]
    rest = [p for name, p in model.named_parameters() if name != "bias"]
    if rest:
        groups.append({"params": rest, "weight_decay": decay_r})
    opt = torch.optim.AdamW(groups, lr=lr)
    probs_t = torch.tensor(np.transpose(probs, (0, 2, 1)))
    logits_t = torch.tensor(np.transpose(_logits(probs), (0, 2, 1)))
    empty_t = torch.tensor(empty)
    gates_t = torch.tensor(gates_np)
    y_fit = torch.tensor(y[fit_mask])
    y_auc = np.where(np.isfinite(y), y, -1.0)

    best_auc = -1.0
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    hist = {"train": [], "val": [], "gold": []}

    def _forward() -> tuple[torch.Tensor, np.ndarray]:
        weight, mixed = model(probs_t, logits_t, empty_t, gates_t)
        return weight, mixed.detach().numpy()

    for epoch in range(epochs + 1):
        model.eval()
        with torch.no_grad():
            _, mixed_np = _forward()
        tr = _auc(y_auc, mixed_np, fit_mask, names)
        va = _auc(y_auc, mixed_np, val_mask, names)
        go = _auc(y_auc, mixed_np, gold_mask, names)
        hist["train"].append(tr)
        hist["val"].append(va)
        hist["gold"].append(go)
        if np.isfinite(tr) and tr > best_auc:
            best_auc = tr
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == epochs:
            break
        model.train()
        opt.zero_grad()
        _, mixed = model(probs_t, logits_t, empty_t, gates_t)
        loss = _macro_bce(mixed[fit_mask], y_fit)
        loss.backward()
        opt.step()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        weight, mixed = model(probs_t, logits_t, empty_t, gates_t)
        weight_np = weight.numpy()
        mixed_np = mixed.numpy()
    n_param = sum(p.numel() for p in model.parameters())
    config = {
        "model_names": list(model_names or [f"m{i}" for i in range(n_models)]),
        "targets": names,
        "slots": list(SLOTS),
        "use_pred": model.use_pred,
        "use_empty": model.use_empty,
        "use_image": use_image,
        "pred_mode": model.pred_mode,
        "empty_mode": model.empty_mode,
        "blend": blend,
        "n_models": n_models,
        "n_targets": n_targets,
        "n_empty": int(empty.shape[1]),
        "gate_dim": int(gates_np.shape[1]),
        "series": "SAG_FS",
        "pca_dim": int(pca["components"].shape[0]) if pca else 0,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "seed": seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "residual_decay": decay_r,
        "n_param": n_param,
    }
    metrics = {
        "train": float(hist["train"][best_epoch]),
        "val": float(hist["val"][best_epoch]),
        "gold": float(hist["gold"][best_epoch]),
        "train_last": float(hist["train"][-1]),
        "val_last": float(hist["val"][-1]),
        "gold_last": float(hist["gold"][-1]),
    }
    return {
        "state": {k: v.numpy() for k, v in best_state.items()},
        "config": config,
        "pca": pca,
        "metrics": metrics,
        "history": hist,
        "mean_weight_fit": weight_np[fit_mask].mean(axis=0),
        "mean_weight_val": weight_np[val_mask].mean(axis=0),
        "mixed": mixed_np,
    }


def _model_from_bundle(bundle: dict) -> MixGate:
    cfg = bundle["config"]
    model = MixGate(
        cfg["n_models"], cfg["n_targets"], cfg["n_empty"], cfg["gate_dim"],
        cfg["use_pred"], cfg["use_empty"], cfg["use_image"],
        pred_mode=cfg.get("pred_mode", "full" if cfg["use_pred"] else "none"),
        empty_mode=cfg.get("empty_mode", "full" if cfg["use_empty"] else "none"),
        blend=cfg.get("blend", "prob"),
    )
    state = {k: torch.tensor(v) for k, v in bundle["state"].items()}
    model.load_state_dict(state)
    model.eval()
    return model


def apply_mix(
    bundle: dict,
    probs: np.ndarray,
    empty: np.ndarray,
    embed: np.ndarray | None = None,
    present: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Poids (N, T, M) et probabilités mélangées (N, T)."""
    cfg = bundle["config"]
    probs = np.clip(np.asarray(probs, np.float32), 0.0, 1.0)
    n = probs.shape[0]
    if cfg["use_image"]:
        if embed is None or present is None or bundle["pca"] is None:
            raise ValueError("ce gate a une branche série")
        gates = project_embed(embed, np.asarray(present, bool), bundle["pca"])
    else:
        gates = np.zeros((n, cfg["gate_dim"]), np.float32)
    model = _model_from_bundle(bundle)
    with torch.no_grad():
        weight, mixed = model(
            torch.tensor(np.transpose(probs, (0, 2, 1))),
            torch.tensor(np.transpose(_logits(probs), (0, 2, 1))),
            torch.tensor(np.asarray(empty, np.float32)),
            torch.tensor(gates),
        )
    return weight.numpy(), mixed.numpy()


def save_bundle(path: str | Path, bundle: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "config": np.array(json.dumps(bundle["config"])),
        "metrics": np.array(json.dumps(bundle["metrics"])),
    }
    for key, value in bundle["state"].items():
        payload[f"state__{key}"] = np.asarray(value)
    if bundle["pca"] is not None:
        for key, value in bundle["pca"].items():
            payload[f"pca__{key}"] = np.asarray(value)
    payload["mean_weight_fit"] = np.asarray(bundle["mean_weight_fit"], np.float32)
    payload["mean_weight_val"] = np.asarray(bundle["mean_weight_val"], np.float32)
    payload["history_train"] = np.asarray(bundle["history"]["train"], np.float64)
    payload["history_val"] = np.asarray(bundle["history"]["val"], np.float64)
    payload["history_gold"] = np.asarray(bundle["history"]["gold"], np.float64)
    np.savez_compressed(path, **payload)


def load_bundle(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    state = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("state__")}
    pca = None
    pca_keys = [k for k in z.files if k.startswith("pca__")]
    if pca_keys:
        pca = {k.split("__", 1)[1]: z[k] for k in pca_keys}
    return {
        "state": state,
        "config": json.loads(str(z["config"])),
        "pca": pca,
        "metrics": json.loads(str(z["metrics"])),
        "history": {
            "train": z["history_train"].tolist(),
            "val": z["history_val"].tolist(),
            "gold": z["history_gold"].tolist(),
        },
        "mean_weight_fit": z["mean_weight_fit"],
        "mean_weight_val": z["mean_weight_val"],
    }


def load_frame(
    root: str | Path,
    models: list[tuple[str, str]],
    distil_uids: str | Path | None = None,
) -> dict:
    """CSV hybrid alignés sur `slot_empty_unseen_rsna.csv`.

    Si un nom de fichier contient « distil », les études de `distil_uids`
    sortent du fit. Labels v3b, gold = train.csv, binarisés à 0,5.
    """
    root = Path(root)
    pred = root / "data" / "hybrid_pred"
    slots = pd.read_csv(pred / "slot_empty_unseen_rsna.csv")
    uids = slots.StudyInstanceUID.astype(str).tolist()
    frames = []
    for name, fn in models:
        df = pd.read_csv(pred / fn)
        got = df.StudyInstanceUID.astype(str).tolist()
        if got != uids:
            raise SystemExit(f"{name} : ordre des UID différent de slot_empty")
        cols = [f"proba_{t}" for t in TARGETS]
        if df[cols].isna().any().any():
            raise SystemExit(f"{name} : proba manquante")
        frames.append(df)
    v3 = pd.read_csv(root / "data/labels/coolin/targets_v3b.csv")
    v3["StudyInstanceUID"] = v3.StudyInstanceUID.astype(str)
    lab = v3.set_index("StudyInstanceUID")[list(TARGETS)]
    gold_df = pd.read_csv(
        root / "data/csv/train.csv",
        usecols=["StudyInstanceUID"] + list(TARGETS),
    )
    gold_df["StudyInstanceUID"] = gold_df.StudyInstanceUID.astype(str)
    gold_lab = (
        gold_df.dropna(subset=list(TARGETS))
        .set_index("StudyInstanceUID")[list(TARGETS)]
    )
    rows, labeled = [], []
    for uid, sp in zip(uids, slots["split"].astype(str)):
        src = gold_lab if sp == "gold" else lab
        if uid not in src.index or src.loc[uid, list(TARGETS)].isna().any():
            labeled.append(False)
            rows.append(np.full(len(TARGETS), np.nan))
        else:
            labeled.append(True)
            rows.append(src.loc[uid, list(TARGETS)].to_numpy(np.float64))
    y_soft = np.vstack(rows)
    labeled_a = np.asarray(labeled)
    y = np.where(labeled_a[:, None], (y_soft >= 0.5).astype(np.float32), -1.0)
    split = slots["split"].astype(str).to_numpy()
    seen: set[str] = set()
    if distil_uids is not None and Path(distil_uids).is_file():
        seen = {
            ln.strip()
            for ln in Path(distil_uids).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
    blocked = (
        np.isin(np.asarray(uids, dtype=object), np.array(list(seen), dtype=object))
        if seen else np.zeros(len(uids), dtype=bool)
    )
    use_distil = any("distil" in fn for _, fn in models)
    fit_base = (split == "fit") & labeled_a
    fit_mask = fit_base & ~blocked if use_distil else fit_base
    probs = np.stack(
        [df[[f"proba_{t}" for t in TARGETS]].to_numpy(np.float64) for df in frames],
        axis=1,
    ).astype(np.float32)
    return {
        "uids": uids,
        "probs": probs,
        "empty": slots[list(SLOTS)].to_numpy(np.float32),
        "y": y.astype(np.float32),
        "split": split,
        "names": [name for name, _ in models],
        "fit_mask": fit_mask,
        "val_mask": (split == "val") & labeled_a,
        "gold_mask": (split == "gold") & labeled_a,
        "fit_blocked": int((fit_base & blocked).sum()) if use_distil else 0,
        "use_distil": use_distil,
    }


def split_auc(y: np.ndarray, score: np.ndarray, frame: dict) -> dict[str, float]:
    names = list(TARGETS)
    return {
        "fit": _auc(y, score, frame["fit_mask"], names),
        "val": _auc(y, score, frame["val_mask"], names),
        "gold": _auc(y, score, frame["gold_mask"], names),
    }


def self_check() -> None:
    """Forme, somme des poids, epoch 0 uniforme, série absente sans effet."""
    rng = np.random.default_rng(0)
    n, m, t = 32, 3, len(TARGETS)
    probs = rng.uniform(0.05, 0.95, size=(n, m, t)).astype(np.float32)
    y = np.full((n, t), -1.0, np.float32)
    y[:24] = rng.integers(0, 2, size=(24, t)).astype(np.float32)
    # chaque cible a les deux classes dans le fit
    y[:24, :] = 0
    y[:12, :] = 1
    empty = np.zeros((n, len(SLOTS)), np.float32)
    empty[::3, 0] = 1
    fit = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    gold = np.zeros(n, dtype=bool)
    fit[:24] = True
    val[24:28] = True
    gold[28:] = True
    y[24:] = rng.integers(0, 2, size=(8, t)).astype(np.float32)
    bias_only = fit_mix_gate(
        probs, y, fit, val, gold, empty,
        model_names=["a", "b", "c"],
        use_pred=False, use_empty=False, use_image=False,
        epochs=0,
    )
    w_bias, mix_bias = apply_mix(bias_only, probs, empty)
    if w_bias.shape != (n, t, m):
        raise SystemExit(f"biais seul : forme {w_bias.shape}")
    if not np.allclose(w_bias, 1.0 / m, atol=1e-5):
        raise SystemExit("biais nul : mélange non uniforme")
    if not np.allclose(mix_bias, probs.mean(axis=1), atol=1e-5):
        raise SystemExit("biais nul : proba différente de la moyenne")
    bundle = fit_mix_gate(
        probs, y, fit, val, gold, empty,
        model_names=["a", "b", "c"], use_pred=True, use_empty=True, use_image=False,
        epochs=2, lr=0.05, weight_decay=0.0, seed=1,
    )
    logits = torch.zeros(4, t, m)
    pr = torch.full((4, t, m), 0.5)
    emp = torch.zeros(4, len(SLOTS))
    fresh = MixGate(m, t, len(SLOTS), 1, True, True, True)
    with torch.no_grad():
        w0, mix0 = fresh(pr, logits, emp, torch.zeros(4, 1))
    if not torch.allclose(w0.sum(-1), torch.ones(4, t), atol=1e-5):
        raise SystemExit("les poids ne somment pas à 1")
    if not torch.allclose(mix0, torch.full((4, t), 0.5), atol=1e-5):
        raise SystemExit("epoch 0 n'est pas le mélange uniforme")
    with torch.no_grad():
        fresh.img_w.zero_()
        fresh.img_w[:, :, 0] = 1.0
        w_off, _ = fresh(pr, logits, emp, torch.zeros(4, 1))
        w_on, _ = fresh(pr, logits, emp, torch.ones(4, 1))
    if torch.allclose(w_off, w_on):
        raise SystemExit("la série ne change pas le mélange")
    if not torch.allclose(w_off, w0):
        raise SystemExit("une série nulle doit laisser le mélange inchangé")
    w, mixed = apply_mix(bundle, probs, empty)
    if w.shape != (n, t, m) or mixed.shape != (n, t):
        raise SystemExit(f"forme apply {w.shape} {mixed.shape}")
    if not np.allclose(w.sum(-1), 1.0, atol=1e-5):
        raise SystemExit("apply : poids hors simplexe")
    # série absente : même sortie qu'un embed nul, present à 0
    embed = rng.normal(size=(n, 8)).astype(np.float32)
    present = np.ones(n, dtype=bool)
    present[:4] = False
    img = fit_mix_gate(
        probs, y, fit, val, gold, empty, embed, present,
        use_pred=False, use_empty=False, use_image=True,
        epochs=1, pca_dim=4, weight_decay=0.0,
    )
    w_a, p_a = apply_mix(img, probs, empty, embed, present)
    embed2 = embed.copy()
    embed2[:4] = 99.0
    w_b, p_b = apply_mix(img, probs, empty, embed2, present)
    if not np.allclose(p_a[:4], p_b[:4], atol=1e-6):
        raise SystemExit("une série absente a encore un effet")
    print("self_check ok", flush=True)
