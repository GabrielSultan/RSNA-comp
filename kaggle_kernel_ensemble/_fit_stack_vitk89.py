# -*- coding: utf-8 -*-
"""Stack des 4 SWA, meme protocole que les notebooks 13 et 15.

Un seul ensemble : ViT-k320, v20b, v20h crop145, CoAt-a.
Ajuste sur les etudes fit des CSV hybrid de ces subsets. La val ne choisit
pas l'epoch. Le checkpoint retenu est la meilleure AUC train.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp.lexicon import TARGETS  # noqa: E402
from src.train.metrics import macro_auc  # noqa: E402

PRED = ROOT / "data" / "hybrid_pred"
OUT = ROOT / "kaggle_kernel_ensemble" / "_stack_vitk89"
MODELS = (
    ("ViT-k320", "hybrid_preds_unseen_rsna_vitkllrd08.csv"),
    ("v20b", "hybrid_preds_unseen_rsna__subset_v20b_dino_ep11_v3b_hybrid025.csv"),
    ("v20h", "hybrid_preds_unseen_rsna__subset_v20h_dino_ep11_v3b_hybrid025_wd01_crop145.csv"),
    ("CoAt-a", "hybrid_preds_unseen_rsna__subset_coat_a_llrd08_v3b_hybrid025.csv"),
)
SLOTS = (
    "empty_SAG_FS", "empty_COR_FS", "empty_AX_FS",
    "empty_SAG_T1", "empty_COR_T1", "empty_AX_T1",
)
EPOCHS = 120
SEED = 42
CLIP = 1e-4
HIDDEN = 8


def _logits(df) -> np.ndarray:
    p = np.clip(df[[f"proba_{t}" for t in TARGETS]].to_numpy(np.float64), CLIP, 1.0 - CLIP)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _design(z: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """z (n, m, 12), empty (n, 6) -> x (12, n, d). Copie du notebook 13."""
    m = z.shape[1]
    s = empty.shape[1]
    x = np.empty((len(TARGETS), z.shape[0], m + s + m * s), np.float32)
    x[:, :, :m] = np.transpose(z, (2, 0, 1))
    x[:, :, m:m + s] = empty[None]
    inter = z[:, :, None, :] * empty[:, None, :, None]
    x[:, :, m + s:] = np.transpose(inter, (3, 0, 1, 2)).reshape(len(TARGETS), z.shape[0], m * s)
    return x


def design_one(logit_m12: np.ndarray, empty6: np.ndarray) -> np.ndarray:
    """Une etude. logit (m, 12), empty (6,) -> (12, d). Meme formule."""
    return _design(np.asarray(logit_m12, np.float32)[None], np.asarray(empty6, np.float32)[None])[:, 0, :]


def _macro_auc_rank(y: np.ndarray, s: np.ndarray) -> float:
    vals = []
    for c in range(y.shape[1]):
        yt = y[:, c]
        ys = s[:, c]
        n_pos = float(yt.sum())
        n = yt.size
        n_neg = n - n_pos
        if n_pos == 0.0 or n_neg == 0.0:
            continue
        order = np.argsort(ys, kind="mergesort")
        ys_s = ys[order]
        obs = np.empty(n, dtype=bool)
        obs[0] = True
        obs[1:] = ys_s[1:] != ys_s[:-1]
        dense = np.cumsum(obs) - 1
        counts = np.bincount(dense)
        starts = np.cumsum(counts) - counts
        avg = (starts + 1 + (starts + counts)) / 2.0
        ranks = np.empty(n, np.float64)
        ranks[order] = avg[dense]
        vals.append((ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return float(np.mean(vals)) if vals else float("nan")


class _TargetMLP(nn.Module):
    def __init__(self, d: int, n_targets: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Parameter(torch.empty(n_targets, d, hidden))
        self.b1 = nn.Parameter(torch.zeros(n_targets, hidden))
        self.fc2 = nn.Parameter(torch.zeros(n_targets, hidden))
        self.b2 = nn.Parameter(torch.zeros(n_targets))
        bound = d ** -0.5
        nn.init.uniform_(self.fc1, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("tnd,tdh->tnh", x, self.fc1) + self.b1[:, None, :]
        h = torch.nn.functional.gelu(h)
        return torch.einsum("tnh,th->tn", h, self.fc2) + self.b2[:, None]


def _load():
    import pandas as pd

    slots = pd.read_csv(PRED / "slot_empty_unseen_rsna.csv")
    uids = slots.StudyInstanceUID.astype(str).tolist()
    frames = []
    for name, fn in MODELS:
        df = pd.read_csv(PRED / fn)
        got = df.StudyInstanceUID.astype(str).tolist()
        if got != uids:
            raise SystemExit(f"{name} : ordre des UID different")
        frames.append(df)
    v3 = pd.read_csv(ROOT / "data/labels/coolin/targets_v3b.csv")
    v3["StudyInstanceUID"] = v3.StudyInstanceUID.astype(str)
    lab = v3.set_index("StudyInstanceUID")[list(TARGETS)]
    gold_df = pd.read_csv(ROOT / "data/csv/train.csv", usecols=["StudyInstanceUID"] + list(TARGETS))
    gold_df["StudyInstanceUID"] = gold_df.StudyInstanceUID.astype(str)
    gold_lab = gold_df.dropna(subset=list(TARGETS)).set_index("StudyInstanceUID")[list(TARGETS)]
    rows, labeled = [], []
    for uid, sp in zip(uids, slots["split"].astype(str)):
        src = gold_lab if sp == "gold" else lab
        if uid not in src.index or src.loc[uid, list(TARGETS)].isna().any():
            labeled.append(False)
            rows.append(np.full(len(TARGETS), np.nan))
        else:
            labeled.append(True)
            rows.append(src.loc[uid, list(TARGETS)].to_numpy(np.float64))
    y_hard = (np.vstack(rows) >= 0.5).astype(np.float64)
    labeled = np.asarray(labeled)
    split = slots["split"].astype(str).to_numpy()
    fit_m = (split == "fit") & labeled
    val_m = (split == "val") & labeled
    gold_m = (split == "gold") & labeled
    model_logits = [_logits(df) for df in frames]
    z = np.stack(model_logits, axis=1)
    empty = slots[list(SLOTS)].to_numpy(np.float32)
    for i in range(len(uids)):
        got = design_one(z[i], empty[i])
        ref = _design(z[i:i + 1], empty[i:i + 1])[:, 0, :]
        if not np.array_equal(got, ref):
            raise SystemExit(f"design_one diverge etude {i}")
    if i < 100:
        raise SystemExit("trop peu d'etudes pour le controle")
    print(
        f"etudes {len(uids)} | fit {int(fit_m.sum())} | val {int(val_m.sum())} | gold {int(gold_m.sum())}",
        flush=True,
    )
    return z, empty, y_hard, fit_m, val_m, gold_m


def _standardize(raw: np.ndarray, fit_m: np.ndarray):
    mu = raw[:, fit_m].mean(1, keepdims=True)
    sd = raw[:, fit_m].std(1, keepdims=True)
    keep = sd > 1e-4
    sd = np.where(keep, sd, 1.0)
    x = (raw - mu) / sd * keep
    return x, mu, sd, keep


def _auc(y: np.ndarray, prob: np.ndarray) -> float:
    return float(macro_auc(y, prob, list(TARGETS))[0])


def fit_logreg(x, y, fit_m, val_m, gold_m) -> dict:
    torch.manual_seed(SEED)
    xt = torch.tensor(x[:, fit_m])
    yt = torch.tensor(y[fit_m].T, dtype=torch.float32)
    xv = torch.tensor(x[:, val_m])
    yv = torch.tensor(y[val_m].T, dtype=torch.float32)
    weight = torch.zeros(len(TARGETS), xt.shape[-1], requires_grad=True)
    bias = torch.zeros(len(TARGETS), requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=0.02, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    y_fit = y[fit_m]
    best_auc = -1.0
    best_step = 0
    best_w = weight.detach().clone()
    best_b = bias.detach().clone()
    for step in range(EPOCHS + 1):
        with torch.no_grad():
            tr = torch.einsum("tnd,td->tn", xt, weight) + bias[:, None]
            ta = _macro_auc_rank(y_fit, torch.sigmoid(tr).numpy().T)
            if ta > best_auc:
                best_auc = ta
                best_step = step
                best_w = weight.detach().clone()
                best_b = bias.detach().clone()
        if step == EPOCHS:
            break
        opt.zero_grad()
        loss = bce(torch.einsum("tnd,td->tn", xt, weight) + bias[:, None], yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr = torch.sigmoid(torch.einsum("tnd,td->tn", xt, best_w) + best_b[:, None]).numpy().T
        va = torch.sigmoid(torch.einsum("tnd,td->tn", xv, best_w) + best_b[:, None]).numpy().T
        xg = torch.tensor(x[:, gold_m])
        gold = torch.sigmoid(torch.einsum("tnd,td->tn", xg, best_w) + best_b[:, None]).numpy().T
    out = {
        "kind": "logreg",
        "epoch": best_step,
        "weight": best_w.numpy(),
        "bias": best_b.numpy(),
        "auc_train": _auc(y[fit_m], tr),
        "auc_val": _auc(y[val_m], va),
        "auc_gold": _auc(y[gold_m], gold),
    }
    print(
        f"logreg epoch {best_step} train {out['auc_train']:.4f} "
        f"val {out['auc_val']:.4f} gold {out['auc_gold']:.4f}",
        flush=True,
    )
    return out


def fit_mlp(x, y, fit_m, val_m, gold_m) -> dict:
    torch.manual_seed(SEED)
    xt = torch.tensor(x[:, fit_m])
    yt = torch.tensor(y[fit_m].T, dtype=torch.float32)
    xv = torch.tensor(x[:, val_m])
    yv = torch.tensor(y[val_m].T, dtype=torch.float32)
    net = _TargetMLP(xt.shape[-1], len(TARGETS), HIDDEN)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    y_fit = y[fit_m]
    best_auc = -1.0
    best_step = 0
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for step in range(EPOCHS + 1):
        with torch.no_grad():
            ta = _macro_auc_rank(y_fit, torch.sigmoid(net(xt)).numpy().T)
            if ta > best_auc:
                best_auc = ta
                best_step = step
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if step == EPOCHS:
            break
        opt.zero_grad()
        loss = bce(net(xt), yt)
        loss.backward()
        opt.step()
    net.load_state_dict(best_state)
    with torch.no_grad():
        tr = torch.sigmoid(net(xt)).numpy().T
        va = torch.sigmoid(net(xv)).numpy().T
        gold = torch.sigmoid(net(torch.tensor(x[:, gold_m]))).numpy().T
    out = {
        "kind": "mlp",
        "epoch": best_step,
        "fc1": best_state["fc1"].numpy(),
        "b1": best_state["b1"].numpy(),
        "fc2": best_state["fc2"].numpy(),
        "b2": best_state["b2"].numpy(),
        "auc_train": _auc(y[fit_m], tr),
        "auc_val": _auc(y[val_m], va),
        "auc_gold": _auc(y[gold_m], gold),
    }
    print(
        f"mlp    epoch {best_step} train {out['auc_train']:.4f} "
        f"val {out['auc_val']:.4f} gold {out['auc_gold']:.4f}",
        flush=True,
    )
    return out


def _check_one(kind: dict, x: np.ndarray, z: np.ndarray, empty: np.ndarray, mu, sd, keep) -> None:
    i = 0
    raw = design_one(z[i], empty[i])
    xs = (raw - mu[:, 0, :]) / sd[:, 0, :] * keep[:, 0, :]
    if not np.allclose(xs, x[:, i, :], atol=1e-6):
        raise SystemExit("standardisation une etude diverge")
    t = torch.from_numpy(xs)
    with torch.no_grad():
        if kind["kind"] == "logreg":
            w = torch.from_numpy(kind["weight"])
            b = torch.from_numpy(kind["bias"])
            logit = torch.einsum("td,td->t", t, w) + b
            batched = torch.einsum(
                "tnd,td->tn", torch.from_numpy(x[:, i:i + 1]), w,
            ) + b[:, None]
        else:
            fc1 = torch.from_numpy(kind["fc1"])
            b1 = torch.from_numpy(kind["b1"])
            fc2 = torch.from_numpy(kind["fc2"])
            b2 = torch.from_numpy(kind["b2"])
            h = torch.nn.functional.gelu(torch.einsum("td,tdh->th", t, fc1) + b1)
            logit = torch.einsum("th,th->t", h, fc2) + b2
            xb = torch.from_numpy(x[:, i:i + 1])
            hb = torch.nn.functional.gelu(torch.einsum("tnd,tdh->tnh", xb, fc1) + b1[:, None, :])
            batched = torch.einsum("tnh,th->tn", hb, fc2) + b2[:, None]
    if not torch.allclose(logit, batched[:, 0], atol=1e-5):
        raise SystemExit(f"forward une etude diverge {kind['kind']}")


def main() -> None:
    z, empty, y, fit_m, val_m, gold_m = _load()
    raw = _design(z, empty)
    x, mu, sd, keep = _standardize(raw, fit_m)
    OUT.mkdir(parents=True, exist_ok=True)
    common = {
        "mu": mu.astype(np.float32),
        "sd": sd.astype(np.float32),
        "keep": keep.astype(np.float32),
        "order": np.array([name for name, _ in MODELS]),
        "slots": np.array(SLOTS),
        "targets": np.array(TARGETS),
    }
    for fit in (fit_logreg, fit_mlp):
        head = fit(x, y, fit_m, val_m, gold_m)
        _check_one(head, x, z, empty, mu, sd, keep)
        payload = {**common, **{k: v for k, v in head.items() if k != "kind"}}
        payload["kind"] = np.array(head["kind"])
        payload["epoch"] = np.array(head["epoch"])
        payload["auc_train"] = np.array(head["auc_train"])
        payload["auc_val"] = np.array(head["auc_val"])
        payload["auc_gold"] = np.array(head["auc_gold"])
        dest = OUT / f"{head['kind']}.npz"
        np.savez(dest, **payload)
        print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
