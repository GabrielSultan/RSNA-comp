# -*- coding: utf-8 -*-
"""Interpretabilite v14 : pooling spatial 7x7 (CAM appris).

Le softmax coupe (attn_score) dit quelle image ; spatial_score dit ou
dans la coupe. Uniforme 1/49 = encore un GAP (v11). Un pic = la tete
a appris a localiser.

Usage :
  python -X utf8 -m src.infer.inspect_spatial --config configs/kneenet_v14_crop130_subset.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.infer.inspect_attention import (
    build_loader, load_ckpt_model, load_config, plane_names,
)
from src.models.kneenet_v14 import MAP_HW

UNIFORM_N = MAP_HW * MAP_HW  # 49
UNIFORM_ENTROPY = float(np.log(UNIFORM_N))


def upsample_cam(cam: np.ndarray, size: int = 224) -> np.ndarray:
    """cam (7, 7) -> (size, size), bilinear."""
    t = torch.from_numpy(np.asarray(cam, dtype=np.float32))[None, None]
    out = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def entropy_hw(p: np.ndarray) -> np.ndarray:
    """p (..., 7, 7) -> entropie nats (...,). Uniforme = log(49) ~ 3.89."""
    flat = np.asarray(p, dtype=np.float64).reshape(*p.shape[:-2], -1)
    flat = flat / np.maximum(flat.sum(axis=-1, keepdims=True), 1e-12)
    return -(flat * np.log(np.clip(flat, 1e-12, 1.0))).sum(axis=-1)


def collect_spatial(model, loader, device: str, targets: list[str],
                    n_examples: int = 6, top_pos: int = 4):
    """Agrege CAM pondere par l'attention-coupe. Garde des etudes pour overlays."""
    if not hasattr(model, "interpret"):
        raise SystemExit("ce modele n'a pas interpret() (v14 seulement)")
    n_series = int(model.n_series)
    n_targets = int(model.n_targets)
    hw = MAP_HW
    # moyenne ponderee des cartes 7x7
    cam_sum = np.zeros((n_series, n_targets, hw, hw), dtype=np.float64)
    cam_w = np.zeros((n_series, n_targets), dtype=np.float64)
    ent_sum = np.zeros(n_targets, dtype=np.float64)
    maxp_sum = np.zeros(n_targets, dtype=np.float64)
    w_tgt = np.zeros(n_targets, dtype=np.float64)
    n_studies = 0
    examples: list[dict] = []
    pos_examples: dict[str, list] = {t: [] for t in targets}

    model.eval()
    for x, y, uids in loader:
        x = x.to(device, non_blocking=True)
        out = model.interpret(x)
        logits = out["logits"].float().cpu()
        slice_attn = out["slice_attn"].float().cpu().numpy()
        spatial = out["spatial_attn"].float().cpu().numpy()
        valid = out["valid"].cpu().numpy()
        probs = torch.sigmoid(logits).numpy()
        y_np = y.numpy()
        b = x.size(0)

        for bi in range(b):
            w = slice_attn[bi] * valid[bi][..., None]  # S,D,T
            for t in range(n_targets):
                wt = w[:, :, t]
                tw = float(wt.sum())
                if tw <= 0:
                    continue
                cam_w[:, t] += wt.sum(axis=1)
                for p in range(n_series):
                    cam_sum[p, t] += np.tensordot(
                        wt[p], spatial[bi, p, :, t], axes=(0, 0),
                    )
                ent = entropy_hw(spatial[bi, :, :, t])  # S,D
                mx = spatial[bi, :, :, t].reshape(n_series, spatial.shape[2], -1).max(-1)
                ent_sum[t] += float((wt * ent).sum())
                maxp_sum[t] += float((wt * mx).sum())
                w_tgt[t] += tw
            n_studies += 1

            rec = {
                "uid": uids[bi] if isinstance(uids[bi], str) else str(uids[bi]),
                "x": x[bi].cpu().numpy(),
                "slice_attn": slice_attn[bi],
                "spatial": spatial[bi],
                "valid": valid[bi],
                "prob": probs[bi],
                "y": y_np[bi],
            }
            if len(examples) < n_examples:
                examples.append(rec)
            for ti, name in enumerate(targets):
                if y_np[bi, ti] == 1.0 and len(pos_examples[name]) < top_pos:
                    pos_examples[name].append(rec)

        print(f"  {n_studies} etudes", flush=True)

    mean_cam = cam_sum / np.maximum(cam_w[:, :, None, None], 1e-12)
    mean_ent = ent_sum / np.maximum(w_tgt, 1e-12)
    mean_maxp = maxp_sum / np.maximum(w_tgt, 1e-12)
    return {
        "n_studies": int(n_studies),
        "targets": targets,
        "planes": plane_names(n_series),
        "mean_cam": mean_cam,
        "mean_entropy": mean_ent,
        "mean_maxp": mean_maxp,
        "neff": np.exp(mean_ent),
        "examples": examples,
        "pos_examples": pos_examples,
    }


def peakiness_table(agg: dict) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(agg["targets"]):
        neff = float(agg["neff"][i])
        rows.append({
            "target": name,
            "entropy": float(agg["mean_entropy"][i]),
            "neff": neff,
            "max_p": float(agg["mean_maxp"][i]),
            "uniforme": UNIFORM_N,
            "lecture": (
                "GAP (pas pique)" if neff > 35 else
                "un peu pique" if neff > 20 else
                "pique"
            ),
        })
    return pd.DataFrame(rows)


def focus_cells(cam7: np.ndarray, keep_mass: float = 0.50,
                min_ratio: float = 2.0, max_cells: int = 6):
    """Cases 7x7 ou le softmax se concentre.

    Garde les cases >= `min_ratio` x uniforme, jusqu'a `keep_mass` de masse
    (et au moins l'argmax). Si tout est plat, liste vide.
    """
    cam = np.asarray(cam7, dtype=np.float64)
    cam = cam / max(float(cam.sum()), 1e-12)
    h, w = cam.shape
    uniform = 1.0 / (h * w)
    peak = float(cam.max())
    if peak < min_ratio * uniform:
        return []
    thresh = min_ratio * uniform
    order = np.argsort(-cam.ravel())
    cells = []
    cum = 0.0
    for idx in order:
        r, c = divmod(int(idx), w)
        p = float(cam[r, c])
        if cells and (p < thresh and cum >= keep_mass):
            break
        cells.append((r, c, p))
        cum += p
        if len(cells) >= max_cells:
            break
        if cum >= keep_mass and p < thresh:
            break
    return cells


def draw_focus_squares(ax, cam7: np.ndarray, img_hw: int | None = None,
                       keep_mass: float = 0.50) -> int:
    """Carres rouges sur l'image (imshow deja fait). Retourne le nb de cases."""
    from matplotlib.patches import Rectangle
    cam = np.asarray(cam7, dtype=np.float64)
    gh, gw = cam.shape
    if img_hw is None:
        arr = ax.get_images()[0].get_array()
        img_hw = int(arr.shape[0])
    cell_h = img_hw / gh
    cell_w = img_hw / gw
    cells = focus_cells(cam, keep_mass=keep_mass)
    if not cells:
        return 0
    pmax = max(p for _r, _c, p in cells)
    for r, c, p in cells:
        lw = 1.6 + 3.5 * (p / pmax)
        ax.add_patch(Rectangle(
            (c * cell_w - 0.5, r * cell_h - 0.5),
            cell_w, cell_h,
            fill=False, edgecolor="#e74c3c", linewidth=lw, zorder=5,
        ))
    return len(cells)


def plot_mean_cams(agg: dict, out: Path | None = None):
    import matplotlib.pyplot as plt
    planes, targets = agg["planes"], agg["targets"]
    mean_cam = agg["mean_cam"]
    n_p, n_t = len(planes), len(targets)
    fig, axes = plt.subplots(n_t, n_p, figsize=(n_p * 2.2, n_t * 1.7))
    axes = np.asarray(axes).reshape(n_t, n_p)
    for t, name in enumerate(targets):
        vmax = float(mean_cam[:, t].max()) + 1e-12
        for p, pname in enumerate(planes):
            ax = axes[t, p]
            ax.imshow(mean_cam[p, t], cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            draw_focus_squares(ax, mean_cam[p, t], img_hw=7, keep_mass=0.35)
            if t == 0:
                ax.set_title(pname, fontsize=9)
            if p == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.suptitle("CAM moyen pondere par l'attention-coupe (7x7, carre = case max)",
                 fontsize=11)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=130)
        plt.close(fig)
    return fig


def plot_top_slices(ex: dict, target: str, targets: list[str],
                    planes: list[str], k: int = 6, out: Path | None = None):
    import matplotlib.pyplot as plt
    ti = targets.index(target)
    attn = ex["slice_attn"][:, :, ti] * ex["valid"]
    flat = attn.ravel()
    top = np.argpartition(-flat, min(k, flat.size) - 1)[:k]
    top = top[np.argsort(-flat[top])]
    n = len(top)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.35, 2.85))
    axes = np.asarray(axes).reshape(n)
    yv = ex["y"][ti]
    lab = "pos" if yv == 1 else ("neg" if yv == 0 else "?")
    uid = ex["uid"][:18]
    for j, tok in enumerate(top):
        p, d = divmod(int(tok), attn.shape[1])
        gray = np.clip(ex["x"][p, d], 0.0, 1.0)
        cam = ex["spatial"][p, d, ti]
        ax = axes[j]
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
        nsq = draw_focus_squares(ax, cam)
        ax.set_xticks([])
        ax.set_yticks([])
        neff = float(np.exp(entropy_hw(cam)))
        note = f"{nsq} case(s)" if nsq else "diffus"
        ax.set_title(
            f"{planes[p][:3]} {d}  {100*flat[tok]:.1f}%\n{note} Neff={neff:.0f}",
            fontsize=8,
        )
    fig.suptitle(
        f"{uid}…  {target}  p={ex['prob'][ti]:.2f}  y={lab}  "
        f"(carre rouge = case 7x7 piquee)",
        fontsize=10,
    )
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=120)
        plt.close(fig)
    return fig


def plot_same_slice(ex: dict, targets: list[str], planes: list[str],
                    names: tuple[str, ...] = ("ACL", "Baker's", "Fracture"),
                    out: Path | None = None):
    """Une coupe (max attn 1re tete) + carres rouges par tete spatiale."""
    import matplotlib.pyplot as plt
    t0 = targets.index(names[0]) if names[0] in targets else 0
    attn = ex["slice_attn"][:, :, t0] * ex["valid"]
    p, d = np.unravel_index(int(attn.argmax()), attn.shape)
    gray = np.clip(ex["x"][p, d], 0.0, 1.0)
    cols = [n for n in names if n in targets]
    fig, axes = plt.subplots(1, 1 + len(cols), figsize=(3.2 * (1 + len(cols)), 3.2))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"{planes[p]} coupe {d}", fontsize=9)
    axes[0].axis("off")
    for ax, name in zip(axes[1:], cols):
        ti = targets.index(name)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
        nsq = draw_focus_squares(ax, ex["spatial"][p, d, ti])
        note = f"{nsq} case(s)" if nsq else "diffus"
        ax.set_title(
            f"{name}  {note}\ncoupe {100*ex['slice_attn'][p, d, ti]:.1f}%  "
            f"p={ex['prob'][ti]:.2f}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(
        f"{ex['uid'][:20]}…  meme coupe, carres = focus spatial",
        fontsize=10,
    )
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=120)
        plt.close(fig)
    return fig


def run(cfg: dict, ckpt_path: Path, split: str, out_dir: Path,
        device: str, batch_size: int, num_workers: int,
        max_studies: int | None, n_examples: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ckpt {ckpt_path}", flush=True)
    model, epoch, val_auc = load_ckpt_model(cfg, ckpt_path, device)
    if not hasattr(model, "interpret"):
        raise SystemExit(f"{cfg.get('model')} n'expose pas interpret() — v14 requis")
    print(f"{cfg.get('model')} epoch {epoch} val AUC {val_auc} | split {split}",
          flush=True)
    loader, targets = build_loader(
        cfg, split, batch_size, num_workers, max_studies,
    )
    print(f"{len(loader.dataset)} etudes, batch {batch_size}", flush=True)
    agg = collect_spatial(model, loader, device, targets, n_examples)
    table = peakiness_table(agg)
    summary = {
        "run_name": cfg.get("run_name"),
        "model": cfg.get("model"),
        "ckpt": str(ckpt_path),
        "epoch": epoch,
        "val_macro_auc": None if val_auc is None else float(val_auc),
        "split": split,
        "n_studies": agg["n_studies"],
        "uniform_entropy": UNIFORM_ENTROPY,
        "uniform_n": UNIFORM_N,
        "targets": targets,
        "mean_entropy": agg["mean_entropy"].tolist(),
        "neff": agg["neff"].tolist(),
        "mean_maxp": agg["mean_maxp"].tolist(),
    }
    table.to_csv(out_dir / "spatial_peakiness.csv", index=False)
    np.savez_compressed(out_dir / "spatial_agg.npz", mean_cam=agg["mean_cam"])
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    plot_mean_cams(agg, out_dir / "mean_cam.png")
    print("\n=== pic spatial (pondere par attention-coupe) ===")
    print(table.round(3).to_string(index=False))
    print(f"\nuniforme 7x7 : entropy={UNIFORM_ENTROPY:.2f}  Neff={UNIFORM_N}")
    print(f"ecrit -> {out_dir}")
    return {"summary": summary, "agg": agg, "table": table}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/kneenet_v14_crop130_subset.yaml",
    )
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="gold",
                        choices=("val", "gold", "train"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-studies", type=int, default=None)
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("model", "kneenet_v14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt) if args.ckpt else (
        Path("checkpoints") / cfg["run_name"] / "best.pt"
    )
    if not ckpt.exists():
        raise SystemExit(f"checkpoint introuvable : {ckpt}")
    out = Path(args.out) if args.out else (
        Path("checkpoints") / cfg["run_name"] / "spatial_inspect"
    )
    bs = int(args.batch_size or cfg.get("batch_size") or 4)
    run(
        cfg, ckpt, args.split, out, device, bs, args.workers,
        args.max_studies, args.n_examples,
    )


if __name__ == "__main__":
    main()
