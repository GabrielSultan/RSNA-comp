# -*- coding: utf-8 -*-
"""Cartes d'attention KneeNet : quelles coupes le softmax ignore.

Ne modifie pas les archis figees. Hook sur `attn_score`, puis le meme
masque / softmax (et gate T1 / temperature v9) que le forward.

Les 24 indices sont le linspace dans la bande profondeur 20-80 %.
Une coupe « morte » = presque jamais dans le top-5 d'aucune tete, et
une masse d'attention tres inferieure a l'uniforme 1/D.

Usage :
  python -X utf8 -m src.infer.inspect_attention --config configs/kneenet_v14_crop130_subset.yaml --split gold
  python -X utf8 -m src.infer.inspect_attention --config configs/kneenet_v11_crop130.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset_npz import PLANES, make_loaders
from src.infer.preprocess_dicom import BAND, N_SLICES
from src.models.registry import build_model
from src.nlp.lexicon import TARGETS
from src.train.train import load_config, make_gold_loader

T1_SLOT = 3
TOPK = 5
KEEP_MASS = 0.90
# sous l'uniforme 1/D, et rarement choisie -> candidate au retrait
DEAD_MASS_RATIO = 0.35
DEAD_TOPK_PCT = 0.03


def plane_names(n_series: int) -> list[str]:
    names = list(PLANES)
    if n_series >= 4:
        names = names + ["T1"]
    return names[:n_series]


def slice_depth_pct(i: int, n: int = N_SLICES,
                    band: tuple[float, float] = BAND) -> float:
    """Percentile de profondeur dans le volume d'origine (bande linspace)."""
    lo, hi = band
    if n <= 1:
        return 50.0 * (lo + hi)
    return (lo + (hi - lo) * i / (n - 1)) * 100.0


def valid_mask(x: torch.Tensor) -> torch.Tensor:
    """Meme convention que les forwards KneeNet : (B, S, D)."""
    if x.ndim == 6:
        return x[:, :, :, 1].amax(dim=(-2, -1)) > 1e-6
    if x.ndim == 5:
        return x.amax(dim=(-2, -1)) > 1e-6
    raise ValueError(f"attendu 5D ou 6D, recu {tuple(x.shape)}")


def _to_gray_volume(x: torch.Tensor) -> torch.Tensor:
    """(S, D, H, W) pour l'affichage, canal G si RGB DICOM."""
    if x.ndim == 5:
        return x[:, :, 1]
    return x


def scores_to_attn(model, x: torch.Tensor,
                   scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """scores (B, S, D, T) -> attn (B, S, D, T), valid (B, S, D)."""
    valid = valid_mask(x)
    s = scores
    temp = float(getattr(model, "attn_temp", 1.0) or 1.0)
    if abs(temp - 1.0) > 1e-8:
        s = s / temp
    n_series = s.size(1)
    if hasattr(model, "t1_gate_logit") and n_series > T1_SLOT:
        log_g = F.logsigmoid(model.t1_gate_logit).to(dtype=s.dtype)
        s = s.clone()
        s[:, T1_SLOT] = s[:, T1_SLOT] + log_g
    s = s.masked_fill(~valid.unsqueeze(-1), -1.0e4)
    b, n_s, d, t = s.shape
    attn = torch.softmax(s.float().reshape(b, n_s * d, t), dim=1)
    return attn.reshape(b, n_s, d, t), valid


@torch.no_grad()
def cam_token_slice_attn(model, cam_tokens: torch.Tensor,
                         valid: torch.Tensor) -> torch.Tensor:
    """Softmax coupe si on scorait le token CAM (diag de attn_score).

    cam_tokens : (B, S, D, T, C) deja + plane/slice embed.
    Retour : (B, S, D, T). Contrefactuel v15, pas le forward v14.
    """
    b, s, d, t, c = cam_tokens.shape
    logits = model.attn_score(cam_tokens.reshape(b * s * d * t, c))
    logits = logits.reshape(b, s, d, t, t)
    scores = torch.diagonal(logits, dim1=3, dim2=4)
    scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e4)
    attn = torch.softmax(scores.float().reshape(b, s * d, t), dim=1)
    return attn.reshape(b, s, d, t).to(cam_tokens.dtype)


@torch.no_grad()
def forward_with_attn(model, x: torch.Tensor):
    """Forward eval + cartes d'attention. Hook, pas de patch des .py figees."""
    captured: dict[str, torch.Tensor] = {}

    def _hook(_mod, _inp, out):
        captured["scores"] = out

    handle = model.attn_score.register_forward_hook(_hook)
    try:
        logits = model(x)
    finally:
        handle.remove()
    if "scores" not in captured:
        raise RuntimeError("attn_score n'a pas ete appele (archi inattendue)")
    attn, valid = scores_to_attn(model, x, captured["scores"])
    return logits, attn, valid


def load_ckpt_model(cfg: dict, ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg_m = dict(cfg)
    cfg_ckpt = ckpt.get("config") or {}
    if "model" not in cfg_m and "model" in cfg_ckpt:
        cfg_m["model"] = cfg_ckpt["model"]
    model = build_model(cfg_m, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    epoch = int(ckpt.get("epoch", -1))
    val_auc = ckpt.get("val_macro_auc")
    return model, epoch, val_auc


def _limit_items(ds, n: int | None):
    if n is None or n >= len(ds.items):
        return ds
    ds.items = list(ds.items)[: int(n)]
    return ds


def build_loader(cfg: dict, split: str, batch_size: int, num_workers: int,
                 max_studies: int | None):
    n_series = int(cfg.get("n_series", 3))
    rgb = bool(cfg.get("rgb_neighbors", False))
    if split == "gold":
        cfg_g = dict(cfg)
        cfg_g["skip_gold"] = False
        gold_ld = make_gold_loader(cfg_g, batch_size, num_workers=0)
        if gold_ld is None:
            raise SystemExit("loader gold indisponible (train.csv / skip_gold)")
        ds = _limit_items(gold_ld.dataset, max_studies)
        targets = list(TARGETS)
    elif split in ("val", "train"):
        train_ld, val_ld, targets = make_loaders(
            labels_csv=cfg["labels_csv"], series_csv=cfg["series_csv"],
            zip_path=cfg["zip_path"], batch_size=batch_size, num_workers=0,
            n_series=n_series,
            train_frac=float(cfg.get("train_frac", 1.0)),
            train_frac_seed=int(cfg.get("train_frac_seed", 42)),
            rgb_neighbors=rgb,
        )
        ds = train_ld.dataset if split == "train" else val_ld.dataset
        ds.train = False
        ds.series_drop_p = 0.0
        ds = _limit_items(ds, max_studies)
    else:
        raise SystemExit(f"split inconnu : {split}")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False,
    )
    return loader, list(targets)


@torch.no_grad()
def collect_attention(model, loader, device: str, targets: list[str],
                      n_examples: int = 4):
    """Agregue l'attention val/gold. Garde quelques volumes pour les mosaics."""
    n_series = int(model.n_series)
    n_targets = int(model.n_targets)
    d = N_SLICES
    sum_attn = np.zeros((n_series, d, n_targets), dtype=np.float64)
    n_valid = np.zeros((n_series, d), dtype=np.float64)
    topk_hits = np.zeros((n_series, d, n_targets), dtype=np.float64)
    topk_any = np.zeros((n_series, d), dtype=np.float64)
    n_studies = 0
    examples: list[dict] = []
    use_cam = hasattr(model, "interpret")
    sum_cam = (
        np.zeros((n_series, d, n_targets), dtype=np.float64) if use_cam else None
    )

    model.eval()
    for x, _y, uids in loader:
        x = x.to(device, non_blocking=True)
        if use_cam:
            out = model.interpret(x)
            attn = out["slice_attn"]
            valid = out["valid"]
            cam = cam_token_slice_attn(model, out["cam_tokens"], valid)
            cam_np = cam.float().cpu().numpy()
        else:
            _, attn, valid = forward_with_attn(model, x)
            cam_np = None
        attn_np = attn.float().cpu().numpy()
        valid_np = valid.cpu().numpy()
        b, s, dd, t = attn_np.shape
        assert dd == d, f"D={dd}, attendu {d}"

        sum_attn += (attn_np * valid_np[..., None]).sum(axis=0)
        n_valid += valid_np.sum(axis=0)
        if cam_np is not None:
            sum_cam += (cam_np * valid_np[..., None]).sum(axis=0)

        flat = attn_np.reshape(b, s * dd, t)
        top_idx = np.argpartition(-flat, kth=min(TOPK, s * dd) - 1, axis=1)
        top_idx = top_idx[:, :TOPK, :]
        for bi in range(b):
            any_mask = np.zeros((s, dd), dtype=bool)
            for ti in range(t):
                for k in range(TOPK):
                    tok = int(top_idx[bi, k, ti])
                    ss, di = divmod(tok, dd)
                    if valid_np[bi, ss, di]:
                        topk_hits[ss, di, ti] += 1.0
                        any_mask[ss, di] = True
            topk_any += any_mask.astype(np.float64)
        n_studies += b

        if len(examples) < n_examples:
            x_cpu = x.cpu()
            for bi in range(b):
                if len(examples) >= n_examples:
                    break
                uid = uids[bi] if isinstance(uids[bi], str) else str(uids[bi])
                examples.append({
                    "uid": uid,
                    "x": _to_gray_volume(x_cpu[bi]).numpy(),
                    "attn": attn_np[bi],
                    "valid": valid_np[bi],
                })
        print(f"  {n_studies} etudes", flush=True)

    mean_sdt = sum_attn / np.maximum(n_valid[..., None], 1.0)
    mean_sd = mean_sdt.mean(axis=2)
    pct_top5_sdt = topk_hits / max(n_studies, 1)
    pct_top5_any = topk_any / max(n_studies, 1)
    mass_d = mean_sd.sum(axis=0)
    mass_d = mass_d / max(float(mass_d.sum()), 1e-12)
    return {
        "n_studies": int(n_studies),
        "targets": targets,
        "planes": plane_names(n_series),
        "mean_sdt": mean_sdt,
        "mean_sd": mean_sd,
        "n_valid": n_valid,
        "pct_top5_sdt": pct_top5_sdt,
        "pct_top5_any": pct_top5_any,
        "mass_d": mass_d,
        "examples": examples,
        "mean_cam_sdt": (
            None if sum_cam is None
            else sum_cam / np.maximum(n_valid[..., None], 1.0)
        ),
    }


def trim_band(mass_d: np.ndarray, keep: float = KEEP_MASS,
              band: tuple[float, float] = BAND):
    """Retire les bords les plus pauvres jusqu'a garder `keep` de la masse."""
    m = np.asarray(mass_d, dtype=np.float64)
    m = m / max(float(m.sum()), 1e-12)
    lo, hi = 0, len(m) - 1
    total = 1.0
    while hi > lo and total - min(m[lo], m[hi]) >= keep:
        if m[lo] <= m[hi]:
            total -= m[lo]
            lo += 1
        else:
            total -= m[hi]
            hi -= 1
    n = len(m)
    return {
        "keep_lo": int(lo),
        "keep_hi": int(hi),
        "n_keep": int(hi - lo + 1),
        "kept_mass": float(m[lo:hi + 1].sum()),
        "band_lo": slice_depth_pct(lo, n, band) / 100.0,
        "band_hi": slice_depth_pct(hi, n, band) / 100.0,
        "drop_idx": [i for i in range(n) if i < lo or i > hi],
    }


def dead_slices(mass_d: np.ndarray, pct_top5_any: np.ndarray) -> list[int]:
    uniform = 1.0 / len(mass_d)
    dead = []
    for i, (mass, top) in enumerate(zip(mass_d, pct_top5_any)):
        if mass < DEAD_MASS_RATIO * uniform and top < DEAD_TOPK_PCT:
            dead.append(i)
    return dead


def slice_table(agg: dict) -> pd.DataFrame:
    n = len(agg["mass_d"])
    planes = agg["planes"]
    mean_sd = agg["mean_sd"]
    rows = []
    for i in range(n):
        row = {
            "slice": i,
            "depth_pct": round(slice_depth_pct(i), 1),
            "mass": float(agg["mass_d"][i]),
            "pct_top5_any": float(agg["pct_top5_any"].mean(axis=0)[i]),
        }
        for p, name in enumerate(planes):
            row[f"mass_{name[:3].lower()}"] = float(mean_sd[p, i])
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_json(cfg: dict, ckpt_path: Path, epoch, val_auc, split: str,
                  agg: dict, suggested: dict, dead: list[int]) -> dict:
    planes = agg["planes"]
    targets = agg["targets"]
    mean_sdt = agg["mean_sdt"]
    mass_by_target = {
        targets[t]: (mean_sdt[:, :, t].sum(axis=0) /
                     max(float(mean_sdt[:, :, t].sum()), 1e-12)).tolist()
        for t in range(len(targets))
    }
    return {
        "run_name": cfg.get("run_name"),
        "model": cfg.get("model"),
        "ckpt": str(ckpt_path),
        "epoch": epoch,
        "val_macro_auc": None if val_auc is None else float(val_auc),
        "split": split,
        "n_studies": agg["n_studies"],
        "planes": planes,
        "targets": targets,
        "depth_pct": [round(slice_depth_pct(i), 2) for i in range(len(agg["mass_d"]))],
        "mass_by_slice": agg["mass_d"].tolist(),
        "mass_by_plane_slice": agg["mean_sd"].tolist(),
        "mass_by_target_slice": mass_by_target,
        "pct_top5_any_by_plane_slice": agg["pct_top5_any"].tolist(),
        "pct_top5_any_mean_planes": agg["pct_top5_any"].mean(axis=0).tolist(),
        "suggested": suggested,
        "dead_idx": dead,
        "band_current": list(BAND),
        "n_slices": N_SLICES,
        "uniform_1_over_d": 1.0 / N_SLICES,
    }


def plot_profiles(agg: dict, out: Path) -> None:
    planes = agg["planes"]
    mass_d = agg["mass_d"]
    mean_sd = agg["mean_sd"]
    xs = np.arange(len(mass_d))
    pct = [slice_depth_pct(i) for i in xs]
    uniform = 1.0 / len(mass_d)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    for p, name in enumerate(planes):
        ax.plot(xs, mean_sd[p], marker="o", ms=3, label=name)
    ax.set_ylabel("attention moyenne / token")
    ax.set_title("Attention moyenne par plan et indice de coupe")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = ["#c0392b" if m < DEAD_MASS_RATIO * uniform else "#2c3e50" for m in mass_d]
    ax.bar(xs, mass_d, color=colors)
    ax.axhline(uniform, color="#888", ls="--", lw=1, label=f"uniforme 1/{len(mass_d)}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i}\n{p:.0f}%" for i, p in zip(xs, pct)], fontsize=7)
    ax.set_xlabel("indice de coupe (percentile profondeur dans le volume d'origine)")
    ax.set_ylabel("masse (somme des plans, moy. cibles)")
    ax.set_title("Profil de profondeur — barres rouges = candidates au retrait")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "attn_profile.png", dpi=140)
    plt.close(fig)


def plot_heatmaps(agg: dict, out: Path) -> None:
    planes = agg["planes"]
    targets = agg["targets"]
    mean_sd = agg["mean_sd"]
    mean_sdt = agg["mean_sdt"]
    d = mean_sd.shape[1]
    tgt_slice = np.stack(
        [mean_sdt[:, :, t].sum(axis=0) for t in range(len(targets))], axis=0,
    )

    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(mean_sd, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(planes)))
    ax.set_yticklabels(planes)
    ax.set_xticks(range(d))
    ax.set_xlabel("indice de coupe (0 = 20 %, 23 = 80 %)")
    ax.set_title("Attention moyenne (moyenne des 12 tetes)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out / "attn_heatmap_planes.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(tgt_slice, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=8)
    ax.set_xticks(range(d))
    ax.set_xlabel("indice de coupe")
    ax.set_title("Masse d'attention par cible (somme des 3 plans)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out / "attn_heatmap_targets.png", dpi=140)
    plt.close(fig)


def plot_examples(agg: dict, out: Path, target_names: tuple[str, ...] = (
        "ACL", "Effusion", "Baker's")) -> None:
    targets = agg["targets"]
    t_idx = [targets.index(n) for n in target_names if n in targets]
    if not t_idx:
        t_idx = [0]
    ex_dir = out / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    planes = agg["planes"]
    for ei, ex in enumerate(agg["examples"]):
        x, attn, valid = ex["x"], ex["attn"], ex["valid"]
        s, d, _h, _w = x.shape
        for ti in t_idx:
            fig, axes = plt.subplots(s, d, figsize=(d * 0.85, s * 1.15))
            axes = np.asarray(axes).reshape(s, d)
            vmax = float(attn[:, :, ti].max()) + 1e-8
            for p in range(s):
                for k in range(d):
                    ax = axes[p, k]
                    ax.imshow(x[p, k], cmap="gray", vmin=0, vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    a = float(attn[p, k, ti])
                    if not valid[p, k]:
                        ax.set_title("—", fontsize=6, color="0.5")
                        ax.set_axis_off()
                        continue
                    ax.set_title(f"{k}:{a:.2f}", fontsize=6,
                                 color="white" if a > 0.15 * vmax else "0.85")
                    for spine in ax.spines.values():
                        spine.set_linewidth(2.0 + 6.0 * a / vmax)
                        spine.set_edgecolor(plt.cm.hot(min(a / vmax, 1.0)))
                axes[p, 0].set_ylabel(planes[p], fontsize=8)
            fig.suptitle(
                f"{ex['uid'][:18]}…  tete {targets[ti]}", fontsize=10,
            )
            fig.tight_layout()
            safe = targets[ti].replace(" ", "_").replace("'", "")
            fig.savefig(ex_dir / f"ex{ei}_{safe}.png", dpi=110)
            plt.close(fig)


def print_report(summary: dict, table: pd.DataFrame) -> None:
    sug = summary["suggested"]
    dead = summary["dead_idx"]
    print("\n=== attention par coupe (masse sommee sur les plans) ===")
    show = table[["slice", "depth_pct", "mass", "pct_top5_any"]].copy()
    show["mass"] = show["mass"].map(lambda v: f"{100 * v:.2f}%")
    show["pct_top5_any"] = show["pct_top5_any"].map(lambda v: f"{100 * v:.1f}%")
    print(show.to_string(index=False))
    print(f"\n90 % de la masse : coupes {sug['keep_lo']}–{sug['keep_hi']} "
          f"({sug['n_keep']}/24) = bande "
          f"{100 * sug['band_lo']:.0f}–{100 * sug['band_hi']:.0f} % "
          f"(masse {100 * sug['kept_mass']:.1f} %)")
    if sug["drop_idx"]:
        print(f"bords a retirer : {sug['drop_idx']}")
    if dead:
        print(f"coupes quasi mortes (masse < {DEAD_MASS_RATIO:.0%} uniforme "
              f"et top-5 < {100 * DEAD_TOPK_PCT:.0f} %) : {dead}")
    else:
        print("aucune coupe « morte » au sens strict : le softmax est juste "
              "concentre au centre de la bande.")
    print("\nPour retirer sans reconvertir : garder les indices "
          f"{sug['keep_lo']}:{sug['keep_hi'] + 1} dans le NPZ (D="
          f"{sug['n_keep']}).")
    print("Pour densifier : reconvertir avec "
          f"BAND=({sug['band_lo']:.2f}, {sug['band_hi']:.2f}) et garder 24 coupes.")


def run(cfg: dict, ckpt_path: Path, split: str, out_dir: Path,
        device: str, batch_size: int, num_workers: int,
        max_studies: int | None, n_examples: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ckpt {ckpt_path}", flush=True)
    model, epoch, val_auc = load_ckpt_model(cfg, ckpt_path, device)
    print(f"{cfg.get('model')} epoch {epoch} val AUC {val_auc} | split {split}",
          flush=True)
    loader, targets = build_loader(
        cfg, split, batch_size, num_workers, max_studies,
    )
    print(f"{len(loader.dataset)} etudes, batch {batch_size}", flush=True)
    agg = collect_attention(model, loader, device, targets, n_examples)
    suggested = trim_band(agg["mass_d"])
    dead = dead_slices(agg["mass_d"], agg["pct_top5_any"].mean(axis=0))
    table = slice_table(agg)
    summary = _summary_json(
        cfg, ckpt_path, epoch, val_auc, split, agg, suggested, dead,
    )
    table.to_csv(out_dir / "slice_attention.csv", index=False)
    np.savez_compressed(
        out_dir / "attention_agg.npz",
        mean_sdt=agg["mean_sdt"], mean_sd=agg["mean_sd"],
        mass_d=agg["mass_d"], pct_top5_any=agg["pct_top5_any"],
        pct_top5_sdt=agg["pct_top5_sdt"], n_valid=agg["n_valid"],
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    plot_profiles(agg, out_dir)
    plot_heatmaps(agg, out_dir)
    plot_examples(agg, out_dir)
    print_report(summary, table)
    if agg.get("mean_cam_sdt") is not None:
        from src.infer.inspect_spatial import save_gap_vs_cam, summarize_gap_vs_cam
        cmp_ = summarize_gap_vs_cam(
            agg["mean_sdt"], agg["mean_cam_sdt"],
            agg["targets"], agg["planes"], agg["n_studies"],
        )
        save_gap_vs_cam(cmp_, out_dir)
        med = float(np.nanmedian([r["corr"] for r in cmp_["rows"]]))
        print(f"GAP vs CAM (contrefactuel v15) corr mediane {med:.3f} "
              f"-> {out_dir / 'gap_vs_cam_slice.csv'}")
    print(f"\necrit -> {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kneenet_v11_crop130.yaml")
    parser.add_argument("--ckpt", default=None,
                        help="defaut : checkpoints/<run_name>/best.pt")
    parser.add_argument("--split", default="val",
                        choices=("val", "gold", "train"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-studies", type=int, default=None)
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("model", "kneenet_v8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt) if args.ckpt else (
        Path("checkpoints") / cfg["run_name"] / "best.pt"
    )
    if not ckpt.exists():
        alt = ckpt.parent / "last.pt"
        if alt.exists():
            ckpt = alt
        else:
            raise SystemExit(f"checkpoint introuvable : {ckpt}")
    if args.out:
        out = Path(args.out)
    else:
        folder = (
            "attention_inspect_gold" if args.split == "gold"
            else "attention_inspect"
        )
        out = Path("checkpoints") / cfg["run_name"] / folder
    bs = int(args.batch_size or cfg.get("batch_size") or 4)
    run(
        cfg, ckpt, args.split, out, device, bs, args.workers,
        args.max_studies, args.n_examples,
    )


if __name__ == "__main__":
    main()
