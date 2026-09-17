# -*- coding: utf-8 -*-
"""P0 plan : D1-D5 + TTA profondeur / serie. Pas de retrain.

  python -X utf8 -m src.infer.p0_diag --stage cpu
  python -X utf8 -m src.infer.p0_diag --stage d1
  python -X utf8 -m src.infer.p0_diag --stage tta
  python -X utf8 -m src.infer.p0_diag --stage all
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset_npz import PROTOCOL_SLOT_NAMES, PROTOCOL_SLOTS, make_loaders
from src.infer.inspect_attention import load_ckpt_model, load_config, run as run_attn
from src.nlp.lang_detect import detect_lang
from src.nlp.lexicon import TARGETS
from src.train.metrics import macro_auc
from src.train.train import evaluate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "checkpoints" / "p0_diag"
GOLD_FOCUS = ("MCL", "Lateral Meniscus", "Lateral OA")
EDGE_IDX = (0, 1, 22, 23)
SAG_SLOTS = ("SAG_FS", "SAG_T1")

TRACE_RE = re.compile(
    r"\b(trace|minimal|mild|small|physiolog|physiologique|"
    r"derrame\s+m[ií]nim|leve|gering|fizyolojik|minime)\b",
    re.I,
)
SYNOV_RE = re.compile(
    r"synovit|sinovit|synovia|synovite",
    re.I,
)


def _dump(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ecrit {p}", flush=True)
    return p


def gold_table() -> pd.DataFrame:
    train = pd.read_csv(ROOT / "data" / "csv" / "train.csv")
    gold = train[train[list(TARGETS)].notna().all(axis=1)].copy()
    llm = pd.read_csv(ROOT / "data" / "labels" / "labels_llm_gpt56sol.csv")
    gold = gold.merge(
        llm[["StudyInstanceUID"] + list(TARGETS)],
        on="StudyInstanceUID", suffixes=("_gold", "_llm"),
    )
    gold["lang"] = gold["Report"].map(detect_lang)
    return gold


def stage_d2() -> dict:
    gold = gold_table()
    out: dict = {"n_gold": int(len(gold)), "by_target": {}}
    for tgt in ("Effusion", "Synovitis"):
        g = gold[f"{tgt}_gold"].astype(float)
        l = gold[f"{tgt}_llm"].astype(float)
        fp = gold[(g == 0) & (l == 1)]
        fn = gold[(g == 1) & (l == 0)]
        tp = int(((g == 1) & (l == 1)).sum())
        tn = int(((g == 0) & (l == 0)).sum())
        rows = []
        for kind, sub in (("fp", fp), ("fn", fn)):
            for _, r in sub.iterrows():
                txt = str(r["Report"])
                rows.append({
                    "kind": kind,
                    "uid": r["StudyInstanceUID"],
                    "lang": r["lang"],
                    "trace_kw": bool(TRACE_RE.search(txt)),
                    "synov_kw": bool(SYNOV_RE.search(txt)),
                    "excerpt": re.sub(r"\s+", " ", txt)[:280],
                })
        out["by_target"][tgt] = {
            "tp": tp, "tn": tn, "fp": int(len(fp)), "fn": int(len(fn)),
            "fp_trace_kw": int(sum(x["trace_kw"] for x in rows if x["kind"] == "fp")),
            "fn_no_synov_kw": int(sum(
                (not x["synov_kw"]) for x in rows if x["kind"] == "fn")),
            "fp_by_lang": fp["lang"].value_counts().to_dict() if len(fp) else {},
            "fn_by_lang": fn["lang"].value_counts().to_dict() if len(fn) else {},
            "rows": rows,
        }
    _dump("d2_gold_vs_llm.json", out)
    return out


def _auc_from_oof(gold: pd.DataFrame) -> dict:
    oof_p = ROOT / "data" / "labels" / "oof_preds_v17.csv"
    if not oof_p.exists():
        return {"error": "pas de oof_preds_v17.csv"}
    oof = pd.read_csv(oof_p)
    m = gold.merge(oof, on="StudyInstanceUID", how="inner")
    y = np.stack([m[f"{t}_gold"].to_numpy() for t in TARGETS], axis=1)
    s = np.stack([m[f"proba_{t}"].to_numpy() for t in TARGETS], axis=1)
    macro, per = macro_auc(y, s, list(TARGETS))
    return {
        "n": int(len(m)),
        "macro": float(macro),
        "per_target": {k: float(v) for k, v in per.items()},
    }


def _hist_gold_curve(run: str) -> dict | None:
    p = ROOT / "checkpoints" / run / "history.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for r in h:
        rows.append({
            "epoch": int(r["epoch"]),
            "val_macro_auc": float(r.get("val_macro_auc", float("nan"))),
            "gold_macro_auc": float(r.get("gold_macro_auc", float("nan"))),
            "proba_std": float(r.get("proba_std", float("nan"))),
            "val_loss": float(r.get("val_loss", float("nan"))),
        })
    if not rows:
        return None
    pic = max(rows, key=lambda x: x["gold_macro_auc"])
    best_val = max(rows, key=lambda x: x["val_macro_auc"])
    return {
        "run": run,
        "n_epochs": len(rows),
        "curve": rows,
        "pic_gold": pic,
        "best_val": best_val,
        "gold_drop_after_pic": [
            x for x in rows if x["epoch"] > pic["epoch"]
        ],
    }


def _gold_sanity(run: str) -> dict | None:
    p = ROOT / "checkpoints" / run / "gold_sanity.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text(encoding="utf-8"))
    return {
        "run": run,
        "epoch": g.get("epoch"),
        "macro": g.get("macro_auc"),
        "per_target": g.get("per_target"),
        "n": g.get("n_studies"),
    }


def stage_d3() -> dict:
    gold = gold_table()
    out = {
        "v17_oof_gold58": _auc_from_oof(gold),
        "gold_sanity": {},
        "gold_macro_curves": {},
        "note": (
            "Pas de snapshots ep5/ep9 pour vit_e_crop130v1 : "
            "courbe gold = macro history, per-cible = gold_sanity du best.pt."
        ),
    }
    for run in (
        "vit_e_crop130v1",
        "v17_crop130v1",
        "v17_crop130_oof_fold0",
        "v17_crop130_oof_fold1",
        "v17_crop130_oof_fold2",
        "subset_vit_k_llrd08_frac025",
        "subset_vit_dinov3_e_crop130_frac025",
        "subset_v20b_crop130_frac025",
    ):
        gs = _gold_sanity(run)
        if gs:
            out["gold_sanity"][run] = gs
        cr = _hist_gold_curve(run)
        if cr:
            out["gold_macro_curves"][run] = cr
    vite = out["gold_macro_curves"].get("vit_e_crop130v1")
    if vite:
        by_ep = {r["epoch"]: r for r in vite["curve"]}
        out["vit_e_full_ep5_vs_ep9"] = {
            "ep5": by_ep.get(5),
            "ep9": by_ep.get(9),
            "pic_gold": vite["pic_gold"],
            "best_val": vite["best_val"],
        }
    _dump("d3_gold_holdout.json", out)
    return out


def _load_manifests() -> pd.DataFrame:
    rows = []
    for man in sorted((ROOT / "data" / "crop130").glob("manifest*.csv")):
        df = pd.read_csv(man)
        df["manifest"] = man.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def stage_d5() -> dict:
    man = _load_manifests()
    if man.empty:
        out = {"error": "pas de manifests crop130"}
        _dump("d5_slices_and_cands.json", out)
        return out
    if "ok" in man.columns:
        man = man[man["ok"] == 1].copy()
    n_raw = pd.to_numeric(man["n_raw"], errors="coerce") if "n_raw" in man.columns else pd.Series(dtype=float)
    n_files = pd.to_numeric(man["n_files"], errors="coerce") if "n_files" in man.columns else n_raw
    native = n_raw.fillna(n_files)

    def _band_n(val) -> float:
        if isinstance(val, str) and val.startswith("["):
            lo, hi = ast.literal_eval(val)
            return float(hi - lo + 1)
        return float("nan")

    band_n = man["band"].map(_band_n) if "band" in man.columns else pd.Series(np.nan, index=man.index)
    # 24 linspace dans band_n coupes natives -> doublons = 24 - unique
    unique_approx = np.minimum(24.0, band_n.to_numpy())
    dups = 24.0 - unique_approx

    by_plane = {}
    for plane, sub in man.groupby("plane"):
        nv = pd.to_numeric(sub["n_raw"], errors="coerce")
        bn = sub["band"].map(_band_n) if "band" in sub.columns else nv * 0.6
        by_plane[str(plane)] = {
            "n_series": int(len(sub)),
            "n_raw_p50": float(nv.median()),
            "n_raw_p10": float(nv.quantile(0.1)),
            "n_raw_p90": float(nv.quantile(0.9)),
            "band_n_p50": float(bn.median()),
            "dup_linspace_p50": float(np.median(24 - np.minimum(24, bn.to_numpy()))),
        }

    series = pd.read_csv(ROOT / "data" / "csv" / "train_series.csv")
    n_cands = []
    multi = 0
    for uid, g in series.groupby("StudyInstanceUID"):
        slot_multi = False
        for plane, fluid in PROTOCOL_SLOTS:
            if fluid:
                c = g[(g.Anatomical_Plane == plane) & (g.Fluid_Sensitive == 1)]
            else:
                c = g[(g.Anatomical_Plane == plane) & (g.Fluid_Sensitive == 0)]
            n = int(len(c))
            n_cands.append(n)
            if n >= 2:
                slot_multi = True
        if slot_multi:
            multi += 1
    n_studies = int(series.StudyInstanceUID.nunique())

    proba_std = {}
    for run in (
        "subset_vit_dinov3_e_crop130_frac025",
        "subset_vit_k_llrd08_frac025",
        "subset_v20b_crop130_frac025",
        "vit_e_crop130v1",
    ):
        cr = _hist_gold_curve(run)
        if cr:
            proba_std[run] = [
                {"epoch": r["epoch"], "proba_std": r["proba_std"]}
                for r in cr["curve"]
            ]

    out = {
        "n_manifest_ok": int(len(man)),
        "n_raw_p50": float(native.median()),
        "band_n_p50": float(np.nanmedian(band_n)),
        "dup_linspace_p50": float(np.nanmedian(dups)),
        "by_plane": by_plane,
        "studies": n_studies,
        "studies_with_ge2_cands_any_slot": multi,
        "pct_studies_multi_cand": round(100.0 * multi / max(n_studies, 1), 1),
        "n_cands_per_slot_mean": float(np.mean(n_cands)) if n_cands else None,
        "proba_std_by_epoch": proba_std,
        "update_norm_per_block": None,
        "update_norm_note": "non logge pendant le train ; indisponible sans retrain",
    }
    _dump("d5_slices_and_cands.json", out)
    return out


def _d1_edges(run: str) -> dict | None:
    npz = ROOT / "checkpoints" / run / "attention_inspect" / "attention_agg.npz"
    summary_p = ROOT / "checkpoints" / run / "attention_inspect" / "summary.json"
    if not npz.exists() or not summary_p.exists():
        return None
    z = np.load(npz)
    summary = json.loads(summary_p.read_text(encoding="utf-8"))
    mean_sdt = z["mean_sdt"]  # S, D, T
    planes = summary["planes"]
    targets = summary["targets"]
    slot_target = {}
    for si, slot in enumerate(planes):
        mass = mean_sdt[si].sum(axis=0)
        mass = mass / max(float(mass.sum()), 1e-12)
        slot_target[slot] = {
            t: float(mass[ti]) for ti, t in enumerate(targets)
        }
    edges = {}
    for tgt in GOLD_FOCUS:
        if tgt not in targets:
            continue
        ti = targets.index(tgt)
        edges[tgt] = {}
        for slot in SAG_SLOTS:
            if slot not in planes:
                continue
            si = planes.index(slot)
            vec = mean_sdt[si, :, ti]
            tot = float(vec.sum()) + 1e-12
            edges[tgt][slot] = {
                "mass_0_1": float(vec[list(EDGE_IDX[:2])].sum() / tot),
                "mass_22_23": float(vec[list(EDGE_IDX[2:])].sum() / tot),
                "n_eff": float(1.0 / ((vec / tot) ** 2).sum()),
            }
    n_eff = {}
    for ti, tgt in enumerate(targets):
        p = np.asarray(summary["mass_by_target_slice"][tgt], dtype=np.float64)
        p = p / max(float(p.sum()), 1e-12)
        n_eff[tgt] = float(1.0 / (p ** 2).sum())
    return {
        "run": run,
        "suggested": summary.get("suggested"),
        "dead_idx": summary.get("dead_idx"),
        "n_studies": summary.get("n_studies"),
        "slot_x_target": slot_target,
        "sagittal_edges": edges,
        "n_eff_by_target": n_eff,
        "expand_band": any(
            edges.get(t, {}).get(s, {}).get("mass_0_1", 0) > 0.12
            or edges.get(t, {}).get(s, {}).get("mass_22_23", 0) > 0.12
            for t in GOLD_FOCUS for s in SAG_SLOTS
        ),
    }


def stage_d1(device: str) -> dict:
    jobs = [
        ("configs/kneenet_v20b_crop130_subset.yaml",
         "checkpoints/subset_v20b_crop130_frac025/best.pt", 4, None),
        ("configs/kneenet_vit_e_crop130_subset.yaml",
         "checkpoints/subset_vit_dinov3_e_crop130_frac025/best.pt", 2, 200),
    ]
    results = {}
    for cfg_p, ckpt_p, bs, max_n in jobs:
        cfg = load_config(str(ROOT / cfg_p))
        ckpt = ROOT / ckpt_p
        if not ckpt.exists():
            results[cfg["run_name"]] = {"error": f"pas de {ckpt}"}
            continue
        out_dir = ROOT / "checkpoints" / cfg["run_name"] / "attention_inspect"
        summary_p = out_dir / "summary.json"
        if summary_p.exists() and cfg["run_name"].startswith("subset_v20b"):
            print(f"D1 skip {cfg['run_name']} (summary deja la)", flush=True)
        else:
            print(f"D1 {cfg['run_name']} max={max_n} -> {out_dir}", flush=True)
            run_attn(
                cfg, ckpt, "val", out_dir, device, bs, 0, max_n, 0,
            )
        extra = _d1_edges(cfg["run_name"])
        results[cfg["run_name"]] = extra
    _dump("d1_attention.json", results)
    return results


def _make_tta_dataset(cfg: dict, batch_size: int, *, rgb_neighbors: bool,
                      rgb_as_gray: bool):
    _, val_ld, targets = make_loaders(
        labels_csv=cfg["labels_csv"], series_csv=cfg["series_csv"],
        zip_path=str(cfg["zip_path"]), batch_size=batch_size, num_workers=0,
        n_series=int(cfg.get("n_series", 6)),
        train_frac=float(cfg.get("train_frac", 1.0)),
        train_frac_seed=int(cfg.get("train_frac_seed", 42)),
        rgb_neighbors=rgb_neighbors,
        img_size=int(cfg.get("img_size", 224)),
        rgb_as_gray=rgb_as_gray,
    )
    ds = val_ld.dataset
    ds.train = False
    ds.series_drop_p = 0.0
    return ds, list(targets)


def _eval_view(ds, targets, model, device: str, *, cand_pick: str,
               rgb_channel: int, batch_size: int, num_workers: int) -> dict:
    ds.cand_pick = cand_pick
    ds.rgb_channel = int(rgb_channel)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False,
    )
    probs, labels, uids, val_loss = evaluate(model, loader, device)
    macro, per = macro_auc(labels, probs, list(targets))
    return {
        "macro": float(macro),
        "val_loss": float(val_loss),
        "n": int(len(uids)),
        "per_target": {k: float(v) for k, v in per.items()},
        "probs": probs,
        "labels": labels,
        "uids": list(uids),
        "targets": list(targets),
    }


def _pack_auc(rec: dict) -> dict:
    return {
        "macro": rec["macro"],
        "val_loss": rec["val_loss"],
        "n": rec["n"],
        "per_target": rec["per_target"],
    }


def _mean_recs(recs: list[dict]) -> dict:
    probs = np.mean([r["probs"] for r in recs], axis=0)
    labels = recs[0]["labels"]
    targets = recs[0]["targets"]
    macro, per = macro_auc(labels, probs, list(targets))
    return {
        "macro": float(macro),
        "n": recs[0]["n"],
        "per_target": {k: float(v) for k, v in per.items()},
    }


def stage_tta(device: str) -> dict:
    jobs = [
        {
            "name": "v20b",
            "config": "configs/kneenet_v20b_crop130_subset.yaml",
            "ckpt": "checkpoints/subset_v20b_crop130_frac025/best.pt",
            "batch_size": 4,
        },
        {
            "name": "vit_e",
            "config": "configs/kneenet_vit_e_crop130_subset.yaml",
            "ckpt": "checkpoints/subset_vit_dinov3_e_crop130_frac025/best.pt",
            "batch_size": 2,
        },
    ]
    out: dict = {}
    for job in jobs:
        cfg = load_config(str(ROOT / job["config"]))
        cfg["zip_path"] = str(ROOT / "data" / "crop130_rgb25d")
        cfg["rgb_neighbors"] = True
        cfg["rgb_as_gray"] = True
        ckpt = ROOT / job["ckpt"]
        if not ckpt.exists():
            out[job["name"]] = {"error": f"pas de {ckpt}"}
            continue
        print(f"TTA load {job['name']}", flush=True)
        model, epoch, val_auc = load_ckpt_model(cfg, ckpt, device)
        bs = int(job["batch_size"])
        ds, targets = _make_tta_dataset(
            cfg, bs, rgb_neighbors=True, rgb_as_gray=True,
        )
        print(f"  val n={len(ds)} rgb25d", flush=True)
        views = {}
        for key, pick, ch in (
            ("G_first", "first", 1),
            ("G_last", "last", 1),
            ("R_first", "first", 0),
            ("B_first", "first", 2),
        ):
            print(f"  eval {job['name']} {key}", flush=True)
            views[key] = _eval_view(
                ds, targets, model, device, cand_pick=pick, rgb_channel=ch,
                batch_size=bs, num_workers=2,
            )
            print(
                f"    {key} macro={views[key]['macro']:.4f} "
                f"n={views[key]['n']}",
                flush=True,
            )
        packed = {k: _pack_auc(v) for k, v in views.items()}
        series = _mean_recs([views["G_first"], views["G_last"]])
        depth = _mean_recs([views["R_first"], views["G_first"], views["B_first"]])
        base = packed["G_first"]["macro"]
        packed["series_tta"] = series
        packed["depth_tta"] = depth
        packed["delta_series"] = round(series["macro"] - base, 4)
        packed["delta_depth"] = round(depth["macro"] - base, 4)
        packed["ckpt_epoch"] = epoch
        packed["ckpt_val"] = None if val_auc is None else float(val_auc)
        focus = {}
        for tgt in ("Fracture", "Contusion", "Lateral Meniscus"):
            if tgt in packed["G_first"]["per_target"]:
                focus[tgt] = {
                    "G": packed["G_first"]["per_target"][tgt],
                    "depth_tta": depth["per_target"][tgt],
                    "d": round(
                        depth["per_target"][tgt] - packed["G_first"]["per_target"][tgt],
                        4,
                    ),
                }
        packed["focus_depth"] = focus
        out[job["name"]] = packed
        del views
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    _dump("d4_tta.json", out)
    return out


def stage_tta_series_crop130(device: str) -> dict:
    """TTA serie sur crop130 (gris) : rgb25d n'a aucun slot a 2 candidats."""
    prev_p = OUT / "d4_tta.json"
    out = json.loads(prev_p.read_text(encoding="utf-8")) if prev_p.exists() else {}
    jobs = [
        {
            "name": "v20b",
            "config": "configs/kneenet_v20b_crop130_subset.yaml",
            "ckpt": "checkpoints/subset_v20b_crop130_frac025/best.pt",
            "batch_size": 4,
        },
        {
            "name": "vit_e",
            "config": "configs/kneenet_vit_e_crop130_subset.yaml",
            "ckpt": "checkpoints/subset_vit_dinov3_e_crop130_frac025/best.pt",
            "batch_size": 2,
        },
    ]
    for job in jobs:
        cfg = load_config(str(ROOT / job["config"]))
        cfg["zip_path"] = str(ROOT / "data" / "crop130")
        cfg["rgb_neighbors"] = False
        cfg["rgb_as_gray"] = False
        ckpt = ROOT / job["ckpt"]
        if not ckpt.exists():
            out.setdefault(job["name"], {})["series_crop130"] = {
                "error": f"pas de {ckpt}",
            }
            continue
        print(f"TTA series crop130 {job['name']}", flush=True)
        model, epoch, val_auc = load_ckpt_model(cfg, ckpt, device)
        bs = int(job["batch_size"])
        ds, targets = _make_tta_dataset(
            cfg, bs, rgb_neighbors=False, rgb_as_gray=False,
        )
        print(f"  val n={len(ds)} crop130", flush=True)
        views = {}
        for key, pick in (("first", "first"), ("last", "last")):
            print(f"  eval {job['name']} crop130 {key}", flush=True)
            views[key] = _eval_view(
                ds, targets, model, device, cand_pick=pick, rgb_channel=1,
                batch_size=bs, num_workers=2,
            )
            print(
                f"    {key} macro={views[key]['macro']:.4f} "
                f"n={views[key]['n']}",
                flush=True,
            )
        series = _mean_recs([views["first"], views["last"]])
        rec = {
            "first": _pack_auc(views["first"]),
            "last": _pack_auc(views["last"]),
            "series_tta": series,
            "delta_series": round(series["macro"] - views["first"]["macro"], 4),
            "n_val": int(len(ds)),
            "ckpt_epoch": epoch,
            "ckpt_val": None if val_auc is None else float(val_auc),
            "note": (
                "crop130 : 38/653 val ont >=2 cands indexes ; "
                "rgb25d en a 0 (T1 alts absents)"
            ),
        }
        out.setdefault(job["name"], {})["series_crop130"] = rec
        del views
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    _dump("d4_tta.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", default="all",
        choices=("cpu", "d1", "tta", "tta_series", "all"),
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | out {OUT}", flush=True)
    if args.stage in ("cpu", "all"):
        stage_d2()
        stage_d3()
        stage_d5()
    if args.stage in ("d1", "all"):
        stage_d1(device)
    if args.stage in ("tta", "all"):
        stage_tta(device)
    if args.stage in ("tta_series", "all"):
        stage_tta_series_crop130(device)
    print("termine", flush=True)


if __name__ == "__main__":
    main()
