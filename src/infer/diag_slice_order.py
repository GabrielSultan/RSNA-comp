# -*- coding: utf-8 -*-
"""Diagnostic : InstanceNumber vs tri geometrique (IPP · normale).

Repond a : faut-il reconvertir les NPZ en triant par ImagePositionPatient
plutot que par InstanceNumber (cle actuelle) ?

Headers seulement (`stop_before_pixels`). Meme selection de series que le
crop130 (1 fluid-prioritaire / plan). Meme bande 20-80 % + linspace 24.

Usage local (si train_series DICOM est dispo) :
  python -X utf8 -m src.infer.diag_slice_order --comp-dir path/vers/competition

Sur Kaggle : notebook `kaggle_DICOM_exctract/diag_slice_order/`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

N_SLICES = 24
BAND = (0.20, 0.80)
PLANES = ["Sagittal", "Coronal", "Axial"]
TAGS = [
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "SOPInstanceUID",
    "NumberOfFrames",
]


def geom_key(ipp, iop) -> float:
    if ipp is None:
        return float("nan")
    ipp = np.asarray(ipp, dtype=np.float64)
    if iop is not None:
        iop = np.asarray(iop, dtype=np.float64)
        if iop.size >= 6:
            row, col = iop[:3], iop[3:6]
            normal = np.cross(row, col)
            nrm = float(np.linalg.norm(normal))
            if nrm > 1e-12:
                normal = normal / nrm
            return float(np.dot(ipp, normal))
    if ipp.size >= 3:
        return float(ipp[2])
    return float("nan")


def inst_key(inst) -> float:
    if inst is None:
        return float("nan")
    try:
        return float(inst)
    except (TypeError, ValueError):
        return float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan")
    aa, bb = a[m], b[m]
    if np.unique(aa).size < 2 or np.unique(bb).size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(aa, kind="mergesort"), kind="mergesort").astype(np.float64)
    rb = np.argsort(np.argsort(bb, kind="mergesort"), kind="mergesort").astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if den < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / den)


def subsample_idx(n: int) -> np.ndarray:
    """Meme formule que `preprocess_dicom.load_series_dicom`."""
    if n <= 0:
        return np.zeros(0, dtype=int)
    a = int(np.floor(BAND[0] * (n - 1)))
    b = int(np.ceil(BAND[1] * (n - 1)))
    if b - a + 1 < 8:
        a, b = 0, n - 1
    idx_band = np.linspace(0, b - a, N_SLICES).round().astype(int)
    return np.clip(a + idx_band, 0, n - 1)


def classify(rho: float) -> str:
    if not np.isfinite(rho):
        return "unknown"
    if rho >= 0.99:
        return "same"
    if rho <= -0.99:
        return "reverse"
    if abs(rho) >= 0.90:
        return "mild"
    return "scramble"


def pick_three_planes(g: pd.DataFrame) -> list[pd.Series]:
    picks = []
    for plane in PLANES:
        gp = g[g.Anatomical_Plane == plane].sort_values("SeriesInstanceUID")
        if len(gp) == 0:
            continue
        fluid = gp[gp.Fluid_Sensitive == 1] if "Fluid_Sensitive" in gp.columns else gp.iloc[0:0]
        if len(fluid) == 0 and "Fat_Suppression" in gp.columns:
            fluid = gp[gp.Fat_Suppression == 1]
        row = fluid.iloc[0] if len(fluid) else gp.iloc[0]
        picks.append(row)
    return picks


def series_path(train_series_dir: Path, study: str, series: str) -> Path:
    p = train_series_dir / study / series
    if p.exists():
        return p
    return train_series_dir / series


def analyze_series(series_dir: Path) -> dict:
    import pydicom

    files = sorted(series_dir.glob("*.dcm"))
    insts, geoms, names, sops = [], [], [], []
    n_fail = 0
    n_multiframe = 0
    for f in files:
        try:
            ds = pydicom.dcmread(
                str(f), stop_before_pixels=True, specific_tags=TAGS,
            )
        except Exception:
            n_fail += 1
            continue
        nf = getattr(ds, "NumberOfFrames", None)
        try:
            if nf is not None and int(nf) > 1:
                n_multiframe += 1
        except (TypeError, ValueError):
            pass
        insts.append(inst_key(getattr(ds, "InstanceNumber", None)))
        geoms.append(geom_key(
            getattr(ds, "ImagePositionPatient", None),
            getattr(ds, "ImageOrientationPatient", None),
        ))
        names.append(f.name)
        sops.append(str(getattr(ds, "SOPInstanceUID", f.stem)))
    n = len(insts)
    out = {
        "n": n,
        "n_files": len(files),
        "n_fail": n_fail,
        "n_multiframe": n_multiframe,
        "n_inst_ok": int(np.isfinite(insts).sum()) if n else 0,
        "n_geom_ok": int(np.isfinite(geoms).sum()) if n else 0,
        "n_inst_unique": int(np.unique(np.asarray(insts)[np.isfinite(insts)]).size) if n else 0,
        "rho_inst_geom": float("nan"),
        "rho_name_geom": float("nan"),
        "kind": "empty",
        "jaccard_24": float("nan"),
        "n_pos_diff_24": -1,
        "n_set_diff_24": -1,
        "jaccard_24_name": float("nan"),
        "mixed_inst": 0,
        "dup_inst": 0,
    }
    if n < 3:
        out["kind"] = "too_few"
        return out
    inst = np.asarray(insts, dtype=np.float64)
    geom = np.asarray(geoms, dtype=np.float64)
    name_rank = np.arange(n, dtype=np.float64)
    out["mixed_inst"] = int(out["n_inst_ok"] not in (0, n))
    out["dup_inst"] = int(
        out["n_inst_ok"] >= 3 and out["n_inst_unique"] < out["n_inst_ok"]
    )
    out["rho_inst_geom"] = spearman(inst, geom)
    out["rho_name_geom"] = spearman(name_rank, geom)
    out["kind"] = classify(out["rho_inst_geom"])
    if out["n_geom_ok"] < 3:
        out["kind"] = "no_geom"
        return out
    if out["n_inst_ok"] < 3:
        out["kind"] = "no_inst"
        return out

    order_inst = np.argsort(inst, kind="mergesort")
    order_geom = np.argsort(geom, kind="mergesort")
    order_name = np.arange(n)
    sub = subsample_idx(n)
    ids = np.asarray(sops)
    set_inst = ids[order_inst[sub]]
    set_geom = ids[order_geom[sub]]
    set_name = ids[order_name[sub]]
    a, b = set(set_inst.tolist()), set(set_geom.tolist())
    union = len(a | b)
    out["jaccard_24"] = (len(a & b) / union) if union else float("nan")
    out["n_pos_diff_24"] = int((set_inst != set_geom).sum())
    out["n_set_diff_24"] = int(len(a ^ b) // 2)
    an, bn = set(set_name.tolist()), set(set_geom.tolist())
    un = len(an | bn)
    out["jaccard_24_name"] = (len(an & bn) / un) if un else float("nan")
    return out


def find_comp_dir(explicit: str | None = None) -> tuple[Path, Path]:
    if explicit:
        base = Path(explicit)
        series_dir = base / "train_series"
        if not series_dir.exists():
            for cand in base.iterdir() if base.is_dir() else []:
                if cand.is_dir() and (cand / "train_series").exists():
                    series_dir = cand / "train_series"
                    break
        return base, series_dir
    if os.path.exists("/kaggle"):
        data = Path("/kaggle/input")
        for root in [data / "competitions", data]:
            if not root.exists():
                continue
            for p in sorted(root.iterdir()):
                if (p / "train.csv").exists() and (p / "train_series.csv").exists():
                    series_dir = p / "train_series"
                    if not series_dir.exists():
                        for cand in p.iterdir():
                            if cand.is_dir() and (cand / "train_series").exists():
                                series_dir = cand / "train_series"
                                break
                    return p, series_dir
        raise FileNotFoundError("competition introuvable sous /kaggle/input")
    raise FileNotFoundError(
        "pas de DICOM local : passer --comp-dir ou lancer le kernel Kaggle"
    )


def build_jobs(series_df: pd.DataFrame, n_studies: int, seed: int) -> list[dict]:
    studies = sorted(series_df.StudyInstanceUID.unique())
    rng = np.random.default_rng(seed)
    if n_studies > 0 and n_studies < len(studies):
        pick = set(rng.choice(studies, size=n_studies, replace=False))
    else:
        pick = set(studies)
    jobs = []
    for uid, g in series_df[series_df.StudyInstanceUID.isin(pick)].groupby(
            "StudyInstanceUID"):
        for row in pick_three_planes(g):
            jobs.append({
                "study": uid,
                "series": row.SeriesInstanceUID,
                "plane": row.Anatomical_Plane,
                "fluid": int(row.Fluid_Sensitive) if "Fluid_Sensitive" in row.index else -1,
            })
    return jobs


def verdict(df: pd.DataFrame) -> dict:
    ok = df[df.n >= 8].copy()
    n = max(len(ok), 1)
    kind_counts = ok["kind"].value_counts().to_dict() if len(ok) else {}
    pct = {k: float(kind_counts.get(k, 0) / n) for k in
           ("same", "reverse", "mild", "scramble", "no_inst", "no_geom", "unknown")}
    j = ok["jaccard_24"].dropna()
    jn = ok["jaccard_24_name"].dropna()
    rho = ok["rho_inst_geom"].dropna()
    rho_n = ok["rho_name_geom"].dropna()
    convert = bool(
        pct["scramble"] > 0.03
        or (len(j) and float((j < 0.95).mean()) > 0.05)
        or (len(j) and float(j.mean()) < 0.97)
        or pct["no_inst"] > 0.05
        or float(ok["mixed_inst"].mean() if len(ok) else 0) > 0.02
    )
    reason = []
    if pct["scramble"] > 0.03:
        reason.append(f"scramble {pct['scramble']:.1%} > 3%")
    if len(j) and float((j < 0.95).mean()) > 0.05:
        reason.append(f"jaccard_24 < 0.95 sur {(j < 0.95).mean():.1%}")
    if len(j) and float(j.mean()) < 0.97:
        reason.append(f"jaccard_24 moyen {float(j.mean()):.3f} < 0.97")
    if pct["no_inst"] > 0.05:
        reason.append(f"InstanceNumber absent {pct['no_inst']:.1%}")
    if float(ok["mixed_inst"].mean() if len(ok) else 0) > 0.02:
        reason.append("cles InstanceNumber/geom melangees")
    if not reason:
        reason.append(
            "InstanceNumber et IPP·normale selectionnent les memes 24 coupes "
            "(a un flip pres)"
        )
    return {
        "n_series": int(len(df)),
        "n_series_n8": int(len(ok)),
        "pct_kind": pct,
        "kind_counts": {str(k): int(v) for k, v in kind_counts.items()},
        "rho_inst_geom_mean": float(rho.mean()) if len(rho) else None,
        "rho_inst_geom_p05": float(rho.quantile(0.05)) if len(rho) else None,
        "rho_name_geom_mean": float(rho_n.mean()) if len(rho_n) else None,
        "jaccard_24_mean": float(j.mean()) if len(j) else None,
        "jaccard_24_lt1": float((j < 1.0).mean()) if len(j) else None,
        "jaccard_24_lt095": float((j < 0.95).mean()) if len(j) else None,
        "jaccard_24_name_mean": float(jn.mean()) if len(jn) else None,
        "pct_dup_inst": float(ok["dup_inst"].mean()) if len(ok) else None,
        "pct_mixed_inst": float(ok["mixed_inst"].mean()) if len(ok) else None,
        "pct_multiframe": float((df.n_multiframe > 0).mean()) if len(df) else None,
        "convert": convert,
        "reason": reason,
        "verdict": "CONVERTIR" if convert else "NE PAS CONVERTIR",
    }


def run(comp_dir: Path, train_series_dir: Path, out_dir: Path,
        n_studies: int, seed: int, workers: int) -> dict:
    csv_path = comp_dir / "train_series.csv"
    series_df = pd.read_csv(csv_path)
    jobs = build_jobs(series_df, n_studies, seed)
    print(f"jobs {len(jobs)} | workers {workers} | series_dir {train_series_dir}",
          flush=True)
    if not train_series_dir.exists():
        raise FileNotFoundError(f"pas de train_series : {train_series_dir}")

    rows = []
    t0 = time.time()

    def _one(job):
        p = series_path(train_series_dir, job["study"], job["series"])
        rec = dict(job)
        if not p.exists():
            rec.update({"n": 0, "kind": "missing_dir", "error": "no_dir"})
            return rec
        try:
            rec.update(analyze_series(p))
        except Exception as e:
            rec.update({"n": 0, "kind": "error", "error": f"{type(e).__name__}: {e}"})
        return rec

    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            rows.append(_one(job))
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 50 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "slice_order_diag.csv"
    df.to_csv(csv_out, index=False)
    summary = verdict(df)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["n_studies_requested"] = n_studies
    summary["seed"] = seed
    (out_dir / "slice_order_verdict.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("csv", csv_out, flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-dir", default=None,
                    help="dossier competition (train_series.csv + train_series/)")
    ap.add_argument("--n-studies", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    comp_dir, series_dir = find_comp_dir(args.comp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("/kaggle/working") if os.path.exists("/kaggle") else Path("checkpoints") / "diag_slice_order"
    )
    print("COMP_DIR", comp_dir, flush=True)
    print("TRAIN_SERIES_DIR", series_dir, series_dir.exists(), flush=True)
    run(comp_dir, series_dir, out_dir, args.n_studies, args.seed, args.workers)


if __name__ == "__main__":
    main()
