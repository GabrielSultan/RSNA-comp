# -*- coding: utf-8 -*-
"""Dump StudyInstanceUID -> L/R depuis les headers DICOM (pas les pixels).

Local :
  python -X utf8 -m src.data.dump_laterality --train-series-dir <dir> \\
      --out data/csv/study_laterality.csv

Kaggle (competition attachee, internet off) : le script trouve tout seul
/kaggle/input/.../train_series. Ecrire le CSV puis le telecharger.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.laterality import (
    image_center_x_mm, lat_of_rows, probe_series_header,
    side_from_geometry_rows, side_from_tag,
)


def _find_kaggle_train_series() -> Path | None:
    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    hits = list(root.rglob("train_series"))
    for p in hits:
        if p.is_dir():
            return p
    return None


def _walk_series(train_series_dir: Path) -> list[tuple[str, str, Path]]:
    items = []
    for study in sorted(p for p in train_series_dir.iterdir() if p.is_dir()):
        for series in sorted(p for p in study.iterdir() if p.is_dir()):
            items.append((study.name, series.name, series))
    return items


def _probe(item: tuple[str, str, Path]) -> dict | None:
    uid, sid, path = item
    rec = probe_series_header(path)
    if rec is None:
        return None
    rec["StudyInstanceUID"] = uid
    rec["SeriesInstanceUID"] = sid
    return rec


def dump_laterality(train_series_dir: Path, out_csv: Path, workers: int = 8) -> pd.DataFrame:
    items = _walk_series(train_series_dir)
    print(f"scan {len(items)} series sous {train_series_dir}", flush=True)
    rows: list[dict] = []
    n_fail = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_probe, it) for it in items]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec is None:
                n_fail += 1
            else:
                rows.append(rec)
            if i % 2000 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} headers ({len(rows)} ok, {n_fail} fail)",
                      flush=True)
    if not rows:
        raise SystemExit("aucun header lu")
    geo = side_from_geometry_rows(rows)
    lat = lat_of_rows(rows, tag="train")
    by_uid: dict[str, list[dict]] = {}
    for r in rows:
        by_uid.setdefault(str(r["StudyInstanceUID"]), []).append(r)
    out_rows = []
    for uid, grp in sorted(by_uid.items()):
        tags = [side_from_tag(r.get("Laterality")) for r in grp]
        tags = [t for t in tags if t is not None]
        tag_side = tags[0] if tags else None
        geo_side = geo.get(uid)
        xs = []
        for r in grp:
            x = image_center_x_mm(
                r.get("ImagePositionPatient"), r.get("ImageOrientationPatient"),
                r.get("PixelSpacing"), r.get("Rows"), r.get("Columns"),
            )
            if x is not None:
                xs.append(x)
        out_rows.append({
            "StudyInstanceUID": uid,
            "laterality": lat.get(uid) or "",
            "source": ("tag" if tag_side else ("geo" if geo_side else "none")),
            "tag_laterality": tag_side or "",
            "geo_laterality": geo_side or "",
            "geo_x_mm": float(np.median(xs)) if xs else "",
            "n_series_probed": len(grp),
            "disagree": int(tag_side is not None and geo_side is not None
                            and tag_side != geo_side),
        })
    df = pd.DataFrame(out_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_l = int((df.laterality == "L").sum())
    n_r = int((df.laterality == "R").sum())
    n_u = int((df.laterality == "").sum())
    print(f"ecrit {out_csv} | {len(df)} etudes | L {n_l} | R {n_r} | "
          f"unresolved {n_u} | disagree {int(df.disagree.sum())}", flush=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-series-dir", default=None)
    parser.add_argument("--out", default="data/csv/study_laterality.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    src = Path(args.train_series_dir) if args.train_series_dir else None
    if src is None:
        src = _find_kaggle_train_series()
    if src is None or not src.is_dir():
        raise SystemExit(
            "train_series introuvable. Passer --train-series-dir "
            "ou lancer sur Kaggle avec le dataset competition."
        )
    dump_laterality(src, Path(args.out), workers=int(args.workers))


if __name__ == "__main__":
    main()
