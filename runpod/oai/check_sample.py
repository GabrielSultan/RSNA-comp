# -*- coding: utf-8 -*-
"""Controle local du convertisseur OAI sur les archives d'echantillon.

1. Egalite bit a bit oai_convert vs load_series_dicom (224 et 320) : les
   coupes gardees sont ecrites en NNN.dcm dans un dossier temporaire.
2. Effet de l'image de projection des MPR si on la gardait (info).
3. Mosaiques PNG par genou (4 series OAI) et une etude RSNA crop130 de
   reference, dans data/oai/derived/qc/.

    python runpod/oai/check_sample.py [--src data/oai/raw_sample]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.oai_convert import convert_datasets, convert_series, iter_raw, _read  # noqa: E402
from src.infer.preprocess_dicom import load_series_dicom  # noqa: E402

ORDER = ["SAG_IW_TSE", "COR_MPR", "AX_MPR", "COR_IW_TSE"]
SHOW = [3, 7, 11, 15, 19]


def _write_dcm(datasets, folder: Path) -> None:
    for i, ds in enumerate(datasets):
        ds.save_as(str(folder / f"{i:03d}.dcm"))


def _strip(vol: np.ndarray, label: str) -> np.ndarray:
    row = np.hstack([vol[k] for k in SHOW])
    row = cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
    cv2.putText(row, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return row


def rsna_reference(out_dir: Path) -> None:
    import pandas as pd
    from src.data.dataset_npz import load_volume, resolve_volume_source

    mapping = resolve_volume_source(str(ROOT / "data" / "crop130"))
    s = pd.read_csv(ROOT / "data" / "csv" / "train_series.csv")
    s = s[s.SeriesInstanceUID.isin(mapping)]
    want = [("Sagittal", 1), ("Coronal", 1), ("Axial", 1), ("Coronal", 0)]
    for uid, g in s.groupby("StudyInstanceUID"):
        picks = []
        for plane, fluid in want:
            gp = g[(g.Anatomical_Plane == plane) & (g.Fluid_Sensitive == fluid)]
            if gp.empty:
                break
            picks.append((f"{plane[:3].upper()} fluid={fluid}",
                          sorted(gp.SeriesInstanceUID)[0]))
        if len(picks) == 4:
            rows = [_strip(load_volume(mapping[sid], sid), lab) for lab, sid in picks]
            cv2.imwrite(str(out_dir / "rsna_reference.png"), np.vstack(rows))
            print(f"reference RSNA : {uid[-12:]}")
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "oai" / "raw_sample"))
    ap.add_argument("--out", default=str(ROOT / "data" / "oai" / "derived" / "qc"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(Path(args.src).rglob("*.tar.gz"))
    if not archives:
        raise SystemExit(f"aucune archive dans {args.src}")
    by_knee: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    n_bad = 0
    for tgz in archives:
        outs, meta = convert_series(tgz)
        all_ds = [d for d in (_read(raw) for _n, raw in iter_raw(tgz))
                  if d is not None and "PixelData" in d]
        kept = [d for d in all_ds
                if "PROJECTION IMAGE" not in str(d.get("ImageType", "")).upper()]
        desc = str(kept[0].get("SeriesDescription", tgz.stem))
        stype, side = desc.rsplit("_", 1)
        knee = f"{tgz.parent.parent.name}_{side[0]}"

        with tempfile.TemporaryDirectory() as tmp:
            _write_dcm(kept, Path(tmp))
            same = {s: bool(np.array_equal(outs[s], load_series_dicom(tmp, img_size=s)))
                    for s in (224, 320)}
        n_bad += not all(same.values())

        note = ""
        if len(all_ds) != len(kept):
            with_proj, _ = convert_datasets(all_ds, sizes=(224,))
            diff = np.abs(with_proj[224].astype(int) - outs[224].astype(int))
            note = (f" | avec projection : {int((diff > 0).mean() * 100)} % pixels "
                    f"differents, max {int(diff.max())}")
        print(f"{tgz.stem:>10s} {desc:18s} fichiers {meta['n_files'] + meta['n_skipped']:3d} "
              f"coupes {meta['n_raw']:3d} bande {meta['band']} src {meta['src_hw']} "
              f"egal 224/320 {same[224]}/{same[320]}{note}")
        by_knee[knee][stype] = outs[224]

    for knee, vols in by_knee.items():
        rows = [_strip(vols[t], t) for t in ORDER if t in vols]
        cv2.imwrite(str(out_dir / f"oai_{knee}.png"), np.vstack(rows))
    rsna_reference(out_dir)
    print(f"mosaiques : {out_dir}")
    if n_bad:
        raise SystemExit(f"{n_bad} serie(s) differente(s) de load_series_dicom")
    print("OK : conversion identique a load_series_dicom pour toutes les series")


if __name__ == "__main__":
    main()
