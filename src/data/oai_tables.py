# -*- coding: utf-8 -*-
"""OAI visite V00 -> tables au format RSNA (series, 12 cibles, liens S3).

Sorties dans data/oai/derived/ (ignore par git : donnees OAI non
redistribuables) :
- oai_series.csv : schema train_series.csv + stype, barcode, image_file,
  file_size, s3_url, batch.
- targets_oai.csv : StudyInstanceUID + 12 cibles, NaN = masque.
- train_series_oai.csv, targets_v3b_oofmask_oai.csv : RSNA + OAI.
- links_v00.txt : une URL S3 par archive (downloadcmd -t).
- prevalence_oai.csv : comptes par cible.

StudyInstanceUID = OAI_{id}_{R|L}, SeriesInstanceUID = OAI_{barcode}
(nom de l'archive .tar.gz, unique par serie).

Regles MOAKS -> cibles (definitions du prompt de src/nlp/gemini_labeler.py) :
- Menisque : morphologie >= 2 (dechirures 2-5, maceration 6-8) ou racine
  posterieure dechiree -> 1 ; les 3 regions <= 1 (1 = signal sans
  dechirure) -> 0 ; sinon masque.
- OA medial / lateral / PF -> 1 si cartilage taille >= 2 (>= 10 % de la
  sous-region) ou perte pleine epaisseur, osteophyte IRM >= 2, ou (TF)
  pincement radio >= 2 / osteophyte radio >= 2. -> 0 si cartilage a 0
  partout (>= 3 sous-regions lues), osteophytes IRM a 0 s'ils sont lus,
  et (TF) pas de pincement radio avec KL <= 1. Le reste est masque.
- Effusion (m_eff_wk) et Synovitis (m_syic, Hoffa) : 0 -> 0, >= 2 -> 1,
  1 masque (frontiere differente entre MOAKS et les comptes rendus).
- Baker's : m_popcys.
- ACL : 0 -> 0, 2 -> 1. Le code 1 (partiel ou degeneratif mucoide, BML a
  l'insertion surrepresentee) et les ligamentoplasties sont masques.
- MCL, Contusion, Fracture : masques (non lus ; BML degeneratives).
Plusieurs lectures d'un genou (projets 65, 22, 30, 63*) : valeur gardee si
elles concordent, masquee sinon.

    python -m src.data.oai_tables
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp.lexicon import TARGETS

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "data" / "oai" / "tabular"
OUT = ROOT / "data" / "oai" / "derived"
NDA_PKG = Path.home() / "NDA" / "nda-tools" / "downloadcmd" / "packages" / "1249964"

# stype image03 -> (plan, fluid-sensitive = fat-suppressed comme chez RSNA)
STYPES: dict[str, tuple[str, int]] = {
    "SAG_IW_TSE": ("Sagittal", 1),
    "COR_MPR": ("Coronal", 1),      # DESS water excitation
    "AX_MPR": ("Axial", 1),         # DESS water excitation
    "COR_IW_TSE": ("Coronal", 0),
}
SIDE_CODE = {1: "R", 2: "L"}
MOAKS_INFO = ("m_comments", "m_technical")

MEN_REGIONS = ("ah", "body", "ph")
CART = {
    "Medial OA": ("cmf", "pmf", "amt", "cmt", "pmt"),
    "Lateral OA": ("clf", "plf", "alt", "clt", "plt"),
    "PF OA": ("alf", "amf", "lp", "mp"),
}
OSTEO = {
    "Medial OA": ("fmc", "fmp", "tm"),
    "Lateral OA": ("flc", "flp", "tl"),
    "PF OA": ("fla", "fma", "pi", "pl", "pm", "ps"),
}
XR = {  # pincement, osteophytes femur / tibia
    "Medial OA": ("xrjsm", "xrosfm", "xrostm"),
    "Lateral OA": ("xrjsl", "xrosfl", "xrostl"),
}


def read_nda(path: Path, **kw) -> pd.DataFrame:
    """Tables NDA : 2e ligne = descriptions des colonnes."""
    return pd.read_csv(path, sep="\t", skiprows=[1], low_memory=False, **kw)


def _grade(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").where(lambda x: x >= 0)


def _tri(pos: pd.Series, neg: pd.Series) -> pd.Series:
    """1 si pos, 0 si neg (et pas pos), NaN sinon."""
    out = pd.Series(np.nan, index=pos.index)
    out[neg.fillna(False).astype(bool)] = 0.0
    out[pos.fillna(False).astype(bool)] = 1.0
    return out


def _cut(x: pd.Series, neg_max: float, pos_min: float) -> pd.Series:
    """<= neg_max -> 0, >= pos_min -> 1, entre les deux ou NaN -> NaN."""
    return _tri(x >= pos_min, x <= neg_max)


def moaks_row_labels(m: pd.DataFrame, xr: pd.DataFrame) -> pd.DataFrame:
    """Une ligne MOAKS -> 12 cibles. xr aligne sur m (memes index)."""
    lab = pd.DataFrame(np.nan, index=m.index, columns=TARGETS)

    for tgt, pre in (("Medial Meniscus", "mm"), ("Lateral Meniscus", "lm")):
        morph = _grade(m, [f"m_{pre}{r}_morph" for r in MEN_REGIONS])
        root = pd.to_numeric(m[f"m_{pre}pr_tear"], errors="coerce")
        pos = (morph >= 2).any(axis=1) | (root == 1)
        neg = morph.notna().all(axis=1) & (morph <= 1).all(axis=1) & (root != 1)
        lab[tgt] = _tri(pos, neg)

    for tgt, regions in CART.items():
        cm = _grade(m, [f"m_cm_{r}" for r in regions])
        tenths = (cm * 10).round()
        size = tenths // 10
        full = tenths % 10
        osm = _grade(m, [f"m_os_{r}" for r in OSTEO[tgt]]).max(axis=1)
        pos = (size >= 2).any(axis=1) | (full >= 1).any(axis=1) | (osm >= 2)
        neg = ((cm.notna().sum(axis=1) >= 3) & (size.fillna(0) == 0).all(axis=1)
               & (full.fillna(0) == 0).all(axis=1) & (osm.isna() | (osm == 0)))
        if tgt in XR:
            jsn, osf, ost = (xr[c] for c in XR[tgt])
            xr_pos = (jsn >= 2) | (np.fmax(osf, ost) >= 2)
            xr_ok = xr["xrkl"].isna() | ((jsn == 0) & (xr["xrkl"] <= 1))
            pos = pos | xr_pos
            neg = neg & xr_ok
        lab[tgt] = _tri(pos, neg)

    lab["Effusion"] = _cut(pd.to_numeric(m["m_eff_wk"], errors="coerce"), 0, 2)
    lab["Synovitis"] = _cut(pd.to_numeric(m["m_syic"], errors="coerce"), 0, 2)
    lab["Baker's"] = _cut(pd.to_numeric(m["m_popcys"], errors="coerce"), 0, 1)
    acl = _cut(pd.to_numeric(m["m_acl_tear"], errors="coerce"), 0, 2)
    acl[pd.to_numeric(m["m_acl_repair"], errors="coerce") == 1] = np.nan
    lab["ACL"] = acl
    return lab


def _agree(s: pd.Series) -> float:
    v = s.dropna().unique()
    return float(v[0]) if len(v) == 1 else np.nan


def build_targets(tab: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (cibles par genou, detail par lecture pour le rapport)."""
    m = read_nda(tab / "oai_kmrisemiquant01.txt")
    m = m[m.visit == "V00"]
    mcols = [c for c in m.columns if c.startswith("m_") and c not in MOAKS_INFO]
    m = m[m[mcols].notna().any(axis=1)].copy()
    m["knee"] = [f"OAI_{int(s)}_{SIDE_CODE[int(d)]}"
                 for s, d in zip(m.src_subject_id, m.side)]

    x = read_nda(tab / "oai_kxrsemiquant01.txt")
    x = x[(x.visit == "V00") & x.side.isin(SIDE_CODE)].copy()
    x["knee"] = [f"OAI_{int(s)}_{SIDE_CODE[int(d)]}"
                 for s, d in zip(x.src_subject_id, x.side)]
    x["_prio"] = (x.readprj.astype(str) != "15").astype(int)  # 15 = lecture centrale
    xcols = ["xrkl", "xrjsm", "xrjsl", "xrosfm", "xrostm", "xrosfl", "xrostl"]
    x = (x.sort_values(["knee", "_prio"]).drop_duplicates("knee")
         .set_index("knee")[xcols].apply(pd.to_numeric, errors="coerce"))
    xr = x.reindex(m.knee.values)
    xr.index = m.index

    rows = moaks_row_labels(m, xr)
    rows.insert(0, "knee", m.knee.values)
    rows.insert(1, "readprj", m.readprj.astype(str).values)
    knees = rows.groupby("knee")[TARGETS].agg(_agree)
    knees.index.name = "StudyInstanceUID"
    return knees.reset_index(), rows


def build_series(knees: set[str], metadata_gz: Path) -> pd.DataFrame:
    im = read_nda(ROOT / "data" / "oai" / "image03" / "image03.txt",
                  usecols=["src_subject_id", "visit", "image_file",
                           "image_description", "accession_number"])
    im = im[im.visit == "V00"]
    parts = im.image_description.str.extract(r"^(.*)_(RIGHT|LEFT)$")
    im = im.assign(stype=parts[0], side=parts[1].map({"RIGHT": "R", "LEFT": "L"}))
    im = im[im.stype.isin(STYPES)].copy()
    im["StudyInstanceUID"] = [f"OAI_{int(s)}_{d}" for s, d in zip(im.src_subject_id, im.side)]
    im = im[im.StudyInstanceUID.isin(knees)]
    im["barcode"] = im.image_file.str.extract(r"/(\d+)\.tar\.gz$")[0]
    im["accession"] = im.accession_number.astype("int64").astype(str)

    meta = read_nda(TAB / "oai_mrimeta01.txt", usecols=["visit", "barcode", "qcreslt"])
    meta = meta[(meta.visit == "V00") & meta.barcode.notna()]
    qc = dict(zip(meta.barcode.astype("int64").astype(str), meta.qcreslt))
    im["qc"] = im.accession.map(qc)
    # Repetitions du meme jour : la serie retenue au QC, sinon la derniere.
    im["_rank"] = (~im.qc.isin(["Y", "YD"])).astype(int)
    im = (im.sort_values(["StudyInstanceUID", "stype", "_rank", "barcode"],
                         ascending=[True, True, True, False])
          .drop_duplicates(["StudyInstanceUID", "stype"]))

    with gzip.open(metadata_gz, "rt") as f:
        pkg = pd.read_csv(f, usecols=["NDA_S3_URL", "FILE_SIZE", "DOWNLOAD_ALIAS"])
    pkg = pkg.drop_duplicates("DOWNLOAD_ALIAS").set_index("DOWNLOAD_ALIAS")
    im["s3_url"] = im.image_file.map(pkg.NDA_S3_URL)
    im["file_size"] = im.image_file.map(pkg.FILE_SIZE)
    missing = int(im.s3_url.isna().sum())
    if missing:
        raise SystemExit(f"{missing} series absentes du package {metadata_gz.name}")

    im["SeriesInstanceUID"] = "OAI_" + im.barcode
    im["Anatomical_Plane"] = im.stype.map(lambda t: STYPES[t][0])
    im["Fluid_Sensitive"] = im.stype.map(lambda t: STYPES[t][1])
    im["Fat_Suppression"] = im.Fluid_Sensitive
    cols = ["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive",
            "Fat_Suppression", "Anatomical_Plane", "stype", "barcode", "qc",
            "image_file", "file_size", "s3_url"]
    return im[cols].sort_values(["StudyInstanceUID", "stype"]).reset_index(drop=True)


def assign_batches(series: pd.DataFrame, knees_per_batch: int) -> pd.DataFrame:
    order = {u: i // knees_per_batch
             for i, u in enumerate(sorted(series.StudyInstanceUID.unique()))}
    return series.assign(batch=series.StudyInstanceUID.map(order))


def prevalence(oai: pd.DataFrame, rsna: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    rec = []
    for t in TARGETS:
        y = oai[t]
        rec.append({
            "target": t,
            "oai_labeled": int(y.notna().sum()),
            "oai_pos": int((y == 1).sum()),
            "oai_prev": round(float(y.mean()), 3) if y.notna().any() else np.nan,
            "rsna_soft_mean": round(float(rsna[t].mean()), 3),
            "gold_prev": round(float(gold[t].mean()), 3),
        })
    return pd.DataFrame(rec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels-rsna", default=str(ROOT / "data/labels/coolin/targets_v3b_oofmask.csv"))
    ap.add_argument("--series-rsna", default=str(ROOT / "data/csv/train_series.csv"))
    ap.add_argument("--metadata-gz", default=str(NDA_PKG / "package_file_metadata_1249964.txt.gz"))
    ap.add_argument("--knees-per-batch", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets, reads = build_targets(TAB)
    targets = targets[targets[TARGETS].notna().any(axis=1)]
    series = build_series(set(targets.StudyInstanceUID), Path(args.metadata_gz))
    targets = targets[targets.StudyInstanceUID.isin(set(series.StudyInstanceUID))]
    series = assign_batches(series, args.knees_per_batch)

    rsna_t = pd.read_csv(args.labels_rsna)
    rsna_s = pd.read_csv(args.series_rsna)
    gold = pd.read_csv(ROOT / "data/csv/train.csv").dropna(subset=TARGETS)
    if set(rsna_t.StudyInstanceUID) & set(targets.StudyInstanceUID):
        raise SystemExit("collision d'uid RSNA / OAI")

    series.to_csv(out / "oai_series.csv", index=False)
    targets.to_csv(out / "targets_oai.csv", index=False)
    pd.concat([rsna_s, series[list(rsna_s.columns)]], ignore_index=True).to_csv(
        out / "train_series_oai.csv", index=False)
    pd.concat([rsna_t, targets[list(rsna_t.columns)]], ignore_index=True).to_csv(
        out / "targets_v3b_oofmask_oai.csv", index=False)
    (out / "links_v00.txt").write_text("\n".join(series.s3_url) + "\n", encoding="utf-8")
    reads.to_csv(out / "moaks_reads_labels.csv", index=False)
    prev = prevalence(targets, rsna_t, gold)
    prev.to_csv(out / "prevalence_oai.csv", index=False)

    n_b = series.batch.nunique()
    unknown = series.file_size <= 1  # FILE_SIZE NDA = 1 : taille inconnue
    est = series.file_size.where(~unknown, series.file_size[~unknown].median())
    gb = est.groupby(series.batch).sum() / 1e9
    print(f"genoux : {len(targets)} | series : {len(series)} "
          f"({series.stype.value_counts().to_dict()})")
    print(f"archives : ~{est.sum() / 1e9:.1f} Go ({int(unknown.sum())} tailles inconnues "
          f"estimees a la mediane) | {n_b} lots, {gb.min():.1f}-{gb.max():.1f} Go par lot")
    print(f"QC mrimeta : {series.qc.value_counts(dropna=False).to_dict()}")
    print(f"lectures MOAKS : {len(reads)} sur {reads.knee.nunique()} genoux")
    print(prev.to_string(index=False))
    print(f"ecrit dans {out}")


if __name__ == "__main__":
    main()
