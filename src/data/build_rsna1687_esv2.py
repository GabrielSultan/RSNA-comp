# -*- coding: utf-8 -*-
"""Train RSNA 1687 sur les soft labels expert-stressed v2, val hybrid.

Meme effectif et memes UID que targets_v3b_rsna1687_distil_main.csv
(taille du train hybrid025, uniquement RSNA, hors val et hors gold).
Les 11 cibles du zip remplacent les labels v3b du train. Synovitis,
absente du zip, reste le label v3b. La val est copiee depuis
targets_v3b_hybrid025.csv.

    python -X utf8 -m src.data.build_rsna1687_esv2
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset_npz import gold_uids_from_train_csv
from src.nlp.lexicon import TARGETS

ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = Path(r"C:\Users\sulta\Downloads\archive (2).zip")
SRC = ROOT / "data/labels/expertsoft_v2"
HYBRID = ROOT / "data/labels/coolin/targets_v3b_hybrid025.csv"
DISTIL = ROOT / "data/labels/coolin/targets_v3b_rsna1687_distil_main.csv"
OUT_DIR = ROOT / "data/labels/coolin"
CFG_DIR = ROOT / "configs"
META = OUT_DIR / "targets_v3b_rsna1687_esv2.json"
N_TRAIN = 1687
ELEVEN = [t for t in TARGETS if t != "Synovitis"]
EXTRACT = [
    "soft_labels.csv",
    "calibrated_soft_labels.csv",
    "accuracy_soft_labels.csv",
    "strict_mri_oof_soft_labels.csv",
    "accuracy_accepted60.csv",
    "study_metadata.csv",
    "DATASET_CARD.txt",
]
VARIANTS = [
    {
        "key": "soft",
        "file": "soft_labels.csv",
        "mask": False,
        "blurb": (
            "soft_labels.csv : cibles continues orientees AUC "
            "(4407 etudes). weak_labels.csv est identique sur les "
            "etudes non gold, donc pas de run separe."
        ),
    },
    {
        "key": "calibrated",
        "file": "calibrated_soft_labels.csv",
        "mask": False,
        "blurb": "calibrated_soft_labels.csv : probabilites calibrees.",
    },
    {
        "key": "accuracy",
        "file": "accuracy_soft_labels.csv",
        "mask": False,
        "blurb": "accuracy_soft_labels.csv : variante orientee accuracy.",
    },
    {
        "key": "strictmri",
        "file": "strict_mri_oof_soft_labels.csv",
        "mask": False,
        "blurb": (
            "strict_mri_oof_soft_labels.csv : scores MRI OOF, "
            "hors ensemble expose aux weak labels."
        ),
    },
    {
        "key": "acc60",
        "file": "accuracy_soft_labels.csv",
        "mask": True,
        "blurb": (
            "accuracy_soft_labels.csv masque par accuracy_accepted60 : "
            "NaN sur les cellules non retenues (~60 % de couverture)."
        ),
    },
]


def _extract() -> None:
    if not ZIP_PATH.is_file():
        raise SystemExit(f"zip absent : {ZIP_PATH}")
    SRC.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        for name in EXTRACT:
            dest = SRC / name
            dest.write_bytes(z.read(name))


def _read_labels(name: str) -> pd.DataFrame:
    df = pd.read_csv(SRC / name)
    df["StudyInstanceUID"] = df.StudyInstanceUID.astype(str)
    missing = [t for t in ELEVEN if t not in df.columns]
    if missing:
        raise SystemExit(f"{name} colonnes manquantes : {missing}")
    if "Synovitis" in df.columns:
        raise SystemExit(f"{name} contient Synovitis, inattendu")
    if df.StudyInstanceUID.duplicated().any():
        raise SystemExit(f"{name} : UID en double")
    return df.set_index("StudyInstanceUID")


def _yaml(key: str, csv_rel: str, blurb: str) -> str:
    run = f"subset_v20h_dino_ep11_v3b_rsna1687_esv2_{key}_crop145"
    cfg = f"kneenet_v20h_crop145_dino_ep11_v3b_rsna1687_esv2_{key}.yaml"
    return f"""# v20h DINO ep11, meme recette que
# subset_v20h_dino_ep11_v3b_hybrid025_wd01_crop145 (wd 0.01, dropout 0.2,
# 4 epochs, cosine 10, LR 1e-4 / 3e-4, crop 145 puis 130).
# Train : 1687 etudes RSNA, les memes UID que
# targets_v3b_rsna1687_distil_main.csv (taille hybrid025, pas d'OAI).
# 11 cibles : {blurb}
# Synovitis : label v3b du subset distil (absente du zip).
# Val : les 652 etudes de targets_v3b_hybrid025.csv, labels inchanges.
# Pas de distillation.
# Juger l'AUC val contre hybrid025 crop145 (0.9584) et le distil
# crop145 (0.9609 a l'epoch 7). Gold = sanity, hors split.
# 3070 8 Go : encode_chunk 8, 2 workers.
#
#   python -X utf8 -m src.train.train --config configs/{cfg}

run_name: {run}
model: kneenet_v20h
n_series: 6
backbone: resnet50
pretrained: true
backbone_weights: checkpoints/v20h_ssl_dino_screen/encoder_ep11.pt
lstm_hidden: 256

labels_csv: {csv_rel}
series_csv: data/oai/derived/train_series_oai.csv
zip_path:
  - data/crop145
  - data/crop130

train_frac: 1.0
val_frac: 0.15
include_gold: false
seed: 42

batch_size: 4
encode_chunk: 8
log_every: 100
num_workers: 2
epochs: 4
cosine_epochs: 10
save_epochs: []
early_stop_patience: 0
skip_gold: false

lr_backbone: 0.0001
lr_head: 0.0003
weight_decay: 0.01
label_smoothing: 0.0
pos_weight_max: null
head_dim: 512
dropout: 0.2
"""


def _build_one(
    spec: dict,
    order: list[str],
    syn: pd.Series,
    val: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    accepted: pd.DataFrame,
) -> dict:
    src = sources[spec["file"]]
    missing = [u for u in order if u not in src.index]
    if missing:
        raise SystemExit(f"{spec['key']} : {len(missing)} UID train absents")
    labels = src.loc[order, ELEVEN].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(labels.to_numpy())
    if spec["mask"]:
        mask = accepted.loc[order, ELEVEN].apply(pd.to_numeric, errors="coerce")
        vals = set(np.unique(mask.to_numpy()))
        if not vals <= {0.0, 1.0}:
            raise SystemExit(f"accepted60 valeurs inattendues : {vals}")
        labels = labels.where(mask == 1.0)
        finite = np.isfinite(labels.to_numpy())
    elif not finite.all():
        raise SystemExit(f"{spec['key']} : NaN dans les 11 cibles train")
    arr = labels.to_numpy()
    if finite.any():
        seen = arr[finite]
        if seen.min() < 0 or seen.max() > 1:
            raise SystemExit(
                f"{spec['key']} hors [0, 1] : {seen.min()} .. {seen.max()}"
            )
    if not finite.any(axis=1).all() and spec["key"] != "acc60":
        raise SystemExit(f"{spec['key']} : etude train sans cible")

    train = pd.DataFrame({"StudyInstanceUID": order})
    for t in ELEVEN:
        train[t] = labels[t].to_numpy()
    train["Synovitis"] = syn.to_numpy()
    train["split"] = "train"
    out = pd.concat([train, val], ignore_index=True)
    out = out[["StudyInstanceUID", "split", *TARGETS]]
    if out.StudyInstanceUID.duplicated().any():
        raise SystemExit(f"{spec['key']} : UID en double")
    if int((out.split == "train").sum()) != N_TRAIN:
        raise SystemExit(f"{spec['key']} train != {N_TRAIN}")
    if int((out.split == "val").sum()) != len(val):
        raise SystemExit(f"{spec['key']} val a bouge")

    csv_name = f"targets_v3b_rsna1687_esv2_{spec['key']}.csv"
    csv_path = OUT_DIR / csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    csv_rel = str(csv_path.relative_to(ROOT)).replace("\\", "/")
    cfg_name = f"kneenet_v20h_crop145_dino_ep11_v3b_rsna1687_esv2_{spec['key']}.yaml"
    (CFG_DIR / cfg_name).write_text(
        _yaml(spec["key"], csv_rel, spec["blurb"]), encoding="utf-8",
    )
    means = {
        t: None if not np.isfinite(labels[t]).any()
        else round(float(labels[t].mean()), 4)
        for t in ELEVEN
    }
    cov = float(finite.mean())
    return {
        "key": spec["key"],
        "source": spec["file"],
        "masked_by_accepted60": bool(spec["mask"]),
        "csv": csv_rel,
        "config": f"configs/{cfg_name}",
        "run_name": f"subset_v20h_dino_ep11_v3b_rsna1687_esv2_{spec['key']}_crop145",
        "train_cell_coverage": round(cov, 4),
        "train_mean": means,
    }


def main() -> None:
    _extract()
    hybrid = pd.read_csv(HYBRID)
    hybrid["StudyInstanceUID"] = hybrid.StudyInstanceUID.astype(str)
    dist = pd.read_csv(DISTIL)
    dist["StudyInstanceUID"] = dist.StudyInstanceUID.astype(str)
    val = hybrid.loc[hybrid.split == "val", ["StudyInstanceUID", *TARGETS]].copy()
    dval = dist.loc[dist.split == "val", ["StudyInstanceUID", *TARGETS]].copy()
    if set(val.StudyInstanceUID) != set(dval.StudyInstanceUID):
        raise SystemExit("val distil != val hybrid")
    merged = val.merge(dval, on="StudyInstanceUID", suffixes=("_h", "_d"))
    for t in TARGETS:
        a = pd.to_numeric(merged[f"{t}_h"], errors="coerce")
        b = pd.to_numeric(merged[f"{t}_d"], errors="coerce")
        if not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True):
            raise SystemExit(f"val distil != hybrid sur {t}")
    if len(val) != 652:
        raise SystemExit(f"val {len(val)} != 652")
    val = val.copy()
    val["split"] = "val"
    val_ids = set(val.StudyInstanceUID)
    if any(u.startswith("OAI_") for u in val_ids):
        raise SystemExit("OAI dans la val")

    train = dist.loc[dist.split == "train"].copy()
    if len(train) != N_TRAIN:
        raise SystemExit(f"distil train {len(train)} != {N_TRAIN}")
    if train.StudyInstanceUID.duplicated().any():
        raise SystemExit("distil train : UID en double")
    order = train.StudyInstanceUID.tolist()
    train_ids = set(order)
    if train_ids & val_ids:
        raise SystemExit("fuite : train intersecte la val")
    if any(u.startswith("OAI_") for u in train_ids):
        raise SystemExit("OAI dans le train")
    gold = gold_uids_from_train_csv(str(HYBRID))
    if train_ids & gold or val_ids & gold:
        raise SystemExit("gold dans le split")
    syn = pd.to_numeric(train["Synovitis"], errors="coerce")
    if syn.isna().any():
        raise SystemExit("Synovitis v3b manquante sur le train")
    syn.index = order

    sources = {name: _read_labels(name) for name in {
        spec["file"] for spec in VARIANTS
    }}
    accepted = _read_labels("accuracy_accepted60.csv")
    soft_ids = set(sources["soft_labels.csv"].index)
    if not train_ids <= soft_ids:
        raise SystemExit(
            f"train hors soft labels : {len(train_ids - soft_ids)}"
        )

    meta_rows = []
    for spec in VARIANTS:
        meta_rows.append(_build_one(spec, order, syn, val, sources, accepted))

    meta_src = pd.read_csv(SRC / "study_metadata.csv")
    meta_src["StudyInstanceUID"] = meta_src.StudyInstanceUID.astype(str)
    meta_i = meta_src.set_index("StudyInstanceUID")
    val_groups = set(meta_i.loc[meta_i.index.intersection(val_ids), "normalized_report_group"])
    tr_meta = meta_i.loc[meta_i.index.intersection(train_ids)]
    n_leak = int(tr_meta.normalized_report_group.isin(val_groups).sum())

    meta = {
        "zip": str(ZIP_PATH),
        "extracted": str(SRC.relative_to(ROOT)).replace("\\", "/"),
        "n_train": N_TRAIN,
        "n_val": len(val_ids),
        "n_oai": 0,
        "train_uids": "memes UID que targets_v3b_rsna1687_distil_main.csv split=train",
        "val_source": "data/labels/coolin/targets_v3b_hybrid025.csv",
        "synovitis": "label v3b du subset distil (colonne absente du zip)",
        "eleven": ELEVEN,
        "report_group_overlap_train_val": n_leak,
        "variants": meta_rows,
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
