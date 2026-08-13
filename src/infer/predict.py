# -*- coding: utf-8 -*-
"""Inference KneeNet sur des etudes DICOM -> submission.csv.

Utilisable tel quel dans le notebook Kaggle (internet off) :
- poids dans un dataset attache (ex. /kaggle/input/<dataset>/best.pt)
- test.csv / test_series.csv / test_series/ fournis par la competition
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset_npz import NORM_MEAN, NORM_STD
from src.infer.preprocess_dicom import study_to_tensor
from src.models.kneenet import KneeNet
from src.nlp.lexicon import TARGETS


def load_model(weights_path: str | Path, device: str = "cuda") -> KneeNet:
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = KneeNet(backbone=cfg.get("backbone", "convnext_tiny"), n_series=3,
                    head_dim=cfg.get("head_dim", 512),
                    dropout=cfg.get("dropout", 0.2), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_studies(model: KneeNet, test_csv: str | Path,
                    test_series_csv: str | Path, test_series_dir: str | Path,
                    device: str = "cuda", tta_hflip: bool = False) -> pd.DataFrame:
    test = pd.read_csv(test_csv)
    series = pd.read_csv(test_series_csv)
    test_series_dir = Path(test_series_dir)

    rows = []
    for uid in test.StudyInstanceUID:
        study_dir = test_series_dir / uid
        srows = series[series.StudyInstanceUID == uid]
        vol = study_to_tensor(study_dir, srows)          # (3, 24, 224, 224) [0,1]
        x = torch.from_numpy(vol).unsqueeze(0).to(device)
        x = (x - NORM_MEAN) / NORM_STD

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
            if tta_hflip:
                logits = 0.5 * (logits + model(torch.flip(x, dims=[4])))
        prob = torch.sigmoid(logits.float()).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
        print(f"  {uid} -> ok", flush=True)

    return pd.DataFrame(rows, columns=["StudyInstanceUID"] + TARGETS)


def write_submission(pred: pd.DataFrame, sample_submission: str | Path | None = None,
                     out: str | Path = "submission.csv") -> None:
    if sample_submission is not None:
        sample = pd.read_csv(sample_submission)
        # ordre des lignes = sample_submission, colonnes = sample_submission
        pred = pred.set_index("StudyInstanceUID").loc[sample.StudyInstanceUID]
        pred = pred.reset_index()[list(sample.columns)]
    pred.to_csv(out, index=False)
    print(f"submission ecrite -> {out} ({len(pred)} lignes)")
