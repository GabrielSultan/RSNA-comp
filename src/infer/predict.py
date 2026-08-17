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

from src.data.transforms import TTA_VARIANTS, apply_affine
from src.infer.preprocess_dicom import study_to_tensor
from src.models.registry import build_model
from src.nlp.lexicon import TARGETS


def load_model(weights_path: str | Path, device: str = "cuda"):
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    cfg.setdefault("model", "kneenet_v8")
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def _predict_logits(model, x: torch.Tensor, device: str) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        return model(x)


@torch.no_grad()
def predict_studies(model, test_csv: str | Path,
                    test_series_csv: str | Path, test_series_dir: str | Path,
                    device: str = "cuda", tta: bool = True) -> pd.DataFrame:
    """tta=True : moyenne des sigmoides sur TTA_VARIANTS (pas de hflip)."""
    test = pd.read_csv(test_csv)
    series = pd.read_csv(test_series_csv)
    test_series_dir = Path(test_series_dir)
    variants = TTA_VARIANTS if tta else ((0.0, 1.0),)
    print(f"TTA : {len(variants)} variantes (hflip off)", flush=True)

    rows = []
    for uid in test.StudyInstanceUID:
        study_dir = test_series_dir / uid
        srows = series[series.StudyInstanceUID == uid]
        vol = study_to_tensor(
            study_dir, srows, n_series=int(getattr(model, "n_series", 3)),
        )
        x0 = torch.from_numpy(vol).unsqueeze(0).to(device)

        probs = []
        for angle_deg, scale in variants:
            xv = apply_affine(x0, angle_deg=angle_deg, scale=scale)
            logits = _predict_logits(model, xv, device)
            probs.append(torch.sigmoid(logits.float()))
        prob = torch.stack(probs, dim=0).mean(dim=0).cpu().numpy()[0]
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
