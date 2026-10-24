#!/usr/bin/env bash
# ViT-j + drop-path 0.1 (recette k). Un levier vs j/k sans dp.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p checkpoints/subset_vit_j_dp01_frac025
exec python -X utf8 -m src.train.train \
  --config configs/archive/kneenet_vit_j_dp_subset.yaml \
  "$@"
