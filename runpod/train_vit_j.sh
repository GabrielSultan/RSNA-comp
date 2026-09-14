#!/usr/bin/env bash
# Run 3 plan : ViT-j fusion [6,9,12] sur recette k (LLRD).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p checkpoints/subset_vit_j_llrd08_frac025
exec python -X utf8 -m src.train.train \
  --config configs/kneenet_vit_j_llrd_subset.yaml \
  "$@"
