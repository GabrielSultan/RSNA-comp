#!/usr/bin/env bash
# Run 2 plan : ViT-e + accum x4 subset.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p checkpoints/subset_vit_e_accum4_frac025
exec python -X utf8 -m src.train.train \
  --config configs/archive/kneenet_vit_e_accum4_subset.yaml \
  "$@"
