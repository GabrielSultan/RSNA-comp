#!/usr/bin/env bash
# Ablation ViT-e + D sur subset. tmux/nohup conseille.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p checkpoints/subset_vit_e_rgb25d_frac025
exec python -X utf8 -m src.train.train \
  --config configs/kneenet_vit_e_rgb25d_subset.yaml \
  "$@"
