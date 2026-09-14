#!/usr/bin/env bash
# Run 1 plan : ViT-k LLRD 0.8 subset. tmux conseille.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p checkpoints/subset_vit_k_llrd08_frac025
exec python -X utf8 -m src.train.train \
  --config configs/kneenet_vit_k_llrd_subset.yaml \
  "$@"
