#!/usr/bin/env bash
# Attend la fin de ViT-e+D puis lance ViT-k LLRD. Ne pas pgrep ce fichier.
set -euo pipefail
cd /workspace
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
LOG=/workspace/train_vit_k.log
echo "$(date -Is) watcher : attente fin vit_e_d" >> "$LOG"
while pgrep -f "src.train.train --config configs/archive/kneenet_vit_e_rgb25d_subset.yaml" >/dev/null 2>&1; do
  sleep 30
done
echo "$(date -Is) vit_e_d termine, lancement ViT-k" >> "$LOG"
python -X utf8 -m src.train.train --config configs/archive/kneenet_vit_k_llrd_subset.yaml >> "$LOG" 2>&1
echo "$(date -Is) vit_k exit $?" >> "$LOG"
