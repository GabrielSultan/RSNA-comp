#!/bin/bash
set -euo pipefail
cd /workspace
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
WTS=342579728
LOG=/workspace/train_vit320.log
echo "attente $(date -Is)" >> /workspace/start_vit320.log
while true; do
  zips=$(find /workspace/data/crop130_320 -name 'volumes_crop130_320_*.zip' | wc -l)
  parts=$(find /workspace/data/crop130_320 -name '*.part' | wc -l)
  wts=$(stat -c %s /workspace/weights/dinov3-vitb16-lvd1689m/model.safetensors 2>/dev/null || echo 0)
  bytes=$(find /workspace/data/crop130_320 -name 'volumes_crop130_320_*.zip' -printf '%s\n' | awk '{s+=$1} END {print s+0}')
  echo "zips=$zips parts=$parts bytes=$bytes wts=$wts $(date -Is)" >> /workspace/start_vit320.log
  if [ "$zips" -eq 6 ] && [ "$parts" -eq 0 ] && [ "$bytes" -eq 41726429237 ] && [ "$wts" -eq "$WTS" ]; then
    break
  fi
  sleep 20
done
rm -f /workspace/kaggle_urls_crop320.json
echo "lancement $(date -Is)" >> /workspace/start_vit320.log
python -X utf8 -m src.train.train \
  --config configs/kneenet_vit_k_llrd08_320_v3b_oofmask_v20h_full.yaml \
  >> "$LOG" 2>&1
echo "train exit $? $(date -Is)" >> /workspace/start_vit320.log
