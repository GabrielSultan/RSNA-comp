#!/bin/bash
# Attend zips + poids + epoch9.pt, puis reprend le full train a l'epoch 10.
# epoch9.pt, pas last.pt : une copie locale de last.pt est restee a l'epoch 4.
set -euo pipefail
cd /workspace
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
CKPT=/workspace/checkpoints/vit_k_llrd08_v3b_oofmask_full/epoch9.pt
EP9=1063328319
WTS=342579728
LOG=/workspace/train_vitk_full.log
echo "attente donnees $(date -Is)" >> /workspace/start_vitk.log
while true; do
  zips=$(find /workspace/data/crop130_rgb25d -name 'volumes_crop130_rgb25d_proto_part*.zip' | wc -l)
  parts=$(find /workspace/data/crop130_rgb25d -name '*.part' | wc -l)
  ep9=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
  wts=$(stat -c %s /workspace/weights/dinov3-vitb16-lvd1689m/model.safetensors 2>/dev/null || echo 0)
  echo "zips=$zips parts=$parts ep9=$ep9 wts=$wts $(date -Is)" >> /workspace/start_vitk.log
  if [ "$zips" -eq 6 ] && [ "$parts" -eq 0 ] && [ "$ep9" -eq "$EP9" ] && [ "$wts" -eq "$WTS" ]; then
    break
  fi
  sleep 20
done
echo "lancement epoch9 $(date -Is)" >> /workspace/start_vitk.log
python -X utf8 -m src.train.train \
  --config configs/kneenet_vit_k_llrd08_v3b_oofmask_full.yaml \
  --resume --ckpt "$CKPT" >> "$LOG" 2>&1
echo "train exit $? $(date -Is)" >> /workspace/start_vitk.log
