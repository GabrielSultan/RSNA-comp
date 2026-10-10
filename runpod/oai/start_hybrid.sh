#!/usr/bin/env bash
# Attend la fin du train en cours (ViT-k oai3), puis enchaine les 4 subset
# hybrides. Un seul GPU : la file est sequentielle.
set -uo pipefail
cd /workspace
LOGS=/workspace/logs
mkdir -p "$LOGS"
echo $$ > "$LOGS/hybrid_wait.pid"
say() { echo "$(date -Is) $*" >> "$LOGS/hybrid_wait.log"; }

busy() {
  pgrep -f 'python -X utf8 -m src.train.train' >/dev/null \
    || pgrep -f 'bash runpod/oai/queue.sh' >/dev/null
}

say "attente des poids from scratch et de la fin du train en cours"
while [[ ! -f /workspace/checkpoints/v20h_ssl_dino_screen/encoder_ep11.pt \
      || ! -f /workspace/weights/coatnet_0_rw_224-sw_in1k/model.safetensors \
      || ! -f /workspace/weights/dinov3-vitb16-lvd1689m/model.safetensors \
      || ! -f /workspace/data/labels/coolin/targets_v3b_hybrid025.csv ]]; do
  sleep 20
done
while busy; do
  sleep 20
done
sleep 15
while busy; do
  sleep 20
done
say "GPU libre, lancement hybrid025"
rm -f "$LOGS/hybrid_wait.pid"
exec bash runpod/oai/queue.sh hybrid \
  configs/hybrid025/kneenet_vit_k_llrd08_320_v3b_hybrid025.yaml \
  configs/hybrid025/kneenet_v20b_crop130_dino_ep11_v3b_hybrid025.yaml \
  configs/hybrid025/kneenet_v20h_crop130_dino_ep11_v3b_hybrid025.yaml \
  configs/hybrid025/kneenet_coat_a_llrd08_v3b_hybrid025.yaml
