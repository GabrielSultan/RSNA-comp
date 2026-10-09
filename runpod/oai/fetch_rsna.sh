#!/usr/bin/env bash
# Zips RSNA depuis les sorties des kernels Kaggle (URL signees envoyees juste
# avant dans /workspace/urls_crop130.json et urls_crop130_320.json).
# crop130 d'abord (bancs 224), puis crop130_320. Les JSON sont effaces des
# que le telechargement est fini ; les URL ne sont jamais affichees.
set -uo pipefail
cd /workspace
for D in crop130 crop130_320; do
  J=/workspace/urls_$D.json
  [ -f "$J" ] || { echo "$(date -Is) $J absent, saute"; continue; }
  echo "$(date -Is) debut $D"
  python -X utf8 runpod/_pod_download_from_urls.py "$J" "/workspace/data/$D"
  echo "$(date -Is) fin $D rc=$?"
  rm -f "$J"
  python -X utf8 runpod/oai/check_data.py "data/$D" --no-oai
done
