#!/bin/bash
set -euo pipefail
cd /workspace
if ps -eo comm,args | awk '$1=="python" && /src.train.train/ {found=1} END {exit !found}'; then
  echo "train deja lance"
  exit 0
fi
mkdir -p /workspace/logs
python - <<'PY'
import subprocess
subprocess.Popen(
    ["python", "-X", "utf8", "-m", "src.train.train",
     "--config", "configs/hybrid025/kneenet_vit_k_llrd08_320_v3b_hybrid025.yaml"],
    cwd="/workspace",
    stdout=open("/workspace/logs/hybrid_vitk320.log", "w"),
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
print("TRAIN_STARTED")
PY
sleep 8
ps -eo pid,etime,comm,args | awk '$3=="python" && /src.train.train/'
echo "=== LOG ==="
wc -l /workspace/logs/hybrid_vitk320.log
tail -n 20 /workspace/logs/hybrid_vitk320.log
