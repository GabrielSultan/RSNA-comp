#!/bin/bash
set -euo pipefail
cd /workspace
pkill -f 'src.train.train' 2>/dev/null || true
sleep 1
setsid bash runpod/train.sh >/workspace/train.log 2>&1 </dev/null &
sleep 15
pgrep -af src.train.train || echo "NO_TRAIN_PROC"
echo '---LOG---'
tail -n 80 /workspace/train.log
