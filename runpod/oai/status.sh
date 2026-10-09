#!/usr/bin/env bash
# Etat du pod en un appel : processus, GPU, logs, snapshots, disques.
cd /workspace
echo "== $(date -Is)"
ps -eo pid,etimes,pcpu,rss,args --sort=-pcpu | grep -E "src.train.train|pipeline.py|downloadcmd|_pod_download|queue.sh" \
  | grep -v grep | cut -c1-150 | head -12
echo "== gpu"
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,temperature.gpu --format=csv,noheader
echo "== queues"
tail -n 3 logs/queue_*.log 2>/dev/null
echo "== trains"
for f in logs/train_*.log; do
  [ -f "$f" ] || continue
  echo "-- $f"
  grep -E "^epoch [0-9]+/|step [0-9]+/|Traceback|Error|out of memory|!!" "$f" | tail -n 3
done
echo "== snapshots"
ls -l --time-style=+%H:%M checkpoints/*_oai3/epoch*.pt checkpoints/*_oai3/history.json 2>/dev/null | awk '{print $5, $6, $7}'
echo "== oai"
tail -n 2 logs/oai_pipeline.log 2>/dev/null
ls data/crop130/oai_crop130_b*.zip 2>/dev/null | wc -l | xargs echo "shards 224 :"
ls data/crop130_320/oai_crop130_320_b*.zip 2>/dev/null | wc -l | xargs echo "shards 320 :"
tail -n 2 logs/fetch_rsna.log 2>/dev/null
echo "== disques"
df -h /workspace / | sed 1d
du -sh /root/oai_raw 2>/dev/null
