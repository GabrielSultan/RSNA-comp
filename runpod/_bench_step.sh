set -euo pipefail
ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}' | while read -r p; do kill "$p" || true; done
for _ in 1 2 3 4 5 6 7 8 9 10; do
  left=$(ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}')
  if [ -z "$left" ]; then break; fi
  sleep 1
done
left=$(ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}')
if [ -n "$left" ]; then kill -9 $left || true; sleep 1; fi
cd /workspace
PYTHONPATH=/workspace python -X utf8 runpod/_bench_step.py
