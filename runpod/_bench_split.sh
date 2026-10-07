set -euo pipefail
ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}' | while read -r p; do kill "$p" || true; done
for _ in 1 2 3 4 5 6 7 8; do
  left=$(ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}')
  if [ -z "$left" ]; then break; fi
  sleep 1
done
cd /workspace
export PYTHONPATH=/workspace
export PYTHONWARNINGS=ignore
unset CUBLAS_WORKSPACE_CONFIG || true
unset OMP_NUM_THREADS || true
python -X utf8 runpod/_bench_pipe.py
