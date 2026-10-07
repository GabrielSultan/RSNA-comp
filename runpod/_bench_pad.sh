set -euo pipefail
python - <<'PY'
from pathlib import Path
p = Path("/workspace/src/models/kneenet_vit_e.py")
t = p.read_text()
old = """
        self.encoder = torch.compile(self.encoder, dynamic=True)
        print("ViT compile (dynamic)", flush=True)
"""
if old not in t:
    raise SystemExit("compile hook missing")
p.write_text(t.replace(old, "\n", 1))
print("compile off")
PY
ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}' | while read -r p; do kill "$p" || true; done
for _ in 1 2 3 4 5 6 7 8; do
  left=$(ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}')
  if [ -z "$left" ]; then break; fi
  sleep 1
done
left=$(ps -eo pid,comm,args | awk '$2=="python" && /src.train.train/ {print $1}')
if [ -n "$left" ]; then kill -9 $left || true; sleep 1; fi
cd /workspace
PYTHONPATH=/workspace PYTHONWARNINGS=ignore CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -X utf8 runpod/_bench_pad.py
