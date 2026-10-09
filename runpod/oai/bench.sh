#!/usr/bin/env bash
# Banc GPU sur la vraie boucle de train : bash runpod/oai/bench.sh cfg.yaml [steps]
# Echantillonne nvidia-smi toutes les 0,5 s pendant les steps mesures et
# resume puissance / util / VRAM (cibles docs : ViT ~0,75 s et >= 420 W).
set -uo pipefail
cd /workspace
CFG=$1; STEPS=${2:-40}
NAME=$(basename "$CFG" .yaml)
OUT=/workspace/logs/bench_$NAME
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits \
  -lms 500 > "$OUT.smi" &
SMI=$!
python -X utf8 -m src.train.train --config "$CFG" --max-steps "$STEPS" > "$OUT.log" 2>&1
RC=$?
kill "$SMI" 2>/dev/null
rm -rf "checkpoints/$(sed -n 's/^run_name:[[:space:]]*//p' "$CFG" | tr -d ' \r')_bench"
grep -E "bench :|Error|error|Traceback|out of memory" "$OUT.log" | tail -5
python - "$OUT.smi" <<'PY'
import sys
rows = [[float(v) for v in l.split(",")] for l in open(sys.argv[1]) if l.strip() and "N/A" not in l]
rows = rows[len(rows) // 2:]
if rows:
    med = lambda i: sorted(r[i] for r in rows)[len(rows) // 2]
    print(f"smi (2e moitie, {len(rows)} points) : util {med(0):.0f} % | {med(1):.0f} W | "
          f"VRAM {max(r[2] for r in rows) / 1024:.1f} Go")
PY
echo "bench rc=$RC"
