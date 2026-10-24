#!/usr/bin/env bash
# Setup RunPod pour l'ablation ViT-e + D (subset).
# Template : PyTorch >= 2.4, CUDA 12.4, Python 3.11/3.12. Workspace = /workspace.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
import sys, torch
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA indisponible"
print("gpu", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} Go")
x = torch.zeros(1, device="cuda")
print("bf16", torch.cuda.is_bf16_supported())
PY

python -m pip install -q --upgrade pip
python -m pip install -q pandas pyyaml "timm>=1.0.15" safetensors

python - <<'PY'
from pathlib import Path
need = [
    "configs/archive/kneenet_vit_e_rgb25d_subset.yaml",
    "src/models/kneenet_vit_e_d.py",
    "data/labels/labels_llm_gpt56sol.csv",
    "data/csv/train_series.csv",
    "data/csv/train.csv",
    "weights/dinov3-vitb16-lvd1689m/model.safetensors",
]
missing = [p for p in need if not Path(p).exists()]
zips = sorted(Path("data/crop130_rgb25d").glob("*.zip")) if Path("data/crop130_rgb25d").is_dir() else []
gb = sum(p.stat().st_size for p in zips) / 1e9
print(f"zips rgb25d : {len(zips)} ({gb:.1f} Go)")
if missing:
    print("MANQUANT :")
    for p in missing:
        print(" ", p)
    raise SystemExit(1)
if len(zips) < 6:
    raise SystemExit(f"attendu 6 zips proto dans data/crop130_rgb25d, recu {len(zips)}")
print("ok")
PY

echo
echo "lancer :  bash runpod/train.sh"
echo "resume :  python -X utf8 -m src.train.train --config configs/archive/kneenet_vit_e_rgb25d_subset.yaml --resume"
