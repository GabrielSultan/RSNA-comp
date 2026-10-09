#!/usr/bin/env bash
# Pod OAI (image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404), depuis
# /workspace apres extraction de bundle.tar : paquets epingles (memes versions
# qu'en local), keyring fichier pour nda-tools (pas de dbus sur le pod),
# test CUDA, chargement strict des 4 init_*.pt.
set -euo pipefail
cd /workspace
mkdir -p logs data/crop130 data/crop130_320

python -m pip install -q --root-user-action=ignore \
  "timm==1.0.28" "pydicom==3.0.2" "opencv-python-headless==5.0.0.93" \
  "scikit-learn==1.6.1" "pandas==2.2.3" pyyaml safetensors requests \
  "nda-tools==0.7.0" keyrings.alt

mkdir -p /root/.config/python_keyring
cat > /root/.config/python_keyring/keyringrc.cfg <<'CFG'
[backend]
default-keyring=keyrings.alt.file.PlaintextKeyring
CFG

python - <<'PY'
import sys, torch, timm, pydicom, cv2, keyring, NDATools
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "| timm", timm.__version__, "| pydicom", pydicom.__version__, "| cv2", cv2.__version__)
print("keyring", type(keyring.get_keyring()).__name__)
assert torch.cuda.is_available(), "CUDA indisponible"
p = torch.cuda.get_device_properties(0)
print("gpu", p.name, f"{p.total_memory / 1e9:.0f} Go | bf16", torch.cuda.is_bf16_supported())
PY

nproc
free -g | head -2
df -h /workspace / | sed 1d
python -X utf8 runpod/oai/dryrun.py --model-only
