# -*- coding: utf-8 -*-
import subprocess
import sys

slugs = []
for ep in (10, 11, 12):
    slugs.append(f"sultent/rsna-knee-vit-k-v3b-full-ep{ep}")
    slugs.append(f"sultent/rsna-knee-vit-k-v3b-ep{ep}")

for slug in slugs:
    proc = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "status", slug],
        capture_output=True, text=True,
    )
    text = (proc.stdout + proc.stderr).strip().replace("\n", " ")
    print(slug, "rc", proc.returncode, text[:240])
