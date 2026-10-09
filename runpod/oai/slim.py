# -*- coding: utf-8 -*-
"""Pod : copie allegee epochN_w.pt (poids + config, sans AdamW, ~1/3 de la
taille) de chaque snapshot epochN.pt des runs *_oai3, pour un rapatriement
sur une ligne locale lente. Les epochN.pt complets restent sur le pod.

    setsid nohup python runpod/oai/slim.py > logs/slim.log 2>&1 &
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
KEEP = ("model", "config", "epoch", "step", "targets", "val_macro_auc")
N_RUNS, EPOCHS = 4, 3


def main() -> None:
    while True:
        snaps = sorted(ROOT.glob("checkpoints/*_oai3/epoch*.pt"))
        for p in snaps:
            if not re.fullmatch(r"epoch\d+\.pt", p.name):
                continue
            out = p.with_name(p.stem + "_w.pt")
            if out.is_file():
                continue
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            tmp = out.with_name(out.name + ".tmp")
            torch.save({k: ckpt[k] for k in KEEP if k in ckpt}, tmp)
            os.replace(tmp, out)
            print(f"{time.strftime('%H:%M:%S')} {out.relative_to(ROOT)} "
                  f"{out.stat().st_size / 1e6:.0f} Mo (complet {p.stat().st_size / 1e6:.0f} Mo)",
                  flush=True)
        if len(list(ROOT.glob("checkpoints/*_oai3/epoch*_w.pt"))) >= N_RUNS * EPOCHS:
            print("12 copies allegees, fin", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
