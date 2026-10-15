# -*- coding: utf-8 -*-
"""Mix lineaire des 4 SWA + solution publique 0.956, rang 60/40. Sans TTA."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
OURS = KERNEL / "_push_ens_vitk89_c145" / "submit.ipynb"
PUBLIC = ROOT / "rsna-knee-abnormality-detection.ipynb"
SLUG = "rsna-knee-ens-vitk89-pub"
PUSH = KERNEL / "_push_ens_vitk89_pub"
OUT = KERNEL / "_out_ens_vitk89_pub"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 3 * 3600
MSG = (
    "ens ViT-k320 vitk OAI3 SWA 8-9 + v20b OAI3 + v20h crop145 SWA 6-8 + CoAt OAI3, "
    "mix lineaire 60% + public 0.956 40%, rang, sans TTA, pas de flip, T4"
)

MD = """# RSNA Knee — rsna-knee-ens-vitk89-pub

**Sans TTA.** Internet off, **T4**. Pas de flip.

1. Mix lineaire des 4 modeles, poids par finding (ViT-k320 / v20b / v20h / CoAt-a).
   ViT-k320 : `vit_k_llrd08_320_v3b_oofmask_vitk_full_oai3` SWA epochs 8-9
   (gold in-sample moyen 0.973). Entree 320, FOV 130 mm.
   v20b DINO : SWA OAI epochs 1-3 (gold 0.970). Entree 224, FOV 130 mm.
   v20h DINO : SWA epochs 6-8 (gold 0.964). Entree 224, FOV 145 mm.
   CoAt-a : SWA OAI epochs 1-3 (gold 0.988). Entree 224, FOV 130 mm.
   Budget 2.5 h, puis la memoire GPU est liberee.
2. Solution publique 0.956 (Raptor CoAtNet-384 SWA + ConvNeXt-S6 R7),
   datasets `nartaa/rsna-knee-publication-swa-weights-20261007`,
   `heliosli/rsna-knee-compact-r7cnxfsss`,
   `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`.
   Echeance 8.3 h depuis le debut du notebook.
3. Fusion par rangs, **60 %** le mix des 4 modeles et **40 %** la solution publique,
   par finding. Une etude non scoree par le mix garde le rang public.
"""

CLOCK = """import time
NB_T0 = time.time()
MIX4_BUDGET_S = 2.5 * 3600
print('horloge depart', flush=True)
print('budget mix4 h', MIX4_BUDGET_S / 3600, '| echeance publique h 8.3', flush=True)
"""

SAVE = """# Scores du mix 4, puis liberation du GPU avant Raptor.
from pathlib import Path as _Out
import gc

out_dir = _Out('/kaggle/working') if _Out('/kaggle/working').exists() else _Out('.')
mix_csv = out_dir / 'mix4_probs.csv'
pred.to_csv(mix_csv, index=False)
print('MIX4_SCORED', len(pred), '|', mix_csv, flush=True)
del model_vit, model_v20b, model_v20h, model_coat
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print('GPU mix4 libere', flush=True)
"""

BLEND = """# Rang 60/40 : 60 % mix4, 40 % submission publique.
import numpy as np
import pandas as pd
from pathlib import Path

W_MIX4, W_PUBLIC = 0.60, 0.40
work = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
pub = pd.read_csv(work / 'submission.csv', dtype={'StudyInstanceUID': str})
mix = pd.read_csv(work / 'mix4_probs.csv', dtype={'StudyInstanceUID': str})
labels = [c for c in pub.columns if c != 'StudyInstanceUID']
assert set(labels) <= set(mix.columns), 'colonnes mix4 incompletes'
pub = pub.set_index('StudyInstanceUID')
mix = mix.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
scored = [u for u in pub.index if u in mix.index]
print('BLEND5 mix4', len(scored), 'public', len(pub), 'poids mix4 0.60 public 0.40', flush=True)
out = pub[labels].copy()
for c in labels:
    pr = pub[c].rank(method='average', pct=True)
    mr = mix[c].reindex(pub.index)
    ok = mr.notna()
    rr = pd.Series(np.nan, index=pub.index, dtype=np.float64)
    if bool(ok.any()):
        rr.loc[ok] = mr.loc[ok].rank(method='average', pct=True).to_numpy()
    blended = W_MIX4 * rr + W_PUBLIC * pr
    blended = blended.fillna(pr)
    out[c] = blended.astype(np.float32)
out.insert(0, 'StudyInstanceUID', pub.index)
out = out.reset_index(drop=True)
dest = work / 'submission.csv'
out.to_csv(dest, index=False)
assert len(out) == len(pub) and np.isfinite(out[labels].to_numpy()).all()
print('BLEND5_OK', dest, 'lignes', len(out), flush=True)
out.head()
"""


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _code(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


def _md(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


def _patch_ours(cell: dict) -> None:
    src = _text(cell)
    old = "    for i, uid in enumerate(test.StudyInstanceUID):\n"
    new = (
        "    for i, uid in enumerate(test.StudyInstanceUID):\n"
        "        if time.time() - NB_T0 > MIX4_BUDGET_S:\n"
        "            print(f'MIX4_BUDGET {i}/{len(test)}', flush=True)\n"
        "            break\n"
    )
    if src.count(old) != 1:
        raise SystemExit(f"boucle mix4 x{src.count(old)}")
    _set(cell, src.replace(old, new, 1))


def _patch_public(cells: list[dict]) -> None:
    run = _text(cells[3])
    old = "NB_T0 = time.time()\n"
    if run.count(old) != 1:
        raise SystemExit(f"NB_T0 x{run.count(old)}")
    _set(cells[3], run.replace(old, "NB_T0 = globals().get('NB_T0', time.time())\n", 1))
    blend = _text(cells[6])
    old_raise = (
        "    raise RuntimeError('our stage did not score %d of %d studies "
        "(more than 50 %%)' % (our_fail.sum(), n))\n"
    )
    new_raise = (
        "    print('PUBLIC_OURS_PARTIAL', int(our_fail.sum()), 'of', n, flush=True)\n"
    )
    if blend.count(old_raise) != 1:
        raise SystemExit("garde 50% introuvable")
    _set(cells[6], blend.replace(old_raise, new_raise, 1))


def write_nb() -> None:
    ours = json.loads(OURS.read_text(encoding="utf-8"))
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    _patch_ours(ours["cells"][7])
    _patch_public(public["cells"])
    cells = [_md(MD), _code(CLOCK)]
    cells.extend(ours["cells"][1:8])
    cells.append(_code(SAVE))
    cells.extend(public["cells"])
    cells.append(_code(BLEND))
    for cell in cells:
        src = _text(cell)
        if cell["cell_type"] != "code" or src.lstrip().startswith("%%"):
            continue
        ast.parse(src)
    blob = "\n".join(_text(c) for c in cells)
    for needle in (
        "swa-8-9",
        "[8, 9]",
        "swa-6-7-8",
        "swa-1-2-3",
        "vit_k_llrd08_320_v3b_oofmask_vitk_full_oai3",
        "vit_k320_vitk_oai3_swa_ep8_9.pt",
        "(0.09, 0.32, 0.55, 0.04)",
        "(0.25, 0.20, 0.28, 0.27)",
        "CROP_MM = 145.0",
        "laterality=False",
        "MIX4_BUDGET_S",
        "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt",
        "rsna-knee-compact-r7cnxfsss",
        "BLEND5_OK",
        "W_MIX4, W_PUBLIC = 0.60, 0.40",
        "sans TTA",
        "gold in-sample moyen 0.973",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "vit_k320_oai3_swa_ep1_3.pt",
        "TTA_VARIANTS",
        "laterality=True",
        "submission.csv manquant",
        "rsna-knee-r6-frozen",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
    )
    meta = {
        "id": KAGGLE_ID,
        "title": SLUG,
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": ACCEL,
        "competition_sources": [COMPETITION],
        "dataset_sources": [
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-knee-compact-r7cnxfsss",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
            "sultent/rsna-knee-vit-k320-vitk-oai3-swa89",
            "sultent/rsna-knee-v20b-dino-oai3-swa",
            "sultent/rsna-knee-v20h-crop145-swa678",
            "sultent/rsna-knee-coat-a-oai3-swa",
        ],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(cells), flush=True)


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        raise


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def push() -> str:
    while True:
        try:
            out = kaggle("kernels", "push", "-p", str(PUSH), "--accelerator", ACCEL, "-t", "32400")
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text):
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if _quota(out):
            print("quota GPU, retry 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        version = m.group(1)
        (PUSH / "_version.txt").write_text(version + "\n", encoding="utf-8")
        print("kernel version", version, flush=True)
        return version


def wait() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KAGGLE_ID).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def _daily_limit(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "maximum number", "submission limit", "daily submission allowance",
        "try again tomorrow", "too many submission", "exceeded",
    ))


def submit(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KAGGLE_ID, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower())
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "swa-8-9",
        "vit_k_llrd08_320_v3b_oofmask_vitk_full_oai3",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop145_dino_ep11_oof_v3b_full_oaiv2_wd01",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "swa-6-7-8",
        "swa-1-2-3",
        "lat False",
        "v20h 145 mm",
        "vit 0.09 v20b 0.32 v20h 0.55 coat 0.04",
        "MIX4_SCORED",
        "GPU mix4 libere",
        "poids mix4 0.60 public 0.40",
        "BLEND5_OK",
        "NOTEBOOK_COMPLETE",
        "sans TTA",
        "lignes 3",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text or "lat True" in text or "v20h_full_oai3" in text:
        raise SystemExit("TTA, flip ou ancien ViT")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    waited = False
    while True:
        print("submit", KAGGLE_ID, "version", version, flush=True)
        try:
            resp = api.competition_submit_code(
                "submission.csv", MSG, COMPETITION, KAGGLE_ID, int(version), quiet=True,
            )
        except Exception as e:
            http = getattr(e, "response", None)
            body = http.text if http is not None else str(e)
            print(body, flush=True)
            if _daily_limit(body) and not waited:
                waited = True
                now = datetime.now(timezone.utc)
                target = now.replace(hour=0, minute=5, second=0, microsecond=0)
                if now >= target:
                    target = target + timedelta(days=1)
                delay = (target - now).total_seconds()
                print(f"ATTENTE_QUOTA {delay:.0f}s", flush=True)
                time.sleep(delay + 5)
                continue
            raise
        print(resp, flush=True)
        print("SOUMIS", KAGGLE_ID, "version", version, flush=True)
        return


def main() -> None:
    write_nb()
    if "--write" in sys.argv:
        return
    version = push()
    wait()
    submit(version)


if __name__ == "__main__":
    main()
