# -*- coding: utf-8 -*-
"""Mix5 sur 2 T4, budget 6,75 h, puis Raptor-384 seul. Rang 80/20."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SPEC = importlib.util.spec_from_file_location(
    "pub70", KERNEL / "_push_ens_cnx_pub70.py",
)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

SLUG = "rsna-knee-ens-cnx-pub80-r384"
PUSH = KERNEL / "_push_ens_cnx_pub80_r384"
OUT = KERNEL / "_out_ens_cnx_pub80_r384"
KAGGLE_ID = f"sultent/{SLUG}"
COMPETITION = base.COMPETITION
ACCEL = base.ACCEL
POLL_S = base.POLL_S
MAX_WAIT_S = base.MAX_WAIT_S
PUBLIC = base.PUBLIC

MSG = (
    "ens ViT-k320 OAI3 SWA 1-3 + v20b OAI SWA + v20h distil crop145 SWA 7-9 "
    "+ CoAt OAI SWA + CNX-r7 SWA 7-10, mix 80% + Raptor-384 20%, rang, "
    "sans TTA, pas de flip, mix sur 2 T4, budget mix 6.75 h, Raptor-384 seul"
)

MD = """# RSNA Knee — rsna-knee-ens-cnx-pub80-r384

**Sans TTA.** Internet off, **deux T4**. Pas de flip.

1. Mix lineaire de 5 modeles, poids par finding
   (ViT-k320 / v20b / v20h distil / CoAt-a / CNX-r7).
   Les 4 premiers gardent les proportions du mix publie.
   CNX-r7 prend, par finding, la part du softmax hybrid T=0.05
   (`subset_cnx_r7_in22k_v3b_hybrid025`, epoch 7). Le reste est
   renormalise sur les 4.
   ViT-k320 : `vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3` SWA 1-3
   (gold 0.990). Entree 320, FOV 130 mm.
   v20b DINO : SWA OAI 1-3 (gold 0.970). Entree 224, FOV 130 mm.
   v20h distil : `v20h_crop145_dino_ep11_v3b_rsna_distil_main_full` SWA 7-9
   (gold 0.969). Entree 224, FOV 145 mm.
   CoAt-a : SWA OAI 1-3 (gold 0.988). Entree 224, FOV 130 mm.
   CNX-r7 : `cnx_r7_in22k_v3b_oofmask_full` SWA 7-10 (gold 0.976).
   Entree 224, FOV 130 mm. ConvNeXt-Small IN-22k.
   Les etudes sont coupees sur les deux T4. Une seule etude est decodee
   a la fois. **Budget 6,75 h** (1 300 etudes a ~17 s en demandent 6,1 h),
   puis la memoire est rendue.
2. Raptor-384 seul (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa`).
   Pas de Raptor-224, pas du reste du notebook public. Timeout 2 h.
3. Fusion par rangs, **80 %** le mix des 5 modeles et **20 %** Raptor-384.
"""

CLOCK = """import time
NB_T0 = time.time()
MIX5_BUDGET_S = 6.75 * 3600
print('horloge depart', flush=True)
print('budget mix5 h', MIX5_BUDGET_S / 3600, '| raptor384 timeout h 2', flush=True)
"""

BLEND = """# Rang 80/20 : 80 % mix5, 20 % Raptor-384.
import numpy as np
import pandas as pd
from pathlib import Path

W_MIX5, W_RAPTOR = 0.80, 0.20
work = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
mix = pd.read_csv(work / 'mix5_probs.csv', dtype={'StudyInstanceUID': str})
raw = pd.read_csv(work / 'raptor_raw.csv', dtype={'StudyInstanceUID': str})
labels = [c for c in mix.columns if c != 'StudyInstanceUID']
assert set(labels) <= set(raw.columns), 'colonnes raptor incompletes'
mix = mix.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
raw = raw.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
order = [str(u) for u in test.StudyInstanceUID]
mix_a = mix.reindex(order)
raw_a = raw.reindex(order)[labels]
present = raw_a.notna().all(axis=1)
rap_fail = present & np.isclose(raw_a.to_numpy(dtype=float), 0.5).all(axis=1)
rap_fail = rap_fail | ~present
print(
    'BLEND80 mix5', int(mix_a[labels[0]].notna().sum()),
    'raptor384', int((~rap_fail).sum()), 'de', len(order),
    'poids mix5 0.80 raptor384 0.20',
    flush=True,
)


def _rank_ok(series, ok):
    out = pd.Series(np.nan, index=series.index, dtype=np.float64)
    if bool(ok.any()):
        out.loc[ok] = series.loc[ok].rank(method='average', pct=True)
    return out


cols = {}
for c in labels:
    mr = _rank_ok(mix_a[c], mix_a[c].notna())
    rr = _rank_ok(raw_a[c], ~rap_fail & raw_a[c].notna())
    blended = W_MIX5 * mr + W_RAPTOR * rr
    blended = blended.fillna(mr).fillna(rr).fillna(0.5)
    cols[c] = blended.astype(np.float32)
sub = pd.DataFrame(cols)
sub.insert(0, 'StudyInstanceUID', order)
dest = work / 'submission.csv'
sub.to_csv(dest, index=False)
assert len(sub) == len(order) and np.isfinite(sub[labels].to_numpy()).all()
print('BLEND80_OK', dest, 'lignes', len(sub), flush=True)
sub.head()
"""


def _patch_raptor384(cell: dict) -> None:
    src = base._text(cell)
    old = (
        "NB_T0 = time.time()\n"
        "rc = subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600).returncode\n"
    )
    new = """NB_T0 = globals().get('NB_T0', time.time())
import gc as _gc
import ctypes as _ct
_gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
try:
    _ct.CDLL('libc.so.6').malloc_trim(0)
except Exception:
    pass
_t384 = time.time()
rc = subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600).returncode
"""
    if src.count(old) != 1:
        raise SystemExit(f"lancement raptor 384 x{src.count(old)}")
    src = src.replace(old, new, 1)
    old_sec = "RAPTOR_SECONDS = time.time() - NB_T0\n"
    if src.count(old_sec) != 1:
        raise SystemExit("RAPTOR_SECONDS introuvable")
    src = src.replace(old_sec, "RAPTOR_SECONDS = time.time() - _t384\n", 1)
    if "print('RAPTOR384_SEUL'" not in src:
        src = src.replace(
            "print('RAPTOR_STAGE_DONE seconds %.0f' % RAPTOR_SECONDS, flush=True)\n",
            "print('RAPTOR_STAGE_DONE seconds %.0f' % RAPTOR_SECONDS, flush=True)\n"
            "print('RAPTOR384_SEUL', flush=True)\n",
            1,
        )
    base._set(cell, src)


def write_nb() -> str:
    rows = base.mix_rows()
    ours = json.loads(base.OURS.read_text(encoding="utf-8"))
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    if len(ours["cells"]) != 9 or len(public["cells"]) != 7:
        raise SystemExit("notebooks source inattendus")
    base._patch_v20h(ours)
    base._patch_cnx(ours, rows)
    ours["cells"].insert(7, base._cnx_cell())
    cells = [base._md(MD), base._code(CLOCK)]
    cells.extend(ours["cells"][1:9])
    cells.append(base._code(base.SAVE))
    _patch_raptor384(public["cells"][1])
    cells.append(public["cells"][0])
    cells.append(public["cells"][1])
    cells.append(base._code(BLEND))
    for cell in cells:
        src = base._text(cell)
        if cell["cell_type"] != "code" or src.lstrip().startswith("%%"):
            continue
        ast.parse(src)
    blob = "\n".join(base._text(c) for c in cells)
    acl = rows[0]
    acl_line = (
        f"vit {acl[0]:.2f} v20b {acl[1]:.2f} v20h {acl[2]:.2f} "
        f"coat {acl[3]:.2f} cnx {acl[4]:.2f}"
    )
    for needle in (
        "MIX5_BUDGET_S = 6.75 * 3600",
        "gpus mix 2 | cuda:0 cuda:1",
        "MIX5_COVERAGE",
        "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt",
        "RAPTOR384_SEUL",
        "W_MIX5, W_RAPTOR = 0.80, 0.20",
        "BLEND80_OK",
        "poids mix5 0.80 raptor384 0.20",
        "sans TTA",
        acl_line.split()[0],
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in (
        "raptor224_run.py",
        "RAPTOR224",
        "DEADLINE_HOURS",
        "W_MIX5, W_PUBLIC = 0.70, 0.30",
        "probe_dual.py",
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
    (PUSH / "submit.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
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
            "sultent/rsna-knee-vit-k320-oai3-swa",
            "sultent/rsna-knee-v20b-dino-oai3-swa",
            "sultent/rsna-knee-v20h-crop145-distil-swa789",
            "sultent/rsna-knee-coat-a-oai3-swa",
            "sultent/rsna-knee-cnx-r7-swa78910",
        ],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    (PUSH / "_acl.txt").write_text(acl_line + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(cells), flush=True)
    return acl_line


def kaggle(*args: str) -> str:
    return base.kaggle(*args)


def push() -> str:
    while True:
        try:
            out = kaggle("kernels", "push", "-p", str(PUSH), "--accelerator", ACCEL, "-t", "32400")
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if base._quota(text):
                print("quota GPU, retry 60s", flush=True)
                time.sleep(60)
                continue
            if base._transient(text):
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
                continue
            raise
        print(out, flush=True)
        if base._quota(out):
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


def submit(version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KAGGLE_ID, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and SLUG in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    acl_line = (PUSH / "_acl.txt").read_text(encoding="utf-8").strip()
    for needle in (
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3",
        "gold 0.9903",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10 lat False gold 0.976",
        "gpus mix 2 | cuda:0 cuda:1",
        "MIX5_COVERAGE 3/3",
        "GPU mix5 libere",
        "RAPTOR_STAGE_DONE",
        "RAPTOR384_SEUL",
        "poids mix5 0.80 raptor384 0.20",
        "BLEND80_OK",
        "lignes 3",
        acl_line,
        "sans TTA",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "RAPTOR224" in text or "TTA x" in text or "probe_dual" in text:
        raise SystemExit("raptor 224, TTA ou ours encore presents")
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
            if base._daily_limit(body) and not waited:
                waited = True
                now = datetime.now(timezone.utc)
                target = now.replace(hour=0, minute=5, second=0, microsecond=0)
                if now >= target:
                    target = target + timedelta(days=1)
                delay = (target - now).total_seconds()
                print(f"ATTENTE_QUOTA {delay:.0f}s", flush=True)
                time.sleep(delay + 5)
                continue
            if base._transient(body):
                time.sleep(20)
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
