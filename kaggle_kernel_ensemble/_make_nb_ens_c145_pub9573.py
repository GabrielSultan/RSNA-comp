# -*- coding: utf-8 -*-
"""Mix soft des 4 modeles + solution publique 0.9573, rang 60/40.

Meme squelette que _make_nb_ens_c145_pub.py. La moitie publique est
heliosli/0-9573 (Raptor 384 + Raptor 224, puis R7), pas le 0.956.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SOFT = KERNEL / "_push_ens_c145_soft" / "submit.ipynb"
PUBLIC = ROOT / "0-9573-latest-public-dataset-score-in-one-click.ipynb"
SLUG = "rsna-knee-ens-c145-pub9573"
PUSH = KERNEL / "_push_ens_c145_pub9573"

MD = """# RSNA Knee — rsna-knee-ens-c145-pub9573

**Sans TTA.** Internet off, **T4**.

1. Mix souple des 4 modeles (softmax hybrid025 T=0.05, sans plancher).
   ViT-k320, v20b et CoAt : SWA OAI 1-3, FOV 130 mm.
   v20h crop145 : SWA epochs 6-8, FOV 145 mm.
   Budget 2.5 h, puis la memoire GPU est liberee.
2. Solution publique 0.9573 (Raptor CoAtNet-384 + Raptor 224, A224=0.35,
   puis ConvNeXt R7), datasets
   `nartaa/rsna-knee-publication-swa-weights-20261007`,
   `heliosli/rsna-knee-compact-r7cnxfsss`,
   `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`.
   Echeance 8.3 h depuis le debut du notebook.
3. Fusion par rangs, **60 %** le mix des 4 modeles et **40 %** la solution publique,
   par finding. Une etude non scoree par le mix garde le rang public.

Pas de flip sur les 4 modeles.
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


def _patch_soft(cell: dict) -> None:
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


def _swap_bundle(cells: list[dict]) -> None:
    """Le dataset gele a une v2 : Kaggle le traite comme non versionne.

    compact-r7 v1 a le meme manifest (sha256 c4bb0f73) et les memes fichiers.
    """
    old = "C_SLUG = 'rsna-knee-r6-frozen-fe055406'"
    new = "C_SLUG = 'rsna-knee-compact-r7cnxfsss'"
    n = 0
    for cell in cells:
        src = _text(cell)
        if old not in src:
            continue
        if src.count(old) != 1:
            raise SystemExit(f"C_SLUG x{src.count(old)}")
        _set(cell, src.replace(old, new, 1))
        n += 1
    if n != 1:
        raise SystemExit(f"C_SLUG remplace x{n}")


def _patch_public(cells: list[dict]) -> None:
    run = _text(cells[3])
    old = (
        "NB_T0 = time.time()\n"
        "rc = subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600).returncode\n"
    )
    new = (
        "NB_T0 = globals().get('NB_T0', time.time())\n"
        "_t384 = time.time()\n"
        "rc = subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600).returncode\n"
    )
    if run.count(old) != 1:
        raise SystemExit(f"NB_T0 x{run.count(old)}")
    run = run.replace(old, new, 1)
    old_sec = "RAPTOR_SECONDS = time.time() - NB_T0\n"
    new_sec = "RAPTOR_SECONDS = time.time() - _t384\n"
    if run.count(old_sec) != 1:
        raise SystemExit("RAPTOR_SECONDS introuvable")
    _set(cells[3], run.replace(old_sec, new_sec, 1))
    blend = _text(cells[8])
    old_raise = (
        "    raise RuntimeError('our stage did not score %d of %d studies "
        "(more than 50 %%)' % (our_fail.sum(), n))\n"
    )
    new_raise = (
        "    print('PUBLIC_OURS_PARTIAL', int(our_fail.sum()), 'of', n, flush=True)\n"
    )
    if blend.count(old_raise) != 1:
        raise SystemExit("garde 50% introuvable")
    _set(cells[8], blend.replace(old_raise, new_raise, 1))


def main() -> None:
    soft = json.loads(SOFT.read_text(encoding="utf-8"))
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    if len(public["cells"]) != 9:
        raise SystemExit(f"notebook 0.9573 : {len(public['cells'])} cellules")
    _patch_soft(soft["cells"][7])
    _patch_public(public["cells"])
    _swap_bundle(public["cells"])
    cells = [_md(MD), _code(CLOCK)]
    cells.extend(soft["cells"][1:8])
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
        "swa-6-7-8",
        "CROP_MM = 145.0",
        "MIX4_BUDGET_S",
        "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt",
        "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt",
        "rsna-knee-compact-r7cnxfsss",
        "A224 = 0.35",
        "BLEND5_OK",
        "W_MIX4, W_PUBLIC = 0.60, 0.40",
        "sans TTA",
        "PUBLIC_OURS_PARTIAL",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    if "submission.csv manquant" in blob:
        raise SystemExit("l ancien writer de submission est encore la")
    if "more than 50 %%" in blob:
        raise SystemExit("la garde 50% du public est encore la")
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
        "id": f"sultent/{SLUG}",
        "title": SLUG,
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-knee-compact-r7cnxfsss",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
            "sultent/rsna-knee-vit-k320-oai3-swa",
            "sultent/rsna-knee-v20b-dino-oai3-swa",
            "sultent/rsna-knee-v20h-crop145-swa678",
            "sultent/rsna-knee-coat-a-oai3-swa",
        ],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("ok", PUSH, "cells", len(cells), "bytes", (PUSH / "submit.ipynb").stat().st_size)


if __name__ == "__main__":
    main()
