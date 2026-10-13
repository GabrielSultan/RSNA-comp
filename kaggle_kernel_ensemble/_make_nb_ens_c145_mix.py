# -*- coding: utf-8 -*-
"""Deux ensembles sans TTA.

soft : softmax T=0.05 des 4 modeles, sans plancher.
hard : porte 0.015 sous la tete, puis softmax T=0.02.

v20h = SWA ep6-8 du run crop145 (FOV 145 mm).
ViT, v20b et CoAt = SWA OAI ep1-3 (FOV 130 mm).
Les poids viennent du best val hybrid025. Pour v20h, le subset crop145 wd01.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_h025_mix4" / "submit.ipynb"
RUN_V20H = "v20h_crop145_dino_ep11_oof_v3b_full_oaiv2_wd01"
WEIGHT_V20H = "v20h_crop145_swa_ep6_8.pt"
DATA_V20H = "sultent/rsna-knee-v20h-crop145-swa678"

JOBS = (
    {
        "tag": "soft",
        "slug": "rsna-knee-ens-c145-soft",
        "mix": "mix4",
        "rule": (
            "softmax des AUC val hybrid025, temperature 0.05, sans plancher. "
            "v20h : subset crop145 wd01, best epoch"
        ),
        "msg": (
            "ens soft ViT/v20b/v20h-crop145/CoAt, softmax hybrid025 T=0.05 "
            "sans plancher, FOV v20h 145 mm, sans TTA, pas de flip, T4"
        ),
    },
    {
        "tag": "hard",
        "slug": "rsna-knee-ens-c145-hard",
        "mix": "spec",
        "rule": (
            "specialiste : AUC a plus de 0.015 sous la tete = 0, "
            "puis softmax temperature 0.02. v20h : subset crop145 wd01, best epoch"
        ),
        "msg": (
            "ens hard ViT/v20b/v20h-crop145/CoAt, porte 0.015 puis softmax T=0.02, "
            "FOV v20h 145 mm, sans TTA, pas de flip, T4"
        ),
    },
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _replace(nb: dict, old: str, new: str, expect: int) -> None:
    found = 0
    for cell in nb["cells"]:
        src = _text(cell)
        n = src.count(old)
        if not n:
            continue
        found += n
        _set(cell, src.replace(old, new))
    if found != expect:
        raise SystemExit(f"remplacement {old!r} x{found}, attendu {expect}")


def _markdown(slug: str, rule: str, golds: dict[str, float], rows) -> str:
    lines = []
    for t, w in rows:
        lines.append(
            f"- `{t}` : ViT-k320 {w[0]:.0%} / v20b DINO {w[1]:.0%} / "
            f"v20h DINO {w[2]:.0%} / CoAt-a {w[3]:.0%}"
        )
    return f"""# RSNA Knee — {slug}

Notebook **self-contained** (internet off, **T4**). **Sans TTA.**

ViT, v20b et CoAt : SWA OAI epochs 1-3, FOV 130 mm.
v20h : SWA epochs 6-8, FOV 145 mm.
{rule}.
Subsets : `subset_vit_k_llrd08_320_v3b_hybrid025`, `subset_v20b_dino_ep11_v3b_hybrid025`, `subset_v20h_dino_ep11_v3b_hybrid025_wd01_crop145`, `subset_coat_a_llrd08_v3b_hybrid025`.

- **ViT-k320** : `vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3` SWA epochs 1-3 (gold in-sample moyen {golds['vit']:.3f}). Entree **320**. Pas de flip.
- **v20b DINO** : `v20b_crop130_dino_ep11_oof_v3b_full_oai3` SWA epochs 1-3 (gold in-sample moyen {golds['v20b']:.3f}). Entree 224. Pas de flip.
- **v20h DINO** : `{RUN_V20H}` SWA epochs 6-8 (gold in-sample moyen {golds['v20h']:.3f}). Entree 224, FOV 145 mm. Pas de flip.
- **CoAt-a** : `coat_a_llrd08_v3b_oofmask_full_oai3` SWA epochs 1-3 (gold in-sample moyen {golds['coat']:.3f}). Entree 224. Pas de flip.

**Mix** (ViT-k320 / v20b DINO / v20h DINO / CoAt-a) :
{chr(10).join(lines)}

Aucun flip de lateralite. Un modele peut valoir 0 %. Blend apres sigmoid.
"""


def _mix_block(rows) -> str:
    body = "\n".join(
        f"    ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}),  # {t}"
        for t, w in rows
    )
    return f"MIX = [\n{body}\n]\n"


VOL_OLD = """        vol_224 = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)
        vol_vit = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=320)
        if vol_vit.shape[-1] != 320 or vol_224.shape[-1] != 224:
            raise RuntimeError((vol_vit.shape, vol_224.shape))
        x_224 = torch.from_numpy(vol_224).unsqueeze(0).to(DEVICE)
        x_vit = torch.from_numpy(vol_vit).unsqueeze(0).to(DEVICE)
        p_vit = torch.sigmoid(_forward(model_vit, x_vit).float())
        p_v20b = torch.sigmoid(_forward(model_v20b, x_224).float())
        p_v20h = torch.sigmoid(_forward(model_v20h, x_224).float())
"""

VOL_NEW = """        CROP_MM = 130.0
        vol_224 = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)
        vol_vit = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=320)
        _SERIES_CACHE.clear()
        CROP_MM = 145.0
        vol_h = study_to_tensor(sdir, srows, n_series=6, laterality=False, img_size=224)
        CROP_MM = 130.0
        if vol_vit.shape[-1] != 320 or vol_224.shape[-1] != 224 or vol_h.shape[-1] != 224:
            raise RuntimeError((vol_vit.shape, vol_224.shape, vol_h.shape))
        x_224 = torch.from_numpy(vol_224).unsqueeze(0).to(DEVICE)
        x_vit = torch.from_numpy(vol_vit).unsqueeze(0).to(DEVICE)
        x_h = torch.from_numpy(vol_h).unsqueeze(0).to(DEVICE)
        p_vit = torch.sigmoid(_forward(model_vit, x_vit).float())
        p_v20b = torch.sigmoid(_forward(model_v20b, x_224).float())
        p_v20h = torch.sigmoid(_forward(model_v20h, x_h).float())
"""


def _patch(nb: dict, job: dict, rows, golds: dict[str, float]) -> None:
    md = next(c for c in nb["cells"] if c["cell_type"] == "markdown")
    _set(md, _markdown(job["slug"], job["rule"], golds, rows))
    cfg = next(
        c for c in nb["cells"]
        if c["cell_type"] == "code" and "MIX = [" in _text(c)
    )
    src = _text(cfg)
    src2, n = re.subn(
        r"MIX = \[\n(?:    \([0-9.]+, [0-9.]+, [0-9.]+, [0-9.]+\),  # .+\n)+\]\n",
        _mix_block(rows),
        src,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"bloc MIX x{n}")
    _set(cfg, src2)
    _replace(nb, "v20h_dino_oai3_swa_ep1_3.pt", WEIGHT_V20H, 2)
    _replace(
        nb,
        "kaggle_dataset_v20h_lat/_upload_dino_oai3_swa/" + WEIGHT_V20H,
        "kaggle_dataset_v20h_lat/_upload_crop145_swa678/" + WEIGHT_V20H,
        1,
    )
    _replace(
        nb,
        "v20h_crop130_dino_ep11_oof_v3b_full_oai3",
        RUN_V20H,
        1,
    )
    _replace(
        nb,
        f"'{RUN_V20H}', 'swa-1-2-3', [1, 2, 3]",
        f"'{RUN_V20H}', 'swa-6-7-8', [6, 7, 8]",
        1,
    )
    _replace(nb, VOL_OLD, VOL_NEW, 1)
    _replace(
        nb,
        "print(len(test), 'etudes | sans TTA | pas de flip')",
        "print(len(test), 'etudes | sans TTA | pas de flip | FOV vit/v20b/coat 130 mm, v20h 145 mm')",
        1,
    )


def _check(nb: dict, rows) -> None:
    blob = "\n".join(_text(c) for c in nb["cells"])
    for needle in (
        RUN_V20H, WEIGHT_V20H, "swa-6-7-8", "[6, 7, 8]",
        "swa-1-2-3", "CROP_MM = 145.0", "sans TTA", "v20h 145 mm",
    ):
        if needle not in blob:
            raise SystemExit(f"notebook incomplet: {needle}")
    for banned in (
        "v20h_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_dino_oai3_swa",
        "TTA_VARIANTS",
        "p_v20h = torch.sigmoid(_forward(model_v20h, x_224)",
    ):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    if "p_coat = torch.sigmoid(_forward(model_coat, x_224)" not in blob:
        raise SystemExit("coat ne lit plus le volume 130 mm")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(_text(cell))
    if len(rows) != 12:
        raise SystemExit("12 findings attendus")


def _write(nb: dict, job: dict) -> Path:
    push = KERNEL / f"_push_ens_c145_{job['tag']}"
    push.mkdir(parents=True, exist_ok=True)
    (push / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
    )
    meta = {
        "id": f"sultent/{job['slug']}",
        "title": job["slug"],
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
            "sultent/rsna-knee-vit-k320-oai3-swa",
            "sultent/rsna-knee-v20b-dino-oai3-swa",
            DATA_V20H,
            "sultent/rsna-knee-coat-a-oai3-swa",
        ],
        "category_ids": [],
    }
    (push / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    (push / "_message.txt").write_text(job["msg"] + "\n", encoding="utf-8")
    print("ok", push)
    return push


def main() -> None:
    preview = _load("h025", KERNEL / "_mix_hybrid025_preview.py")
    preview.NAMES = (
        ("ViT-k320", "subset_vit_k_llrd08_320_v3b_hybrid025"),
        ("v20b DINO", "subset_v20b_dino_ep11_v3b_hybrid025"),
        ("v20h DINO", "subset_v20h_dino_ep11_v3b_hybrid025_wd01_crop145"),
        ("CoAt-a", "subset_coat_a_llrd08_v3b_hybrid025"),
    )
    mixes = preview.ensemble_rows()
    base_swa = _load("oai3", KERNEL / "_make_nb_ens_oai3_swa.py")
    golds = {
        "vit": base_swa._gold_mean(
            "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3", (1, 2, 3),
        ),
        "v20b": base_swa._gold_mean(
            "v20b_crop130_dino_ep11_oof_v3b_full_oai3", (1, 2, 3),
        ),
        "v20h": base_swa._gold_mean(RUN_V20H, (6, 7, 8)),
        "coat": base_swa._gold_mean(
            "coat_a_llrd08_v3b_oofmask_full_oai3", (1, 2, 3),
        ),
    }
    print("golds", {k: round(v, 4) for k, v in golds.items()})
    src = json.loads(SRC.read_text(encoding="utf-8"))
    seen = []
    for job in JOBS:
        rows = mixes[job["mix"]]
        for t, w in rows:
            if len(w) != 4 or abs(sum(w) - 1.0) > 1e-6 or min(w) < -1e-9:
                raise SystemExit(f"poids invalides {job['tag']} {t} {w}")
            print(f"{job['tag']} {t:<18} " + " ".join(f"{v:.2f}" for v in w))
        nb = copy.deepcopy(src)
        _patch(nb, job, rows, golds)
        _check(nb, rows)
        _write(nb, job)
        seen.append(rows[0][1])
    if seen[0] == seen[1]:
        raise SystemExit("soft et hard identiques sur ACL")


if __name__ == "__main__":
    main()
