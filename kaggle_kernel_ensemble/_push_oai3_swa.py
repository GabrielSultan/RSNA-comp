# -*- coding: utf-8 -*-
"""Quatre kernels T4 des SWA OAI ep1-3, un seul forward, sans TTA.

Pas de soumission competition. Datasets neufs : les SWA RSNA deja en
ligne restent attaches aux ensembles. encode_chunk du YAML OAI
(288 / 576) est rabattu pour la T4.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UPLOAD_PY = ROOT / "kaggle_kernel_vit_e_dino_last" / "_upload_chunked.py"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
ACCEL = "NvidiaTeslaT4"
EPOCHS = (1, 2, 3)
SWA_LABEL = "swa-1-2-3"

JOBS = (
    {
        "tag": "vit",
        "run": "vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3",
        "model": "kneenet_vit_k320",
        "weight": "vit_k320_oai3_swa_ep1_3.pt",
        "dataset": "sultent/rsna-knee-vit-k320-oai3-swa",
        "kernel": "sultent/rsna-knee-vit-k320-oai3-swa-sans-tta",
        "title": "RSNA Knee ViT-k320 OAI3 SWA sans TTA",
        "ds_title": "RSNA Knee ViT-k320 OAI3 SWA ep1-3",
        "upload": ROOT / "kaggle_dataset_vit_k" / "_upload_k320_oai3_swa",
        "push": ROOT / "kaggle_kernel_vit_e_dino_last" / "_push_k320_oai3_swa",
        "out": ROOT / "kaggle_kernel_vit_e_dino_last" / "_out_k320_oai3_swa",
        "tta": False,
        "min_bytes": 200_000_000,
    },
    {
        "tag": "v20h",
        "run": "v20h_crop130_dino_ep11_oof_v3b_full_oai3",
        "model": "kneenet_v20h",
        "weight": "v20h_dino_oai3_swa_ep1_3.pt",
        "dataset": "sultent/rsna-knee-v20h-dino-oai3-swa",
        "kernel": "sultent/rsna-knee-v20h-dino-oai3-swa-sans-tta",
        "title": "RSNA Knee v20h DINO OAI3 SWA sans TTA",
        "ds_title": "RSNA Knee v20h DINO OAI3 SWA ep1-3",
        "upload": ROOT / "kaggle_dataset_v20h_lat" / "_upload_dino_oai3_swa",
        "push": ROOT / "kaggle_kernel_ensemble" / "_push_v20h_dino_oai3_swa",
        "out": ROOT / "kaggle_kernel_ensemble" / "_out_v20h_dino_oai3_swa",
        "tta": False,
        "min_bytes": 80_000_000,
        "nb_src": ROOT / "kaggle_kernel_ensemble" / "_push_v20h_dino_full_ep7" / "submit.ipynb",
    },
    {
        "tag": "v20b",
        "run": "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "model": "kneenet_v20b",
        "weight": "v20b_dino_oai3_swa_ep1_3.pt",
        "dataset": "sultent/rsna-knee-v20b-dino-oai3-swa",
        "kernel": "sultent/rsna-knee-v20b-dino-oai3-swa-sans-tta",
        "title": "RSNA Knee v20b DINO OAI3 SWA sans TTA",
        "ds_title": "RSNA Knee v20b DINO OAI3 SWA ep1-3",
        "upload": ROOT / "kaggle_dataset_v20b" / "_upload_dino_oai3_swa",
        "push": ROOT / "kaggle_kernel_ensemble" / "_push_v20b_dino_oai3_swa",
        "out": ROOT / "kaggle_kernel_ensemble" / "_out_v20b_dino_oai3_swa",
        "tta": False,
        "min_bytes": 80_000_000,
        "nb_src": ROOT / "kaggle_kernel_ensemble" / "_push_v20b_dino_full_ep8" / "submit.ipynb",
    },
    {
        "tag": "coat",
        "run": "coat_a_llrd08_v3b_oofmask_full_oai3",
        "model": "kneenet_coat_a",
        "weight": "coat_a_oai3_swa_ep1_3.pt",
        "dataset": "sultent/rsna-knee-coat-a-oai3-swa",
        "kernel": "sultent/rsna-knee-coat-a-oai3-swa-sans-tta",
        "title": "RSNA Knee CoAt-a OAI3 SWA sans TTA",
        "ds_title": "RSNA Knee CoAt-a OAI3 SWA ep1-3",
        "upload": ROOT / "kaggle_dataset_coat" / "_upload_oai3_swa",
        "push": ROOT / "kaggle_kernel_coat" / "_push_oai3_swa",
        "out": ROOT / "kaggle_kernel_coat" / "_out_oai3_swa",
        "tta": False,
        "min_bytes": 80_000_000,
    },
)


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


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set_source(cell: dict, text: str) -> None:
    src = text if text.endswith("\n") else text + "\n"
    cell["source"] = [line + "\n" for line in src.rstrip("\n").split("\n")]


def _replace_cells(nb: dict, old: str, new: str, expect: int) -> None:
    found = 0
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        n = src.count(old)
        if not n:
            continue
        found += n
        _set_source(cell, src.replace(old, new))
    if found != expect:
        raise SystemExit(f"remplacement {old!r} x{found}, attendu {expect}")


def _blob(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def _gold(run: Path) -> tuple[float, dict[str, float]]:
    hist = json.loads((run / "history.json").read_text(encoding="utf-8"))
    by = {}
    for ep in EPOCHS:
        rec = next(r for r in hist if int(r["epoch"]) == ep)
        if rec.get("gold_macro_auc") is None:
            raise SystemExit(f"gold absent {run.name} epoch {ep}")
        by[str(ep)] = float(rec["gold_macro_auc"])
    return sum(by.values()) / len(by), by


def _build_model(name: str, cfg: dict):
    common = dict(
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=8,
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
        pretrained=False,
    )
    if name == "kneenet_vit_k320":
        from src.models.kneenet_vit_k320 import KneeNet
        return KneeNet(
            img_size=320,
            drop_path_rate=float(cfg.get("drop_path_rate") or 0.0),
            grad_checkpoint=False,
            **common,
        )
    if name == "kneenet_v20h":
        from src.models.kneenet_v20h import KneeNet
        return KneeNet(**common)
    if name == "kneenet_v20b":
        from src.models.kneenet_v20b import KneeNet
        return KneeNet(**common)
    if name == "kneenet_coat_a":
        from src.models.kneenet_coat_a import KneeNet
        return KneeNet(
            backbone=str(cfg.get("backbone", "coatnet_0_rw_224.sw_in1k")),
            img_size=224,
            **common,
        )
    raise SystemExit(name)


def package(job: dict) -> float:
    run = ROOT / "checkpoints" / job["run"]
    src = run / "swa_ep1_3.pt"
    if not src.is_file():
        raise SystemExit(f"SWA manquant: {src}")
    gold, by = _gold(run)
    print("package", job["tag"], src.name, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    if str(cfg.get("model")) != job["model"]:
        raise SystemExit(cfg.get("model"))
    if str(cfg.get("run_name")) != job["run"]:
        raise SystemExit(cfg.get("run_name"))
    if str(ckpt.get("epoch")) != SWA_LABEL:
        raise SystemExit(ckpt.get("epoch"))
    got = tuple(int(e) for e in (ckpt.get("swa_epochs") or []))
    if got != EPOCHS:
        raise SystemExit(f"swa_epochs {got}")
    if cfg.get("laterality"):
        raise SystemExit(f"{job['tag']} laterality inattendue")
    cfg["laterality"] = False
    stored = ckpt.get("gold_by_epoch") or {}
    for ep in EPOCHS:
        prev = stored.get(str(ep))
        if prev is None or abs(float(prev) - by[str(ep)]) > 1e-6:
            raise SystemExit(f"gold incoherent {job['tag']} ep {ep}")
    if job["model"] == "kneenet_vit_k320" and int(cfg.get("img_size", 0)) != 320:
        raise SystemExit(f"img_size {cfg.get('img_size')}")
    out = {
        "model": ckpt["model"],
        "config": cfg,
        "epoch": SWA_LABEL,
        "swa_epochs": list(EPOCHS),
        "gold_by_epoch": by,
        "gold_macro_auc": gold,
        "val_macro_auc": -1.0,
    }
    model = _build_model(job["model"], cfg)
    model.load_state_dict(out["model"], strict=True)
    print(
        "load strict OK", job["model"],
        "gold", round(gold, 4),
        "encode_chunk cfg", cfg.get("encode_chunk"),
        flush=True,
    )
    del model, ckpt
    gc.collect()
    dest_dir = job["upload"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / job["weight"]
    torch.save(out, dest)
    if dest.stat().st_size < job["min_bytes"]:
        raise SystemExit(f"poids trop petit {dest} {dest.stat().st_size}")
    meta = {
        "title": job["ds_title"],
        "id": job["dataset"],
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dest_dir / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("wrote", dest, dest.stat().st_size, flush=True)
    del out
    gc.collect()
    return gold


def _write_meta(push: Path, job: dict) -> None:
    meta = {
        "id": job["kernel"],
        "title": job["title"],
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": ACCEL,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [job["dataset"]],
        "category_ids": [],
    }
    (push / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )


def _guard(run: str, model_line_old: str, model_line_new: str) -> tuple[str, str]:
    extra = (
        f"{model_line_new}\n"
        f"    if str(cfg.get('run_name')) != {run!r}:\n"
        f"        raise RuntimeError(f\"run inattendu {{cfg.get('run_name')}}\")\n"
        f"    if str(ckpt.get('epoch')) != {SWA_LABEL!r}:\n"
        f"        raise RuntimeError(f\"epoch inattendue {{ckpt.get('epoch')}}\")\n"
        f"    if list(ckpt.get('swa_epochs') or []) != [1, 2, 3]:\n"
        f"        raise RuntimeError(f\"swa_epochs inattendu {{ckpt.get('swa_epochs')}}\")\n"
    )
    return model_line_old, extra


def write_vit(job: dict, gold: float) -> None:
    mod = _load_mod(
        ROOT / "kaggle_kernel_vit_e_dino_last" / "_make_nb.py", "make_nb_vit",
    )
    local = str(job["upload"].relative_to(ROOT)).replace("\\", "/")
    mod.build(
        1,
        gold,
        job["weight"],
        run_name=job["run"],
        dataset=job["dataset"],
        local_dir=local,
        tta=False,
        out_dir=job["push"],
        kernel_id=job["kernel"],
        title=job["title"],
        img_size=320,
        encode_chunk_cap=2,
    )
    nb_path = job["push"] / "submit.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    _replace_cells(
        nb,
        '    if int(ckpt.get("epoch", -1)) != 1:\n'
        '        raise RuntimeError(f"epoch inattendue {ckpt.get(\'epoch\')}")\n',
        '    if str(ckpt.get("epoch")) != "swa-1-2-3":\n'
        '        raise RuntimeError(f"epoch inattendue {ckpt.get(\'epoch\')}")\n'
        '    if list(ckpt.get("swa_epochs") or []) != [1, 2, 3]:\n'
        '        raise RuntimeError(f"swa_epochs inattendu {ckpt.get(\'swa_epochs\')}")\n',
        1,
    )
    _replace_cells(
        nb,
        "# RSNA Knee — ViT-k LLRD 0.8 v3b (epoch 1) — T4 — sans TTA",
        "# RSNA Knee — ViT-k320 OAI3 SWA ep1-3 — T4 — sans TTA",
        1,
    )
    _replace_cells(
        nb,
        f"train `{job['run']}` **epoch 1**",
        f"train `{job['run']}` **SWA OAI epochs 1-3**",
        1,
    )
    _replace_cells(
        nb,
        "          '| epoch', ckpt.get('epoch'), '| gold in-sample', gold_txt,\n",
        "          '| epoch', ckpt.get('epoch'), '| swa', ckpt.get('swa_epochs'),\n"
        "          '| gold in-sample', gold_txt,\n",
        1,
    )
    blob = _blob(nb)
    for needle in (
        job["run"], job["weight"], "img_size=320", "kneenet_vit_k320",
        "swa-1-2-3", "sans TTA", "encode_chunk = 2",
    ):
        if needle not in blob:
            raise SystemExit(f"vit incomplet: {needle}")
    if "TTA_VARIANTS" in blob or '!= 1' in blob or "img_size=224" in blob:
        raise SystemExit("vit garde le TTA, un garde epoch 1 ou du 224")
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_meta(job["push"], job)
    print("ecrit", nb_path, flush=True)


def write_coat(job: dict, gold: float) -> None:
    mod = _load_mod(ROOT / "kaggle_kernel_coat" / "_make_nb.py", "make_nb_coat")
    local = str(job["upload"].relative_to(ROOT)).replace("\\", "/")
    mod.build(
        epoch=0,
        gold=gold,
        weight=job["weight"],
        dataset=job["dataset"],
        local_dir=local,
        out_dir=job["push"],
        kernel_id=job["kernel"],
        title=job["title"],
        swa=True,
    )
    nb_path = job["push"] / "submit.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parent = "coat_a_llrd08_v3b_oofmask_full"
    _replace_cells(nb, parent, job["run"], 2)
    _replace_cells(nb, "swa-7-8-9-10", SWA_LABEL, 1)
    _replace_cells(nb, "[7, 8, 9, 10]", "[1, 2, 3]", 1)
    _replace_cells(
        nb,
        "epochs 7, 8, 9 et 10",
        "epochs OAI 1, 2 et 3",
        1,
    )
    blob = _blob(nb)
    for needle in (
        job["run"], job["weight"], "sans TTA", "kneenet_coat_a",
        SWA_LABEL, "[1, 2, 3]", "img_size=224", "encode_chunk = 8",
    ):
        if needle not in blob:
            raise SystemExit(f"coat incomplet: {needle}")
    if re.search(rf"{re.escape(parent)}(?!_oai3)", blob) or "swa-7-8-9-10" in blob or "TTA_VARIANTS" in blob:
        raise SystemExit("coat garde le SWA RSNA ou le TTA")
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_meta(job["push"], job)
    print("ecrit", nb_path, flush=True)


_TTA_BLOCK = """TTA_VARIANTS = (
    (0.0, 1.00),
    (5.0, 1.00),
    (-5.0, 1.00),
    (0.0, 0.95),
    (0.0, 1.05),
)
"""

_AFFINE = """def apply_affine(x, angle_deg=0.0, scale=1.0):
    if angle_deg == 0.0 and scale == 1.0:
        return x
    b, s, d, h, w = x.shape
    xf = x.reshape(b * s * d, 1, h, w)
    angle = x.new_tensor(angle_deg * 3.141592653589793 / 180.0)
    sc = x.new_tensor(scale)
    cos, sin = torch.cos(angle) / sc, torch.sin(angle) / sc
    theta = xf.new_zeros(b, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta = theta.repeat_interleave(s * d, dim=0)
    grid = F.affine_grid(theta, xf.shape, align_corners=False)
    xf = F.grid_sample(xf, grid, align_corners=False, padding_mode='zeros')
    return xf.reshape(b, s, d, h, w)


"""

_TTA_LOOP = """        ps = []
        for angle_deg, scale in TTA_VARIANTS:
            xa = apply_affine(x, angle_deg, scale)
            ps.append(torch.sigmoid(_forward(xa, use_amp=True).float()))
        prob = torch.stack(ps, dim=0).mean(dim=0).cpu().numpy()[0]
"""

_PLAIN_LOOP = """        prob = torch.sigmoid(_forward(x, use_amp=True).float()).cpu().numpy()[0]
"""


def _drop_tta(nb: dict) -> None:
    _replace_cells(nb, " | TTA x{len(TTA_VARIANTS)} | ", " | sans TTA | ", 1)
    _replace_cells(nb, "print('TTA variantes   :', TTA_VARIANTS)\n", "", 1)
    _replace_cells(nb, _TTA_BLOCK, "", 1)
    _replace_cells(nb, _AFFINE, "", 1)
    _replace_cells(nb, _TTA_LOOP, _PLAIN_LOOP, 1)
    _replace_cells(nb, "→ TTA affine ×5 →", "→ un seul forward, sans TTA →", 1)
    _replace_cells(
        nb,
        "# --- Inference : TTA, un seul modele,",
        "# --- Inference : sans TTA, un seul modele,",
        1,
    )
    for old, new in (
        (
            "TTA : identite + rotation ±5° + scale 0.95/1.05. Pas de flip horizontal.\n",
            "Un seul forward, **sans TTA**. Pas de flip horizontal.\n",
        ),
        (
            "TTA : identite + rotation ±5° + scale 0.95/1.05. Pas de hflip.\n",
            "Un seul forward, **sans TTA**. Pas de hflip.\n",
        ),
    ):
        found = sum("".join(c["source"]).count(old) for c in nb["cells"])
        if found:
            _replace_cells(nb, old, new, found)


def write_resnet(job: dict, gold: float) -> None:
    nb = json.loads(job["nb_src"].read_text(encoding="utf-8"))
    if job["tag"] == "v20h":
        pairs = (
            (
                "sultent/rsna-knee-v20h-dino-full-ep7",
                job["dataset"],
                1,
            ),
            ("v20h_dino_full_ep7.pt", job["weight"], 2),
            (
                "../kaggle_dataset_v20h_lat/_upload_dino_full_ep7",
                str(job["upload"].relative_to(ROOT)).replace("\\", "/"),
                1,
            ),
            (
                "# RSNA Knee — v20h DINO ep11 full oofmask v3b, sans flip L/R, **solo** epoch 7",
                "# RSNA Knee — v20h DINO OAI3 SWA ep1-3, sans flip L/R, sans TTA",
                1,
            ),
            (
                f"**sans flip** gauche/droite → **KneeNet v20h** (`v20h_crop130_dino_ep11_oof_v3b_full`, encodeur DINO, epoch 7, gold in-sample macro-AUC 0.970)",
                f"**sans flip** gauche/droite → **KneeNet v20h** (`{job['run']}`, encodeur DINO, SWA OAI ep1-3, gold in-sample moyen {gold:.3f})",
                1,
            ),
            (
                "Full train apres masque OOF, labels Coolin targets_v3b_oofmask, gold dans le train (poids x4). Snapshot epoch 7 (gold in-sample 0.970). Pas de canon laterality : le run a ete train sans flip.",
                "Trois epochs OAI (MOAKS) moyennes en SWA, apres le SWA RSNA ep6-7. Pas de canon laterality.",
                1,
            ),
            ("dataset v20h dino full ep7 attache", "dataset v20h DINO OAI3 SWA attache", 1),
            ("print('modele          : v20h dino full ep7 epoch 7')",
             "print('modele          : v20h DINO OAI3 SWA ep1-3')", 1),
            ("| v20h dino full ep7')", "| v20h DINO OAI3 SWA')", 1),
            (
                "    print('modele', cfg.get('model'), '| epoch', ckpt.get('epoch'),\n",
                "    print('modele', cfg.get('model'), '| run', cfg.get('run_name'),\n"
                "          '| epoch', ckpt.get('epoch'),\n",
                1,
            ),
            (
                "        encode_chunk=max(8, int(cfg.get('encode_chunk', 8))),\n",
                "        encode_chunk=8,\n",
                1,
            ),
        )
        guard_old = (
            "    if str(cfg.get('model')) != 'kneenet_v20h':\n"
            "        raise RuntimeError(f\"model={cfg.get('model')!r}, attendu kneenet_v20h\")\n"
        )
    else:
        pairs = (
            ("v20b_dino_full_ep8.pt", job["weight"], 2),
            (
                "sultent/rsna-knee-v20b-dino-full-ep8",
                job["dataset"],
                1,
            ),
            (
                "kaggle_dataset_v20b/_upload_dino_full_ep8",
                str(job["upload"].relative_to(ROOT)).replace("\\", "/"),
                1,
            ),
            (
                "# RSNA Knee — v20b DINO ep11 full oofmask v3b, **solo** epoch 8",
                "# RSNA Knee — v20b DINO OAI3 SWA ep1-3, sans TTA",
                1,
            ),
            (
                "epoch 8, gold in-sample macro-AUC 0.971)",
                f"SWA OAI ep1-3, gold in-sample moyen {gold:.3f})",
                1,
            ),
            (
                "`v20b_crop130_dino_ep11_oof_v3b_full`",
                f"`{job['run']}`",
                1,
            ),
            ("dataset v20b dino full ep8 attache", "dataset v20b DINO OAI3 SWA attache", 1),
            ("print('modele          : v20b dino full ep8')",
             "print('modele          : v20b DINO OAI3 SWA ep1-3')", 1),
            ("| v20b dino full ep8')", "| v20b DINO OAI3 SWA')", 1),
            (
                "          '| epoch', ckpt.get('epoch'),\n"
                "          '| gold macro-AUC :', None if gold is None else round(float(gold), 4),\n",
                "          '| epoch', ckpt.get('epoch'),\n"
                "          '| swa', ckpt.get('swa_epochs'),\n"
                "          '| gold macro-AUC :', None if gold is None else round(float(gold), 4),\n",
                1,
            ),
        )
        guard_old = (
            "    if str(cfg.get('model')) != 'kneenet_v20b':\n"
            "        raise RuntimeError(f\"model={cfg.get('model')!r}, attendu kneenet_v20b\")\n"
        )
    for old, new, expect in pairs:
        _replace_cells(nb, old, new, expect)
    _replace_cells(nb, guard_old, _guard(job["run"], guard_old, guard_old)[1], 1)
    _drop_tta(nb)
    blob = _blob(nb)
    for needle in (
        job["run"], job["weight"], SWA_LABEL, "[1, 2, 3]",
        "LATERALITY_CANON_L = False", "sans TTA", "encode_chunk=8",
    ):
        if needle not in blob:
            raise SystemExit(f"{job['tag']} incomplet: {needle}")
    if "ep7" in blob or "ep8" in blob or "encode_chunk=max" in blob or "TTA_VARIANTS" in blob:
        raise SystemExit(f"{job['tag']} garde un epoch parent, le TTA ou encode_chunk 576")
    job["push"].mkdir(parents=True, exist_ok=True)
    nb_path = job["push"] / "submit.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_meta(job["push"], job)
    print("ecrit", nb_path, flush=True)


def write_notebook(job: dict, gold: float) -> None:
    if job["tag"] == "vit":
        write_vit(job, gold)
    elif job["tag"] == "coat":
        write_coat(job, gold)
    else:
        write_resnet(job, gold)


def _files_text(dataset: str) -> str | None:
    try:
        return kaggle("datasets", "files", dataset)
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        low = text.lower()
        missing = ("404", "403", "not found", "notfound", "does not exist", "forbidden")
        if any(s in low for s in missing):
            return None
        raise


def _weight_size(text: str, weight: str) -> int | None:
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] != weight:
            continue
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return None
    return None


def _status_dataset(dataset: str) -> str:
    try:
        return kaggle("datasets", "status", dataset).strip()
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        low = text.lower()
        missing = ("404", "403", "not found", "notfound", "does not exist", "forbidden")
        if any(s in low for s in missing):
            return ""
        raise


def _ready(status: str) -> bool:
    low = status.lower()
    return "ready" in low and "pending" not in low


def _dataset_state(job: dict) -> str:
    listed = _files_text(job["dataset"])
    if listed is None:
        st = _status_dataset(job["dataset"])
        if _ready(st):
            return "ready"
        if st and "pending" in st.lower():
            return "pending"
        return "absent"
    size = _weight_size(listed, job["weight"])
    if size is not None and size >= job["min_bytes"]:
        return "ready"
    return "pending"


def _quota(text: str) -> bool:
    low = text.lower()
    return "maximum batch gpu" in low or "quota" in low


def ensure_dataset(job: dict) -> None:
    pt = job["upload"] / job["weight"]
    if not pt.is_file() or pt.stat().st_size < job["min_bytes"]:
        raise SystemExit(f"poids manquant: {pt}")
    state = _dataset_state(job)
    print(f"[ds {job['tag']}]", state, flush=True)
    if state == "ready":
        print(f"[ds {job['tag']}] deja pret, pas de reupload", flush=True)
        return
    version = state == "pending"
    cmd = [
        sys.executable, "-X", "utf8", str(UPLOAD_PY),
        "--folder", str(job["upload"]),
        "--message", f"OAI3 SWA ep1-3 {job['tag']}",
    ]
    if version:
        cmd.append("--version")
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    t0 = time.time()
    while True:
        state = _dataset_state(job)
        print(f"[ds {job['tag']}]", state, flush=True)
        if state == "ready":
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout dataset {job['tag']}")
        time.sleep(20)


def push_kernel(job: dict) -> str:
    push = job["push"]
    meta = json.loads((push / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("id") != job["kernel"]:
        raise SystemExit(meta.get("id"))
    if meta.get("dataset_sources") != [job["dataset"]]:
        raise SystemExit(meta.get("dataset_sources"))
    if meta.get("machine_shape") != ACCEL or meta.get("enable_internet"):
        raise SystemExit("kernel meta GPU/internet")
    if meta.get("id") == job["dataset"]:
        raise SystemExit("slug kernel = slug dataset")
    nb = (push / "submit.ipynb").read_text(encoding="utf-8")
    if job["tta"]:
        if "TTA_VARIANTS" not in nb:
            raise SystemExit(f"{job['tag']} sans TTA")
    elif "TTA_VARIANTS" in nb or "sans TTA" not in nb:
        raise SystemExit(f"{job['tag']} TTA incoherent")
    print("push", job["kernel"], "T4", flush=True)
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(push),
                "--accelerator", ACCEL, "-t", "32400",
            )
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text):
                print(f"quota GPU {job['tag']}, retry dans 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if _quota(out):
            print(f"quota GPU {job['tag']}, retry dans 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(f"version introuvable: {out!r}")
        version = m.group(1)
        (push / "_version.txt").write_text(version, encoding="utf-8")
        print("kernel version", job["kernel"], version, flush=True)
        return version


def wait_kernel(job: dict) -> None:
    t0 = time.time()
    seen_running = False
    while True:
        raw = kaggle("kernels", "status", job["kernel"]).strip()
        up = raw.upper()
        print(f"[{job['tag']}]", raw, flush=True)
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            job["out"].mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", job["kernel"], "-p", str(job["out"])), flush=True)
            except subprocess.CalledProcessError:
                pass
            raise SystemExit(f"kernel {job['tag']} en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen_running = True
        done = ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up
        if done and (seen_running or (time.time() - t0) > 90):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout kernel {job['tag']}")
        time.sleep(POLL_S)


def collect(job: dict, version: str) -> None:
    dest = job["out"]
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", job["kernel"], "-p", str(dest)), flush=True)
    csv = dest / "submission.csv"
    if not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit(f"submission.csv absent: {dest}")
    log = next(dest.glob("*.log"), None)
    if log is None:
        raise SystemExit(f"log absent: {dest}")
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (job["run"], SWA_LABEL, "lignes 3", "[1, 2, 3]"):
        if needle not in text:
            raise SystemExit(f"log {job['tag']} sans {needle}")
    if job["tta"]:
        if "TTA x" not in text:
            raise SystemExit(f"log {job['tag']} sans TTA")
    elif "sans TTA" not in text or "TTA x" in text:
        raise SystemExit(f"log {job['tag']} TTA incoherent")
    print(
        "KERNEL_PRET", job["kernel"], "version", version,
        "| pas de soumission competition",
        flush=True,
    )


def selected(only: list[str] | None) -> list[dict]:
    if not only:
        return list(JOBS)
    wanted = set(only)
    known = {job["tag"] for job in JOBS}
    missing = wanted - known
    if missing:
        raise SystemExit(f"tags inconnus: {sorted(missing)}")
    return [job for job in JOBS if job["tag"] in wanted]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--make-only", action="store_true")
    p.add_argument("--skip-make", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()
    jobs = selected(args.only)
    if not args.skip_make:
        for job in jobs:
            gold = package(job)
            write_notebook(job, gold)
    if args.make_only:
        print("MAKE_ONLY", flush=True)
        return
    for job in jobs:
        ensure_dataset(job)
    for job in jobs:
        version = push_kernel(job)
        wait_kernel(job)
        collect(job, version)
        print("DONE", job["tag"], version, flush=True)


if __name__ == "__main__":
    main()
