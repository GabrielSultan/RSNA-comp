# -*- coding: utf-8 -*-
"""Attend epoch8 du run crop145, SWA 6-7-8, soumission sans TTA."""
from __future__ import annotations

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

RUN_NAME = "v20h_crop145_dino_ep11_oof_v3b_full_oaiv2_wd01"
RUN = ROOT / "checkpoints" / RUN_NAME
EPOCHS = (6, 7, 8)
SWA_LABEL = "swa-6-7-8"
WEIGHT = "v20h_crop145_swa_ep6_8.pt"
DATASET = "sultent/rsna-knee-v20h-crop145-swa678"
SLUG = "rsna-knee-v20h-crop145-swa678-sans-tta"
KERNEL = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
UPLOAD = ROOT / "kaggle_dataset_v20h_lat" / "_upload_crop145_swa678"
PUSH = ROOT / "kaggle_kernel_ensemble" / "_push_v20h_crop145_swa678"
OUT = ROOT / "kaggle_kernel_ensemble" / "_out_v20h_crop145_swa678"
SRC_NB = ROOT / "kaggle_kernel_ensemble" / "_push_v20h_oaiv2_wd01_ep6" / "submit.ipynb"
UPLOAD_PY = ROOT / "kaggle_kernel_vit_e_dino_last" / "_upload_chunked.py"
POLL_S = 30
MAX_WAIT_S = 8 * 3600


def _swa():
    spec = importlib.util.spec_from_file_location(
        "swa_oai3", ROOT / "kaggle_kernel_ensemble" / "_swa_oai3.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _golds() -> dict[str, float]:
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    out = {}
    for ep in EPOCHS:
        rec = next(r for r in hist if int(r["epoch"]) == ep)
        if rec.get("gold_macro_auc") is None:
            raise SystemExit(f"gold absent epoch {ep}")
        out[str(ep)] = float(rec["gold_macro_auc"])
    return out


def _epoch8_ready() -> bool:
    snap = RUN / "epoch8.pt"
    if not snap.is_file() or snap.stat().st_size < 80_000_000:
        return False
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    if not any(int(r.get("epoch", -1)) == 8 and r.get("gold_macro_auc") is not None for r in hist):
        return False
    size = snap.stat().st_size
    time.sleep(15)
    return snap.is_file() and snap.stat().st_size == size


def build_swa(swa) -> float:
    dst = RUN / "swa_ep6_8.pt"
    print("SWA", EPOCHS, flush=True)
    golds = _golds()
    acc: dict = {}
    cfg = None
    ref_sd = None
    for i, ep in enumerate(EPOCHS):
        src = RUN / f"epoch{ep}.pt"
        print("load", src.name, src.stat().st_size, flush=True)
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        c = dict(ckpt.get("config") or {})
        if str(c.get("model")) != "kneenet_v20h":
            raise SystemExit(c.get("model"))
        if str(c.get("run_name")) != RUN_NAME:
            raise SystemExit(c.get("run_name"))
        if int(ckpt.get("epoch", -1)) != ep:
            raise SystemExit(f"{src.name} epoch {ckpt.get('epoch')}")
        if c.get("laterality"):
            raise SystemExit("laterality inattendue")
        if cfg is None:
            cfg = c
        swa._add(acc, ckpt["model"], first=i == 0)
        if ref_sd is None:
            ref_sd = {k: v.detach().cpu() for k, v in ckpt["model"].items()}
        del ckpt
        gc.collect()
        print(f"  epoch {ep} gold {golds[str(ep)]:.4f}", flush=True)
    avg = swa._divide(acc, ref_sd, len(EPOCHS))
    del acc, ref_sd
    gc.collect()
    cfg["laterality"] = False
    from src.models.kneenet_v20h import KneeNet
    model = KneeNet(
        pretrained=False,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.2)),
        encode_chunk=8,
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    )
    model.load_state_dict(avg, strict=True)
    print("load strict OK", flush=True)
    del model
    gold = sum(golds.values()) / len(golds)
    payload = {
        "model": avg,
        "config": cfg,
        "epoch": SWA_LABEL,
        "swa_epochs": list(EPOCHS),
        "gold_by_epoch": golds,
        "gold_macro_auc": gold,
        "val_macro_auc": -1.0,
    }
    torch.save(payload, dst)
    print("wrote", dst, dst.stat().st_size, "gold", round(gold, 4), flush=True)
    UPLOAD.mkdir(parents=True, exist_ok=True)
    up = UPLOAD / WEIGHT
    torch.save(payload, up)
    meta = {
        "title": "RSNA Knee v20h crop145 SWA ep6-8",
        "id": DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (UPLOAD / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("wrote", up, up.stat().st_size, flush=True)
    return gold


def _set(cell: dict, text: str) -> None:
    text = text if text.endswith("\n") else text + "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _replace(nb: dict, old: str, new: str, expect: int) -> None:
    found = 0
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        n = src.count(old)
        if not n:
            continue
        found += n
        _set(cell, src.replace(old, new))
    if found != expect:
        raise SystemExit(f"remplacement {old!r} x{found}, attendu {expect}")


def write_notebook(gold: float) -> None:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    local = str(UPLOAD.relative_to(ROOT)).replace("\\", "/")
    pairs = (
        (
            "v20h_crop130_dino_ep11_oof_v3b_full_oaiv2_wd01",
            RUN_NAME,
            2,
        ),
        ("v20h_oaiv2_wd01_ep6.pt", WEIGHT, 2),
        ("sultent/rsna-knee-v20h-oaiv2-wd01-ep6", DATASET, 1),
        ("kaggle_dataset_v20h_lat/_upload_oaiv2_wd01_ep6", local, 1),
        ("CROP_MM, CROP_REL, BAND = 130.0, 0.70, (0.20, 0.80)",
         "CROP_MM, CROP_REL, BAND = 145.0, 0.70, (0.20, 0.80)", 1),
        ("crop 130 mm", "crop 145 mm", 2),
        (
            "# RSNA Knee — v20h DINO oaiv2 wd01 epoch 6, sans flip, sans TTA",
            "# RSNA Knee — v20h DINO crop145 oaiv2 wd01 SWA ep6-8, sans flip, sans TTA",
            1,
        ),
        (
            "encodeur DINO, epoch 6, gold in-sample 0.960, wd 0.01, dropout 0.2",
            f"encodeur DINO, SWA epochs 6-8, gold in-sample moyen {gold:.3f}, wd 0.01, dropout 0.2",
            1,
        ),
        (
            "Full oaiv2, weight decay 0.01 et dropout 0.2. Pas de canon laterality.",
            "SWA uniforme des epochs 6, 7 et 8. FOV 145 mm. Pas de canon laterality.",
            1,
        ),
        ("dataset v20h oaiv2 wd01 ep6 attache", "dataset v20h crop145 SWA ep6-8 attache", 1),
        (
            "print('modele          : v20h DINO oaiv2 wd01 ep6')",
            "print('modele          : v20h DINO crop145 SWA ep6-8')",
            1,
        ),
        ("| v20h DINO oaiv2 wd01 ep6')", "| v20h DINO crop145 SWA ep6-8')", 1),
        (
            "    if int(ckpt.get('epoch', -1)) != 6:\n"
            "        raise RuntimeError(f\"epoch inattendue {ckpt.get('epoch')}\")\n",
            "    if str(ckpt.get('epoch')) != 'swa-6-7-8':\n"
            "        raise RuntimeError(f\"epoch inattendue {ckpt.get('epoch')}\")\n"
            "    if list(ckpt.get('swa_epochs') or []) != [6, 7, 8]:\n"
            "        raise RuntimeError(f\"swa_epochs inattendu {ckpt.get('swa_epochs')}\")\n",
            1,
        ),
    )
    for old, new, expect in pairs:
        _replace(nb, old, new, expect)
    blob = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in (RUN_NAME, WEIGHT, "sans TTA", "145.0", "swa-6-7-8", "[6, 7, 8]"):
        if needle not in blob:
            raise SystemExit(f"notebook incomplet: {needle}")
    if "TTA_VARIANTS" in blob or "CROP_MM, CROP_REL, BAND = 130.0" in blob:
        raise SystemExit("TTA ou crop 130 encore present")
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8",
    )
    meta = {
        "id": KERNEL,
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
        "dataset_sources": [DATASET],
        "category_ids": [],
    }
    (PUSH / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("ecrit", PUSH / "submit.ipynb", flush=True)


def _status_dataset() -> str:
    try:
        return kaggle("datasets", "status", DATASET).strip()
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        if any(s in text.lower() for s in ("404", "403", "not found", "forbidden")):
            return ""
        raise


def ensure_dataset() -> None:
    pt = UPLOAD / WEIGHT
    if not pt.is_file() or pt.stat().st_size < 80_000_000:
        raise SystemExit(f"poids manquant {pt}")
    st = _status_dataset().lower()
    print("[ds]", st or "(absent)", flush=True)
    if "pending" in st:
        cmd = None
    else:
        cmd = [
            sys.executable, "-X", "utf8", str(UPLOAD_PY),
            "--folder", str(UPLOAD),
            "--message", "v20h crop145 SWA ep6-8",
        ]
        if "ready" in st:
            cmd.append("--version")
    if cmd:
        print("+", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
    t0 = time.time()
    while True:
        st = _status_dataset().lower()
        print("[ds]", st or "(pending)", flush=True)
        if "ready" in st and "pending" not in st:
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout dataset")
        time.sleep(20)


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def push_kernel() -> str:
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(PUSH),
                "--accelerator", ACCEL, "-t", "32400",
            )
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
        print("kernel version", m.group(1), flush=True)
        return m.group(1)


def wait_kernel() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KERNEL).strip()
        print(raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            OUT.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
            except subprocess.CalledProcessError:
                pass
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


def submit(version: str, gold: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    log = next(OUT.glob("*.log"))
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (RUN_NAME, "swa-6-7-8", "sans TTA", "lignes 3", "laterality False"):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text:
        raise SystemExit("TTA dans le log")
    msg = (
        f"v20h DINO crop145 oaiv2 wd01 SWA ep6-8 "
        f"(gold moyen {gold:.3f}, FOV 145 mm, dropout 0.2, wd 0.01, sans TTA, pas de flip, T4)"
    )
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", version,
        "-f", "submission.csv", "-m", msg,
    ), flush=True)
    print("SOUMIS", KERNEL, "version", version, flush=True)


def main() -> None:
    print("attente epoch8.pt", flush=True)
    while not _epoch8_ready():
        snap = RUN / "epoch8.pt"
        if snap.is_file():
            print(f"  {snap.name} {snap.stat().st_size}", flush=True)
        else:
            print("  pas encore de epoch8.pt", flush=True)
        time.sleep(60)
    print("epoch8.pt pret", flush=True)
    gold = build_swa(_swa())
    write_notebook(gold)
    ensure_dataset()
    version = push_kernel()
    wait_kernel()
    submit(version, gold)


if __name__ == "__main__":
    main()
