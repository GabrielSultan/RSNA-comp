# -*- coding: utf-8 -*-
"""Solo sans TTA : v20h DINO oaiv2 wd01 epoch 6, puis soumission competition."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN = ROOT / "checkpoints" / "v20h_crop130_dino_ep11_oof_v3b_full_oaiv2_wd01"
RUN_NAME = "v20h_crop130_dino_ep11_oof_v3b_full_oaiv2_wd01"
EPOCH = 6
WEIGHT = f"v20h_oaiv2_wd01_ep{EPOCH}.pt"
DATASET = f"sultent/rsna-knee-v20h-oaiv2-wd01-ep{EPOCH}"
SLUG = f"rsna-knee-v20h-oaiv2-wd01-ep{EPOCH}-sans-tta"
KERNEL = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
UPLOAD = ROOT / "kaggle_dataset_v20h_lat" / f"_upload_oaiv2_wd01_ep{EPOCH}"
PUSH = ROOT / "kaggle_kernel_ensemble" / f"_push_v20h_oaiv2_wd01_ep{EPOCH}"
OUT = ROOT / "kaggle_kernel_ensemble" / f"_out_v20h_oaiv2_wd01_ep{EPOCH}"
SRC_NB = ROOT / "kaggle_kernel_ensemble" / "_push_v20h_dino_oai3_swa" / "submit.ipynb"
UPLOAD_PY = ROOT / "kaggle_kernel_vit_e_dino_last" / "_upload_chunked.py"
POLL_S = 30
MAX_WAIT_S = 8 * 3600


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


def _gold() -> float:
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    rec = next(r for r in hist if int(r["epoch"]) == EPOCH)
    return float(rec["gold_macro_auc"])


def package(gold: float) -> None:
    src = RUN / f"epoch{EPOCH}.pt"
    if not src.is_file():
        raise SystemExit(f"poids manquant: {src}")
    print("load", src.name, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    if str(cfg.get("model")) != "kneenet_v20h":
        raise SystemExit(cfg.get("model"))
    if str(cfg.get("run_name")) != RUN_NAME:
        raise SystemExit(cfg.get("run_name"))
    if int(ckpt.get("epoch", -1)) != EPOCH:
        raise SystemExit(ckpt.get("epoch"))
    if cfg.get("laterality"):
        raise SystemExit("laterality inattendue")
    if abs(float(cfg.get("dropout", -1)) - 0.2) > 1e-6:
        raise SystemExit(f"dropout {cfg.get('dropout')}")
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
    model.load_state_dict(ckpt["model"], strict=True)
    print("load strict OK gold", round(gold, 4), "encode_chunk", cfg.get("encode_chunk"), flush=True)
    del model
    out = {
        "model": ckpt["model"],
        "config": cfg,
        "epoch": EPOCH,
        "gold_macro_auc": gold,
        "val_macro_auc": -1.0,
    }
    UPLOAD.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD / WEIGHT
    torch.save(out, dest)
    if dest.stat().st_size < 80_000_000:
        raise SystemExit(f"poids trop petit {dest.stat().st_size}")
    meta = {
        "title": f"RSNA Knee v20h oaiv2 wd01 ep{EPOCH}",
        "id": DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (UPLOAD / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("wrote", dest, dest.stat().st_size, flush=True)


def _set_source(cell: dict, text: str) -> None:
    src = text if text.endswith("\n") else text + "\n"
    cell["source"] = [line + "\n" for line in src.rstrip("\n").split("\n")]


def _replace(nb: dict, old: str, new: str, expect: int) -> None:
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


def write_notebook(gold: float) -> None:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    local = str(UPLOAD.relative_to(ROOT)).replace("\\", "/")
    pairs = (
        (
            "v20h_crop130_dino_ep11_oof_v3b_full_oai3",
            RUN_NAME,
            2,
        ),
        ("v20h_dino_oai3_swa_ep1_3.pt", WEIGHT, 2),
        ("sultent/rsna-knee-v20h-dino-oai3-swa", DATASET, 1),
        ("kaggle_dataset_v20h_lat/_upload_dino_oai3_swa", local, 1),
        (
            "# RSNA Knee — v20h DINO OAI3 SWA ep1-3, sans flip L/R, sans TTA",
            f"# RSNA Knee — v20h DINO oaiv2 wd01 epoch {EPOCH}, sans flip, sans TTA",
            1,
        ),
        (
            "encodeur DINO, SWA OAI ep1-3, gold in-sample moyen 0.972",
            f"encodeur DINO, epoch {EPOCH}, gold in-sample {gold:.3f}, wd 0.01, dropout 0.2",
            1,
        ),
        (
            "Trois epochs OAI (MOAKS) moyennes en SWA, apres le SWA RSNA ep6-7. Pas de canon laterality.",
            "Full oaiv2, weight decay 0.01 et dropout 0.2. Pas de canon laterality.",
            1,
        ),
        ("dataset v20h DINO OAI3 SWA attache", f"dataset v20h oaiv2 wd01 ep{EPOCH} attache", 1),
        (
            "print('modele          : v20h DINO OAI3 SWA ep1-3')",
            f"print('modele          : v20h DINO oaiv2 wd01 ep{EPOCH}')",
            1,
        ),
        ("| v20h DINO OAI3 SWA')", f"| v20h DINO oaiv2 wd01 ep{EPOCH}')", 1),
        (
            "    if str(ckpt.get('epoch')) != 'swa-1-2-3':\n"
            "        raise RuntimeError(f\"epoch inattendue {ckpt.get('epoch')}\")\n"
            "    if list(ckpt.get('swa_epochs') or []) != [1, 2, 3]:\n"
            "        raise RuntimeError(f\"swa_epochs inattendu {ckpt.get('swa_epochs')}\")\n",
            f"    if int(ckpt.get('epoch', -1)) != {EPOCH}:\n"
            "        raise RuntimeError(f\"epoch inattendue {ckpt.get('epoch')}\")\n",
            1,
        ),
    )
    for old, new, expect in pairs:
        _replace(nb, old, new, expect)
    blob = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in (RUN_NAME, WEIGHT, "sans TTA", "encode_chunk=8", f"!= {EPOCH}"):
        if needle not in blob:
            raise SystemExit(f"notebook incomplet: {needle}")
    if "TTA_VARIANTS" in blob or "swa-1-2-3" in blob or "oai3" in blob:
        raise SystemExit("notebook garde le SWA OAI3 ou le TTA")
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
        low = text.lower()
        if any(s in low for s in ("404", "403", "not found", "forbidden")):
            return ""
        raise


def ensure_dataset() -> None:
    pt = UPLOAD / WEIGHT
    if not pt.is_file() or pt.stat().st_size < 80_000_000:
        raise SystemExit(f"poids manquant: {pt}")
    st = _status_dataset().lower()
    print("[ds]", st or "(absent)", flush=True)
    if "ready" in st and "pending" not in st:
        print("[ds] deja pret, nouvelle version", flush=True)
        cmd = [
            sys.executable, "-X", "utf8", str(UPLOAD_PY),
            "--folder", str(UPLOAD), "--version",
            "--message", f"v20h oaiv2 wd01 epoch {EPOCH} sans optim",
        ]
    elif "pending" in st:
        cmd = None
    else:
        cmd = [
            sys.executable, "-X", "utf8", str(UPLOAD_PY),
            "--folder", str(UPLOAD),
            "--message", f"v20h oaiv2 wd01 epoch {EPOCH} sans optim",
        ]
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
    print("push", KERNEL, flush=True)
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
        version = m.group(1)
        (PUSH / "_version.txt").write_text(version, encoding="utf-8")
        print("kernel version", version, flush=True)
        return version


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
    log = next(OUT.glob("*.log"), None)
    csv = OUT / "submission.csv"
    if log is None or not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit("sortie incomplete")
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (RUN_NAME, "sans TTA", "lignes 3", f"epoch {EPOCH}", "laterality False"):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text or "swa-1-2-3" in text:
        raise SystemExit("log TTA ou SWA")
    msg = (
        f"v20h DINO oaiv2 wd01 epoch {EPOCH} "
        f"(gold in-sample {gold:.3f}, dropout 0.2, wd 0.01, sans TTA, pas de flip, T4)"
    )
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", version,
        "-f", "submission.csv", "-m", msg,
    ), flush=True)
    print("SOUMIS", KERNEL, "version", version, flush=True)


def _epoch_ready() -> bool:
    snap = RUN / f"epoch{EPOCH}.pt"
    hist_p = RUN / "history.json"
    if not snap.is_file() or snap.stat().st_size < 80_000_000 or not hist_p.is_file():
        return False
    hist = json.loads(hist_p.read_text(encoding="utf-8"))
    if not any(int(r.get("epoch", -1)) == EPOCH for r in hist):
        return False
    size = snap.stat().st_size
    time.sleep(15)
    return snap.is_file() and snap.stat().st_size == size


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, default=6)
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()
    global EPOCH, WEIGHT, DATASET, SLUG, KERNEL, UPLOAD, PUSH, OUT
    EPOCH = int(args.epoch)
    WEIGHT = f"v20h_oaiv2_wd01_ep{EPOCH}.pt"
    DATASET = f"sultent/rsna-knee-v20h-oaiv2-wd01-ep{EPOCH}"
    SLUG = f"rsna-knee-v20h-oaiv2-wd01-ep{EPOCH}-sans-tta"
    KERNEL = f"sultent/{SLUG}"
    UPLOAD = ROOT / "kaggle_dataset_v20h_lat" / f"_upload_oaiv2_wd01_ep{EPOCH}"
    PUSH = ROOT / "kaggle_kernel_ensemble" / f"_push_v20h_oaiv2_wd01_ep{EPOCH}"
    OUT = ROOT / "kaggle_kernel_ensemble" / f"_out_v20h_oaiv2_wd01_ep{EPOCH}"
    if args.wait:
        print(f"attente epoch{EPOCH}.pt", flush=True)
        while not _epoch_ready():
            snap = RUN / f"epoch{EPOCH}.pt"
            if snap.is_file():
                print(f"  {snap.name} {snap.stat().st_size} encore instable", flush=True)
            else:
                print(f"  pas encore de {snap.name}", flush=True)
            time.sleep(60)
        print(f"epoch{EPOCH}.pt pret", flush=True)
    gold = _gold()
    package(gold)
    write_notebook(gold)
    ensure_dataset()
    version = push_kernel()
    wait_kernel()
    submit(version, gold)


if __name__ == "__main__":
    main()
