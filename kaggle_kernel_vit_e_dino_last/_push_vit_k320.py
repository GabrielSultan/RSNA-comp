# -*- coding: utf-8 -*-
"""Datasets + kernels T4 ViT-k320 v3b oofmask v20h, TTA x5, puis soumission."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kneenet_vit_k320 import KneeNet as KneeNetK320  # noqa: E402

KERNEL_DIR = ROOT / "kaggle_kernel_vit_e_dino_last"
MAKE = KERNEL_DIR / "_make_nb.py"
UPLOAD_PY = KERNEL_DIR / "_upload_chunked.py"
RUN = ROOT / "checkpoints" / "vit_k_llrd08_320_v3b_oofmask_v20h_full"
RUN_NAME = "vit_k_llrd08_320_v3b_oofmask_v20h_full"
COMPETITION = "rsna-knee-abnormality-detection"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
ACCEL = "NvidiaTeslaT4"
IMG_SIZE = 320
CHUNK_CAP = 2
EPOCHS = (10, 9, 8, 7)
KEEP = ("model", "config", "epoch", "gold_macro_auc")


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        print(text, flush=True)
        raise


def gold_of(epoch: int) -> float:
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    rec = next(r for r in hist if int(r["epoch"]) == epoch)
    return float(rec["gold_macro_auc"])


def dataset_id(epoch: int) -> str:
    return f"sultent/rsna-knee-vit-k320-v3b-ep{epoch}"


def kernel_id(epoch: int) -> str:
    return f"sultent/rsna-knee-vit-k320-v3b-ep{epoch}-tta-x5"


def upload_dir(epoch: int) -> Path:
    return (ROOT / "kaggle_dataset_vit_k" / f"_upload_k320_ep{epoch}").resolve()


def push_dir(epoch: int) -> Path:
    return KERNEL_DIR / f"_push_k320_ep{epoch}"


def out_dir(epoch: int) -> Path:
    return KERNEL_DIR / f"_out_k320_ep{epoch}"


def weight_name(epoch: int) -> str:
    return f"vit_k320_v3b_ep{epoch}.pt"


def strip_epoch(epoch: int) -> float:
    src = RUN / f"epoch{epoch}.pt"
    if not src.is_file():
        raise SystemExit(f"poids manquant: {src}")
    gold = gold_of(epoch)
    print("load", src.name, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    if cfg.get("run_name") != RUN_NAME:
        raise SystemExit(f"run inattendu {cfg.get('run_name')}")
    if str(cfg.get("model")) != "kneenet_vit_k320":
        raise SystemExit(f"model={cfg.get('model')!r}")
    if int(cfg.get("img_size", 0)) != IMG_SIZE:
        raise SystemExit(f"img_size={cfg.get('img_size')!r}")
    if cfg.get("laterality"):
        raise SystemExit("laterality inattendue")
    if int(ckpt.get("epoch", -1)) != epoch:
        raise SystemExit(f"epoch {ckpt.get('epoch')}, attendu {epoch}")
    out = {k: ckpt[k] for k in KEEP if k in ckpt and k != "gold_macro_auc"}
    out["config"] = cfg
    out["epoch"] = epoch
    out["gold_macro_auc"] = gold
    dest_dir = upload_dir(epoch)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / weight_name(epoch)
    torch.save(out, dest)
    model = KneeNetK320(
        pretrained=False,
        img_size=IMG_SIZE,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=CHUNK_CAP,
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
        grad_checkpoint=False,
    )
    model.load_state_dict(out["model"], strict=True)
    print(
        "strip", dest.name, dest.stat().st_size,
        "epoch", epoch, "gold", round(gold, 4), flush=True,
    )
    meta = {
        "title": f"RSNA Knee ViT-k320 v3b ep{epoch}",
        "id": dataset_id(epoch),
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dest_dir / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    del ckpt, model
    return gold


def write_notebook(epoch: int, gold: float) -> None:
    local = str(upload_dir(epoch).relative_to(ROOT)).replace("\\", "/")
    subprocess.check_call([
        sys.executable, "-X", "utf8", str(MAKE),
        "--epoch", str(epoch),
        "--gold", f"{gold:.6f}",
        "--weight", weight_name(epoch),
        "--run-name", RUN_NAME,
        "--dataset", dataset_id(epoch),
        "--local-dir", local,
        "--out-dir", str(push_dir(epoch)),
        "--img-size", str(IMG_SIZE),
        "--encode-chunk-cap", str(CHUNK_CAP),
        "--kernel-id", kernel_id(epoch),
        "--title", f"RSNA Knee ViT-k320 v3b ep{epoch} TTA x5",
    ])
    nb = (push_dir(epoch) / "submit.ipynb").read_text(encoding="utf-8")
    if "img_size=320" not in nb or "TTA_VARIANTS" not in nb:
        raise SystemExit(f"notebook ep{epoch} incomplet")
    if "img_size=224" in nb:
        raise SystemExit(f"notebook ep{epoch} encore en 224")


def _status_dataset(epoch: int) -> str:
    try:
        return kaggle("datasets", "status", dataset_id(epoch)).strip()
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


def _quota(text: str) -> bool:
    low = text.lower()
    return "maximum batch gpu" in low or "quota" in low


def ensure_dataset(epoch: int) -> None:
    folder = upload_dir(epoch)
    pt = folder / weight_name(epoch)
    if not pt.is_file() or pt.stat().st_size < 50_000_000:
        raise SystemExit(f"poids manquant: {pt}")
    st = _status_dataset(epoch)
    print(f"[ds ep{epoch}]", st or "(absent)", flush=True)
    if _ready(st):
        print(f"[ds ep{epoch}] deja pret", flush=True)
        return
    if not (st and "pending" in st.lower()):
        cmd = [sys.executable, "-X", "utf8", str(UPLOAD_PY), "--folder", str(folder)]
        print("+", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
    t0 = time.time()
    while True:
        st = _status_dataset(epoch)
        print(f"[ds ep{epoch}]", st or "(pending)", flush=True)
        if _ready(st):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout dataset ep{epoch}")
        time.sleep(20)


def push_kernel(epoch: int) -> str:
    push = push_dir(epoch)
    meta = json.loads((push / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("id") != kernel_id(epoch):
        raise SystemExit(f"kernel id {meta.get('id')}")
    if meta.get("machine_shape") != ACCEL:
        raise SystemExit(f"machine_shape={meta.get('machine_shape')!r}")
    print("push", kernel_id(epoch), "T4", flush=True)
    while True:
        try:
            out = kaggle(
                "kernels", "push", "-p", str(push),
                "--accelerator", ACCEL, "-t", "32400",
            )
        except subprocess.CalledProcessError as e:
            text = e.output if isinstance(e.output, str) else str(e)
            if _quota(text):
                print(f"quota GPU ep{epoch}, retry dans 60s", flush=True)
                time.sleep(60)
                continue
            raise
        print(out, flush=True)
        if _quota(out):
            print(f"quota GPU ep{epoch}, retry dans 60s", flush=True)
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+) successfully pushed", out, flags=re.I)
        if not m:
            raise RuntimeError(f"version introuvable: {out!r}")
        print("kernel version", kernel_id(epoch), m.group(1), flush=True)
        return m.group(1)


def wait_kernel(epoch: int) -> None:
    t0 = time.time()
    seen_running = False
    while True:
        raw = kaggle("kernels", "status", kernel_id(epoch)).strip()
        up = raw.upper()
        print(f"[ep{epoch}]", raw, flush=True)
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            dest = out_dir(epoch)
            dest.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", kernel_id(epoch), "-p", str(dest)), flush=True)
            except subprocess.CalledProcessError:
                pass
            raise SystemExit(f"kernel ep{epoch} en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen_running = True
        done = ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up
        if done and (seen_running or (time.time() - t0) > 90):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout kernel ep{epoch}")
        time.sleep(POLL_S)


def submit(epoch: int, version: str) -> None:
    gold = gold_of(epoch)
    msg = (
        f"ViT-k320 LLRD 0.8 v3b oofmask v20h full epoch {epoch} "
        f"(gold in-sample {gold:.3f}, pas de flip, TTA x5, T4)"
    )
    dest = out_dir(epoch)
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", kernel_id(epoch), "-p", str(dest)), flush=True)
    csv = dest / "submission.csv"
    if not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit(f"submission.csv absent: {dest}")
    log = next(dest.glob("*.log"), None)
    if log is None:
        raise SystemExit(f"log absent: {dest}")
    text = log.read_text(encoding="utf-8", errors="replace")
    if "TTA x5" not in text or "kneenet_vit_k320" not in text:
        raise SystemExit(f"log ep{epoch} inattendu")
    if "lignes 3" not in text:
        raise SystemExit(f"submission ep{epoch} pas 3 lignes")
    print(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", kernel_id(epoch), "-v", version,
        "-f", "submission.csv", "-m", msg,
    ), flush=True)
    print("submitted", kernel_id(epoch), "version", version, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, nargs="*", default=list(EPOCHS))
    p.add_argument("--skip-make", action="store_true")
    args = p.parse_args()
    if not args.skip_make:
        for epoch in args.epoch:
            gold = strip_epoch(epoch)
            write_notebook(epoch, gold)
    for epoch in args.epoch:
        ensure_dataset(epoch)
        version = push_kernel(epoch)
        wait_kernel(epoch)
        submit(epoch, version)
        print("DONE", epoch, version, flush=True)


if __name__ == "__main__":
    main()
