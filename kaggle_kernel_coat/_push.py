# -*- coding: utf-8 -*-
"""Datasets + kernels T4 CoAt-a v3b oofmask, sans TTA.

Trois kernels distincts (epoch 7, epoch 9, SWA ep7-10). Pas de
soumission competition : la limite journaliere est deja atteinte.
"""
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

from src.models.kneenet_coat_a import KneeNet  # noqa: E402

from _make_nb import RUN_NAME, build  # noqa: E402

KERNEL_DIR = ROOT / "kaggle_kernel_coat"
UPLOAD_PY = ROOT / "kaggle_kernel_vit_e_dino_last" / "_upload_chunked.py"
RUN = ROOT / "checkpoints" / RUN_NAME
POLL_S = 30
MAX_WAIT_S = 8 * 3600
ACCEL = "NvidiaTeslaT4"
KEEP_EPOCH = ("model", "config", "epoch", "val_macro_auc")
SWA_EPOCHS = (7, 8, 9, 10)
SWA_LABEL = "swa-7-8-9-10"

JOBS = (
    {
        "tag": "ep7",
        "src": "epoch7.pt",
        "weight": "coat_a_v3b_ep7.pt",
        "dataset": "sultent/rsna-knee-coat-a-v3b-ep7",
        "kernel": "sultent/rsna-knee-coat-a-v3b-ep7-sans-tta",
        "title": "RSNA Knee CoAt-a v3b ep7 sans TTA",
        "ds_title": "RSNA Knee CoAt-a v3b oofmask ep7",
        "swa": False,
        "epoch": 7,
    },
    {
        "tag": "ep9",
        "src": "epoch9.pt",
        "weight": "coat_a_v3b_ep9.pt",
        "dataset": "sultent/rsna-knee-coat-a-v3b-ep9",
        "kernel": "sultent/rsna-knee-coat-a-v3b-ep9-sans-tta",
        "title": "RSNA Knee CoAt-a v3b ep9 sans TTA",
        "ds_title": "RSNA Knee CoAt-a v3b oofmask ep9",
        "swa": False,
        "epoch": 9,
    },
    {
        "tag": "swa",
        "src": "swa_ep7_10.pt",
        "weight": "coat_a_v3b_swa_ep7_10.pt",
        "dataset": "sultent/rsna-knee-coat-a-v3b-swa",
        "kernel": "sultent/rsna-knee-coat-a-v3b-swa-sans-tta",
        "title": "RSNA Knee CoAt-a v3b SWA sans TTA",
        "ds_title": "RSNA Knee CoAt-a v3b oofmask SWA",
        "swa": True,
        "epoch": 0,
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


def _golds() -> dict[int, float]:
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    out: dict[int, float] = {}
    for rec in hist:
        if rec.get("gold_macro_auc") is None:
            continue
        out[int(rec["epoch"])] = float(rec["gold_macro_auc"])
    return out


def gold_of(job: dict, golds: dict[int, float]) -> float:
    if job["swa"]:
        vals = [golds[ep] for ep in SWA_EPOCHS]
        return sum(vals) / len(vals)
    return golds[int(job["epoch"])]


def upload_dir(job: dict) -> Path:
    return (ROOT / "kaggle_dataset_coat" / f"_upload_{job['tag']}").resolve()


def push_dir(job: dict) -> Path:
    return KERNEL_DIR / f"_push_{job['tag']}"


def out_dir(job: dict) -> Path:
    return KERNEL_DIR / f"_out_{job['tag']}"


def _check_common(cfg: dict, ckpt: dict) -> None:
    if cfg.get("run_name") != RUN_NAME:
        raise SystemExit(f"run inattendu {cfg.get('run_name')}")
    if str(cfg.get("model")) != "kneenet_coat_a":
        raise SystemExit(f"model={cfg.get('model')!r}")
    if str(cfg.get("backbone")) != "coatnet_0_rw_224.sw_in1k":
        raise SystemExit(f"backbone={cfg.get('backbone')!r}")
    if cfg.get("laterality"):
        raise SystemExit("laterality inattendue")
    if int(cfg.get("n_series", 0)) != 6:
        raise SystemExit(f"n_series={cfg.get('n_series')!r}")
    if "model" not in ckpt or len(ckpt["model"]) < 100:
        raise SystemExit("state_dict vide")


def _load_strict(cfg: dict, state: dict) -> None:
    model = KneeNet(
        pretrained=False,
        backbone=str(cfg.get("backbone")),
        img_size=224,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=8,
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    )
    model.load_state_dict(state, strict=True)
    print("load strict OK local", flush=True)
    del model


def strip(job: dict, gold: float, golds: dict[int, float]) -> None:
    src = RUN / job["src"]
    if not src.is_file():
        raise SystemExit(f"poids manquant: {src}")
    print("load", src.name, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    _check_common(cfg, ckpt)
    if job["swa"]:
        if str(ckpt.get("epoch")) != SWA_LABEL:
            raise SystemExit(f"epoch SWA {ckpt.get('epoch')!r}")
        got = tuple(int(e) for e in (ckpt.get("swa_epochs") or []))
        if got != SWA_EPOCHS:
            raise SystemExit(f"swa_epochs {got}")
        by_ep = ckpt.get("gold_by_epoch") or {}
        for ep in SWA_EPOCHS:
            stored = by_ep.get(str(ep))
            if stored is None or abs(float(stored) - golds[ep]) > 1e-6:
                raise SystemExit(f"gold SWA epoch {ep} incoherent")
        out = {
            "model": ckpt["model"],
            "config": cfg,
            "epoch": SWA_LABEL,
            "swa_epochs": list(SWA_EPOCHS),
            "gold_by_epoch": {str(ep): golds[ep] for ep in SWA_EPOCHS},
            "gold_macro_auc": gold,
            "val_macro_auc": -1.0,
        }
    else:
        epoch = int(job["epoch"])
        if int(ckpt.get("epoch", -1)) != epoch:
            raise SystemExit(f"epoch {ckpt.get('epoch')}, attendu {epoch}")
        out = {k: ckpt[k] for k in KEEP_EPOCH if k in ckpt and k != "gold_macro_auc"}
        out["config"] = cfg
        out["epoch"] = epoch
        out["gold_macro_auc"] = gold
    _load_strict(cfg, out["model"])
    dest_dir = upload_dir(job)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / job["weight"]
    torch.save(out, dest)
    if dest.stat().st_size < 80_000_000:
        raise SystemExit(f"poids trop petit: {dest} {dest.stat().st_size}")
    meta = {
        "title": job["ds_title"],
        "id": job["dataset"],
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dest_dir / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "strip", dest.name, dest.stat().st_size,
        "gold", round(gold, 4), flush=True,
    )
    del ckpt, out


def write_notebook(job: dict, gold: float) -> None:
    local = str(upload_dir(job).relative_to(ROOT)).replace("\\", "/")
    build(
        epoch=int(job["epoch"]),
        gold=gold,
        weight=job["weight"],
        dataset=job["dataset"],
        local_dir=local,
        out_dir=push_dir(job),
        kernel_id=job["kernel"],
        title=job["title"],
        swa=bool(job["swa"]),
    )


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


def _files_text(dataset: str) -> str | None:
    """None si le dataset n'existe pas. Le endpoint status renvoie 403."""
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


def _ready(status: str) -> bool:
    low = status.lower()
    return "ready" in low and "pending" not in low


def _dataset_state(job: dict) -> str:
    """absent | pending | ready. files prime sur status (403 systematique)."""
    listed = _files_text(job["dataset"])
    if listed is None:
        st = _status_dataset(job["dataset"])
        if _ready(st):
            return "ready"
        if st and "pending" in st.lower():
            return "pending"
        return "absent"
    size = _weight_size(listed, job["weight"])
    if size is not None and size >= 80_000_000:
        return "ready"
    return "pending"


def _quota(text: str) -> bool:
    low = text.lower()
    return "maximum batch gpu" in low or "quota" in low


def ensure_dataset(job: dict, *, reupload: bool) -> None:
    folder = upload_dir(job)
    pt = folder / job["weight"]
    if not pt.is_file() or pt.stat().st_size < 80_000_000:
        raise SystemExit(f"poids manquant: {pt}")
    state = _dataset_state(job)
    print(f"[ds {job['tag']}]", state, flush=True)
    if state == "ready" and not reupload:
        print(f"[ds {job['tag']}] deja pret", flush=True)
        return
    version = state != "absent"
    cmd = [
        sys.executable, "-X", "utf8", str(UPLOAD_PY),
        "--folder", str(folder),
        "--message", f"CoAt-a v3b oofmask {job['tag']} sans optim",
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
    push = push_dir(job)
    meta = json.loads((push / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("id") != job["kernel"]:
        raise SystemExit(f"kernel id {meta.get('id')}")
    if meta.get("machine_shape") != ACCEL:
        raise SystemExit(f"machine_shape={meta.get('machine_shape')!r}")
    if meta.get("enable_internet"):
        raise SystemExit("internet doit rester off")
    nb = (push / "submit.ipynb").read_text(encoding="utf-8")
    if "sans TTA" not in nb or "TTA_VARIANTS" in nb:
        raise SystemExit(f"notebook {job['tag']} TTA incoherent")
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
    kernel = job["kernel"]
    while True:
        raw = kaggle("kernels", "status", kernel).strip()
        up = raw.upper()
        print(f"[{job['tag']}]", raw, flush=True)
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            dest = out_dir(job)
            dest.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", kernel, "-p", str(dest)), flush=True)
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
    dest = out_dir(job)
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", job["kernel"], "-p", str(dest)), flush=True)
    csv = dest / "submission.csv"
    if not csv.is_file() or csv.stat().st_size == 0:
        raise SystemExit(f"submission.csv absent: {dest}")
    log = next(dest.glob("*.log"), None)
    if log is None:
        raise SystemExit(f"log absent: {dest}")
    text = log.read_text(encoding="utf-8", errors="replace")
    if "kneenet_coat_a" not in text or "lignes 3" not in text:
        raise SystemExit(f"log {job['tag']} inattendu")
    if "sans TTA" not in text or "TTA x" in text:
        raise SystemExit(f"log {job['tag']} TTA incoherent")
    if "load strict OK" not in text:
        raise SystemExit(f"log {job['tag']} sans load strict")
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
    p.add_argument("--reupload", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()
    jobs = selected(args.only)
    golds = _golds()
    if not args.skip_make:
        for job in jobs:
            gold = gold_of(job, golds)
            strip(job, gold, golds)
            write_notebook(job, gold)
    if args.make_only:
        print("MAKE_ONLY", flush=True)
        return
    for job in jobs:
        ensure_dataset(job, reupload=args.reupload)
    for job in jobs:
        version = push_kernel(job)
        wait_kernel(job)
        collect(job, version)
        print("DONE", job["tag"], version, flush=True)


if __name__ == "__main__":
    main()
