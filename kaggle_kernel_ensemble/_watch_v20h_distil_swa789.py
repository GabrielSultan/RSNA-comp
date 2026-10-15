# -*- coding: utf-8 -*-
"""Attend la fin du full v20h distil, SWA 7-8-9, soumission Kaggle sans TTA.

Ne pas importer torch tant que le train tourne : la 3070 est prise.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_NAME = "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full"
RUN = ROOT / "checkpoints" / RUN_NAME
EPOCHS = (7, 8, 9)
LAST = 9
SWA_LABEL = "swa-7-8-9"
WEIGHT = "v20h_crop145_distil_swa_ep7_9.pt"
DATASET = "sultent/rsna-knee-v20h-crop145-distil-swa789"
SLUG = "rsna-knee-v20h-crop145-distil-swa789-sans-tta"
KERNEL = f"sultent/{SLUG}"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
UPLOAD = ROOT / "kaggle_dataset_v20h_lat" / "_upload_crop145_distil_swa789"
PUSH = ROOT / "kaggle_kernel_ensemble" / "_push_v20h_crop145_distil_swa789"
OUT = ROOT / "kaggle_kernel_ensemble" / "_out_v20h_crop145_distil_swa789"
SRC_NB = ROOT / "kaggle_kernel_ensemble" / "_push_v20h_crop145_swa678" / "submit.ipynb"
UPLOAD_PY = ROOT / "kaggle_kernel_vit_e_dino_last" / "_upload_chunked.py"
LOG = ROOT / "kaggle_kernel_ensemble" / "_watch_v20h_distil_swa789.log"
POLL_S = 60
TRAIN_WAIT_S = 18 * 3600
KERNEL_WAIT_S = 6 * 3600


def say(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    say("+ " + " ".join(cmd))
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as e:
        text = e.output if isinstance(e.output, str) else str(e)
        say(text)
        raise


def _train_alive() -> bool | None:
    try:
        out = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                "| Select-Object -ExpandProperty CommandLine",
            ],
            stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
    except (subprocess.CalledProcessError, OSError) as e:
        say(f"process check ignore : {e}")
        return None
    return "src.train.train" in out and RUN_NAME in out


def _history() -> list[dict]:
    p = RUN / "history.json"
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _epoch_ready(ep: int) -> bool:
    snap = RUN / f"epoch{ep}.pt"
    if not snap.is_file() or snap.stat().st_size < 80_000_000:
        return False
    hist = _history()
    if not any(
        int(r.get("epoch", -1)) == ep and r.get("gold_macro_auc") is not None
        for r in hist
    ):
        return False
    size = snap.stat().st_size
    time.sleep(15)
    return snap.is_file() and snap.stat().st_size == size


def wait_train() -> None:
    say(f"attente epoch{LAST}.pt de {RUN_NAME}")
    t0 = time.time()
    seen_alive = False
    while True:
        alive = _train_alive()
        if alive:
            seen_alive = True
        hist = _history()
        done = [int(r["epoch"]) for r in hist]
        snaps = sorted(
            p.name for p in RUN.glob("epoch*.pt") if p.stat().st_size > 80_000_000
        )
        say(
            f"attente ep{LAST} | history {done or '-'} | snaps {snaps or '-'} "
            f"| train {'oui' if alive else 'non' if alive is False else '?'}"
        )
        finished = (RUN / "gold_sanity.json").is_file()
        ready = _epoch_ready(LAST)
        if ready and finished and alive is not True:
            for ep in EPOCHS:
                if not (RUN / f"epoch{ep}.pt").is_file():
                    raise SystemExit(f"epoch{ep}.pt manquant")
            say(f"PRET epoch{LAST}")
            return
        if ready and not finished:
            say(f"epoch{LAST}.pt ecrit, attente gold_sanity.json")
        elif alive is False and seen_alive and (time.time() - t0) > 120:
            raise SystemExit(
                f"TRAIN_MORT avant epoch{LAST} | history {done} | snaps {snaps}"
            )
        if time.time() - t0 > TRAIN_WAIT_S:
            raise SystemExit(f"timeout train avant epoch{LAST}")
        time.sleep(POLL_S)


def _swa():
    spec = importlib.util.spec_from_file_location(
        "swa_oai3", ROOT / "kaggle_kernel_ensemble" / "_swa_oai3.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _golds() -> dict[str, float]:
    hist = _history()
    out = {}
    for ep in EPOCHS:
        rec = next(r for r in hist if int(r["epoch"]) == ep)
        if rec.get("gold_macro_auc") is None:
            raise SystemExit(f"gold absent epoch {ep}")
        out[str(ep)] = float(rec["gold_macro_auc"])
    return out


def build_swa(swa) -> float:
    import torch

    dst = RUN / "swa_ep7_9.pt"
    say(f"SWA {EPOCHS}")
    golds = _golds()
    acc: dict = {}
    cfg = None
    ref_sd = None
    for i, ep in enumerate(EPOCHS):
        src = RUN / f"epoch{ep}.pt"
        say(f"load {src.name} {src.stat().st_size}")
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
        if c.get("oai_sample_weight") not in (None, 0, 0.0):
            raise SystemExit(f"oai_sample_weight {c.get('oai_sample_weight')}")
        if cfg is None:
            cfg = c
        swa._add(acc, ckpt["model"], first=i == 0)
        if ref_sd is None:
            ref_sd = {k: v.detach().cpu() for k, v in ckpt["model"].items()}
        del ckpt
        gc.collect()
        say(f"  epoch {ep} gold {golds[str(ep)]:.4f}")
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
    say("load strict OK")
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
    say(f"wrote {dst} {dst.stat().st_size} gold {gold:.4f}")
    UPLOAD.mkdir(parents=True, exist_ok=True)
    up = UPLOAD / WEIGHT
    torch.save(payload, up)
    meta = {
        "title": "RSNA Knee v20h crop145 distil SWA ep7-9",
        "id": DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (UPLOAD / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    say(f"wrote {up} {up.stat().st_size}")
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


def notebook(gold: float) -> dict:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    local = str(UPLOAD.relative_to(ROOT)).replace("\\", "/")
    pairs = (
        (
            "v20h_crop145_dino_ep11_oof_v3b_full_oaiv2_wd01",
            RUN_NAME,
            2,
        ),
        ("v20h_crop145_swa_ep6_8.pt", WEIGHT, 2),
        ("sultent/rsna-knee-v20h-crop145-swa678", DATASET, 1),
        ("kaggle_dataset_v20h_lat/_upload_crop145_swa678", local, 1),
        (
            "oaiv2 wd01 SWA ep6-8",
            "distil SWA ep7-9",
            1,
        ),
        (
            "encodeur DINO, SWA epochs 6-8, gold in-sample moyen 0.964, wd 0.01, dropout 0.2",
            "encodeur DINO, SWA epochs 7-9, gold in-sample moyen "
            f"{gold:.3f}, wd 0.01, dropout 0.2, RSNA seul",
            1,
        ),
        (
            "SWA uniforme des epochs 6, 7 et 8. FOV 145 mm. Pas de canon laterality.",
            "SWA uniforme des epochs 7, 8 et 9. FOV 145 mm. RSNA seul, pas d'OAI. "
            "Pas de canon laterality.",
            1,
        ),
        (
            "dataset v20h crop145 SWA ep6-8 attache",
            "dataset v20h crop145 distil SWA ep7-9 attache",
            1,
        ),
        (
            "v20h DINO crop145 SWA ep6-8",
            "v20h DINO crop145 distil SWA ep7-9",
            2,
        ),
        (
            "    if str(ckpt.get('epoch')) != 'swa-6-7-8':\n"
            "        raise RuntimeError(f\"epoch inattendue {ckpt.get('epoch')}\")\n"
            "    if list(ckpt.get('swa_epochs') or []) != [6, 7, 8]:\n"
            "        raise RuntimeError(f\"swa_epochs inattendu {ckpt.get('swa_epochs')}\")\n",
            "    if str(ckpt.get('epoch')) != 'swa-7-8-9':\n"
            "        raise RuntimeError(f\"epoch inattendue {ckpt.get('epoch')}\")\n"
            "    if list(ckpt.get('swa_epochs') or []) != [7, 8, 9]:\n"
            "        raise RuntimeError(f\"swa_epochs inattendu {ckpt.get('swa_epochs')}\")\n",
            1,
        ),
    )
    for old, new, expect in pairs:
        _replace(nb, old, new, expect)
    blob = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in (
        RUN_NAME, WEIGHT, "sans TTA", "145.0", "swa-7-8-9", "[7, 8, 9]",
        "pas d'OAI",
    ):
        if needle not in blob:
            raise SystemExit(f"notebook incomplet: {needle}")
    if "oaiv2" in blob or "TTA_VARIANTS" in blob or "swa-6-7-8" in blob:
        raise SystemExit("ancien run ou TTA encore present")
    if "CROP_MM, CROP_REL, BAND = 130.0" in blob:
        raise SystemExit("crop 130 encore present")
    return nb


def write_notebook(gold: float) -> None:
    nb = notebook(gold)
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
    say(f"ecrit {PUSH / 'submit.ipynb'}")


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
    say(f"[ds] {st or '(absent)'}")
    if "pending" in st:
        cmd = None
    else:
        cmd = [
            sys.executable, "-X", "utf8", str(UPLOAD_PY),
            "--folder", str(UPLOAD),
            "--message", "v20h crop145 distil SWA ep7-9",
        ]
        if "ready" in st:
            cmd.append("--version")
    if cmd:
        say("+ " + " ".join(cmd))
        subprocess.check_call(cmd)
    t0 = time.time()
    while True:
        st = _status_dataset().lower()
        say(f"[ds] {st or '(pending)'}")
        if "ready" in st and "pending" not in st:
            return
        if time.time() - t0 > KERNEL_WAIT_S:
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
                say("quota GPU, retry 60s")
                time.sleep(60)
                continue
            raise
        say(out)
        if _quota(out):
            say("quota GPU, retry 60s")
            time.sleep(60)
            continue
        m = re.search(r"Kernel version (\d+)", out, flags=re.I)
        if not m:
            raise RuntimeError(out)
        say(f"kernel version {m.group(1)}")
        return m.group(1)


def wait_kernel() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KERNEL).strip()
        say(raw)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            OUT.mkdir(parents=True, exist_ok=True)
            try:
                say(kaggle("kernels", "output", KERNEL, "-p", str(OUT)))
            except subprocess.CalledProcessError:
                pass
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > KERNEL_WAIT_S:
            raise SystemExit("timeout kernel")
        time.sleep(POLL_S)


def submit(version: str, gold: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    say(kaggle("kernels", "output", KERNEL, "-p", str(OUT)))
    log = next(OUT.glob("*.log"))
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (RUN_NAME, "swa-7-8-9", "sans TTA", "lignes 3", "laterality False"):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text:
        raise SystemExit("TTA dans le log")
    msg = (
        f"v20h DINO crop145 distil Original_main SWA ep7-9 "
        f"(gold moyen {gold:.3f}, FOV 145 mm, RSNA seul, dropout 0.2, "
        f"wd 0.01, sans TTA, pas de flip, T4)"
    )
    say(kaggle(
        "competitions", "submit", COMPETITION,
        "-k", KERNEL, "-v", version,
        "-f", "submission.csv", "-m", msg,
    ))
    say(f"SOUMIS {KERNEL} version {version}")


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if "--check-nb" in sys.argv:
        notebook(0.964)
        say("notebook OK")
        return
    wait_train()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gold = build_swa(_swa())
    write_notebook(gold)
    ensure_dataset()
    version = push_kernel()
    wait_kernel()
    submit(version, gold)


if __name__ == "__main__":
    main()
