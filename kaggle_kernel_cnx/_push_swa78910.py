# -*- coding: utf-8 -*-
"""SWA epochs 7-10 de cnx_r7_in22k_v3b_oofmask_full, puis soumission T4 sans TTA."""
from __future__ import annotations

import gc
import importlib.util
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_NAME = "cnx_r7_in22k_v3b_oofmask_full"
RUN = ROOT / "checkpoints" / RUN_NAME
EPOCHS = (7, 8, 9, 10)
SWA_LABEL = "swa-7-8-9-10"
SWA_NAME = "swa_ep7_10.pt"
WEIGHT = "cnx_r7_swa_ep7_10.pt"
DATASET = "sultent/rsna-knee-cnx-r7-swa78910"
KERNEL = "sultent/rsna-knee-cnx-r7-swa78910-sans-tta"
TITLE = "rsna-knee-cnx-r7-swa78910-sans-tta"
DS_TITLE = "RSNA Knee CNX-r7 IN22k v3b SWA ep7-10"
UPLOAD = ROOT / "kaggle_dataset_cnx" / "_upload_swa78910"
PUSH = ROOT / "kaggle_kernel_cnx" / "_push_swa78910"
OUT = ROOT / "kaggle_kernel_cnx" / "_out_swa78910"
UPLOAD_PY = ROOT / "kaggle_kernel_vit_e_dino_last" / "_upload_chunked.py"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
MIN_BYTES = 80_000_000
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


def _gold() -> tuple[float, dict[str, float]]:
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    by = {}
    for ep in EPOCHS:
        rec = next(r for r in hist if int(r["epoch"]) == ep)
        if rec.get("gold_macro_auc") is None:
            raise SystemExit(f"gold absent epoch {ep}")
        by[str(ep)] = float(rec["gold_macro_auc"])
    return sum(by.values()) / len(by), by


def _add(acc: dict, sd: dict, first: bool) -> tuple[int, int]:
    n_float = n_other = 0
    if first:
        if not sd:
            raise SystemExit("state_dict vide")
        for k, t in sd.items():
            t = t.detach().cpu()
            if t.is_floating_point():
                acc[k] = t.float().clone()
                n_float += 1
            else:
                acc[k] = t.clone()
                n_other += 1
        return n_float, n_other
    if list(sd.keys()) != list(acc.keys()):
        raise SystemExit("cles state_dict differentes entre epochs")
    for k, t in sd.items():
        t = t.detach().cpu()
        a = acc[k]
        if a.is_floating_point():
            if not t.is_floating_point() or t.shape != a.shape:
                raise SystemExit(f"{k}: type/shape incompatible")
            a += t.float()
            n_float += 1
        else:
            if not torch.equal(a, t):
                print(f"  warn buffer {k} non identique, on garde le dernier", flush=True)
                acc[k] = t.clone()
            n_other += 1
    return n_float, n_other


def _divide(acc: dict, sd_ref: dict, n: int) -> dict:
    out = {}
    for k, a in acc.items():
        ref = sd_ref[k]
        if a.is_floating_point() and ref.is_floating_point():
            out[k] = (a / float(n)).to(dtype=ref.dtype)
        else:
            out[k] = a
    return out


def _build(cfg: dict):
    from src.models.kneenet_cnx_r7 import KneeNet
    return KneeNet(
        backbone=str(cfg.get("backbone")),
        n_series=int(cfg.get("n_series", 6)),
        hidden=int(cfg.get("hidden", 384)),
        dropout=float(cfg.get("dropout", 0.1)),
        pretrained=False,
        encode_chunk=int(cfg.get("encode_chunk", 16)),
        max_depth=int(cfg.get("max_depth", 64) or 64),
    )


def build_swa() -> None:
    dst = RUN / SWA_NAME
    if dst.is_file() and dst.stat().st_size > MIN_BYTES:
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        got = tuple(int(e) for e in (ckpt.get("swa_epochs") or []))
        if str(ckpt.get("epoch")) == SWA_LABEL and got == EPOCHS:
            print("SWA deja present", dst.name, dst.stat().st_size, flush=True)
            del ckpt
            return
        del ckpt
    print(f"SWA {RUN_NAME} epochs {EPOCHS}", flush=True)
    acc: dict = {}
    cfg = None
    ref_sd = None
    n_float = n_other = 0
    for i, ep in enumerate(EPOCHS):
        src = RUN / f"epoch{ep}.pt"
        if not src.is_file():
            raise FileNotFoundError(src)
        print(f"load {src.name} {src.stat().st_size}", flush=True)
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        c = ckpt.get("config") or {}
        if str(c.get("model")) != "kneenet_cnx_r7":
            raise SystemExit(f"model {c.get('model')}")
        if str(c.get("run_name")) != RUN_NAME:
            raise SystemExit(c.get("run_name"))
        if str(c.get("backbone")) != "convnext_small.fb_in22k_ft_in1k":
            raise SystemExit(c.get("backbone"))
        if int(ckpt.get("epoch", -1)) != ep:
            raise SystemExit(f"epoch ckpt {ckpt.get('epoch')} != {ep}")
        if cfg is None:
            cfg = c
        sd = ckpt["model"]
        n_float, n_other = _add(acc, sd, first=i == 0)
        if ref_sd is None:
            ref_sd = {k: v.detach().cpu() for k, v in sd.items()}
        del sd, ckpt
        gc.collect()
    avg = _divide(acc, ref_sd, len(EPOCHS))
    del acc, ref_sd
    gc.collect()
    print(f"moyenne {len(EPOCHS)} : {n_float} tenseurs float, {n_other} buffers", flush=True)
    model = _build(cfg)
    model.load_state_dict(avg, strict=True)
    print("load strict OK SWA", flush=True)
    del model
    gc.collect()
    out = {
        "model": avg,
        "config": cfg,
        "epoch": SWA_LABEL,
        "swa_epochs": list(EPOCHS),
        "val_macro_auc": -1.0,
    }
    torch.save(out, dst)
    print(f"wrote {dst} {dst.stat().st_size}", flush=True)
    del out, avg
    gc.collect()


def package() -> float:
    src = RUN / SWA_NAME
    gold, by = _gold()
    print("package", src.name, src.stat().st_size, "gold", round(gold, 4), flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    if str(ckpt.get("epoch")) != SWA_LABEL:
        raise SystemExit(ckpt.get("epoch"))
    got = tuple(int(e) for e in (ckpt.get("swa_epochs") or []))
    if got != EPOCHS:
        raise SystemExit(f"swa_epochs {got}")
    if cfg.get("laterality"):
        raise SystemExit("laterality inattendue")
    if "crop130" not in str(cfg.get("zip_path", "")):
        raise SystemExit(f"zip_path {cfg.get('zip_path')}")
    cfg["laterality"] = False
    out = {
        "model": ckpt["model"],
        "config": cfg,
        "epoch": SWA_LABEL,
        "swa_epochs": list(EPOCHS),
        "gold_by_epoch": by,
        "gold_macro_auc": gold,
        "val_macro_auc": -1.0,
    }
    model = _build(cfg)
    model.load_state_dict(out["model"], strict=True)
    print("load strict OK package", flush=True)
    del model, ckpt
    gc.collect()
    UPLOAD.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD / WEIGHT
    torch.save(out, dest)
    if dest.stat().st_size < MIN_BYTES:
        raise SystemExit(f"poids trop petit {dest} {dest.stat().st_size}")
    meta = {
        "title": DS_TITLE,
        "id": DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (UPLOAD / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print("wrote", dest, dest.stat().st_size, flush=True)
    del out
    gc.collect()
    return gold


def write_notebook(gold: float) -> None:
    spec = importlib.util.spec_from_file_location(
        "make_nb_cnx", ROOT / "kaggle_kernel_cnx" / "_make_nb.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    local = str(UPLOAD.relative_to(ROOT)).replace("\\", "/")
    mod.build(
        gold=gold,
        weight=WEIGHT,
        dataset=DATASET,
        local_dir=local,
        out_dir=PUSH,
        kernel_id=KERNEL,
        title=TITLE,
    )


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
        if weight not in line:
            continue
        parts = line.split()
        for part in parts:
            if part.isdigit() and int(part) >= MIN_BYTES:
                return int(part)
    return None


def _dataset_state() -> str:
    listed = _files_text(DATASET)
    if listed is None:
        return "absent"
    size = _weight_size(listed, WEIGHT)
    if size is not None and size >= MIN_BYTES:
        return "ready"
    return "pending"


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def _transient(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "expecting value",
        "remote end closed",
        "connection aborted",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "502",
        "503",
        "504",
    ))


def ensure_dataset() -> None:
    pt = UPLOAD / WEIGHT
    if not pt.is_file() or pt.stat().st_size < MIN_BYTES:
        raise SystemExit(f"poids manquant: {pt}")
    state = _dataset_state()
    print("[ds]", state, flush=True)
    if state == "ready":
        return
    cmd = [
        sys.executable, "-X", "utf8", str(UPLOAD_PY),
        "--folder", str(UPLOAD),
        "--message", "CNX-r7 IN22k v3b SWA ep7-10",
    ]
    if state != "absent":
        cmd.append("--version")
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    t0 = time.time()
    while True:
        state = _dataset_state()
        print("[ds]", state, flush=True)
        if state == "ready":
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout dataset")
        time.sleep(20)


def push_kernel() -> str:
    meta = json.loads((PUSH / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("id") != KERNEL or meta.get("enable_internet"):
        raise SystemExit(f"metadata {meta.get('id')}")
    if meta.get("machine_shape") != ACCEL:
        raise SystemExit(meta.get("machine_shape"))
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
            if _transient(text):
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
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


def wait_kernel() -> None:
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", KERNEL).strip()
        up = raw.upper()
        print(raw, flush=True)
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            OUT.mkdir(parents=True, exist_ok=True)
            try:
                print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
            except subprocess.CalledProcessError:
                pass
            raise SystemExit("kernel en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        done = ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up
        if done and (seen or time.time() - t0 > 90):
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


def submit(version: str, gold: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and "cnx" in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    gold_txt = str(round(float(gold), 4))
    for needle in (
        "kneenet_cnx_r7",
        RUN_NAME,
        "epoch swa-7-8-9-10",
        "[7, 8, 9, 10]",
        f"gold in-sample {gold_txt}",
        "crop 130",
        "load strict OK",
        "sans TTA",
        "cnx forward fp32",
        "lignes 3",
        "submission.csv",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text or "lat True" in text:
        raise SystemExit("TTA ou flip")
    from kaggle.api.kaggle_api_extended import KaggleApi
    msg = (
        f"CNX-r7 IN22k v3b oofmask SWA ep7-10 (gold in-sample moyen {gold:.3f}, "
        "FOV 130 mm, ConvNeXt-Small, sans TTA, pas de flip, T4)"
    )
    (PUSH / "_message.txt").write_text(msg + "\n", encoding="utf-8")
    api = KaggleApi()
    api.authenticate()
    waited = False
    while True:
        print("submit", KERNEL, "version", version, flush=True)
        try:
            resp = api.competition_submit_code(
                "submission.csv", msg, COMPETITION, KERNEL, int(version), quiet=True,
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
            if _transient(body):
                print("api kaggle instable, retry 20s", flush=True)
                time.sleep(20)
                continue
            raise
        print(resp, flush=True)
        print("SOUMIS", KERNEL, "version", version, flush=True)
        return


def main() -> None:
    build_swa()
    gold = package()
    write_notebook(gold)
    if "--write" in sys.argv:
        print("WRITE_ONLY", round(gold, 4), flush=True)
        return
    ensure_dataset()
    version = push_kernel()
    wait_kernel()
    submit(version, gold)


if __name__ == "__main__":
    main()
