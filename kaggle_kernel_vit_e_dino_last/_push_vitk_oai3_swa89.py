# -*- coding: utf-8 -*-
"""SWA ep8-9 de vit_k_llrd08_320_v3b_oofmask_vitk_full_oai3, puis soumission T4 sans TTA."""
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

RUN_NAME = "vit_k_llrd08_320_v3b_oofmask_vitk_full_oai3"
RUN = ROOT / "checkpoints" / RUN_NAME
EPOCHS = (8, 9)
SWA_LABEL = "swa-8-9"
SWA_NAME = "swa_ep8_9.pt"
WEIGHT = "vit_k320_vitk_oai3_swa_ep8_9.pt"
DATASET = "sultent/rsna-knee-vit-k320-vitk-oai3-swa89"
KERNEL = "sultent/rsna-knee-vit-k320-vitk-oai3-swa89-sans-tta"
TITLE = "rsna-knee-vit-k320-vitk-oai3-swa89-sans-tta"
DS_TITLE = "RSNA Knee ViT-k320 vitk OAI3 SWA ep8-9"
UPLOAD = ROOT / "kaggle_dataset_vit_k" / "_upload_k320_vitk_oai3_swa89"
PUSH = ROOT / "kaggle_kernel_vit_e_dino_last" / "_push_k320_vitk_oai3_swa89"
OUT = ROOT / "kaggle_kernel_vit_e_dino_last" / "_out_k320_vitk_oai3_swa89"
UPLOAD_PY = ROOT / "kaggle_kernel_vit_e_dino_last" / "_upload_chunked.py"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
MIN_BYTES = 200_000_000
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


def _swa():
    spec = importlib.util.spec_from_file_location(
        "swa_four", ROOT / "kaggle_kernel_ensemble" / "_swa_four.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gold() -> tuple[float, dict[str, float]]:
    hist = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    by = {}
    for ep in EPOCHS:
        rec = next(r for r in hist if int(r["epoch"]) == ep)
        if rec.get("gold_macro_auc") is None:
            raise SystemExit(f"gold absent epoch {ep}")
        by[str(ep)] = float(rec["gold_macro_auc"])
    return sum(by.values()) / len(by), by


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
    swa = _swa()
    swa._one({
        "run": RUN_NAME,
        "epochs": EPOCHS,
        "expect_model": "kneenet_vit_k320",
        "out": SWA_NAME,
    })


def package() -> float:
    src = RUN / SWA_NAME
    if not src.is_file():
        raise SystemExit(f"SWA manquant: {src}")
    gold, by = _gold()
    print("package", src.name, src.stat().st_size, flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    if str(cfg.get("model")) != "kneenet_vit_k320":
        raise SystemExit(cfg.get("model"))
    if str(cfg.get("run_name")) != RUN_NAME:
        raise SystemExit(cfg.get("run_name"))
    if str(ckpt.get("epoch")) != SWA_LABEL:
        raise SystemExit(ckpt.get("epoch"))
    got = tuple(int(e) for e in (ckpt.get("swa_epochs") or []))
    if got != EPOCHS:
        raise SystemExit(f"swa_epochs {got}")
    if cfg.get("laterality"):
        raise SystemExit("laterality inattendue")
    if int(cfg.get("img_size", 0)) != 320:
        raise SystemExit(f"img_size {cfg.get('img_size')}")
    cfg["laterality"] = False
    stored = ckpt.get("gold_by_epoch") or {}
    for ep in EPOCHS:
        prev = stored.get(str(ep))
        if prev is None or abs(float(prev) - by[str(ep)]) > 1e-6:
            raise SystemExit(f"gold incoherent ep {ep}: {prev} vs {by[str(ep)]}")
    out = {
        "model": ckpt["model"],
        "config": cfg,
        "epoch": SWA_LABEL,
        "swa_epochs": list(EPOCHS),
        "gold_by_epoch": by,
        "gold_macro_auc": gold,
        "val_macro_auc": -1.0,
    }
    from src.models.kneenet_vit_k320 import KneeNet
    model = KneeNet(
        img_size=320,
        n_series=int(cfg.get("n_series", 6)),
        dropout=float(cfg.get("dropout", 0.4)),
        encode_chunk=8,
        head_dim=int(cfg.get("head_dim", 512)),
        lstm_hidden=int(cfg.get("lstm_hidden", 256)),
        drop_path_rate=float(cfg.get("drop_path_rate") or 0.0),
        pretrained=False,
        grad_checkpoint=False,
    )
    model.load_state_dict(out["model"], strict=True)
    print("load strict OK gold", round(gold, 4), "encode_chunk", cfg.get("encode_chunk"), flush=True)
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
    torch.save(out, src)
    print("wrote", dest, dest.stat().st_size, flush=True)
    del out
    gc.collect()
    return gold


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
    spec = importlib.util.spec_from_file_location(
        "make_nb_vit", ROOT / "kaggle_kernel_vit_e_dino_last" / "_make_nb.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    local = str(UPLOAD.relative_to(ROOT)).replace("\\", "/")
    mod.build(
        8,
        gold,
        WEIGHT,
        run_name=RUN_NAME,
        dataset=DATASET,
        local_dir=local,
        tta=False,
        out_dir=PUSH,
        kernel_id=KERNEL,
        title=TITLE,
        img_size=320,
        encode_chunk_cap=2,
    )
    nb_path = PUSH / "submit.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    _replace(
        nb,
        '    if int(ckpt.get("epoch", -1)) != 8:\n'
        '        raise RuntimeError(f"epoch inattendue {ckpt.get(\'epoch\')}")\n',
        '    if str(ckpt.get("epoch")) != "swa-8-9":\n'
        '        raise RuntimeError(f"epoch inattendue {ckpt.get(\'epoch\')}")\n'
        '    if list(ckpt.get("swa_epochs") or []) != [8, 9]:\n'
        '        raise RuntimeError(f"swa_epochs inattendu {ckpt.get(\'swa_epochs\')}")\n',
        1,
    )
    _replace(
        nb,
        "# RSNA Knee — ViT-k LLRD 0.8 v3b (epoch 8) — T4 — sans TTA",
        "# RSNA Knee — ViT-k320 vitk OAI3 SWA ep8-9 — T4 — sans TTA",
        1,
    )
    _replace(
        nb,
        f"train `{RUN_NAME}` **epoch 8**",
        f"train `{RUN_NAME}` **SWA epochs 8-9**",
        1,
    )
    _replace(
        nb,
        "          '| epoch', ckpt.get('epoch'), '| gold in-sample', gold_txt,\n",
        "          '| epoch', ckpt.get('epoch'), '| swa', ckpt.get('swa_epochs'),\n"
        "          '| gold in-sample', gold_txt,\n",
        1,
    )
    blob = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in (
        RUN_NAME, WEIGHT, "img_size=320", "kneenet_vit_k320",
        "swa-8-9", "[8, 9]", "sans TTA", "encode_chunk = 2",
    ):
        if needle not in blob:
            raise SystemExit(f"notebook incomplet: {needle}")
    if "TTA_VARIANTS" in blob or "!= 8" in blob or "img_size=224" in blob:
        raise SystemExit("notebook garde le TTA, un garde epoch 8 ou du 224")
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": KERNEL,
        "title": TITLE,
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
    print("ecrit", nb_path, flush=True)


def _quota(text: str) -> bool:
    low = text.lower()
    return "maximum batch gpu" in low or "quota" in low


def _status_dataset() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    api = KaggleApi()
    api.authenticate()
    owner, slug = DATASET.split("/")
    try:
        with api.build_kaggle_client() as client:
            req = ApiGetDatasetRequest()
            req.owner_slug = owner
            req.dataset_slug = slug
            ds = client.datasets.dataset_api_client.get_dataset(req)
    except Exception as e:
        text = str(e).lower()
        if any(s in text for s in ("404", "403", "not found", "forbidden")):
            return ""
        raise
    statuses = [str(v.status or "") for v in (ds.versions or [])]
    if any("pending" in s.lower() for s in statuses):
        return "pending"
    if ds.current_version_number and any(s.lower() == "ready" for s in statuses):
        return "ready"
    return "pending"


def ensure_dataset() -> None:
    pt = UPLOAD / WEIGHT
    if not pt.is_file() or pt.stat().st_size < MIN_BYTES:
        raise SystemExit(f"poids manquant: {pt}")
    st = _status_dataset()
    print("[ds]", st or "(absent)", flush=True)
    ready = "ready" in st.lower() and "pending" not in st.lower()
    if ready:
        print("[ds] deja pret", flush=True)
        return
    cmd = [
        sys.executable, "-X", "utf8", str(UPLOAD_PY),
        "--folder", str(UPLOAD),
        "--message", "ViT-k320 vitk OAI3 SWA ep8-9",
    ]
    if st and "pending" in st.lower():
        cmd.append("--version")
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    t0 = time.time()
    while True:
        st = _status_dataset()
        print("[ds]", st or "(pending)", flush=True)
        if "ready" in st.lower() and "pending" not in st.lower():
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit("timeout dataset")
        time.sleep(20)


def push_kernel() -> str:
    meta = json.loads((PUSH / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("id") != KERNEL or meta.get("dataset_sources") != [DATASET]:
        raise SystemExit(meta)
    if meta.get("machine_shape") != ACCEL or meta.get("enable_internet"):
        raise SystemExit("kernel meta GPU/internet")
    nb = (PUSH / "submit.ipynb").read_text(encoding="utf-8")
    if "TTA_VARIANTS" in nb or "sans TTA" not in nb or "swa-8-9" not in nb:
        raise SystemExit("notebook incoherent avant push")
    print("push", KERNEL, "T4", flush=True)
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


def submit(version: str, gold: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower())
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (RUN_NAME, SWA_LABEL, "[8, 9]", "sans TTA", "lignes 3", "kneenet_vit_k320"):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text:
        raise SystemExit("log avec TTA")
    from kaggle.api.kaggle_api_extended import KaggleApi
    msg = (
        f"ViT-k320 vitk OAI3 SWA ep8-9 (gold in-sample {gold:.3f}, "
        "pas de flip, sans TTA, T4)"
    )
    api = KaggleApi()
    api.authenticate()
    print("submit", KERNEL, "version", version, flush=True)
    resp = api.competition_submit_code(
        "submission.csv", msg, COMPETITION, KERNEL, int(version), quiet=True,
    )
    print(resp, flush=True)
    print("SOUMIS", KERNEL, "version", version, flush=True)


def main() -> None:
    build_swa()
    gold = package()
    write_notebook(gold)
    ensure_dataset()
    version = push_kernel()
    wait_kernel()
    submit(version, gold)


if __name__ == "__main__":
    main()
