# -*- coding: utf-8 -*-
"""Envoie le code, le checkpoint complet et lance la reprise Kaggle."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
import torch
from nbformat.v4 import new_code_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "vit_k_llrd08_v3b_oofmask_full" / "epoch9.pt"
YAML = ROOT / "configs" / "kneenet_vit_k_llrd08_v3b_oofmask_kaggle.yaml"
STAGE = Path(tempfile.mkdtemp(prefix="vitk_kaggle_"))
DATASETS = {
    "code": "sultent/rsna-knee-vitk-v3b-resume-code",
    "ckpt": "sultent/rsna-knee-vit-k-v3b-ep9-resume",
    "api": "sultent/rsna-knee-kaggle-api",
}


def kaggle(*args: str) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        text = exc.output if isinstance(exc.output, str) else str(exc)
        print(text, flush=True)
        raise


def meta(folder: Path, title: str, slug: str) -> None:
    (folder / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        ),
        encoding="utf-8",
    )


def ignore_pyc(_dir: str, names: list[str]) -> list[str]:
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]


def stage_code() -> Path:
    folder = STAGE / "code"
    folder.mkdir(parents=True)
    shutil.copytree(ROOT / "src", folder / "src", ignore=ignore_pyc)
    (folder / "configs").mkdir()
    shutil.copy2(YAML, folder / "configs" / YAML.name)
    for rel in (
        "data/labels/coolin/targets_v3b_oofmask_vitk.csv",
        "data/csv/train_series.csv",
        "data/csv/train.csv",
    ):
        dst = folder / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    meta(folder, "RSNA Knee ViT-k v3b resume code", DATASETS["code"])
    return folder


def stage_ckpt() -> Path:
    folder = STAGE / "ckpt"
    folder.mkdir()
    try:
        os.link(CKPT, folder / "epoch9.pt")
    except OSError:
        shutil.copy2(CKPT, folder / "epoch9.pt")
    meta(folder, "RSNA Knee ViT-k v3b ep9 resume", DATASETS["ckpt"])
    return folder


def stage_api() -> Path:
    folder = STAGE / "api"
    folder.mkdir()
    src = Path.home() / ".kaggle" / "kaggle.json"
    shutil.copy2(src, folder / "kaggle.json")
    meta(folder, "RSNA Knee Kaggle API", DATASETS["api"])
    return folder


def ensure(folder: Path, slug: str, message: str) -> None:
    try:
        status = kaggle("datasets", "status", slug)
    except subprocess.CalledProcessError:
        status = ""
    low = status.lower()
    if "ready" in low:
        print(kaggle(
            "datasets", "version", "-p", str(folder),
            "-m", message, "--dir-mode", "zip",
        ), flush=True)
    else:
        print(kaggle(
            "datasets", "create", "-p", str(folder), "--dir-mode", "zip",
        ), flush=True)
    deadline = time.time() + 45 * 60
    while time.time() < deadline:
        try:
            status = kaggle("datasets", "status", slug).strip()
        except subprocess.CalledProcessError:
            status = "(pending)"
        print("[ds]", slug, status, flush=True)
        if "ready" in status.lower() and "pending" not in status.lower():
            return
        time.sleep(20)
    raise SystemExit(f"timeout dataset {slug}")


def write_notebook() -> None:
    cell = (HERE / "train_cell.py").read_text(encoding="utf-8")
    nb = new_notebook(cells=[new_code_cell(cell)])
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    path = HERE / "train.ipynb"
    nbformat.write(nb, path)
    print("notebook", path, flush=True)


def main() -> None:
    print("stage", STAGE, flush=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if int(ckpt.get("epoch", -1)) != 9 or "optim" not in ckpt:
        raise SystemExit(f"epoch9 inutilisable: epoch={ckpt.get('epoch')}")
    print("epoch", ckpt["epoch"], "step", ckpt["step"], "adam", len(ckpt["optim"]["state"]), flush=True)
    del ckpt
    code = stage_code()
    api = stage_api()
    weights = stage_ckpt()
    ensure(code, DATASETS["code"], "code reprise ViT-k")
    ensure(api, DATASETS["api"], "credentials")
    ensure(weights, DATASETS["ckpt"], "epoch9 avec AdamW")
    write_notebook()
    print(kaggle(
        "kernels", "push", "-p", str(HERE),
        "--accelerator", "NvidiaTeslaT4", "-t", "43200",
    ), flush=True)


if __name__ == "__main__":
    main()
