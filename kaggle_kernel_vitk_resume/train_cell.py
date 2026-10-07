# -*- coding: utf-8 -*-
"""Reprise ViT-k epoch 9 -> 12 sur T4. Le batch, l'AdamW et le cosinus restent ceux du full train."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "timm==1.0.28", "kaggle"]
)


def one_dir(fragment: str) -> Path:
    # Kaggle monte les entrees sous /kaggle/input/datasets et .../notebooks.
    hits = [
        p for p in Path("/kaggle/input").rglob("*")
        if p.is_dir() and fragment in p.name
    ]
    if len(hits) != 1:
        names = sorted(p.name for p in Path("/kaggle/input").iterdir())
        raise SystemExit(f"{fragment} -> {[str(h) for h in hits]} | input {names}")
    return hits[0]


def show(path: Path, limit: int = 40) -> None:
    print("contenu", path, flush=True)
    n = 0
    for p in sorted(path.rglob("*")):
        kind = "dir" if p.is_dir() else str(p.stat().st_size)
        print(" ", p.relative_to(path), kind, flush=True)
        n += 1
        if n >= limit:
            print("  ...", flush=True)
            break


code_mount = one_dir("vitk-v3b-resume-code")
show(code_mount)
unpack = Path("/kaggle/working/unpacked")
if unpack.exists():
    shutil.rmtree(unpack)
unpack.mkdir()
for zpath in code_mount.rglob("*.zip"):
    print("unzip", zpath.name, flush=True)
    shutil.unpack_archive(str(zpath), unpack / zpath.stem)


def find_one(name: str) -> Path:
    hits: list[Path] = []
    seen: set[str] = set()
    for base in (code_mount, unpack, Path("/kaggle/input")):
        for hit in base.rglob(name):
            key = str(hit)
            if key not in seen:
                seen.add(key)
                hits.append(hit)
    if len(hits) != 1:
        raise SystemExit(f"{name} -> {[str(h) for h in hits]}")
    return hits[0]


ckpt = next(one_dir("vit-k-v3b-ep9-resume").rglob("*.pt"))
cred = json.loads(find_one("kaggle.json").read_text())
os.environ["KAGGLE_USERNAME"] = cred["username"]
os.environ["KAGGLE_KEY"] = cred["key"]
print("ckpt", ckpt.name, ckpt.stat().st_size, flush=True)
if ckpt.stat().st_size < 1_000_000_000:
    raise SystemExit("checkpoint trop petit : AdamW absent")

model_py = find_one("kneenet_vit_e.py")
src_dir = model_py.parents[1]
cfg = find_one("kneenet_vit_k_llrd08_v3b_oofmask_kaggle.yaml")
labels = find_one("targets_v3b_oofmask_vitk.csv")
series = find_one("train_series.csv")
train_csv = find_one("train.csv")
print("src", src_dir, flush=True)

root = Path("/kaggle/working/proj")
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
os.symlink(src_dir, root / "src", target_is_directory=True)
(root / "configs").mkdir()
os.symlink(cfg, root / "configs" / cfg.name)
for src, rel in (
    (labels, "data/labels/coolin/targets_v3b_oofmask_vitk.csv"),
    (series, "data/csv/train_series.csv"),
    (train_csv, "data/csv/train.csv"),
):
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)

vol = root / "data" / "crop130_rgb25d"
vol.mkdir(parents=True)
n_zip = 0
for src in Path("/kaggle/input").rglob("volumes_crop130_rgb25d_proto_part*.zip"):
    os.symlink(src, vol / src.name)
    n_zip += 1
for src in Path("/kaggle/input").rglob("manifest_rgb25d_proto_part*.csv"):
    dst = vol / src.name
    if not dst.exists():
        os.symlink(src, dst)
print("zips", n_zip, flush=True)
if n_zip != 6:
    raise SystemExit(f"attendu 6 zips, vu {n_zip}")
probe = vol / "volumes_crop130_rgb25d_proto_part0.zip"
if probe.stat().st_size < 1_000_000_000:
    raise SystemExit(f"zip part0 trop petit : {probe.stat().st_size}")

run_dir = root / "checkpoints" / "vit_k_llrd08_v3b_oofmask_full"
pushed: set[int] = set()


def kaggle_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "kaggle", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def push_epoch(pt: Path, epoch: int) -> None:
    stage = Path(f"/tmp/upload_ep{epoch}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy2(pt, stage / pt.name)
    hist = run_dir / "history.json"
    if hist.is_file():
        shutil.copy2(hist, stage / "history.json")
    slug = f"sultent/rsna-knee-vit-k-v3b-full-ep{epoch}"
    meta = {
        "title": f"RSNA Knee ViT-k v3b full ep{epoch}",
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    status = kaggle_cli("datasets", "status", slug)
    blob = (status.stdout + status.stderr).lower()
    if status.returncode == 0 and "ready" in blob:
        cmd = ["datasets", "version", "-p", str(stage), "-m", f"epoch {epoch} avec AdamW", "--dir-mode", "zip"]
    else:
        cmd = ["datasets", "create", "-p", str(stage), "--dir-mode", "zip"]
    done = kaggle_cli(*cmd)
    print(done.stdout, done.stderr, flush=True)
    if done.returncode != 0:
        raise RuntimeError(f"upload epoch {epoch} rc={done.returncode}")
    shutil.rmtree(stage, ignore_errors=True)
    pushed.add(epoch)
    print(f"dataset publie {slug}", flush=True)


def push_ready() -> None:
    if not run_dir.is_dir():
        return
    for pt in sorted(run_dir.glob("epoch*.pt")):
        epoch = int(pt.stem.replace("epoch", ""))
        if epoch < 10 or epoch in pushed:
            continue
        size = pt.stat().st_size
        if size < 1_000_000_000:
            continue
        time.sleep(3)
        if pt.stat().st_size != size:
            continue
        try:
            push_epoch(pt, epoch)
        except Exception as exc:
            print(f"upload epoch {epoch} en echec : {exc}", flush=True)


env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["OMP_NUM_THREADS"] = "1"
proc = subprocess.Popen(
    [
        sys.executable, "-u", "-m", "src.train.train",
        "--config", "configs/kneenet_vit_k_llrd08_v3b_oofmask_kaggle.yaml",
        "--resume", "--ckpt", str(ckpt),
    ],
    cwd=root,
    env=env,
)
while proc.poll() is None:
    push_ready()
    time.sleep(30)
push_ready()
if proc.returncode != 0:
    raise SystemExit(proc.returncode)
print("train termine", flush=True)
