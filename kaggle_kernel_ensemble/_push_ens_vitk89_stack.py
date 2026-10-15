# -*- coding: utf-8 -*-
"""Deux soumissions sans TTA : stack logreg (notebook 13) et MLP (notebook 15)."""
from __future__ import annotations

import ast
import base64
import io
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_vitk89_c145" / "submit.ipynb"
STACK_DIR = KERNEL / "_stack_vitk89"
COMPETITION = "rsna-knee-abnormality-detection"
ACCEL = "NvidiaTeslaT4"
POLL_S = 30
MAX_WAIT_S = 8 * 3600
DATASETS = [
    "sultent/rsna-knee-vit-k320-vitk-oai3-swa89",
    "sultent/rsna-knee-v20b-dino-oai3-swa",
    "sultent/rsna-knee-v20h-crop145-swa678",
    "sultent/rsna-knee-coat-a-oai3-swa",
]
JOBS = (
    {
        "kind": "logreg",
        "slug": "rsna-knee-ens-vitk89-logreg",
        "msg": (
            "ens ViT-k320 SWA 8-9 + v20b OAI3 + v20h crop145 SWA 6-8 + CoAt OAI3, "
            "stack logreg notebook 13, sans TTA, pas de flip, T4"
        ),
    },
    {
        "kind": "mlp",
        "slug": "rsna-knee-ens-vitk89-mlp",
        "msg": (
            "ens ViT-k320 SWA 8-9 + v20b OAI3 + v20h crop145 SWA 6-8 + CoAt OAI3, "
            "stack MLP notebook 15, sans TTA, pas de flip, T4"
        ),
    },
)

MIX_OLD = """# Ordre : ViT-k320, v20b DINO, v20h DINO, CoAt-a. Somme 1, plancher 0.
N_MODELS = 4
MIX = [
    (0.09, 0.32, 0.55, 0.04),  # ACL
    (0.03, 0.44, 0.52, 0.01),  # MCL
    (0.27, 0.29, 0.30, 0.14),  # Medial Meniscus
    (0.18, 0.36, 0.41, 0.05),  # Lateral Meniscus
    (0.17, 0.32, 0.33, 0.18),  # Medial OA
    (0.21, 0.29, 0.29, 0.21),  # Lateral OA
    (0.14, 0.32, 0.42, 0.12),  # PF OA
    (0.28, 0.23, 0.22, 0.27),  # Effusion
    (0.30, 0.26, 0.24, 0.20),  # Synovitis
    (0.26, 0.27, 0.31, 0.16),  # Baker's
    (0.19, 0.31, 0.35, 0.15),  # Contusion
    (0.25, 0.20, 0.28, 0.27),  # Fracture
]
"""

PRINT_OLD = """print(len(test), 'etudes | sans TTA | pas de flip | FOV vit/v20b/coat 130 mm, v20h 145 mm')
for t, w in zip(TARGETS, MIX):
    if abs(sum(w) - 1.0) > 1e-6 or min(w) < -1e-9 or len(w) != 4:
        raise RuntimeError((t, w))
    print(f'  {t:<18} vit {w[0]:.2f} v20b {w[1]:.2f} v20h {w[2]:.2f} coat {w[3]:.2f}')
"""

BLEND_OLD = """        w = torch.tensor(MIX, dtype=torch.float32, device=p_vit.device)
        prob = (
            w[:, 0] * p_vit + w[:, 1] * p_v20b + w[:, 2] * p_v20h + w[:, 3] * p_coat
        ).cpu().numpy()[0]
        rows.append([uid] + prob.tolist())
"""

MD_OLD = """**Mix** (ViT-k320 / v20b DINO / v20h DINO / CoAt-a) :
- `ACL` : ViT-k320 9% / v20b DINO 32% / v20h DINO 55% / CoAt-a 4%
- `MCL` : ViT-k320 3% / v20b DINO 44% / v20h DINO 52% / CoAt-a 1%
- `Medial Meniscus` : ViT-k320 27% / v20b DINO 29% / v20h DINO 30% / CoAt-a 14%
- `Lateral Meniscus` : ViT-k320 18% / v20b DINO 36% / v20h DINO 41% / CoAt-a 5%
- `Medial OA` : ViT-k320 17% / v20b DINO 32% / v20h DINO 33% / CoAt-a 18%
- `Lateral OA` : ViT-k320 21% / v20b DINO 29% / v20h DINO 29% / CoAt-a 21%
- `PF OA` : ViT-k320 14% / v20b DINO 32% / v20h DINO 42% / CoAt-a 12%
- `Effusion` : ViT-k320 28% / v20b DINO 23% / v20h DINO 22% / CoAt-a 27%
- `Synovitis` : ViT-k320 30% / v20b DINO 26% / v20h DINO 24% / CoAt-a 20%
- `Baker's` : ViT-k320 26% / v20b DINO 27% / v20h DINO 31% / CoAt-a 16%
- `Contusion` : ViT-k320 19% / v20b DINO 31% / v20h DINO 35% / CoAt-a 15%
- `Fracture` : ViT-k320 25% / v20b DINO 20% / v20h DINO 28% / CoAt-a 27%

Aucun flip de lateralite. Un modele peut valoir 0 %. Blend apres sigmoid.
"""


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _replace(nb: dict, old: str, new: str, expect: int) -> None:
    found = 0
    for cell in nb["cells"]:
        src = _text(cell)
        n = src.count(old)
        if not n:
            continue
        found += n
        _set(cell, src.replace(old, new))
    if found != expect:
        raise SystemExit(f"remplacement x{found}, attendu {expect}: {old[:80]!r}")


def _b64(kind: str) -> tuple[str, dict]:
    blob = (STACK_DIR / f"{kind}.npz").read_bytes()
    got = np.load(io.BytesIO(blob))
    if str(got["kind"]) != kind:
        raise SystemExit(got["kind"])
    order = [str(x) for x in got["order"].tolist()]
    if order != ["ViT-k320", "v20b", "v20h", "CoAt-a"]:
        raise SystemExit(order)
    if got["mu"].shape[-1] != 4 + 6 + 4 * 6:
        raise SystemExit(got["mu"].shape)
    return base64.b64encode(blob).decode("ascii"), {
        "epoch": int(got["epoch"]),
        "auc_train": float(got["auc_train"]),
        "auc_val": float(got["auc_val"]),
        "auc_gold": float(got["auc_gold"]),
    }


def _stack_code(kind: str, b64: str, meta: dict) -> str:
    tag = "STACK_LOGREG" if kind == "logreg" else "STACK_MLP"
    head = "logreg" if kind == "logreg" else "MLP 8 unites GELU"
    if kind == "logreg":
        head_code = (
            'out = torch.einsum("td,td->t", t, torch.from_numpy(_STACK["weight"]))'
            ' + torch.from_numpy(_STACK["bias"])'
        )
    else:
        head_code = """h = torch.nn.functional.gelu(
        torch.einsum("td,tdh->th", t, torch.from_numpy(_STACK["fc1"]))
        + torch.from_numpy(_STACK["b1"])
    )
    out = torch.einsum("th,th->t", h, torch.from_numpy(_STACK["fc2"])) + torch.from_numpy(_STACK["b2"])"""
    return f"""import base64
import io

_STACK = np.load(io.BytesIO(base64.b64decode({b64!r})))
_MU = _STACK["mu"][:, 0, :]
_SD = _STACK["sd"][:, 0, :]
_KEEP = _STACK["keep"][:, 0, :]
if [str(x) for x in _STACK["order"].tolist()] != ["ViT-k320", "v20b", "v20h", "CoAt-a"]:
    raise RuntimeError(_STACK["order"])
if str(_STACK["kind"]) != {kind!r}:
    raise RuntimeError(_STACK["kind"])
if _MU.shape != (12, 34):
    raise RuntimeError(_MU.shape)


def _design_one(logit_m12, empty6):
    z = np.asarray(logit_m12, np.float32)[None]
    empty_b = np.asarray(empty6, np.float32)[None]
    m = z.shape[1]
    s = empty_b.shape[1]
    n_t = z.shape[2]
    x = np.empty((n_t, z.shape[0], m + s + m * s), np.float32)
    x[:, :, :m] = np.transpose(z, (2, 0, 1))
    x[:, :, m:m + s] = empty_b[None]
    inter = z[:, :, None, :] * empty_b[:, None, :, None]
    x[:, :, m + s:] = np.transpose(inter, (3, 0, 1, 2)).reshape(n_t, z.shape[0], m * s)
    return x[:, 0, :]


def _stack_prob(probs, empty6):
    p = np.clip(np.asarray(probs, np.float64), 1e-4, 1.0 - 1e-4)
    logit = np.log(p / (1.0 - p)).astype(np.float32)
    raw = _design_one(logit, empty6)
    xs = (raw - _MU) / _SD * _KEEP
    t = torch.from_numpy(np.ascontiguousarray(xs))
    {head_code}
    prob = torch.sigmoid(out).numpy()
    if prob.shape != (12,) or not np.isfinite(prob).all():
        raise RuntimeError(prob.shape)
    return prob


print(
    {tag!r}, "epoch", int(_STACK["epoch"]),
    "train", round(float(_STACK["auc_train"]), 4),
    "val", round(float(_STACK["auc_val"]), 4),
    "gold_stack", round(float(_STACK["auc_gold"]), 4),
    "d", int(_MU.shape[1]), "head", {head!r},
)
print(len(test), 'etudes | sans TTA | pas de flip | FOV vit/v20b/coat 130 mm, v20h 145 mm')
_ = ({meta["epoch"]}, {meta["auc_train"]:.6f})
"""


def _blend(kind: str) -> str:
    return """        probs = np.stack([
            p_vit.detach().float().cpu().numpy().reshape(-1),
            p_v20b.detach().float().cpu().numpy().reshape(-1),
            p_v20h.detach().float().cpu().numpy().reshape(-1),
            p_coat.detach().float().cpu().numpy().reshape(-1),
        ], axis=0)
        empty = np.array([
            0.0 if _pick_protocol_row(srows, plane, fluid) is not None else 1.0
            for plane, fluid in PROTOCOL_SLOTS
        ], dtype=np.float32)
        empty_acc += empty
        prob = _stack_prob(probs, empty)
        rows.append([uid] + prob.tolist())
"""


def _markdown(kind: str, meta: dict) -> str:
    label = "regression logistique" if kind == "logreg" else "MLP par cible (8 unites, GELU)"
    return f"""**Stack {label}** (notebook {"13" if kind == "logreg" else "15"}).
Entrees par cible : logit de chaque modele (clip 1e-4), les 6 slots vides, et le produit logit x slot.
Ajuste sur les etudes fit des predictions hybrid de ces quatre subsets.
120 epochs, log loss, L2 1e-3, Adam. Epoch retenue : meilleure AUC train ({meta["epoch"]}),
train {meta["auc_train"]:.4f}, val {meta["auc_val"]:.4f} (la val ne choisit pas l'epoch).
Pas de flip. Sans TTA. La sortie est le sigmoid du stack, pas une moyenne ponderee.
"""


def write_one(job: dict) -> Path:
    kind = job["kind"]
    b64, meta = _b64(kind)
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    _replace(nb, "# RSNA Knee — rsna-knee-ens-vitk89-c145", f"# RSNA Knee — {job['slug']}", 1)
    _replace(nb, MD_OLD, _markdown(kind, meta), 1)
    _replace(nb, MIX_OLD, "# Le blend lineaire est remplace par le stack charge plus bas.\n", 1)
    _replace(nb, PRINT_OLD, _stack_code(kind, b64, meta), 1)
    _replace(nb, "rows = []\n", "rows = []\nempty_acc = np.zeros(6, np.float32)\n", 1)
    _replace(nb, BLEND_OLD, _blend(kind), 1)
    _replace(
        nb,
        "pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)\n",
        "print('STACK_EMPTY_SUM', empty_acc.round(3).tolist())\n"
        "pred = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)\n",
        1,
    )
    blob = "\n".join(_text(c) for c in nb["cells"])
    tag = "STACK_LOGREG" if kind == "logreg" else "STACK_MLP"
    for needle in (
        tag, "swa-8-9", "[8, 9]", "swa-6-7-8", "swa-1-2-3",
        "vit_k_llrd08_320_v3b_oofmask_vitk_full_oai3",
        "_design_one", "_stack_prob", "STACK_EMPTY_SUM",
        "sans TTA", "CROP_MM = 145.0", "laterality=False",
        "1e-4",
    ):
        if needle not in blob:
            raise SystemExit(f"{kind} incomplet: {needle}")
    for banned in ("w[:, 0] * p_vit", "MIX = [", "TTA_VARIANTS", "laterality=True"):
        if banned in blob:
            raise SystemExit(f"{kind} residu {banned}")
    if kind == "logreg" and "functional.gelu" in blob:
        raise SystemExit("logreg contient le MLP")
    if kind == "mlp" and "functional.gelu" not in blob:
        raise SystemExit("mlp sans GELU")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(_text(cell))
    push = KERNEL / f"_push_ens_vitk89_{kind}"
    push.mkdir(parents=True, exist_ok=True)
    (push / "submit.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta_k = {
        "id": f"sultent/{job['slug']}",
        "title": job["slug"],
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
        "dataset_sources": DATASETS,
        "category_ids": [],
    }
    (push / "kernel-metadata.json").write_text(json.dumps(meta_k, indent=2) + "\n", encoding="utf-8")
    (push / "_message.txt").write_text(job["msg"] + "\n", encoding="utf-8")
    print("ecrit", push, kind, "epoch", meta["epoch"], flush=True)
    return push


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


def _quota(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "maximum batch gpu" in low


def push_one(job: dict) -> str:
    push = KERNEL / f"_push_ens_vitk89_{job['kind']}"
    kernel = f"sultent/{job['slug']}"
    print("push", kernel, flush=True)
    while True:
        try:
            out = kaggle("kernels", "push", "-p", str(push), "--accelerator", ACCEL, "-t", "32400")
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
        (push / "_version.txt").write_text(version + "\n", encoding="utf-8")
        print("kernel version", kernel, version, flush=True)
        return version


def wait_one(job: dict) -> None:
    kernel = f"sultent/{job['slug']}"
    t0 = time.time()
    seen = False
    while True:
        raw = kaggle("kernels", "status", kernel).strip()
        print(job["slug"], raw, flush=True)
        up = raw.upper()
        if "FAILED" in up or "ERROR" in up or "CANCEL" in up:
            raise SystemExit(f"{job['slug']} en erreur")
        if "RUNNING" in up or "QUEUED" in up or "PENDING" in up:
            seen = True
        if ("COMPLETE" in up or "SUCCEEDED" in up) and "RUNNING" not in up and (
            seen or time.time() - t0 > 90
        ):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"timeout {job['slug']}")
        time.sleep(POLL_S)


def _daily_limit(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in (
        "maximum number", "submission limit", "daily submission allowance",
        "try again tomorrow", "too many submission", "exceeded",
    ))


def submit_one(job: dict, version: str) -> None:
    kernel = f"sultent/{job['slug']}"
    dest = KERNEL / f"_out_ens_vitk89_{job['kind']}"
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", kernel, "-p", str(dest)), flush=True)
    log = next(p for p in dest.glob("*.log") if "cufile" not in p.name.lower())
    text = log.read_text(encoding="utf-8", errors="replace")
    tag = "STACK_LOGREG" if job["kind"] == "logreg" else "STACK_MLP"
    for needle in (
        tag, "STACK_EMPTY_SUM", "sans TTA", "lignes 3", "swa-8-9",
        "swa-6-7-8", "swa-1-2-3", "lat False", "v20h 145 mm",
        "vit_k_llrd08_320_v3b_oofmask_vitk_full_oai3",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop145_dino_ep11_oof_v3b_full_oaiv2_wd01",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
    ):
        if needle not in text:
            raise SystemExit(f"{job['kind']} log sans {needle}")
    if "TTA x" in text or "lat True" in text or "vit 0.09 v20b 0.32" in text:
        raise SystemExit(f"{job['kind']} TTA, flip ou ancien mix lineaire")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    msg = (KERNEL / f"_push_ens_vitk89_{job['kind']}" / "_message.txt").read_text(encoding="utf-8").strip()
    waited = False
    while True:
        print("submit", kernel, "version", version, flush=True)
        try:
            resp = api.competition_submit_code(
                "submission.csv", msg, COMPETITION, kernel, int(version), quiet=True,
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
            raise
        print(resp, flush=True)
        print("SOUMIS", kernel, "version", version, flush=True)
        return


def verify(job: dict) -> None:
    """Le code embarque doit donner le meme vecteur que l'entrainement."""
    import importlib.util

    import torch

    spec = importlib.util.spec_from_file_location(
        "fit_stack_vitk89", KERNEL / "_fit_stack_vitk89.py",
    )
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    kind = job["kind"]
    nb = json.loads((KERNEL / f"_push_ens_vitk89_{kind}" / "submit.ipynb").read_text(encoding="utf-8"))
    src = "\n".join(_text(c) for c in nb["cells"])
    start = src.index("import base64\n")
    end = src.index("\nprint(\n    ", start)
    ns: dict = {"np": np, "torch": torch}
    exec(src[start:end], ns)
    blob = np.load(STACK_DIR / f"{kind}.npz")
    rng = np.random.default_rng(0)
    for i in range(8):
        logit = rng.normal(size=(4, 12)).astype(np.float32)
        empty = rng.integers(0, 2, size=6).astype(np.float32)
        probs = 1.0 / (1.0 + np.exp(-logit.astype(np.float64)))
        probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
        back = np.log(probs / (1.0 - probs)).astype(np.float32)
        raw = fit.design_one(back, empty)
        if not np.array_equal(ns["_design_one"](back, empty), raw):
            raise SystemExit(f"{kind} design notebook diverge")
        mu = blob["mu"][:, 0, :]
        sd = blob["sd"][:, 0, :]
        keep = blob["keep"][:, 0, :]
        xs = (raw - mu) / sd * keep
        t = torch.from_numpy(np.ascontiguousarray(xs))
        with torch.no_grad():
            if kind == "logreg":
                w = torch.from_numpy(blob["weight"])
                b = torch.from_numpy(blob["bias"])
                expect = torch.sigmoid(torch.einsum("td,td->t", t, w) + b).numpy()
            else:
                fc1 = torch.from_numpy(blob["fc1"])
                b1 = torch.from_numpy(blob["b1"])
                fc2 = torch.from_numpy(blob["fc2"])
                b2 = torch.from_numpy(blob["b2"])
                h = torch.nn.functional.gelu(torch.einsum("td,tdh->th", t, fc1) + b1)
                expect = torch.sigmoid(torch.einsum("th,th->t", h, fc2) + b2).numpy()
        got = ns["_stack_prob"](probs, empty)
        if not np.allclose(got, expect, atol=1e-6):
            raise SystemExit(f"{kind} forward notebook diverge {np.max(np.abs(got - expect))}")
    print("verifie", kind, flush=True)


def main() -> None:
    for job in JOBS:
        write_one(job)
        verify(job)
    if "--write" in sys.argv:
        return
    versions = {}
    for job in JOBS:
        versions[job["kind"]] = push_one(job)
        wait_one(job)
        submit_one(job, versions[job["kind"]])


if __name__ == "__main__":
    main()
