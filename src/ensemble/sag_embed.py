# -*- coding: utf-8 -*-
"""Vecteur gelé de la série sagittale fat-sat, une par étude.

Toujours le même slot : Sagittal, fluide (SAG_FS). Pas de repli vers un
autre plan : une coronale à la place apprendrait une autre anatomie.
Si le slot manque, le vecteur est nul et `present` vaut 0.

Huit coupes réparties sur la profondeur, canon gauche (même flip que
l'entraînement). ConvNeXt-Tiny ImageNet, gelé, moyenne et écart-type
des huit descripteurs. Le gate ne voit ensuite qu'une PCA de ce vecteur.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset_npz import load_volume, make_series_index
from src.data.laterality import load_laterality_csv, normalise_laterality

ENCODER_NAME = "convnext_tiny.fb_in22k_ft_in1k"
DEFAULT_WEIGHTS = "weights/convnext_tiny-fb_in22k_ft_in1k"
IMG = 224
N_SLICES = 8
FEAT = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _weights_file(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin"):
            cand = p / name
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"aucun poids dans {p.resolve()}")
    if not p.is_file():
        raise FileNotFoundError(f"poids introuvables : {p.resolve()}")
    return p


def _encoder(weights: str | Path, device: torch.device) -> torch.nn.Module:
    import timm

    ckpt = _weights_file(weights)
    model = timm.create_model(
        ENCODER_NAME,
        pretrained=True,
        pretrained_cfg_overlay={"file": str(ckpt)},
        num_classes=0,
        global_pool="avg",
    )
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n = int(model.num_features)
    if n != FEAT:
        raise RuntimeError(f"ConvNeXt-Tiny : {n} canaux, attendu {FEAT}")
    print(f"encodeur gelé : {ENCODER_NAME} | {ckpt}", flush=True)
    return model


def _slice_indices(depth: int, n: int) -> np.ndarray:
    n = min(int(n), int(depth))
    idx = np.rint(np.linspace(0, depth - 1, n)).astype(np.int64)
    return np.unique(idx)


def _flush(
    model: torch.nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    images: list[np.ndarray],
    rows: list[int],
    counts: list[int],
    out: np.ndarray,
) -> None:
    if not images:
        return
    batch = np.stack(images, axis=0)
    x = torch.from_numpy(batch).to(device, non_blocking=True)
    x = x.float().div_(255.0)
    if x.shape[-2] != IMG or x.shape[-1] != IMG:
        x = F.interpolate(
            x.unsqueeze(1), size=(IMG, IMG),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
    x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    x = (x - mean) / std
    feat = model(x).detach().float().cpu().numpy()
    offset = 0
    for row, count in zip(rows, counts):
        part = feat[offset:offset + count]
        offset += count
        mu = part.mean(axis=0)
        sd = part.std(axis=0)
        out[row, :FEAT] = mu
        out[row, FEAT:] = sd
    images.clear()
    rows.clear()
    counts.clear()


def extract_sag_fs(
    uids: list[str],
    root: str | Path,
    zip_path: str | Path = "data/crop130",
    out_path: str | Path | None = None,
    laterality_csv: str | Path = "data/csv/study_laterality.csv",
    series_csv: str | Path = "data/csv/train_series.csv",
    n_slices: int = N_SLICES,
    batch_images: int = 128,
    weights: str | Path = DEFAULT_WEIGHTS,
) -> Path:
    """Écrit un npz aligné sur `uids`. Retourne le chemin."""
    root = Path(root)
    zip_path = Path(zip_path)
    if not zip_path.is_absolute():
        zip_path = root / zip_path
    if not zip_path.exists():
        raise FileNotFoundError(f"volumes introuvables : {zip_path}")
    out_path = Path(out_path) if out_path else root / "data" / "hybrid_pred" / "sag_fs_embed.npz"
    if not out_path.is_absolute():
        out_path = root / out_path
    series_csv_p = Path(series_csv)
    if not series_csv_p.is_absolute():
        series_csv_p = root / series_csv_p
    lat_path = Path(laterality_csv)
    if not lat_path.is_absolute():
        lat_path = root / lat_path

    index, volume_index = make_series_index(series_csv_p, zip_path)
    lat_map = load_laterality_csv(lat_path) if lat_path.is_file() else {}
    if not lat_map:
        print(f"laterality absente ({lat_path}) : pas de flip droite", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _encoder(root / weights if not Path(weights).is_absolute() else weights, device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    n = len(uids)
    embed = np.zeros((n, FEAT * 2), np.float32)
    present = np.zeros(n, np.uint8)
    series_uid = np.full(n, "", dtype="U128")
    depth_other = 0
    read_fail = 0

    # Regroupe par zip pour relire chaque archive d'une traite.
    groups: dict[str, list[tuple[int, str]]] = {}
    missing = 0
    for i, uid in enumerate(uids):
        sid = index.pick_protocol(uid, "Sagittal", 1, None, which="first")
        if sid is None:
            missing += 1
            continue
        src = zip_path if volume_index is None else volume_index.get(sid)
        if src is None:
            missing += 1
            continue
        groups.setdefault(str(src), []).append((i, sid))

    images: list[np.ndarray] = []
    rows: list[int] = []
    counts: list[int] = []
    t0 = time.perf_counter()
    done = 0
    with torch.inference_mode():
        for src, items in groups.items():
            for i, sid in items:
                uid = uids[i]
                try:
                    vol = load_volume(src, sid)
                except Exception:
                    read_fail += 1
                    continue
                if vol.ndim != 3 or vol.shape[0] < 1:
                    read_fail += 1
                    continue
                if np.issubdtype(vol.dtype, np.floating):
                    vmax = float(vol.max())
                    if vmax <= 1.5:
                        vol = np.clip(vol * 255.0, 0, 255)
                    vol = np.clip(vol, 0, 255).astype(np.uint8)
                if int(vol.max()) == 0:
                    read_fail += 1
                    continue
                if vol.shape[0] != 24:
                    depth_other += 1
                vol = normalise_laterality(vol, "Sagittal", lat_map.get(uid))
                idx = _slice_indices(vol.shape[0], n_slices)
                sl = np.ascontiguousarray(vol[idx])
                if sl.shape[-2] != IMG or sl.shape[-1] != IMG:
                    tsl = torch.from_numpy(sl.astype(np.float32) / 255.0).unsqueeze(1)
                    tsl = F.interpolate(
                        tsl, size=(IMG, IMG), mode="bilinear", align_corners=False,
                    )
                    sl = (tsl.squeeze(1).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
                images.extend(sl[k] for k in range(sl.shape[0]))
                rows.append(i)
                counts.append(int(sl.shape[0]))
                present[i] = 1
                series_uid[i] = sid
                done += 1
                if sum(counts) >= batch_images:
                    _flush(model, device, mean, std, images, rows, counts, embed)
                if done % 400 == 0:
                    rate = done / max(time.perf_counter() - t0, 1e-6)
                    print(f"  {done} séries  {rate:.1f} études/s", flush=True)
        _flush(model, device, mean, std, images, rows, counts, embed)

    elapsed = time.perf_counter() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        uid=np.asarray(uids),
        embed=embed,
        present=present,
        series_uid=series_uid,
        encoder=np.array(ENCODER_NAME),
        zip_path=np.array(str(zip_path)),
        n_slices=np.int32(n_slices),
    )
    print(
        f"SAG_FS {int(present.sum())}/{n} présentes | "
        f"sans slot {missing} | lecture {read_fail} | "
        f"profondeur ≠ 24 {depth_other} | {elapsed:.0f} s | {out_path}",
        flush=True,
    )
    return out_path


def load_embed(path: str | Path, uids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    got = [str(u) for u in z["uid"].tolist()]
    if got != [str(u) for u in uids]:
        raise SystemExit(f"{path} : ordre des UID différent du tableau d'experts")
    return z["embed"].astype(np.float32), z["present"].astype(bool)
