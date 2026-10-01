# -*- coding: utf-8 -*-
"""SimCLR sur les coupes crop130, encodeur v20h (ResNet-50 dilated).

Depart : weights/ResNet50.pt (RadImageNet). Les deux vues d'une coupe
sont deux augmentations. La loss NT-Xent rapproche ces vues et eloigne
les autres coupes du batch. Le projecteur est jete a la fin.

Sortie : checkpoints/<run>/encoder.pt
State dict brut, cles `backbone.*`, chargeable par KneeNet v20h
(`_load_radimagenet`, strict=True). Les BN restent en train ici :
v20h les gele ensuite, donc les running stats ecrites sont celles du SSL.

Entree encodeur : R=G=B = coupe dans [0, 1], comme stem_mix a l'init
de v20h (le canal central copie sur les 3 canaux, mean 0, std 1).
Pas de flip horizontal (medial / lateral).

Usage :
  python -X utf8 -m src.train.ssl_pretrain --config configs/ssl_v20h_simclr.yaml
  python -X utf8 -m src.train.ssl_pretrain --config configs/ssl_v20h_simclr.yaml --resume
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.ssl_slices import make_ssl_loader
from src.models.kneenet_v20h import _ResNet50Encoder


class Projector(nn.Module):
    """Tete SimCLR. Absente de encoder.pt."""

    def __init__(self, in_dim: int = 2048, hidden: int = 2048,
                 out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor,
            temperature: float) -> torch.Tensor:
    """z1, z2 : (N, D). Positif de i = la meme coupe, autre vue."""
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / float(temperature)
    eye = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1.0e4)
    labels = torch.cat([
        torch.arange(n, 2 * n, device=z.device),
        torch.arange(0, n, device=z.device),
    ], dim=0)
    return F.cross_entropy(sim, labels)


def sample_slices(vol: torch.Tensor, k: int) -> torch.Tensor:
    """vol (B, D, H, W) -> (B, K, H, W). Coupes vides evitees si possible."""
    b, d, _h, _w = vol.shape
    k = max(1, min(int(k), d))
    valid = vol.amax(dim=(-1, -2)) > 1.0e-6
    picked = []
    for i in range(b):
        ix = valid[i].nonzero(as_tuple=False).view(-1)
        if ix.numel() == 0:
            ix = torch.arange(d, device=vol.device)
        perm = ix[torch.randperm(ix.numel(), device=vol.device)]
        if perm.numel() < k:
            extra = ix[torch.randint(0, ix.numel(), (k - perm.numel(),),
                                     device=vol.device)]
            perm = torch.cat([perm, extra], dim=0)
        picked.append(vol[i, perm[:k]])
    return torch.stack(picked, dim=0)


def augment_views(
    x: torch.Tensor,
    scale_range: tuple[float, float] = (0.75, 1.0),
    max_rotate_deg: float = 12.0,
    max_translate: float = 0.08,
    brightness: float = 0.2,
    contrast: float = 0.2,
    gamma_range: tuple[float, float] = (0.8, 1.2),
) -> torch.Tensor:
    """x : (N, 3, H, W) dans [0, 1]. Une aug independante par coupe.

    scale < 1 zoome (crop), meme convention que augment_batch.
    Pas de flip horizontal.
    """
    n, _c, _h, _w = x.shape
    device = x.device
    max_rot = max_rotate_deg * math.pi / 180.0
    angle = (torch.rand(n, device=device) * 2 - 1) * max_rot
    scale = torch.empty(n, device=device).uniform_(*scale_range)
    tx = (torch.rand(n, device=device) * 2 - 1) * max_translate
    ty = (torch.rand(n, device=device) * 2 - 1) * max_translate
    cos = torch.cos(angle) / scale
    sin = torch.sin(angle) / scale
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    bshift = (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * brightness
    cscale = torch.empty(n, 1, 1, 1, device=device).uniform_(
        1 - contrast, 1 + contrast)
    gamma = torch.empty(n, 1, 1, 1, device=device).uniform_(*gamma_range)
    x = ((x + bshift).clamp(0, 1) - 0.5) * cscale + 0.5
    x = x.clamp(1.0e-4, 1).pow(gamma)
    return x.clamp(0, 1)


def _pair(cfg: dict, key: str) -> tuple[float, float]:
    a, b = cfg[key]
    return float(a), float(b)


def _views(slices: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """slices (N, H, W) -> deux RGB (N, 3, H, W). R=G=B = coupe."""
    x = slices.unsqueeze(1).repeat(1, 3, 1, 1)
    kw = dict(
        scale_range=_pair(cfg, "scale_range"),
        max_rotate_deg=float(cfg["max_rotate_deg"]),
        max_translate=float(cfg["max_translate"]),
        brightness=float(cfg["brightness"]),
        contrast=float(cfg["contrast"]),
        gamma_range=_pair(cfg, "gamma_range"),
    )
    return augment_views(x, **kw), augment_views(x, **kw)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    print(f"seed {seed} (cudnn benchmark)", flush=True)


def _lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * p))


def _load_init(encoder: nn.Module, path: str) -> None:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"init SSL introuvable : {p.resolve()}")
    sd = torch.load(p, map_location="cpu", weights_only=False)
    encoder.load_state_dict(sd, strict=True)
    print(f"init encodeur : {p} | {len(sd)} tenseurs", flush=True)


def _embed(encoder: nn.Module, projector: nn.Module,
           x: torch.Tensor) -> torch.Tensor:
    maps = encoder(x)
    gap = maps.mean(dim=(-2, -1))
    return projector(gap)


def main() -> None:
    from src.train.train import _atomic_torch_save, load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.get("out_dir", f"checkpoints/{cfg['run_name']}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    loader = make_ssl_loader(
        series_csv=cfg["series_csv"],
        zip_path=cfg["zip_path"],
        series_batch=int(cfg["series_batch"]),
        num_workers=int(cfg.get("num_workers", 4)),
        img_size=int(cfg.get("img_size", 224)),
        seed=seed,
    )
    steps_per_epoch = len(loader)
    if steps_per_epoch < 1:
        raise SystemExit("loader SSL vide (series_batch trop grand ?)")
    slices_per = int(cfg["slices_per_series"])
    n_img = int(cfg["series_batch"]) * slices_per
    epochs = int(cfg["epochs"])
    total_steps = epochs * steps_per_epoch
    warmup = int(0.05 * total_steps)
    chance = math.log(max(2 * n_img - 1, 2))
    print(
        f"device {device} | series/step {cfg['series_batch']} | "
        f"coupes/serie {slices_per} | N {n_img} | vues {2 * n_img} | "
        f"steps/epoch {steps_per_epoch} | epochs {epochs}",
        flush=True,
    )
    print(f"NT-Xent hasard ~ {chance:.3f}", flush=True)

    encoder = _ResNet50Encoder().to(device)
    projector = Projector(
        in_dim=encoder.num_features,
        hidden=int(cfg.get("proj_hidden", 2048)),
        out_dim=int(cfg.get("proj_dim", 128)),
    ).to(device)
    _load_init(encoder, cfg["init_weights"])

    lr_enc = float(cfg["lr_encoder"])
    lr_proj = float(cfg["lr_projector"])
    optim = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": lr_enc},
            {"params": projector.parameters(), "lr": lr_proj},
        ],
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
    )
    base_lrs = [lr_enc, lr_proj]
    temperature = float(cfg.get("temperature", 0.2))
    log_every = int(cfg.get("log_every", 50))

    start_epoch = 0
    history: list[dict] = []
    step = 0
    resume_path = out_dir / "last.pt"
    if args.resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume sans {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"], strict=True)
        projector.load_state_dict(ckpt["projector"], strict=True)
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt["epoch"])
        step = start_epoch * steps_per_epoch
        hist_p = out_dir / "history.json"
        if hist_p.is_file():
            history = json.loads(hist_p.read_text(encoding="utf-8"))
            history = [h for h in history if int(h["epoch"]) <= start_epoch]
        print(f"resume epoch {start_epoch} | step {step}", flush=True)

    use_amp = device == "cuda"
    for epoch in range(start_epoch, epochs):
        encoder.train()
        projector.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        t_log = t0
        n_log = 0
        for vol in loader:
            vol = vol.to(device, non_blocking=True)
            slices = sample_slices(vol, slices_per)
            flat = slices.reshape(-1, slices.size(-2), slices.size(-1))
            v1, v2 = _views(flat, cfg)
            for group, base in zip(optim.param_groups, base_lrs):
                group["lr"] = _lr_at(step, total_steps, base, warmup)
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                z1 = _embed(encoder, projector, v1)
                z2 = _embed(encoder, projector, v2)
            loss = nt_xent(z1, z2, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(projector.parameters()),
                1.0,
            )
            optim.step()
            step += 1
            running += float(loss.detach())
            nb += 1
            n_log += flat.size(0) * 2
            if step % log_every == 0:
                dt = max(time.time() - t_log, 1.0e-6)
                print(
                    f"  epoch {epoch + 1} step {nb}/{steps_per_epoch} | "
                    f"loss {running / nb:.4f} | {n_log / dt:.0f} vues/s | "
                    f"lr {optim.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
                t_log = time.time()
                n_log = 0

        avg = running / max(nb, 1)
        dt = time.time() - t0
        rec = {
            "epoch": epoch + 1,
            "loss": avg,
            "chance_loss": chance,
            "lr_encoder": optim.param_groups[0]["lr"],
            "seconds": round(dt, 1),
        }
        history.append(rec)
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8",
        )
        payload = {
            "epoch": epoch + 1,
            "encoder": {k: v.detach().cpu()
                        for k, v in encoder.state_dict().items()},
            "projector": {k: v.detach().cpu()
                          for k, v in projector.state_dict().items()},
            "optim": optim.state_dict(),
        }
        enc_path = out_dir / "encoder.pt"
        _atomic_torch_save(payload["encoder"], enc_path)
        _atomic_torch_save(payload, resume_path)
        print(
            f"epoch {epoch + 1}/{epochs} | loss {avg:.4f} "
            f"(hasard {chance:.3f}) | {dt / 60:.1f} min | -> {enc_path}",
            flush=True,
        )

    print(
        f"SSL termine. Init v20h : backbone_weights: {out_dir / 'encoder.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
