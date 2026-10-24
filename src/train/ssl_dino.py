# -*- coding: utf-8 -*-
"""DINO sur les coupes crop130, encodeur v20h (ResNet-50 dilated).

Recette du candidat : 2 vues globales (zoom 40-100 %) pour l'eleve et le
professeur, 6 vues locales (zoom 10-40 %) pour l'eleve seul. Le professeur
est une moyenne mobile de l'eleve. Il produit une loi sur 4096 categories.
L'eleve s'en rapproche, y compris depuis un petit morceau de la coupe.

La tete est jetee. Sortie : checkpoints/<run>/encoder.pt
State dict brut, cles `backbone.*`.
`encoder: resnet50` (defaut) : chargeable par KneeNet v20h.
`encoder: vit_dinov3` : chargeable par ViT-k (`backbone_weights` en `.pt`).
BN de l'eleve en train : v20h les gele ensuite. Le ViT n'a pas de BN.

Entree : R=G=B = coupe dans [0, 1]. Pas de flip horizontal.
ViT-k : ensuite la meme normalisation ImageNet que KneeNet-ViT-e.

Usage :
  python -X utf8 -m src.train.ssl_dino --config configs/archive/ssl_v20h_dino.yaml
  python -X utf8 -m src.train.ssl_dino --config configs/archive/ssl_v20h_dino.yaml --resume
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
from src.train.ssl_pretrain import _load_init, _set_seed, sample_slices


class DINOHead(nn.Module):
    """Tete DINO. Absente de encoder.pt."""

    def __init__(
        self,
        in_dim: int = 2048,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        use_bn: bool = True,
        norm_last_layer: bool = True,
    ) -> None:
        super().__init__()
        if nlayers < 1:
            raise ValueError(f"nlayers={nlayers}")
        layers: list[nn.Module] = []
        if nlayers == 1:
            layers.append(nn.Linear(in_dim, bottleneck_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        last = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.last_layer = torch.nn.utils.parametrizations.weight_norm(last)
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINOLoss(nn.Module):
    """Entropie croisee eleve -> professeur. Pas de comparaison d'une vue a elle-meme."""

    def __init__(
        self,
        out_dim: int,
        ncrops: int,
        teacher_temp: float,
        warmup_teacher_temp: float,
        warmup_teacher_temp_epochs: int,
        nepochs: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.ncrops = int(ncrops)
        self.register_buffer("center", torch.zeros(1, out_dim))
        warm = max(0, int(warmup_teacher_temp_epochs))
        warm = min(warm, int(nepochs))
        if warm == 0:
            schedule = np.full(int(nepochs), float(teacher_temp), dtype=np.float64)
        else:
            schedule = np.concatenate([
                np.linspace(float(warmup_teacher_temp), float(teacher_temp), warm),
                np.full(int(nepochs) - warm, float(teacher_temp)),
            ])
        self.register_buffer(
            "teacher_temp_schedule", torch.tensor(schedule, dtype=torch.float32))

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        student_out = student_output / self.student_temp
        student_chunks = student_out.chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_probs = F.softmax(
            (teacher_output - self.center) / temp, dim=-1)
        teacher_chunks = teacher_probs.detach().chunk(2)

        total = student_output.new_zeros(())
        n_terms = 0
        for iq, q in enumerate(teacher_chunks):
            for v, s in enumerate(student_chunks):
                if v == iq:
                    continue
                logp = F.log_softmax(s.float(), dim=-1)
                total = total + torch.sum(-q.float() * logp, dim=-1).mean()
                n_terms += 1
        total = total / max(n_terms, 1)
        self.update_center(teacher_output)
        return total

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum)

    @torch.no_grad()
    def teacher_confidence(self, teacher_output: torch.Tensor, epoch: int) -> float:
        temp = self.teacher_temp_schedule[epoch]
        probs = F.softmax((teacher_output - self.center) / temp, dim=-1)
        return float(probs.max(dim=-1).values.mean())


def _pair(cfg: dict, key: str) -> tuple[float, float]:
    a, b = cfg[key]
    return float(a), float(b)


def _gauss_kernel(k: int, sigma: float, device: torch.device) -> torch.Tensor:
    ax = torch.arange(k, device=device, dtype=torch.float32) - (k // 2)
    g = torch.exp(-0.5 * (ax / max(sigma, 1.0e-3)) ** 2)
    return g / g.sum()


def _blur_mask(x: torch.Tensor, mask: torch.Tensor, k: int = 9) -> torch.Tensor:
    if not bool(mask.any()):
        return x
    sigma = float(torch.empty(1).uniform_(0.1, 2.0))
    g = _gauss_kernel(k, sigma, x.device)
    c = x.size(1)
    kx = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    ky = g.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    xb = x[mask]
    xb = F.conv2d(xb, kx, padding=(0, k // 2), groups=c)
    xb = F.conv2d(xb, ky, padding=(k // 2, 0), groups=c)
    out = x.clone()
    out[mask] = xb
    return out


def dino_crop(
    x: torch.Tensor,
    area_range: tuple[float, float],
    max_rotate_deg: float,
    brightness: float,
    contrast: float,
    gamma_range: tuple[float, float],
    blur_p: float,
) -> torch.Tensor:
    """x (N, 3, H, W) dans [0, 1].

    area_range est la fraction d'aire de la coupe, comme DINO
    (0.4-1 = on garde 40 a 100 % de l'image, puis on la ramene a H, W).
    Pas de flip horizontal.
    """
    n, _c, _h, _w = x.shape
    device = x.device
    lo, hi = area_range
    area = torch.empty(n, device=device).uniform_(lo, hi)
    log_r = torch.empty(n, device=device).uniform_(math.log(0.75), math.log(4.0 / 3.0))
    ratio = log_r.exp()
    w = (area * ratio).sqrt().clamp(max=1.0)
    h = (area / ratio).sqrt().clamp(max=1.0)
    max_rot = max_rotate_deg * math.pi / 180.0
    angle = (torch.rand(n, device=device) * 2 - 1) * max_rot
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    tx = (torch.rand(n, device=device) * 2 - 1) * (1.0 - w)
    ty = (torch.rand(n, device=device) * 2 - 1) * (1.0 - h)
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = cos * w
    theta[:, 0, 1] = -sin * h
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * w
    theta[:, 1, 1] = cos * h
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    y = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    bshift = (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * brightness
    cscale = torch.empty(n, 1, 1, 1, device=device).uniform_(1 - contrast, 1 + contrast)
    gamma = torch.empty(n, 1, 1, 1, device=device).uniform_(*gamma_range)
    y = ((y + bshift).clamp(0, 1) - 0.5) * cscale + 0.5
    y = y.clamp(1.0e-4, 1).pow(gamma).clamp(0, 1)
    if blur_p > 0:
        mask = torch.rand(n, device=device) < blur_p
        y = _blur_mask(y, mask)
    return y


def make_views(slices: torch.Tensor, cfg: dict) -> list[torch.Tensor]:
    """slices (N, H, W) -> [global, global, local, ...]. RGB, R=G=B."""
    x = slices.unsqueeze(1).repeat(1, 3, 1, 1)
    common = dict(
        max_rotate_deg=float(cfg.get("max_rotate_deg", 12)),
        brightness=float(cfg.get("brightness", 0.4)),
        contrast=float(cfg.get("contrast", 0.4)),
        gamma_range=_pair(cfg, "gamma_range"),
    )
    g_area = _pair(cfg, "global_crops_scale")
    l_area = _pair(cfg, "local_crops_scale")
    n_local = int(cfg["local_crops_number"])
    views = [
        dino_crop(x, g_area, blur_p=1.0, **common),
        dino_crop(x, g_area, blur_p=0.1, **common),
    ]
    for _ in range(n_local):
        views.append(dino_crop(x, l_area, blur_p=0.5, **common))
    return views


def _lr_at(step: int, total: int, base: float, warmup: int, min_lr: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return min_lr + (base - min_lr) * cosine


def _momentum_at(step: int, total: int, base: float) -> float:
    p = min(max(step, 0) / max(total, 1), 1.0)
    return 1.0 + 0.5 * (base - 1.0) * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def _ema_update(student: nn.Module, teacher: nn.Module, m: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        if bt.dtype.is_floating_point:
            bt.data.mul_(m).add_(bs.data, alpha=1.0 - m)
        else:
            bt.data.copy_(bs.data)


def _sd_cpu(module: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _forward_logits(
    encoder: nn.Module,
    head: nn.Module,
    x: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    gaps = []
    step = max(1, int(chunk))
    for i in range(0, x.size(0), step):
        maps = encoder(x[i:i + step])
        gaps.append(maps.mean(dim=(-2, -1)))
    return head(torch.cat(gaps, dim=0))


def _cancel_last_layer_grad(head: DINOHead) -> None:
    for p in head.last_layer.parameters():
        p.grad = None


def _freeze_layerscale(encoder: nn.Module) -> None:
    """gamma DINOv3 : ViT-k les gele aussi au finetuning."""
    n = 0
    for name, p in encoder.named_parameters():
        if name.rsplit(".", 1)[-1].startswith("gamma"):
            p.requires_grad = False
            n += 1
    if n:
        print(f"LayerScale gele : {n} tenseurs gamma", flush=True)


def _new_encoder(cfg: dict, load: bool) -> nn.Module:
    kind = str(cfg.get("encoder", "resnet50"))
    if kind == "resnet50":
        enc = _ResNet50Encoder()
        if load:
            _load_init(enc, cfg["init_weights"])
        return enc
    if kind == "vit_dinov3":
        from src.models.kneenet_vit_e import _Dinov3VitEncoder
        if load:
            enc = _Dinov3VitEncoder(
                weights_path=cfg["init_weights"], pretrained=True,
            )
        else:
            enc = _Dinov3VitEncoder(pretrained=False)
        _freeze_layerscale(enc)
        return enc
    raise SystemExit(f"encoder SSL inconnu : {kind}")


def _imagenet_norm(views: list[torch.Tensor]) -> list[torch.Tensor]:
    from src.models.kneenet_vit_e import IMAGENET_MEAN, IMAGENET_STD
    ref = views[0]
    mean = ref.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = ref.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return [(v - mean) / std for v in views]


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
        num_workers=int(cfg.get("num_workers", 8)),
        img_size=int(cfg.get("img_size", 224)),
        seed=seed,
    )
    steps_per_epoch = len(loader)
    if steps_per_epoch < 1:
        raise SystemExit("loader SSL vide (series_batch trop grand ?)")

    slices_per = int(cfg["slices_per_series"])
    n_img = int(cfg["series_batch"]) * slices_per
    n_local = int(cfg["local_crops_number"])
    n_crops = 2 + n_local
    epochs = int(cfg["epochs"])
    total_steps = epochs * steps_per_epoch
    warmup_epochs = int(cfg.get("warmup_epochs", 10))
    warmup_steps = warmup_epochs * steps_per_epoch
    out_dim = int(cfg.get("out_dim", 4096))
    uniform = math.log(out_dim)
    print(
        f"device {device} | series/step {cfg['series_batch']} | "
        f"coupes/serie {slices_per} | images {n_img} | "
        f"vues eleve {n_crops} | vues prof 2 | "
        f"steps/epoch {steps_per_epoch} | epochs {epochs}",
        flush=True,
    )
    print(f"DINO uniforme ln({out_dim}) = {uniform:.3f}", flush=True)

    encoder = _new_encoder(cfg, load=True).to(device)
    head = DINOHead(
        in_dim=encoder.num_features,
        out_dim=out_dim,
        hidden_dim=int(cfg.get("hidden_dim", 2048)),
        bottleneck_dim=int(cfg.get("bottleneck_dim", 256)),
        use_bn=bool(cfg.get("use_bn_in_head", True)),
        norm_last_layer=bool(cfg.get("norm_last_layer", True)),
    ).to(device)

    teacher_encoder = _new_encoder(cfg, load=False).to(device)
    teacher_head = DINOHead(
        in_dim=encoder.num_features,
        out_dim=out_dim,
        hidden_dim=int(cfg.get("hidden_dim", 2048)),
        bottleneck_dim=int(cfg.get("bottleneck_dim", 256)),
        use_bn=bool(cfg.get("use_bn_in_head", True)),
        norm_last_layer=bool(cfg.get("norm_last_layer", True)),
    ).to(device)
    teacher_encoder.load_state_dict(encoder.state_dict())
    teacher_head.load_state_dict(head.state_dict())
    for p in list(teacher_encoder.parameters()) + list(teacher_head.parameters()):
        p.requires_grad = False

    student = nn.ModuleDict({"encoder": encoder, "head": head})
    teacher = nn.ModuleDict({"encoder": teacher_encoder, "head": teacher_head})

    lr = float(cfg["lr"])
    min_lr = float(cfg.get("min_lr", 1.0e-6))
    wd = float(cfg.get("weight_decay", 1.0e-4))
    decay, no_decay = [], []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    opt_name = str(cfg.get("optim", "sgd")).lower()
    if opt_name == "sgd":
        optim = torch.optim.SGD(groups, lr=lr, momentum=0.9)
    elif opt_name == "adamw":
        optim = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))
    else:
        raise SystemExit(f"optim SSL inconnu : {opt_name}")
    print(f"optim {opt_name} | lr {lr:g} | wd {wd:g}", flush=True)
    criterion = DINOLoss(
        out_dim=out_dim,
        ncrops=n_crops,
        teacher_temp=float(cfg.get("teacher_temp", 0.04)),
        warmup_teacher_temp=float(cfg.get("warmup_teacher_temp", 0.04)),
        warmup_teacher_temp_epochs=int(cfg.get("warmup_teacher_temp_epochs", 0)),
        nepochs=epochs,
        student_temp=float(cfg.get("student_temp", 0.1)),
        center_momentum=float(cfg.get("center_momentum", 0.9)),
    ).to(device)

    log_every = int(cfg.get("log_every", 50))
    forward_chunk = int(cfg.get("forward_chunk", 64))
    clip_grad = float(cfg.get("clip_grad", 3.0))
    mom_base = float(cfg.get("momentum_teacher", 0.996))
    freeze_last = int(cfg.get("freeze_last_layer", 1))

    start_epoch = 0
    history: list[dict] = []
    step = 0
    resume_path = out_dir / "last.pt"
    if args.resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume sans {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"], strict=True)
        head.load_state_dict(ckpt["head"], strict=True)
        teacher_encoder.load_state_dict(ckpt["teacher_encoder"], strict=True)
        teacher_head.load_state_dict(ckpt["teacher_head"], strict=True)
        crit_sd = ckpt["criterion"]
        key = "teacher_temp_schedule"
        cur = criterion.state_dict()[key]
        old = crit_sd[key]
        if old.shape != cur.shape:
            extended = cur.clone()
            n = min(int(old.numel()), int(extended.numel()))
            extended[:n] = old[:n].to(device=extended.device, dtype=extended.dtype)
            if extended.numel() > old.numel():
                extended[n:] = old[-1].to(device=extended.device, dtype=extended.dtype)
            crit_sd = dict(crit_sd)
            crit_sd[key] = extended
            print(
                f"teacher_temp_schedule {tuple(old.shape)} -> {tuple(extended.shape)}",
                flush=True,
            )
        criterion.load_state_dict(crit_sd)
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
        student.train()
        teacher.eval()
        t0 = time.time()
        running = 0.0
        conf_sum = 0.0
        nb = 0
        t_log = t0
        n_log = 0
        for vol in loader:
            vol = vol.to(device, non_blocking=True)
            slices = sample_slices(vol, slices_per)
            flat = slices.reshape(-1, slices.size(-2), slices.size(-1))
            views = make_views(flat, cfg)
            if cfg.get("imagenet_norm"):
                views = _imagenet_norm(views)
            lr_now = _lr_at(step, total_steps, lr, warmup_steps, min_lr)
            for group in optim.param_groups:
                group["lr"] = lr_now
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                student_out = torch.cat([
                    _forward_logits(encoder, head, v, forward_chunk) for v in views
                ], dim=0)
                with torch.no_grad():
                    teacher_out = torch.cat([
                        _forward_logits(teacher_encoder, teacher_head, v, forward_chunk)
                        for v in views[:2]
                    ], dim=0)
            loss = criterion(student_out, teacher_out, epoch)
            loss.backward()
            if epoch < freeze_last:
                _cancel_last_layer_grad(head)
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    clip_grad,
                )
            optim.step()
            m = _momentum_at(step, total_steps, mom_base)
            _ema_update(student, teacher, m)
            step += 1
            running += float(loss.detach())
            conf_sum += criterion.teacher_confidence(teacher_out.detach(), epoch)
            nb += 1
            n_log += flat.size(0) * n_crops
            if step % log_every == 0:
                dt = max(time.time() - t_log, 1.0e-6)
                print(
                    f"  epoch {epoch + 1} step {nb}/{steps_per_epoch} | "
                    f"loss {running / nb:.4f} | conf {conf_sum / nb:.3f} | "
                    f"{n_log / dt:.0f} vues/s | lr {lr_now:.2e}",
                    flush=True,
                )
                t_log = time.time()
                n_log = 0

        avg = running / max(nb, 1)
        dt = time.time() - t0
        rec = {
            "epoch": epoch + 1,
            "loss": avg,
            "uniform_loss": uniform,
            "teacher_conf": conf_sum / max(nb, 1),
            "lr": optim.param_groups[0]["lr"],
            "seconds": round(dt, 1),
        }
        history.append(rec)
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8",
        )
        payload = {
            "epoch": epoch + 1,
            "encoder": _sd_cpu(encoder),
            "head": _sd_cpu(head),
            "teacher_encoder": _sd_cpu(teacher_encoder),
            "teacher_head": _sd_cpu(teacher_head),
            "criterion": {k: v.detach().cpu() for k, v in criterion.state_dict().items()},
            "optim": optim.state_dict(),
        }
        enc_path = out_dir / "encoder.pt"
        _atomic_torch_save(payload["encoder"], enc_path)
        _atomic_torch_save(payload, resume_path)
        if epoch + 1 in {int(e) for e in cfg.get("snapshot_epochs", [])}:
            _atomic_torch_save(payload["encoder"], out_dir / f"encoder_ep{epoch + 1}.pt")
        print(
            f"epoch {epoch + 1}/{epochs} | loss {avg:.4f} "
            f"(uniforme {uniform:.3f}) | {dt / 60:.1f} min | -> {enc_path}",
            flush=True,
        )
        stop_after = cfg.get("stop_after_epoch")
        if stop_after is not None and epoch + 1 >= int(stop_after):
            print(f"arret apres l'epoch {epoch + 1} (stop_after_epoch, cosine sur {epochs})", flush=True)
            break

    print(
        f"DINO termine. Init v20h : backbone_weights: {out_dir / 'encoder.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
