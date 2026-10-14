# -*- coding: utf-8 -*-
"""MV-LQM Mamba — copie du checkpoint Original_main (fold_*.pt).

Graphe fige par les cles du state_dict `mv_lqm_mamba` :

  encoder.backbone : ResNet-50 torchvision, stride 32, sans FC.
    `layer4` est le meme module que `encoder.backbone.7` (alias du
    groupe de LR, pas un second layer4).
  aggregator.input_proj : LayerNorm(2048) + Linear(2048 -> 256)
  aggregator.plane_embed (3, 256), slice_embed (24, 256)
  aggregator.ssm : Mamba d_model 256, expand 2, d_state 1, conv k=4
  aggregator.query (12, 256) + cross_attn + query_norm + tete
    Linear(256->256) / GELU / Dropout(0) / Linear(256->1)

Entree dataset : (B, 3, 24, 224, 224) gris [0, 1], crop130.
Le forward applique le FOV 120 mm du checkpoint (crop 207 px, offset 8,
resize 224), repete le gris sur 3 canaux, norme ImageNet, GAP 7x7.

Trois choix absents des tenseurs, tenus fixes :
  - conv causale (padding 3, coupe a L) et scan d_state=1 en Euler
  - cross-attention a 8 tetes (les shapes QKV ne donnent pas le nombre)
  - query_norm apres l'attention, avant la tete
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50

N_TARGETS = 12
N_SERIES = 3
N_SLICES = 24
D_MODEL = 256
D_INNER = 512
D_CONV = 4
N_HEADS = 8
FEAT_DIM = 2048
MAP_HW = 7
FOV_OFFSET = 8
FOV_CROP = 207
DEFAULT_WEIGHTS = "weights/ResNet50.pt"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _ResNet50Encoder(nn.Module):
    """ResNet-50 jusqu'a layer4. Cles `backbone.N.*` = weights/ResNet50.pt."""

    def __init__(self) -> None:
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.num_features = FEAT_DIM
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _MambaDState1(nn.Module):
    """Bloc Mamba diagonal, un etat par canal (d_state=1)."""

    def __init__(self, d_model: int = D_MODEL, d_inner: int = D_INNER,
                 d_conv: int = D_CONV) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.local_conv = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner,
            padding=d_conv - 1, bias=True,
        )
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.b_proj = nn.Linear(d_inner, d_inner)
        self.c_proj = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.log_a = nn.Parameter(torch.zeros(d_inner))
        self.d_skip = nn.Parameter(torch.ones(d_inner))
        self._init_ssm()

    def _init_ssm(self) -> None:
        """dt petit, sortie nulle : le bloc demarre comme un residu identite."""
        nn.init.zeros_(self.dt_proj.weight)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        dt = torch.exp(
            torch.rand(self.d_inner) * (torch.log(torch.tensor(0.1))
                                         - torch.log(torch.tensor(1e-3)))
            + torch.log(torch.tensor(1e-3))
        )
        inv_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus)

    def _scan(self, u: torch.Tensor, dt: torch.Tensor, b_coef: torch.Tensor,
              c_coef: torch.Tensor) -> torch.Tensor:
        """u, dt, B, C : (B, L, C). h_t = exp(dt A) h + dt B u, y = C h + D u."""
        with torch.autocast("cuda", enabled=False):
            u32 = u.float()
            dt32 = dt.float()
            b32 = b_coef.float()
            c32 = c_coef.float()
            a = -self.log_a.float().exp()
            h = u32.new_zeros(u32.shape[0], u32.shape[2])
            ys = []
            for t in range(u32.shape[1]):
                dt_t = dt32[:, t]
                h = torch.exp(dt_t * a) * h + dt_t * b32[:, t] * u32[:, t]
                ys.append(c32[:, t] * h)
            y = torch.stack(ys, dim=1)
            y = y + self.d_skip.float() * u32
        return y.to(dtype=u.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, d_model)."""
        residual = x
        z = self.norm(x)
        x_branch, gate = self.in_proj(z).chunk(2, dim=-1)
        seq = x_branch.transpose(1, 2)
        seq = self.local_conv(seq)[..., : x_branch.shape[1]]
        seq = F.silu(seq.transpose(1, 2))
        dt = F.softplus(self.dt_proj(seq))
        y = self._scan(seq, dt, self.b_proj(seq), self.c_proj(seq))
        y = y * F.silu(gate)
        return residual + self.out_proj(y)


class _Aggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.plane_embed = nn.Parameter(torch.zeros(N_SERIES, D_MODEL))
        self.slice_embed = nn.Parameter(torch.zeros(N_SLICES, D_MODEL))
        self.query = nn.Parameter(torch.empty(N_TARGETS, D_MODEL))
        nn.init.normal_(self.query, std=0.02)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(FEAT_DIM),
            nn.Linear(FEAT_DIM, D_MODEL),
        )
        self.ssm = _MambaDState1()
        self.cross_attn = nn.MultiheadAttention(
            D_MODEL, N_HEADS, batch_first=True,
        )
        self.query_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """feats (B, S, D, 2048), valid (B, S, D) -> logits (B, 12)."""
        b, s, d, _c = feats.shape
        tok = self.input_proj(feats)
        tok = tok + self.plane_embed.view(1, s, 1, -1)
        tok = tok + self.slice_embed.view(1, 1, d, -1)
        tok = tok.masked_fill(~valid.unsqueeze(-1), 0.0)
        tok = self.ssm(tok.reshape(b * s, d, D_MODEL)).reshape(b, s * d, D_MODEL)
        keep = valid.reshape(b, s * d)
        tok = tok.masked_fill(~keep.unsqueeze(-1), 0.0)

        q = self.query.unsqueeze(0).expand(b, -1, -1)
        pad = ~keep
        all_bad = pad.all(dim=1)
        if bool(all_bad.any()):
            pad = pad.clone()
            pad[all_bad, 0] = False
        ctx, _ = self.cross_attn(
            q, tok, tok, key_padding_mask=pad, need_weights=False,
        )
        logits = self.head(self.query_norm(ctx)).squeeze(-1)
        if bool(all_bad.any()):
            logits = logits.masked_fill(all_bad.unsqueeze(-1), 0.0)
        return logits


class KneeNet(nn.Module):
    arch_id = "kneenet_mvlqm"

    def __init__(self, backbone: str = "resnet50", n_series: int = N_SERIES,
                 n_targets: int = N_TARGETS, pretrained: bool = True,
                 encode_chunk: int = 16,
                 backbone_weights: str = DEFAULT_WEIGHTS) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"mv_lqm_mamba est un ResNet-50, recu {backbone}")
        if int(n_series) != N_SERIES:
            raise ValueError(f"mv_lqm_mamba a {N_SERIES} plans, recu {n_series}")
        if int(n_targets) != N_TARGETS:
            raise ValueError(f"mv_lqm_mamba a {N_TARGETS} cibles, recu {n_targets}")
        self.n_series = N_SERIES
        self.n_targets = N_TARGETS
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = "resnet50"

        self.encoder = _ResNet50Encoder()
        # Meme objet que backbone[7] : le state_dict porte les deux prefixes.
        self.layer4 = self.encoder.backbone[7]
        self.aggregator = _Aggregator()
        if pretrained:
            self._load_radimagenet(backbone_weights)
        print(
            "MV-LQM Mamba | ResNet-50 stride 32 | d_state 1 | "
            f"3x24 | GAP {MAP_HW}x{MAP_HW} | FOV {FOV_CROP}px+"
            f"{FOV_OFFSET} | encode_chunk {self.encode_chunk}",
            flush=True,
        )

    def _load_radimagenet(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"poids RadImageNet introuvables : {p.resolve()}"
            )
        sd = torch.load(p, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(sd, strict=True)
        print(f"RadImageNet ResNet-50 charge : {p}", flush=True)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def encoder_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("encoder."):
                yield p

    def _fov120(self, x: torch.Tensor) -> torch.Tensor:
        """Crop centre 207 px (offset 8) puis resize 224. x (B,S,D,224,224)."""
        if x.shape[-2:] != (224, 224):
            raise ValueError(
                f"FOV 120 mm attend 224x224, recu {tuple(x.shape[-2:])}"
            )
        y0 = FOV_OFFSET
        y1 = FOV_OFFSET + FOV_CROP
        cropped = x[..., y0:y1, y0:y1]
        b, s, d, h, w = cropped.shape
        flat = cropped.reshape(b * s * d, 1, h, w)
        flat = F.interpolate(flat, size=(224, 224), mode="bilinear",
                             align_corners=False)
        return flat.reshape(b, s, d, 224, 224)

    def _repeat_gray(self, sl: torch.Tensor) -> torch.Tensor:
        """sl (N, H, W) dans [0, 1] -> RGB repete, norme ImageNet."""
        rgb = sl.unsqueeze(1).expand(-1, 3, -1, -1)
        mean = sl.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = sl.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        return (rgb - mean) / std

    def _encode_gap(self, part: torch.Tensor) -> torch.Tensor:
        """(N, 3, 224, 224) -> (N, 2048). Le GAP est dans le checkpoint :
        le backward ne garde pas les cartes 7x7."""
        maps = self.encoder(part)
        if maps.shape[-2:] != (MAP_HW, MAP_HW):
            raise RuntimeError(
                f"ResNet-50 stride 32 : carte {tuple(maps.shape[-2:])}, "
                f"attendu {MAP_HW}x{MAP_HW}"
            )
        return maps.mean(dim=(-2, -1))

    def _encode_chunks(self, rgb: torch.Tensor) -> torch.Tensor:
        n = rgb.size(0)
        if n == 0:
            return rgb.new_zeros(0, FEAT_DIM)
        outs = []
        chunk = max(1, self.encode_chunk)
        for i in range(0, n, chunk):
            part = rgb[i:i + chunk]
            if self.training:
                feats = checkpoint(self._encode_gap, part, use_reentrant=False)
            else:
                feats = self._encode_gap(part)
            outs.append(feats)
        return torch.cat(outs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, 24, 224, 224) gris [0, 1] -> logits (B, 12)."""
        b, s, d, _h, _w = x.shape
        if s != N_SERIES or d != N_SLICES:
            raise ValueError(
                f"attendu ({N_SERIES}, {N_SLICES}) series/coupes, recu {(s, d)}"
            )
        valid = x.amax(dim=(-2, -1)) > 1e-6
        flat = self._fov120(x).reshape(b * s * d, 224, 224)
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return x.new_zeros(b, N_TARGETS)
        rgb = self._repeat_gray(flat.index_select(0, idx)).contiguous()
        gaps = self._encode_chunks(rgb)
        feats = gaps.new_zeros(b * s * d, FEAT_DIM)
        feats.index_copy_(0, idx, gaps)
        return self.aggregator(feats.view(b, s, d, FEAT_DIM), valid)
