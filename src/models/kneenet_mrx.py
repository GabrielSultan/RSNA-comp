# -*- coding: utf-8 -*-
"""KneeNet MRX — transformer IRM factorise (profondeur puis espace).

v20h, ViT-k, CoAt et le Mamba MV-LQM encodent chaque coupe, puis
poolent l'espace (GAP ou CAM) avant tout melange entre coupes. Une
dechirure meniscale, le trajet du LCA ou une fracture se jugent au
meme site, d'une coupe a la suivante. Ce signal est detruit par le
pool. MRX le garde.

  1. ResNet-50 dilate (meme stem A+B que v20h) -> carte, puis 1x1 et
     pool adaptatif `grid` x `grid` (7 par defaut).
  2. Attention en profondeur a chaque site (suite de 24 coupes).
  3. Attention spatiale sur la grille, coupe par coupe.
  4. 12 requetes croisent tous les tokens valides. Tete diagonale
     (epoch 0 = tete partagee, comme v20h).
  5. Residu MRNet : max sur les coupes valides, moyenne des plans,
     Linear init 0. Il demarre muet et peut s'allumer.

Entree : (B, S, D, H, W) grayscale [0, 1]. S = 6 slots v20b par
defaut. Coupe vide (slot absent) = masque, pas un token.

Critere d'arret sur le subset hybrid025 : macro val sous le v20h du
meme split de plus de 0.01 apres 7 epochs, ou erreurs quasi
colineaires avec v20h. Dans ce cas le membre d'ensemble n'existe pas.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50

N_TARGETS = 12
DEFAULT_WEIGHTS = "weights/ResNet50.pt"
N_PROTOCOL_SLOTS = 6


def _dilate_layer4(layer4: nn.Sequential, dilate: int = 2) -> None:
    """stride 32 -> 16, in-place. Poids 3x3 / 1x1 inchanges."""
    for i, block in enumerate(layer4):
        if i == 0:
            block.conv2.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        block.conv2.dilation = (dilate, dilate)
        block.conv2.padding = (dilate, dilate)


class _ResNet50Encoder(nn.Module):
    """ResNet-50 jusqu'a layer4 dilated. Cles `backbone.N.*` = v20h."""

    def __init__(self) -> None:
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        _dilate_layer4(self.backbone[7])
        self.num_features = 2048
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _encoder_layer(d_model: int, nhead: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


def _run_encoder(enc: nn.TransformerEncoder, seq: torch.Tensor,
                 pad: torch.Tensor) -> torch.Tensor:
    """Attention avec masque. Une rangee toute masquee est mise a zero.

    PyTorch renvoie des NaN si `src_key_padding_mask` est vrai partout.
    """
    all_bad = pad.all(dim=1)
    if bool(all_bad.any()):
        pad = pad.clone()
        pad[all_bad, 0] = False
    out = enc(seq, src_key_padding_mask=pad)
    if bool(all_bad.any()):
        out = out.masked_fill(all_bad.view(-1, 1, 1), 0.0)
    return out


class KneeNet(nn.Module):
    arch_id = "kneenet_mrx"

    def __init__(self, backbone: str = "resnet50", n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, head_dim: int = 256,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 8, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 d_model: int = 128, grid: int = 7, nhead: int = 4,
                 n_depth_layers: int = 1, n_space_layers: int = 1,
                 attn_dropout: float = 0.1,
                 encoder_ckpt: str | None = None) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"MRX est un ResNet-50, recu {backbone}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} non divisible par nhead {nhead}")
        if int(grid) < 1 or int(n_depth_layers) < 1 or int(n_space_layers) < 1:
            raise ValueError("grid, n_depth_layers et n_space_layers >= 1")
        self.n_series = int(n_series)
        self.n_targets = int(n_targets)
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = "resnet50"
        self.d_model = int(d_model)
        self.grid = int(grid)
        self.max_depth = int(max_depth)

        self.encoder = _ResNet50Encoder()
        self.proj = nn.Conv2d(self.encoder.num_features, self.d_model, kernel_size=1)
        self.grid_pool = nn.AdaptiveAvgPool2d(self.grid)

        self.plane_embed = nn.Parameter(torch.zeros(self.n_series, self.d_model))
        self.slice_embed = nn.Parameter(torch.zeros(self.max_depth, self.d_model))
        self.grid_embed = nn.Parameter(torch.zeros(self.grid, self.grid, self.d_model))

        depth_layer = _encoder_layer(self.d_model, int(nhead), float(attn_dropout))
        space_layer = _encoder_layer(self.d_model, int(nhead), float(attn_dropout))
        self.depth_enc = nn.TransformerEncoder(
            depth_layer, num_layers=int(n_depth_layers), enable_nested_tensor=False,
        )
        self.space_enc = nn.TransformerEncoder(
            space_layer, num_layers=int(n_space_layers), enable_nested_tensor=False,
        )
        self.query = nn.Parameter(torch.empty(self.n_targets, self.d_model))
        nn.init.normal_(self.query, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            self.d_model, int(nhead), dropout=float(attn_dropout), batch_first=True,
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_out = nn.Linear(head_dim, self.n_targets)
        with torch.no_grad():
            w0 = self.head_out.weight[0].detach().clone()
            b0 = self.head_out.bias[0].detach().clone()
            self.head_out.weight.copy_(w0.unsqueeze(0).expand_as(self.head_out.weight))
            self.head_out.bias.copy_(b0.expand_as(self.head_out.bias))

        # Max-pool MRNet, init 0 : n'ajoute rien tant qu'il n'a pas appris.
        self.mrnet_proj = nn.Linear(self.d_model, self.n_targets)
        nn.init.zeros_(self.mrnet_proj.weight)
        nn.init.zeros_(self.mrnet_proj.bias)

        self.stem_mix = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self._init_stem_mix()
        self.register_buffer("img_mean", torch.zeros(1, 3, 1, 1), persistent=True)
        self.register_buffer("img_std", torch.ones(1, 3, 1, 1), persistent=True)

        if pretrained:
            self._load_radimagenet(backbone_weights)
        if encoder_ckpt:
            self._load_encoder_ckpt(encoder_ckpt)
        print(
            f"KneeNet-MRX | ResNet-50 dilate | d {self.d_model} | "
            f"grille {self.grid}x{self.grid} | profondeur x{int(n_depth_layers)} "
            f"| espace x{int(n_space_layers)} | tetes {int(nhead)} | "
            f"slots {self.n_series} | residu MRNet init 0",
            flush=True,
        )

    def extra_log(self) -> dict[str, float]:
        w = self.head_out.weight.detach()
        return {
            "head_out_w_div": float((w - w[0]).abs().mean()),
            "mrnet_abs": float(self.mrnet_proj.weight.detach().abs().mean()),
            "query_abs": float(self.query.detach().abs().mean()),
        }

    def _init_stem_mix(self) -> None:
        """out_c = in_G pour les 3 canaux -> R=G=B=k au depart."""
        with torch.no_grad():
            self.stem_mix.weight.zero_()
            self.stem_mix.weight[:, 1, 0, 0] = 1.0

    def _load_radimagenet(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"poids encodeur introuvables : {p.resolve()}"
            )
        sd = torch.load(p, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(sd, strict=True)
        print(f"encodeur ResNet-50 charge : {p}", flush=True)

    def _load_encoder_ckpt(self, path: str) -> None:
        """Reprend encoder + stem d'un v20h (ou MRX) deja entraine.

        Le melangeur et la tete restent neufs : c'est le test du biais
        spatial en profondeur, pas un nouvel encodeur.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"encoder_ckpt introuvable : {p.resolve()}")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        src = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        own = self.state_dict()
        picked = {
            k: v for k, v in src.items()
            if k in own and own[k].shape == v.shape
            and (k.startswith("encoder.") or k.startswith("stem_mix.")
                 or k in ("img_mean", "img_std"))
        }
        need = [k for k in own if k.startswith("encoder.")]
        missing = [k for k in need if k not in picked]
        if missing:
            raise RuntimeError(
                f"encoder_ckpt {p} : {len(missing)} tenseurs encodeur "
                f"absents ou de mauvaise forme (ex. {missing[0]})"
            )
        self.load_state_dict(picked, strict=False)
        print(f"encoder_ckpt {p} | {len(picked)} tenseurs | tete MRX neuve",
              flush=True)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def set_encoder_frozen(self, flag: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not flag
        for p in self.stem_mix.parameters():
            p.requires_grad = not flag

    def encoder_parameters(self):
        for p in self.encoder.parameters():
            if p.requires_grad:
                yield p

    def head_parameters(self):
        for name, p in self.named_parameters():
            if p.requires_grad and not name.startswith("encoder."):
                yield p

    def _stem_rgb(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Triplet A+B pour les tokens valides. x (B,S,D,H,W), idx plat."""
        _b, s, d, _h, _w = x.shape
        bs = idx // (s * d)
        rest = idx % (s * d)
        ss = rest // d
        dd = rest % d
        d_m = (dd - 1).clamp(0, d - 1)
        d_p = (dd + 1).clamp(0, d - 1)
        center = x[bs, ss, dd]
        rgb = torch.stack(
            [
                0.5 + 0.5 * (center - x[bs, ss, d_m]),
                center,
                0.5 + 0.5 * (x[bs, ss, d_p] - center),
            ],
            dim=1,
        )
        rgb = self.stem_mix(rgb)
        return (rgb - self.img_mean) / self.img_std

    def _encode_grid(self, part: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, d, grid, grid)."""
        maps = self.encoder(part)
        return self.grid_pool(self.proj(maps))

    def _encode_chunks(self, rgb: torch.Tensor) -> torch.Tensor:
        n = rgb.size(0)
        g = self.grid
        if n == 0:
            return rgb.new_zeros(0, self.d_model, g, g)
        outs = []
        chunk = max(1, self.encode_chunk)
        for i in range(0, n, chunk):
            part = rgb[i:i + chunk]
            if self.training:
                feats = checkpoint(self._encode_grid, part, use_reentrant=False)
            else:
                feats = self._encode_grid(part)
            outs.append(feats)
        return torch.cat(outs, dim=0)

    def _depth_mix(self, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """tok (B,S,D,H,W,C), valid (B,S,D) -> meme forme, melange en D."""
        b, s, d, h, w, c = tok.shape
        seq = tok.permute(0, 1, 3, 4, 2, 5).reshape(b * s * h * w, d, c)
        pad = ((~valid).view(b, s, 1, 1, d)
               .expand(b, s, h, w, d)
               .reshape(b * s * h * w, d))
        seq = _run_encoder(self.depth_enc, seq, pad)
        dead = (~valid).view(b, s, d, 1, 1, 1)
        out = seq.reshape(b, s, h, w, d, c).permute(0, 1, 4, 2, 3, 5)
        return out.masked_fill(dead, 0.0)

    def _space_mix(self, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """tok (B,S,D,H,W,C) -> meme forme, melange sur la grille."""
        b, s, d, h, w, c = tok.shape
        seq = tok.reshape(b * s * d, h * w, c)
        pad = ((~valid).reshape(b * s * d, 1).expand(-1, h * w))
        seq = _run_encoder(self.space_enc, seq, pad)
        dead = (~valid).reshape(b, s, d, 1, 1, 1)
        return seq.reshape(b, s, d, h, w, c).masked_fill(dead, 0.0)

    def _mrnet_logits(self, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Max des coupes, moyenne des plans. tok (B,S,D,H,W,C)."""
        gap = tok.mean(dim=(-3, -2))
        neg = torch.finfo(gap.dtype).min
        masked = gap.masked_fill(~valid.unsqueeze(-1), neg)
        mx = masked.max(dim=2).values
        plane_ok = valid.any(dim=2)
        mx = mx.masked_fill(~plane_ok.unsqueeze(-1), 0.0)
        denom = plane_ok.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = mx.sum(dim=1) / denom
        return self.mrnet_proj(pooled)

    def _readout(self, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Requetes cibles + residu MRNet. tok (B,S,D,H,W,C) -> (B, T)."""
        b, s, d, h, w, c = tok.shape
        flat = tok.reshape(b, s * d * h * w, c)
        keep = (valid.view(b, s, d, 1, 1)
                .expand(b, s, d, h, w)
                .reshape(b, s * d * h * w))
        pad = ~keep
        all_bad = pad.all(dim=1)
        if bool(all_bad.any()):
            pad = pad.clone()
            pad[all_bad, 0] = False
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        ctx, _ = self.cross_attn(q, flat, flat, key_padding_mask=pad, need_weights=False)
        hidden = self.head(self.norm(ctx))
        logits = self.head_out(hidden).diagonal(dim1=-2, dim2=-1)
        logits = logits + self._mrnet_logits(tok, valid)
        if bool(all_bad.any()):
            logits = logits.masked_fill(all_bad.unsqueeze(-1), 0.0)
        return logits

    def _mix(self, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Melange fp32. tok (B,S,D,H,W,C) -> logits (B, T)."""
        with torch.autocast("cuda", enabled=False):
            tok = tok.float()
            b, s, d, h, w, _c = tok.shape
            if s > self.plane_embed.size(0) or d > self.slice_embed.size(0):
                raise ValueError(
                    f"S={s} D={d} hors embeds "
                    f"({self.plane_embed.size(0)}, {self.slice_embed.size(0)})"
                )
            if (h, w) != (self.grid, self.grid):
                raise RuntimeError(
                    f"grille {h}x{w}, attendu {self.grid}x{self.grid}"
                )
            tok = tok + self.plane_embed[:s].view(1, s, 1, 1, 1, -1)
            tok = tok + self.slice_embed[:d].view(1, 1, d, 1, 1, -1)
            tok = tok + self.grid_embed.view(1, 1, 1, h, w, -1)
            dead = (~valid).view(b, s, d, 1, 1, 1)
            tok = tok.masked_fill(dead, 0.0)
            tok = self._depth_mix(tok, valid)
            tok = self._space_mix(tok, valid)
            return self._readout(tok, valid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, S, D, H, W) dans [0, 1] -> logits (B, n_targets)."""
        b, s, d, _h, _w = x.shape
        if s != self.n_series:
            raise ValueError(f"attendu {self.n_series} series, recu {s}")
        if d > self.max_depth:
            raise ValueError(f"D={d} > max_depth={self.max_depth}")

        valid = x.amax(dim=(-2, -1)) > 1e-6
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        g = self.grid
        grids = x.new_zeros(n_tok, self.d_model, g, g)
        if idx.numel() > 0:
            maps = self._encode_chunks(self._stem_rgb(x, idx))
            if grids.dtype != maps.dtype:
                grids = grids.to(dtype=maps.dtype)
            grids[idx] = maps
        tok = grids.reshape(b, s, d, self.d_model, g, g).permute(0, 1, 2, 4, 5, 3)
        return self._mix(tok, valid).to(dtype=x.dtype)
