# -*- coding: utf-8 -*-
"""KneeNet-CNX-a — graphe v20h, encodeur ConvNeXt-T timm en stride 32.

Seul l'encodeur change vs v20h. ConvNeXt-T (`convnext_tiny`) tel que
pre-entraine : carte (768, 7, 7) a 224. Pas de `output_stride=16` :
timm remplace alors le downsample 2x2 stride 2 du stage 3 par un
stride 1, et dilate les dwconv 7x7. Les poids se chargent (memes
shapes) mais le stage 3, le seul lu par la tete, ne calcule plus ce
qu'il a appris. Mesure : val 0.616 / 0.623 sur les 2 premieres epochs
hybrid025, contre 0.757 des l'epoch 1 pour CoAt-a (carte 7x7, memes
donnees).

  - `head.norm` (LayerNorm2d pre-entrainee) appliquee par position :
    forward_features la saute, la carte sortirait non normalisee.
  - LayerNorm partout : pas de BN a geler. LayerScale (`gamma`) gele,
    recette ViT-k.
  - stem 2.5D R=k-1 / G=k / B=k+1 (ViT-k / CoAt-a), stem_mix init G :
    epoch 0 = coupe grise repliquee. Norm = pretrained_cfg (ImageNet).
  - CAM 12 tetes + GAP, BiLSTM init 0, attention coupe par cible,
    label_embed + tete diagonale init liee : v20h.

LLRD : 18 blocs `stages.S.blocks.B`, downsample = 1er bloc de son
stage, head.norm = dernier (src/train/param_groups.py). llrd 0.86 sur
18 blocs ~ 0.8 sur les 12 de ViT-k (stem a 0.066 x pic vs 0.069).

`backbone` YAML : `convnext_tiny.dinov3_lvd1689m` ou
`convnext_tiny.fb_in22k_ft_in1k`, `backbone_weights` assortis.
Sans poids (inference), timm construit `convnext_tiny` sans tag : pas
de dependance au registre de tags de la version timm du kernel.
Fichier autonome (torch + timm) : inline dans les kernels Kaggle.
Entree : (B, 6, 24, 224, 224) grayscale [0, 1]. Slots = v20b.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

N_TARGETS = 12
TIMM_NAME = "convnext_tiny.dinov3_lvd1689m"
DEFAULT_WEIGHTS = "weights/convnext_tiny-dinov3_lvd1689m"
LSTM_HIDDEN = 256
N_PROTOCOL_SLOTS = 6
OUTPUT_STRIDE = 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _weights_file(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin"):
            cand = p / name
            if cand.is_file():
                return cand
        raise FileNotFoundError(
            f"aucun checkpoint dans {p.resolve()} "
            "(attendu model.safetensors)"
        )
    if not p.is_file():
        raise FileNotFoundError(f"poids ConvNeXt introuvables : {p.resolve()}")
    return p


class _ConvNextEncoder(nn.Module):
    """ConvNeXt timm stride 32 -> cartes (N, C, H/32, W/32) apres head.norm."""

    def __init__(self, name: str = TIMM_NAME,
                 weights_path: str = DEFAULT_WEIGHTS,
                 pretrained: bool = True) -> None:
        super().__init__()
        kwargs = dict(num_classes=0, global_pool="",
                      output_stride=OUTPUT_STRIDE)
        if pretrained:
            ckpt = _weights_file(weights_path)
            self.backbone = timm.create_model(
                name, pretrained=True,
                pretrained_cfg_overlay={"file": str(ckpt)}, **kwargs,
            )
            print(f"ConvNeXt charge : {name} | {ckpt} | stride {OUTPUT_STRIDE}",
                  flush=True)
        else:
            self.backbone = timm.create_model(
                name.split(".", 1)[0], pretrained=False, **kwargs,
            )
        self.num_features = int(self.backbone.num_features)
        cfg = self.backbone.pretrained_cfg if pretrained else {}
        self.mean = tuple(float(v) for v in cfg.get("mean", IMAGENET_MEAN))
        self.std = tuple(float(v) for v in cfg.get("std", IMAGENET_STD))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.head.norm(self.backbone.forward_features(x))


class KneeNet(nn.Module):
    arch_id = "kneenet_cnx_a"

    def __init__(self, backbone: str = TIMM_NAME,
                 n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 8, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 lstm_hidden: int = LSTM_HIDDEN) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.lstm_hidden = int(lstm_hidden)

        self.encoder = _ConvNextEncoder(
            name=self.backbone_name, weights_path=backbone_weights,
            pretrained=pretrained,
        )
        self._freeze_layer_scale()
        feat_dim = self.encoder.num_features

        self.stem_mix = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self._init_stem_mix()

        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        self.attn_score = nn.Linear(feat_dim, n_targets)
        self.spatial_score = nn.Conv2d(feat_dim, n_targets, kernel_size=1)
        nn.init.zeros_(self.spatial_score.weight)
        nn.init.zeros_(self.spatial_score.bias)

        self.slice_lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Linear(2 * self.lstm_hidden, feat_dim)
        nn.init.zeros_(self.lstm_proj.weight)
        nn.init.zeros_(self.lstm_proj.bias)

        self.label_embed = nn.Parameter(torch.zeros(n_targets, feat_dim))
        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_out = nn.Linear(head_dim, n_targets)
        with torch.no_grad():
            w0 = self.head_out.weight[0].detach().clone()
            b0 = self.head_out.bias[0].detach().clone()
            self.head_out.weight.copy_(w0.unsqueeze(0).expand_as(self.head_out.weight))
            self.head_out.bias.copy_(b0.expand_as(self.head_out.bias))

        self.register_buffer(
            "img_mean", torch.tensor(self.encoder.mean).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "img_std", torch.tensor(self.encoder.std).view(1, 3, 1, 1),
            persistent=True,
        )
        print(
            f"KneeNet-CNX-a | {self.backbone_name} | stride {OUTPUT_STRIDE} "
            f"| CAM + BiLSTM | RGB k-1/k/k+1 | dim {feat_dim} "
            f"| slots {self.n_series} | tete par cible (diag, init liee) "
            f"| norm {self.encoder.mean}/{self.encoder.std}",
            flush=True,
        )

    def _init_stem_mix(self) -> None:
        """out = in_G pour les 3 canaux -> R=G=B=k au depart."""
        with torch.no_grad():
            self.stem_mix.weight.zero_()
            self.stem_mix.weight[:, 1, 0, 0] = 1.0

    @staticmethod
    def _is_layerscale(name: str) -> bool:
        return name.rsplit(".", 1)[-1].startswith("gamma")

    def _freeze_layer_scale(self) -> None:
        n = 0
        for name, p in self.encoder.named_parameters():
            if self._is_layerscale(name):
                p.requires_grad = False
                n += 1
        print(f"LayerScale gele : {n} tenseurs gamma", flush=True)

    def set_encoder_frozen(self, flag: bool) -> None:
        for name, p in self.encoder.named_parameters():
            p.requires_grad = (not flag) and not self._is_layerscale(name)
        self.stem_mix.weight.requires_grad = not flag

    def encoder_parameters(self):
        for p in self.encoder.parameters():
            if p.requires_grad:
                yield p
        if self.stem_mix.weight.requires_grad:
            yield self.stem_mix.weight

    def head_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith("encoder.") or name.startswith("stem_mix."):
                continue
            yield p

    @torch.no_grad()
    def extra_log(self) -> dict[str, float]:
        """Part de |stem_mix| par coupe + divergence des lignes de head_out."""
        s = self.stem_mix.weight.detach().abs().sum(dim=(0, 2, 3))
        s = s / s.sum().clamp(min=1e-8)
        w = self.head_out.weight.detach()
        b = self.head_out.bias.detach()
        return {
            "stem_mix_km1": float(s[0]),
            "stem_mix_k": float(s[1]),
            "stem_mix_kp1": float(s[2]),
            "head_out_w_div": float((w - w[0]).abs().mean()),
            "head_out_b_div": float((b - b[0]).abs().mean()),
            "label_embed_abs": float(self.label_embed.detach().abs().mean()),
        }

    def _stem_rgb(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """R=k-1, G=k, B=k+1 puis stem_mix. x (B,S,D,H,W), idx plat."""
        _b, s, d, _h, _w = x.shape
        bs = idx // (s * d)
        rest = idx % (s * d)
        ss = rest // d
        dd = rest % d
        d_m = (dd - 1).clamp(0, d - 1)
        d_p = (dd + 1).clamp(0, d - 1)
        rgb = torch.stack(
            [x[bs, ss, d_m], x[bs, ss, dd], x[bs, ss, d_p]],
            dim=1,
        )
        rgb = self.stem_mix(rgb)
        return (rgb - self.img_mean) / self.img_std

    def _encode_indexed(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Stem + encodeur par paquets de `encode_chunk` coupes.

        Le stem par paquet evite de materialiser le RGB fp32 de tout le
        batch d'un bloc (meme raison que coat_a320). Memes gradients.
        """
        chunk = max(1, self.encode_chunk)
        outs = []
        for i in range(0, int(idx.numel()), chunk):
            rgb = self._stem_rgb(x, idx[i:i + chunk])
            if self.training:
                feats = checkpoint(self.encoder, rgb, use_reentrant=False)
            else:
                feats = self.encoder(rgb)
            outs.append(feats)
        return torch.cat(outs, dim=0)

    def _spatial_pool(self, maps: torch.Tensor, return_attn: bool = False):
        """maps (N, C, H, W) -> gap (N, C), local (N, T, C) [, attn (N, T, H, W)]."""
        n, c, h, w = maps.shape
        gap = maps.mean(dim=(-2, -1))
        logits = self.spatial_score(maps)
        attn = torch.softmax(
            logits.float().reshape(n, self.n_targets, h * w), dim=-1,
        ).to(maps.dtype)
        flat = maps.reshape(n, c, h * w)
        local = torch.einsum("nth,nch->ntc", attn, flat)
        if return_attn:
            return gap, local, attn.reshape(n, self.n_targets, h, w)
        return gap, local

    def _temporal_delta(self, gaps: torch.Tensor, b: int, s: int,
                        d: int) -> torch.Tensor:
        """gaps (BSD, C) -> residu LSTM (B, S, D, C). Epoch 0 : 0."""
        c = gaps.size(-1)
        seq = gaps.reshape(b * s, d, c)
        with torch.autocast("cuda", enabled=False):
            h, _ = self.slice_lstm(seq.float())
            delta = self.lstm_proj(h)
        return delta.to(dtype=gaps.dtype).reshape(b, s, d, c)

    def _head_logits(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled (B, T, C) -> logits (B, T). Epoch 0 = tete partagee."""
        pooled = pooled + self.label_embed.to(dtype=pooled.dtype)
        hidden = self.head(self.norm(pooled))
        return self.head_out(hidden).diagonal(dim1=-2, dim2=-1)

    def _slice_pool(self, x: torch.Tensor, valid: torch.Tensor,
                    gaps: torch.Tensor, local: torch.Tensor):
        """gaps (BSD,C), local (BSD,T,C) -> logits (B,T), slice_attn (B,S,D,T)."""
        b, s, d = valid.shape
        delta = self._temporal_delta(gaps, b, s, d)
        pe = self.plane_embed[:s].view(1, s, 1, -1)
        se = self.slice_embed[:d].view(1, 1, d, -1)
        feats = gaps.reshape(b, s, d, -1) + delta + pe + se
        scores = self.attn_score(feats)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        scores = scores.reshape(b, s * d, self.n_targets)
        attn = torch.softmax(scores.float(), dim=1).to(feats.dtype)
        tokens = local.reshape(b, s, d, self.n_targets, -1)
        tokens = tokens + delta.unsqueeze(3) + pe.unsqueeze(3) + se.unsqueeze(3)
        tokens = tokens.reshape(b, s * d, self.n_targets, -1)
        pooled = torch.einsum("bnt,bntc->btc", attn, tokens)
        logits = self._head_logits(pooled)
        return logits, attn.reshape(b, s, d, self.n_targets)

    def _encode_valid(self, x: torch.Tensor, valid: torch.Tensor,
                      return_attn: bool = False):
        b, s, d, img_h, img_w = x.shape
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        gaps = x.new_zeros(n_tok, feat_dim)
        local = x.new_zeros(n_tok, self.n_targets, feat_dim)
        spatial = None
        if idx.numel() > 0:
            maps = self._encode_indexed(x, idx)
            pooled = self._spatial_pool(maps, return_attn=return_attn)
            gap, loc = pooled[0], pooled[1]
            gaps = gaps.to(dtype=gap.dtype)
            local = local.to(dtype=loc.dtype)
            gaps[idx] = gap
            local[idx] = loc
            if return_attn:
                spat = pooled[2]
                spatial = spat.new_zeros(n_tok, self.n_targets, *spat.shape[-2:])
                spatial[idx] = spat
        if return_attn and spatial is None:
            spatial = x.new_zeros(
                n_tok, self.n_targets,
                img_h // OUTPUT_STRIDE, img_w // OUTPUT_STRIDE,
            )
        return gaps, local, spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, S, D, H, W) dans [0, 1] -> logits (B, 12)."""
        _b, s, d, _h, _w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        if d > self.slice_embed.size(0):
            raise ValueError(
                f"D={d} > max_depth={self.slice_embed.size(0)} : "
                "augmenter max_depth"
            )
        valid = x.amax(dim=(-2, -1)) > 1e-6
        gaps, local, _ = self._encode_valid(x, valid)
        logits, _attn = self._slice_pool(x, valid, gaps, local)
        return logits

    @torch.no_grad()
    def interpret(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward eval + cartes. spatial_attn : (B, S, D, T, h, w)."""
        self.eval()
        b, s, d, _h, _w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        valid = x.amax(dim=(-2, -1)) > 1e-6
        gaps, local, spatial = self._encode_valid(x, valid, return_attn=True)
        logits, slice_attn = self._slice_pool(x, valid, gaps, local)
        delta = self._temporal_delta(gaps, b, s, d)
        pe = self.plane_embed[:s].view(1, s, 1, 1, -1)
        se = self.slice_embed[:d].view(1, 1, d, 1, -1)
        cam_tokens = (
            local.reshape(b, s, d, self.n_targets, -1)
            + delta.unsqueeze(3) + pe + se
        )
        hw_h, hw_w = spatial.shape[-2:]
        return {
            "logits": logits,
            "slice_attn": slice_attn,
            "spatial_attn": spatial.reshape(b, s, d, self.n_targets, hw_h, hw_w),
            "cam_tokens": cam_tokens,
            "valid": valid,
        }
