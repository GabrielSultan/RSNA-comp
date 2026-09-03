# -*- coding: utf-8 -*-
"""KneeNet-ViT-c — DINOv3 + graphe v8 (pas v20b).

v1/vB collaient CAM + BiLSTM + GAP patches sur le ViT : plancher 0.80,
B collapse 0.62 puis recul au degel. Ici on isole le pooling natif :

  - stem : coupe centrale x3 + mean/std ImageNet (comme v1, pas A+B)
  - 1 vecteur par coupe = CLS DINOv3 (registers / patches jetes)
  - attention 12 tetes sur les 6x24 tokens (comme v8)
  - pas de CAM, pas de LSTM
  - out_bias = logit des prevalences train (comme v8bis)

Recette YAML : freeze 1 epoch, lr_backbone 1e-5, LayerScale gele.
Gel 2 + 3e-5 detruit le CLS au degel (val 0.69 -> 0.61, std 0.22 -> 0.13).
Poids : `weights/dinov3-vitb16-lvd1689m/model.safetensors`
Entree : (B, 6, 24, H, W) grayscale [0, 1]. Slots = v20b.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

N_TARGETS = 12
DEFAULT_WEIGHTS = "weights/dinov3-vitb16-lvd1689m"
N_PROTOCOL_SLOTS = 6
N_PREFIX = 5  # CLS + 4 registers DINOv3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TIMM_NAME = "vit_base_patch16_dinov3.lvd1689m"


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
        raise FileNotFoundError(f"poids DINOv3 introuvables : {p.resolve()}")
    return p


class _Dinov3ClsEncoder(nn.Module):
    """ViT-B/16 DINOv3 -> CLS (N, 768). Patches et registers jetes."""

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS,
                 pretrained: bool = True) -> None:
        super().__init__()
        kwargs = dict(
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
        )
        if pretrained:
            ckpt = _weights_file(weights_path)
            self.backbone = timm.create_model(
                TIMM_NAME,
                pretrained=True,
                pretrained_cfg_overlay={"file": str(ckpt)},
                **kwargs,
            )
            print(f"DINOv3 ViT-B/16 charge : {ckpt}", flush=True)
        else:
            self.backbone = timm.create_model(
                TIMM_NAME, pretrained=False, **kwargs,
            )
        self.backbone.set_grad_checkpointing(True)
        self.num_features = int(self.backbone.num_features)
        self.n_prefix = int(getattr(
            self.backbone, "num_prefix_tokens", N_PREFIX))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (N, 3, H, W) -> CLS (N, C)."""
        tokens = self.backbone.forward_features(x)
        return tokens[:, 0]


class KneeNet(nn.Module):
    arch_id = "kneenet_vit_c"

    def __init__(self, backbone: str = TIMM_NAME,
                 n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 4, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 bias_init: list[float] | None = None) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.bias_init = bias_init

        self.encoder = _Dinov3ClsEncoder(
            weights_path=backbone_weights, pretrained=pretrained,
        )
        self._freeze_layer_scale()
        feat_dim = self.encoder.num_features

        self.plane_embed = nn.Parameter(torch.zeros(n_series, feat_dim))
        self.slice_embed = nn.Parameter(torch.zeros(max_depth, feat_dim))
        self.attn_score = nn.Linear(feat_dim, n_targets)
        self.norm = nn.LayerNorm(feat_dim)
        self.out_bias = nn.Parameter(torch.zeros(n_targets))
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.register_buffer(
            "img_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "img_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=True,
        )
        self._init_head()
        print(
            f"KneeNet-ViT-c | CLS | attn v8 | dim {feat_dim} "
            f"| bias_init {bias_init is not None} | slots {self.n_series}",
            flush=True,
        )

    def _init_head(self) -> None:
        if self.bias_init is None:
            return
        b = torch.tensor(self.bias_init, dtype=self.out_bias.dtype)
        if b.numel() != self.out_bias.numel():
            raise ValueError(
                f"bias_init de taille {b.numel()}, "
                f"attendu {self.out_bias.numel()}"
            )
        with torch.no_grad():
            self.out_bias.copy_(b)

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
        """Gele/degel de l'encodeur. Les gamma LayerScale restent geles."""
        for name, p in self.encoder.named_parameters():
            p.requires_grad = (not flag) and not self._is_layerscale(name)

    def encoder_parameters(self):
        for p in self.encoder.parameters():
            if p.requires_grad:
                yield p

    def head_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("encoder."):
                yield p

    def _encode_chunks(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb ImageNet -> CLS (N, C)."""
        n = rgb.size(0)
        c = self.encoder.num_features
        if n == 0:
            return rgb.new_zeros(0, c)
        outs = []
        chunk = max(1, self.encode_chunk)
        for i in range(0, n, chunk):
            part = rgb[i:i + chunk]
            if self.training:
                feats = checkpoint(self.encoder, part, use_reentrant=False)
            else:
                feats = self.encoder(part)
            outs.append(feats)
        return torch.cat(outs, dim=0)

    def _stem_rgb(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Coupe centrale x3 + ImageNet. x (B,S,D,H,W), idx plat."""
        _b, s, d, _h, _w = x.shape
        bs = idx // (s * d)
        rest = idx % (s * d)
        ss = rest // d
        dd = rest % d
        center = x[bs, ss, dd]
        rgb = center.unsqueeze(1).expand(-1, 3, -1, -1)
        return (rgb - self.img_mean) / self.img_std

    def _pool(self, feats: torch.Tensor, valid: torch.Tensor):
        """feats (B,S,D,C) CLS -> logits (B,T), slice_attn (B,S,D,T)."""
        b, s, d, _c = feats.shape
        feats = feats + self.plane_embed[:s].view(1, s, 1, -1)
        feats = feats + self.slice_embed[:d].view(1, 1, d, -1)
        scores = self.attn_score(feats)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e4)
        scores = scores.reshape(b, s * d, self.n_targets)
        attn = torch.softmax(scores.float(), dim=1).to(feats.dtype)
        tokens = feats.reshape(b, s * d, -1)
        pooled = torch.einsum("bnt,bnf->btf", attn, tokens)
        logits = self.head(self.norm(pooled)).squeeze(-1) + self.out_bias
        return logits, attn.reshape(b, s, d, self.n_targets)

    def _fill_cls(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, s, d, _h, _w = x.shape
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        feats = x.new_zeros(n_tok, feat_dim)
        if idx.numel() > 0:
            cls = self._encode_chunks(self._stem_rgb(x, idx))
            if feats.dtype != cls.dtype:
                feats = feats.to(dtype=cls.dtype)
            feats[idx] = cls
        return feats.reshape(b, s, d, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, S, D, H, W) dans [0, 1] -> logits (B, 12)."""
        b, s, d, _h, _w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        if d > self.slice_embed.size(0):
            raise ValueError(
                f"D={d} > max_depth={self.slice_embed.size(0)} : "
                "augmenter max_depth"
            )
        valid = x.amax(dim=(-2, -1)) > 1e-6
        feats = self._fill_cls(x, valid)
        logits, _attn = self._pool(feats, valid)
        return logits

    @torch.no_grad()
    def interpret(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward eval + attention de coupe. Pas de CAM (pas de spatial_attn)."""
        self.eval()
        b, s, d, _h, _w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        valid = x.amax(dim=(-2, -1)) > 1e-6
        feats = self._fill_cls(x, valid)
        logits, slice_attn = self._pool(feats, valid)
        return {
            "logits": logits,
            "slice_attn": slice_attn,
            "gap_tokens": feats,
            "valid": valid,
        }
