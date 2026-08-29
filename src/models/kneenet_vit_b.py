# -*- coding: utf-8 -*-
"""KneeNet-ViT-b — v1 + CLS pour les scores de coupe + bias prior.

Meme corps que kneenet_vit (6 slots, CAM patch, BiLSTM init 0, stem gris x3).
Changements vs le subset v1 (freeze + lr 3e-5, GAP des patches, pas de prior) :

  - CLS DINOv3 -> LSTM / attn_score (quelle coupe)
  - grille patches -> spatial_score (ou dans la coupe)
  - out_bias = logit des prevalences train (comme v8bis)
  - pas de gel encodeur ; lr_backbone 1e-4 (YAML)

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
LSTM_HIDDEN = 256
N_PROTOCOL_SLOTS = 6
PATCH_SIZE = 16
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


class _Dinov3VitEncoder(nn.Module):
    """ViT-B/16 DINOv3 -> cartes (N, 768, h, w) + CLS (N, 768)."""

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
        self.patch_size = PATCH_SIZE
        self.n_prefix = N_PREFIX

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (N, 3, H, W) -> maps (N, C, h, w), cls (N, C)."""
        tokens = self.backbone.forward_features(x)
        cls = tokens[:, 0]
        patches = tokens[:, self.n_prefix:]
        n, n_hw, c = patches.shape
        side = int(n_hw ** 0.5)
        if side * side != n_hw:
            raise ValueError(
                f"grille patch non carree : {n_hw} tokens "
                f"(H={x.shape[-2]}, W={x.shape[-1]})"
            )
        maps = patches.transpose(1, 2).reshape(n, c, side, side)
        return maps, cls


class KneeNet(nn.Module):
    arch_id = "kneenet_vit_b"

    def __init__(self, backbone: str = TIMM_NAME,
                 n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 4, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 lstm_hidden: int = LSTM_HIDDEN,
                 bias_init: list[float] | None = None) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.patch_size = PATCH_SIZE
        self.lstm_hidden = int(lstm_hidden)
        self.bias_init = bias_init

        self.encoder = _Dinov3VitEncoder(
            weights_path=backbone_weights, pretrained=pretrained,
        )
        feat_dim = self.encoder.num_features

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
            f"KneeNet-ViT-b | CAM patch {PATCH_SIZE} | dim {feat_dim} "
            f"| CLS+BiLSTM | bias_init {bias_init is not None} "
            f"| slots {self.n_series}",
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

    def set_encoder_frozen(self, flag: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not flag

    def encoder_parameters(self):
        for p in self.encoder.parameters():
            if p.requires_grad:
                yield p

    def head_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("encoder."):
                yield p

    def _encode_chunks(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """rgb ImageNet -> maps (N, C, h, w), cls (N, C)."""
        n = rgb.size(0)
        c = self.encoder.num_features
        h = rgb.size(-2) // self.patch_size
        w = rgb.size(-1) // self.patch_size
        if n == 0:
            return rgb.new_zeros(0, c, h, w), rgb.new_zeros(0, c)
        maps_out, cls_out = [], []
        chunk = max(1, self.encode_chunk)
        for i in range(0, n, chunk):
            part = rgb[i:i + chunk]
            if self.training:
                maps, cls = checkpoint(self.encoder, part, use_reentrant=False)
            else:
                maps, cls = self.encoder(part)
            maps_out.append(maps)
            cls_out.append(cls)
        return torch.cat(maps_out, dim=0), torch.cat(cls_out, dim=0)

    def _spatial_pool(self, maps: torch.Tensor, return_attn: bool = False):
        """maps (N, C, H, W) -> local (N, T, C) [, attn]."""
        n, c, h, w = maps.shape
        logits = self.spatial_score(maps)
        attn = torch.softmax(
            logits.float().reshape(n, self.n_targets, h * w), dim=-1,
        ).to(maps.dtype)
        flat = maps.reshape(n, c, h * w)
        local = torch.einsum("nth,nch->ntc", attn, flat)
        if return_attn:
            return local, attn.reshape(n, self.n_targets, h, w)
        return local

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

    def _temporal_delta(self, gaps: torch.Tensor, b: int, s: int,
                        d: int) -> torch.Tensor:
        """gaps (BSD, C) = CLS -> residu LSTM (B, S, D, C). Epoch 0 : 0."""
        c = gaps.size(-1)
        seq = gaps.reshape(b * s, d, c)
        with torch.autocast("cuda", enabled=False):
            h, _ = self.slice_lstm(seq.float())
            delta = self.lstm_proj(h)
        return delta.to(dtype=gaps.dtype).reshape(b, s, d, c)

    def _slice_pool(self, x: torch.Tensor, valid: torch.Tensor,
                    gaps: torch.Tensor, local: torch.Tensor):
        """CLS+LSTM (gaps) et tokens CAM (local) -> logits (B, T)."""
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
        logits = self.head(self.norm(pooled)).squeeze(-1) + self.out_bias
        return logits, attn.reshape(b, s, d, self.n_targets)

    def _fill_tokens(self, x: torch.Tensor, valid: torch.Tensor,
                     return_spatial: bool = False):
        b, s, d, img_h, img_w = x.shape
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        gaps = x.new_zeros(n_tok, feat_dim)
        local = x.new_zeros(n_tok, self.n_targets, feat_dim)
        spatial = None
        if return_spatial:
            hw_h, hw_w = img_h // self.patch_size, img_w // self.patch_size
            spatial = x.new_zeros(n_tok, self.n_targets, hw_h, hw_w)
        if idx.numel() > 0:
            maps, cls = self._encode_chunks(self._stem_rgb(x, idx))
            if return_spatial:
                loc, spat = self._spatial_pool(maps, return_attn=True)
            else:
                loc = self._spatial_pool(maps)
                spat = None
            if gaps.dtype != cls.dtype:
                gaps = gaps.to(dtype=cls.dtype)
            if local.dtype != loc.dtype:
                local = local.to(dtype=loc.dtype)
            gaps[idx] = cls
            local[idx] = loc
            if spatial is not None and spat is not None:
                if spatial.dtype != spat.dtype:
                    spatial = spatial.to(dtype=spat.dtype)
                spatial[idx] = spat
        return gaps, local, spatial

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
        gaps, local, _ = self._fill_tokens(x, valid)
        logits, _attn = self._slice_pool(x, valid, gaps, local)
        return logits

    @torch.no_grad()
    def interpret(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward eval + cartes. spatial_attn : (B, S, D, T, h, w)."""
        self.eval()
        b, s, d, img_h, img_w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        valid = x.amax(dim=(-2, -1)) > 1e-6
        gaps, local, spatial = self._fill_tokens(x, valid, return_spatial=True)
        logits, slice_attn = self._slice_pool(x, valid, gaps, local)
        delta = self._temporal_delta(gaps, b, s, d)
        pe = self.plane_embed[:s].view(1, s, 1, 1, -1)
        se = self.slice_embed[:d].view(1, 1, d, 1, -1)
        cam_tokens = (
            local.reshape(b, s, d, self.n_targets, -1)
            + delta.unsqueeze(3) + pe + se
        )
        hw_h, hw_w = img_h // self.patch_size, img_w // self.patch_size
        return {
            "logits": logits,
            "slice_attn": slice_attn,
            "spatial_attn": spatial.reshape(
                b, s, d, self.n_targets, hw_h, hw_w,
            ),
            "cam_tokens": cam_tokens,
            "valid": valid,
        }
