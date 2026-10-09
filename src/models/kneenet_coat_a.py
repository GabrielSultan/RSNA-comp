# -*- coding: utf-8 -*-
"""KneeNet-CoAt-a — baseline CoAtNet 2.5D @224, graphe ViT-e / v20b.

Seul l'encodeur change vs ViT-e : CoAtNet-0 timm (stem conv + 2 stages
MBConv BN + 2 stages Transformer LN), ImageNet-1k, 27M params (~ResNet-50).
Le reste est le graphe qui score deja :

  - stem 2.5D : R=k-1, G=k, B=k+1 (voisins linspace crop130), stem_mix
    1x1 init G -> epoch 0 = coupe grise repliquee (in-distribution)
  - carte finale 7x7 (stride 32 @224) : CAM 12 tetes + GAP
  - BiLSTM profondeur residuel init 0, attention coupe par cible, 6 slots
  - BN encodeur gelee (running stats ImageNet, affine entrainable),
    comme v20b : chunks correles + checkpoint = stats instables

Normalisation = pretrained_cfg timm (0.5 / 0.5 pour les CoAtNet rw),
pas ImageNet. `backbone` YAML = autre CoAtNet timm (poids assortis).
Entree : (B, 6, 24, H, W) grayscale [0, 1]. Slots = v20b.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

N_TARGETS = 12
TIMM_NAME = "coatnet_0_rw_224.sw_in1k"
DEFAULT_WEIGHTS = "weights/coatnet_0_rw_224-sw_in1k"
LSTM_HIDDEN = 256
N_PROTOCOL_SLOTS = 6


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
        raise FileNotFoundError(f"poids CoAtNet introuvables : {p.resolve()}")
    return p


class _CoatEncoder(nn.Module):
    """CoAtNet timm -> cartes (N, C, H/32, W/32) apres LayerNorm finale."""

    def __init__(self, name: str = TIMM_NAME,
                 weights_path: str = DEFAULT_WEIGHTS,
                 pretrained: bool = True,
                 img_size: int = 224) -> None:
        super().__init__()
        self.img_size = int(img_size)
        kwargs = dict(num_classes=0, global_pool="", img_size=self.img_size)
        if pretrained:
            ckpt = _weights_file(weights_path)
            self.backbone = timm.create_model(
                name, pretrained=True,
                pretrained_cfg_overlay={"file": str(ckpt)}, **kwargs,
            )
            print(f"CoAtNet charge : {name} | {ckpt} | entree {self.img_size}",
                  flush=True)
        else:
            self.backbone = timm.create_model(name, pretrained=False, **kwargs)
        self.num_features = int(self.backbone.num_features)
        cfg = self.backbone.pretrained_cfg
        self.mean = tuple(float(v) for v in cfg.get("mean", (0.5, 0.5, 0.5)))
        self.std = tuple(float(v) for v in cfg.get("std", (0.5, 0.5, 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)


class KneeNet(nn.Module):
    arch_id = "kneenet_coat_a"

    def __init__(self, backbone: str = TIMM_NAME,
                 n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 8, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 lstm_hidden: int = LSTM_HIDDEN,
                 img_size: int = 224) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.lstm_hidden = int(lstm_hidden)
        self.img_size = int(img_size)

        self.encoder = _CoatEncoder(
            name=self.backbone_name, weights_path=backbone_weights,
            pretrained=pretrained, img_size=self.img_size,
        )
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

        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.register_buffer(
            "img_mean", torch.tensor(self.encoder.mean).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "img_std", torch.tensor(self.encoder.std).view(1, 3, 1, 1),
            persistent=True,
        )
        print(
            f"KneeNet-CoAt-a | {self.backbone_name} | CAM + BiLSTM "
            f"| RGB k-1/k/k+1 | dim {feat_dim} | slots {self.n_series} "
            f"| norm {self.encoder.mean}/{self.encoder.std}",
            flush=True,
        )

    def _init_stem_mix(self) -> None:
        with torch.no_grad():
            self.stem_mix.weight.zero_()
            self.stem_mix.weight[:, 1, 0, 0] = 1.0

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def set_encoder_frozen(self, flag: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not flag
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
        """Part de |stem_mix| sur chaque coupe d'entree (G seul = pas de 2.5D)."""
        w = self.stem_mix.weight.detach().abs().sum(dim=(0, 2, 3))
        w = w / w.sum().clamp(min=1e-8)
        return {"stem_mix_km1": float(w[0]), "stem_mix_k": float(w[1]),
                "stem_mix_kp1": float(w[2])}

    def _encode_chunks(self, rgb: torch.Tensor) -> torch.Tensor:
        n = rgb.size(0)
        if n == 0:
            h, w = rgb.size(-2) // 32, rgb.size(-1) // 32
            return rgb.new_zeros(0, self.encoder.num_features, h, w)
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

    def _temporal_delta(self, gaps: torch.Tensor, b: int, s: int,
                        d: int) -> torch.Tensor:
        c = gaps.size(-1)
        seq = gaps.reshape(b * s, d, c)
        with torch.autocast("cuda", enabled=False):
            h, _ = self.slice_lstm(seq.float())
            delta = self.lstm_proj(h)
        return delta.to(dtype=gaps.dtype).reshape(b, s, d, c)

    def _slice_pool(self, x: torch.Tensor, valid: torch.Tensor,
                    gaps: torch.Tensor, local: torch.Tensor):
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
        logits = self.head(self.norm(pooled)).squeeze(-1)
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
            maps = self._encode_chunks(self._stem_rgb(x, idx))
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
            spatial = x.new_zeros(n_tok, self.n_targets, img_h // 32, img_w // 32)
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
