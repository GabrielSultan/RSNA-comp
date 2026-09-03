# -*- coding: utf-8 -*-
"""KneeNet-ViT-f — e + CAM mid-depth (block 8), GAP/LSTM last layer.

E ep7 (CAM last + RGB) : val 0.807 / gold 0.814 vs v1 0.797 / 0.812.
Le trou restant est le ménisque (0.804 vs v20b ~0.91) : last-layer
DINOv3 trop mélangée.

Un levier : CAM sur les cartes patch du block 8/12 ; GAP + BiLSTM sur
la last layer. Stem RGB k-1/k/k+1 inchangé.

1er run (lr 3e-5, CAM dans le graphe) : collapse ep3 0.71 -> 0.60 dès
que le re-warmup finit. Double backprop mid+last. Fix : stop-grad sur
le CAM, residu LSTM seulement sur le routage slice, lr_backbone 1e-5.

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
N_PREFIX = 5
CAM_BLOCK = 7  # 0-index : block 8 / 12
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
    """ViT-B/16 DINOv3 -> (last maps, cam maps), chacune (N, 768, h, w)."""

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS,
                 pretrained: bool = True, cam_block: int = CAM_BLOCK) -> None:
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
        self.n_prefix = int(getattr(
            self.backbone, "num_prefix_tokens", N_PREFIX))
        n_blocks = len(self.backbone.blocks)
        if not 0 <= int(cam_block) < n_blocks:
            raise ValueError(
                f"cam_block={cam_block} hors 0..{n_blocks - 1}"
            )
        self.cam_block = int(cam_block)
        self.n_blocks = n_blocks

    def _tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        patches = tokens[:, self.n_prefix:]
        n, n_hw, c = patches.shape
        side = int(n_hw ** 0.5)
        if side * side != n_hw:
            raise ValueError(
                f"grille patch non carree : {n_hw} tokens"
            )
        return patches.transpose(1, 2).reshape(n, c, side, side)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (N, 3, H, W) -> last_maps, cam_maps (N, C, h, w)."""
        last, mids = self.backbone.forward_intermediates(
            x,
            indices=[self.cam_block],
            norm=False,
            output_fmt="NCHW",
            intermediates_only=False,
        )
        return self._tokens_to_map(last), mids[0]


class KneeNet(nn.Module):
    arch_id = "kneenet_vit_f"

    def __init__(self, backbone: str = TIMM_NAME,
                 n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 4, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 lstm_hidden: int = LSTM_HIDDEN,
                 cam_block: int = CAM_BLOCK) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.patch_size = PATCH_SIZE
        self.lstm_hidden = int(lstm_hidden)

        self.encoder = _Dinov3VitEncoder(
            weights_path=backbone_weights, pretrained=pretrained,
            cam_block=cam_block,
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

        self.norm = nn.LayerNorm(feat_dim)
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
        print(
            f"KneeNet-ViT-f | CAM block {self.encoder.cam_block + 1}/"
            f"{self.encoder.n_blocks} (stop-grad) | GAP+BiLSTM last "
            f"| RGB k-1/k/k+1 | dim {feat_dim} | slots {self.n_series}",
            flush=True,
        )

    def _init_stem_mix(self) -> None:
        """out = in_G -> epoch 0 identique au stem e."""
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

    def _encode_chunks(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = rgb.size(0)
        c = self.encoder.num_features
        if n == 0:
            h = rgb.size(-2) // self.patch_size
            w = rgb.size(-1) // self.patch_size
            z = rgb.new_zeros(0, c, h, w)
            return z, z
        lasts, cams = [], []
        chunk = max(1, self.encode_chunk)
        for i in range(0, n, chunk):
            part = rgb[i:i + chunk]
            if self.training:
                last, cam = checkpoint(
                    self.encoder, part, use_reentrant=False,
                )
            else:
                last, cam = self.encoder(part)
            lasts.append(last)
            cams.append(cam)
        # Stop-grad sur le CAM : un seul chemin encodeur (GAP last),
        # comme e. Sans ca, 3e-5 detruit DINOv3 a l'epoch 3 (val 0.71->0.60).
        return torch.cat(lasts, dim=0), torch.cat(cams, dim=0).detach()

    def _spatial_pool(self, last_maps: torch.Tensor, cam_maps: torch.Tensor,
                      return_attn: bool = False):
        """GAP last (N,C) ; CAM mid -> local (N,T,C) [, attn]."""
        n, c, h, w = cam_maps.shape
        gap = last_maps.mean(dim=(-2, -1))
        logits = self.spatial_score(cam_maps)
        attn = torch.softmax(
            logits.float().reshape(n, self.n_targets, h * w), dim=-1,
        ).to(cam_maps.dtype)
        flat = cam_maps.reshape(n, c, h * w)
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
        # CAM mid : pas de residu LSTM (espace last). Slice attn reste last.
        tokens = local.reshape(b, s, d, self.n_targets, -1)
        tokens = tokens + pe.unsqueeze(3) + se.unsqueeze(3)
        tokens = tokens.reshape(b, s * d, self.n_targets, -1)
        pooled = torch.einsum("bnt,bntc->btc", attn, tokens)
        logits = self.head(self.norm(pooled)).squeeze(-1)
        return logits, attn.reshape(b, s, d, self.n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d, _h, _w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        if d > self.slice_embed.size(0):
            raise ValueError(
                f"D={d} > max_depth={self.slice_embed.size(0)} : "
                "augmenter max_depth"
            )
        valid = x.amax(dim=(-2, -1)) > 1e-6
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        gaps = x.new_zeros(n_tok, feat_dim)
        local = x.new_zeros(n_tok, self.n_targets, feat_dim)
        if idx.numel() > 0:
            last_maps, cam_maps = self._encode_chunks(self._stem_rgb(x, idx))
            gap, loc = self._spatial_pool(last_maps, cam_maps)
            if gaps.dtype != gap.dtype:
                gaps = gaps.to(dtype=gap.dtype)
            if local.dtype != loc.dtype:
                local = local.to(dtype=loc.dtype)
            gaps[idx] = gap
            local[idx] = loc
        logits, _attn = self._slice_pool(x, valid, gaps, local)
        return logits

    @torch.no_grad()
    def interpret(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        b, s, d, img_h, img_w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        valid = x.amax(dim=(-2, -1)) > 1e-6
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        hw_h, hw_w = img_h // self.patch_size, img_w // self.patch_size
        gaps = x.new_zeros(n_tok, feat_dim)
        local = x.new_zeros(n_tok, self.n_targets, feat_dim)
        spatial = x.new_zeros(n_tok, self.n_targets, hw_h, hw_w)
        if idx.numel() > 0:
            last_maps, cam_maps = self._encode_chunks(self._stem_rgb(x, idx))
            gap, loc, spat = self._spatial_pool(
                last_maps, cam_maps, return_attn=True,
            )
            if gaps.dtype != gap.dtype:
                gaps = gaps.to(dtype=gap.dtype)
            if local.dtype != loc.dtype:
                local = local.to(dtype=loc.dtype)
            if spatial.dtype != spat.dtype:
                spatial = spatial.to(dtype=spat.dtype)
            gaps[idx] = gap
            local[idx] = loc
            spatial[idx] = spat
        logits, slice_attn = self._slice_pool(x, valid, gaps, local)
        delta = self._temporal_delta(gaps, b, s, d)
        pe = self.plane_embed[:s].view(1, s, 1, 1, -1)
        se = self.slice_embed[:d].view(1, 1, d, 1, -1)
        cam_tokens = local.reshape(b, s, d, self.n_targets, -1) + pe + se
        return {
            "logits": logits,
            "slice_attn": slice_attn,
            "spatial_attn": spatial.reshape(
                b, s, d, self.n_targets, hw_h, hw_w,
            ),
            "cam_tokens": cam_tokens,
            "valid": valid,
        }
