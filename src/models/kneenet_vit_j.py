# -*- coding: utf-8 -*-
"""KneeNet-ViT-j — ViT-e + fusion multi-niveaux blocs {6, 9, 12}.

Graphe e (CAM+LSTM, stem RGB linspace). Un seul forward :
`forward_intermediates` blocs 6 et 9 + last (bloc 12 + LN pretrained).
LN dédiée sur 6 et 9. Somme w init [0, 0, 1] → epoch 0 = ViT-e
(last seul, pas le double CAM de ViT-f).

Recette parent = k (LLRD 0.8 + no_decay), pas accum.
Option `drop_path_rate` (YAML) : stochastic depth timm, OFF pendant
le freeze. Défaut 0 = graphe j inchangé.
Succès plan : val MedMen / LatMen / Frac +0.02. Bruit si w reste
[0, 0, 1] et Δ macro < 0.01.

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
# 1-index des blocs ViT-B (12 blocs). Last = 12 via forward_features/norm.
MIX_BLOCKS = (6, 9, 12)
MID_INDICES = (5, 8)  # 0-index : après blocs 6 et 9
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
    """ViT-B/16 DINOv3 -> last + mid maps (N, 768, H/16, W/16)."""

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS,
                 pretrained: bool = True,
                 drop_path_rate: float = 0.0) -> None:
        super().__init__()
        kwargs = dict(
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
        )
        dp = float(drop_path_rate or 0.0)
        if dp > 0:
            kwargs["drop_path_rate"] = dp
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
        self.mid_indices = list(MID_INDICES)
        self.drop_path_rate = dp
        self._drop_path: list[tuple[nn.Module, float]] = []
        for m in self.backbone.modules():
            if m.__class__.__name__ == "DropPath":
                self._drop_path.append((m, float(m.drop_prob)))
        if dp > 0:
            print(
                f"drop_path {dp:g} | {len(self._drop_path)} modules "
                f"(off pendant freeze)",
                flush=True,
            )

    def _tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        patches = tokens[:, self.n_prefix:]
        n, n_hw, c = patches.shape
        side = int(n_hw ** 0.5)
        if side * side != n_hw:
            raise ValueError(
                f"grille patch non carree : {n_hw} tokens"
            )
        return patches.transpose(1, 2).reshape(n, c, side, side)

    def _as_maps(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4:
            return t
        if t.ndim == 3:
            return self._tokens_to_map(t)
        raise ValueError(f"feat ndim {t.ndim}, attendu 3 (NLC) ou 4 (NCHW)")

    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (N, 3, H, W) -> last, mid6, mid9 (N, C, h, w).

        last = bloc 12 + LN pretrained (= ViT-e). mids = sorties brutes
        des blocs 6 et 9 (prefix retiré, NCHW).
        """
        last, mids = self.backbone.forward_intermediates(
            x,
            indices=self.mid_indices,
            norm=False,
            output_fmt="NCHW",
            intermediates_only=False,
        )
        if len(mids) != 2:
            raise RuntimeError(
                f"attendu 2 intermediates, recu {len(mids)}"
            )
        return (
            self._as_maps(last),
            self._as_maps(mids[0]),
            self._as_maps(mids[1]),
        )


class KneeNet(nn.Module):
    arch_id = "kneenet_vit_j"

    def __init__(self, backbone: str = TIMM_NAME,
                 n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 4, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 lstm_hidden: int = LSTM_HIDDEN,
                 drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.patch_size = PATCH_SIZE
        self.lstm_hidden = int(lstm_hidden)
        self.drop_path_rate = float(drop_path_rate or 0.0)

        self.encoder = _Dinov3VitEncoder(
            weights_path=backbone_weights, pretrained=pretrained,
            drop_path_rate=self.drop_path_rate,
        )
        self._freeze_layer_scale()
        feat_dim = self.encoder.num_features

        self.stem_mix = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self._init_stem_mix()

        self.mix_ln = nn.ModuleList([
            nn.LayerNorm(feat_dim),
            nn.LayerNorm(feat_dim),
        ])
        self.level_mix = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]))

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
        dp_txt = (f" | drop_path {self.drop_path_rate:g}"
                  if self.drop_path_rate > 0 else "")
        print(
            f"KneeNet-ViT-j | fusion blocs {list(MIX_BLOCKS)} "
            f"init {self.level_mix.detach().tolist()}{dp_txt} | CAM patch "
            f"{PATCH_SIZE} | BiLSTM | dim {feat_dim} | slots {self.n_series}",
            flush=True,
        )

    def extra_log(self) -> dict[str, float]:
        w = self.level_mix.detach().float()
        return {
            "mix_block06": float(w[0]),
            "mix_block09": float(w[1]),
            "mix_block12": float(w[2]),
        }

    def _init_stem_mix(self) -> None:
        """out = in_G -> epoch 0 identique a v1 / e."""
        with torch.no_grad():
            self.stem_mix.weight.zero_()
            self.stem_mix.weight[:, 1, 0, 0] = 1.0

    @staticmethod
    def _ln_map(ln: nn.LayerNorm, maps: torch.Tensor) -> torch.Tensor:
        return ln(maps.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _mix_levels(self, last: torch.Tensor, mid6: torch.Tensor,
                    mid9: torch.Tensor) -> torch.Tensor:
        w = self.level_mix
        return (
            w[0] * self._ln_map(self.mix_ln[0], mid6)
            + w[1] * self._ln_map(self.mix_ln[1], mid9)
            + w[2] * last
        )

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
        # Gel : mix reste [0,0,1] → freeze epoch = ViT-e.
        self.level_mix.requires_grad = not flag
        for p in self.mix_ln.parameters():
            p.requires_grad = not flag
        # DropPath stochastique seulement une fois l'encodeur degele,
        # sinon l'epoch 1 n'est plus identity-e.
        for m, p0 in self.encoder._drop_path:
            m.drop_prob = 0.0 if flag else p0

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

    def _encode_chunks(self, rgb: torch.Tensor) -> torch.Tensor:
        n = rgb.size(0)
        c = self.encoder.num_features
        if n == 0:
            h = rgb.size(-2) // self.patch_size
            w = rgb.size(-1) // self.patch_size
            return rgb.new_zeros(0, c, h, w)
        last_parts, m6_parts, m9_parts = [], [], []
        chunk = max(1, self.encode_chunk)
        for i in range(0, n, chunk):
            part = rgb[i:i + chunk]
            if self.training:
                last, m6, m9 = checkpoint(
                    self.encoder, part, use_reentrant=False)
            else:
                last, m6, m9 = self.encoder(part)
            last_parts.append(last)
            m6_parts.append(m6)
            m9_parts.append(m9)
        return self._mix_levels(
            torch.cat(last_parts, dim=0),
            torch.cat(m6_parts, dim=0),
            torch.cat(m9_parts, dim=0),
        )

    def _spatial_pool(self, maps: torch.Tensor, return_attn: bool = False):
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
            maps = self._encode_chunks(self._stem_rgb(x, idx))
            gap, loc = self._spatial_pool(maps)
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
            maps = self._encode_chunks(self._stem_rgb(x, idx))
            gap, loc, spat = self._spatial_pool(maps, return_attn=True)
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
        cam_tokens = (
            local.reshape(b, s, d, self.n_targets, -1)
            + delta.unsqueeze(3) + pe + se
        )
        return {
            "logits": logits,
            "slice_attn": slice_attn,
            "spatial_attn": spatial.reshape(
                b, s, d, self.n_targets, hw_h, hw_w,
            ),
            "cam_tokens": cam_tokens,
            "valid": valid,
        }
