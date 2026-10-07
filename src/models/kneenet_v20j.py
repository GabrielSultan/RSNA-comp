# -*- coding: utf-8 -*-
"""KneeNet v20j — v20h + D (voisins DICOM vrais, pas linspace).

Base v20h inchangee : ResNet-50, CAM 14x14 dilated, BiLSTM residuel
init 0, BN gelee, 6 slots protocole, tete par cible (label_embed init 0
+ Linear 512->12 diagonale, init liee = v20b a l'epoch 0).

Levier D (v12 / v20i) : entree (B, 6, D, 3, H, W) deja R=k-1, G=k, B=k+1
dans le volume ORIGINAL. A+B sur ces canaux (pas sur les 24 linspace) :

    R = 0.5 + 0.5*(G - R_dicom)
    G = G
    B = 0.5 + 0.5*(B_dicom - G)

stem_mix 1x1 init « tout depuis G » -> epoch 0 = centre seul = v20h.

Dataset : crop130_rgb25d proto, rgb_neighbors=True.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50

N_TARGETS = 12
DEFAULT_WEIGHTS = "weights/ResNet50.pt"
MAP_HW = 14
LSTM_HIDDEN = 256
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
    """ResNet-50 jusqu'a layer4 dilated. Cles `backbone.N.*` = v14/v16/v17."""

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


class KneeNet(nn.Module):
    arch_id = "kneenet_v20j"
    rgb_neighbors = True

    def __init__(self, backbone: str = "resnet50", n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, head_dim: int = 512,
                 dropout: float = 0.2, pretrained: bool = True,
                 encode_chunk: int = 16, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS,
                 lstm_hidden: int = LSTM_HIDDEN) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.map_hw = MAP_HW
        self.lstm_hidden = int(lstm_hidden)

        self.encoder = _ResNet50Encoder()
        if pretrained:
            self._load_radimagenet(backbone_weights)
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

        self.stem_mix = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self._init_stem_mix()
        self.register_buffer(
            "img_mean", torch.zeros(1, 3, 1, 1), persistent=True,
        )
        self.register_buffer(
            "img_std", torch.ones(1, 3, 1, 1), persistent=True,
        )

    def extra_log(self) -> dict[str, float]:
        """Divergence de la tete par cible, et ouverture du stem RGB."""
        w = self.head_out.weight.detach()
        b = self.head_out.bias.detach()
        sw = self.stem_mix.weight.detach()
        return {
            "head_out_w_div": float((w - w[0]).abs().mean()),
            "head_out_b_div": float((b - b[0]).abs().mean()),
            "label_embed_abs": float(self.label_embed.detach().abs().mean()),
            "stem_mix_from_R": float(sw[:, 0].abs().sum()),
            "stem_mix_from_G": float(sw[:, 1].abs().sum()),
            "stem_mix_from_B": float(sw[:, 2].abs().sum()),
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
        print(
            f"ResNet-50 charge : {p} | CAM {MAP_HW}x{MAP_HW} "
            f"| BiLSTM hidden {self.lstm_hidden} | slots {self.n_series} "
            f"| RGB DICOM k-1/k/k+1 + tete par cible",
            flush=True,
        )

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

    def _encode_chunks(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb : (N, 3, H, W) dans [0, 1] -> (N, C, 14, 14)."""
        n = rgb.size(0)
        c = self.encoder.num_features
        if n == 0:
            return rgb.new_zeros(0, c, MAP_HW, MAP_HW)
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

    @staticmethod
    def _diff_channels(rgb: torch.Tensor) -> torch.Tensor:
        """rgb (N, 3, H, W) R=k-1 G=k B=k+1 DICOM -> G=k, R/B = diffs A+B."""
        prev, center, nxt = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        return torch.stack((
            (0.5 + 0.5 * (center - prev)).clamp(0, 1),
            center,
            (0.5 + 0.5 * (nxt - center)).clamp(0, 1),
        ), dim=1)

    def _stem_rgb(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """A+B sur les canaux DICOM stockes. x (B,S,D,3,H,W), idx plat."""
        n_tok = x.shape[0] * x.shape[1] * x.shape[2]
        rgb = x.reshape(n_tok, 3, x.shape[-2], x.shape[-1]).index_select(0, idx)
        rgb = self._diff_channels(rgb)
        rgb = self.stem_mix(rgb)
        return (rgb - self.img_mean) / self.img_std

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
        """pooled (B, T, C) -> logits (B, T). Epoch 0 = tete v20b partagee."""
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

    def _unpack(self, x: torch.Tensor) -> tuple[int, int, int]:
        if x.ndim != 6:
            raise ValueError(
                f"v20j attend (B, S, D, 3, H, W) rgb DICOM, recu {tuple(x.shape)}"
            )
        b, s, d, c, _h, _w = x.shape
        assert s == self.n_series, f"attendu {self.n_series} series, recu {s}"
        if c != 3:
            raise ValueError(f"attendu 3 canaux RGB, recu C={c}")
        if d > self.slice_embed.size(0):
            raise ValueError(
                f"D={d} > max_depth={self.slice_embed.size(0)} : "
                "augmenter max_depth"
            )
        return b, s, d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, S, D, 3, H, W) dans [0, 1] -> logits (B, 12)."""
        b, s, d = self._unpack(x)
        valid = x[:, :, :, 1].amax(dim=(-2, -1)) > 1e-6
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        gaps = x.new_zeros(n_tok, feat_dim)
        local = x.new_zeros(n_tok, self.n_targets, feat_dim)
        if idx.numel() > 0:
            maps = self._encode_chunks(self._stem_rgb(x, idx))
            if gaps.dtype != maps.dtype:
                gaps = gaps.to(dtype=maps.dtype)
                local = local.to(dtype=maps.dtype)
            gap, loc = self._spatial_pool(maps)
            gaps[idx] = gap
            local[idx] = loc
        logits, _attn = self._slice_pool(x, valid, gaps, local)
        return logits

    @torch.no_grad()
    def interpret(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward eval + cartes. spatial_attn : (B, S, D, T, 14, 14)."""
        self.eval()
        b, s, d = self._unpack(x)
        valid = x[:, :, :, 1].amax(dim=(-2, -1)) > 1e-6
        n_tok = b * s * d
        idx = valid.reshape(-1).nonzero(as_tuple=False).view(-1)
        feat_dim = self.encoder.num_features
        gaps = x.new_zeros(n_tok, feat_dim)
        local = x.new_zeros(n_tok, self.n_targets, feat_dim)
        spatial = x.new_zeros(n_tok, self.n_targets, MAP_HW, MAP_HW)
        if idx.numel() > 0:
            maps = self._encode_chunks(self._stem_rgb(x, idx))
            gap, loc, spat = self._spatial_pool(maps, return_attn=True)
            if gaps.dtype != gap.dtype:
                gaps = gaps.to(dtype=gap.dtype)
            if local.dtype != loc.dtype:
                local = local.to(dtype=loc.dtype)
            hw_h, hw_w = spat.shape[-2], spat.shape[-1]
            if spatial.shape[-2:] != (hw_h, hw_w):
                spatial = x.new_zeros(n_tok, self.n_targets, hw_h, hw_w)
            if spatial.dtype != spat.dtype:
                spatial = spatial.to(dtype=spat.dtype)
            gaps[idx] = gap
            local[idx] = loc
            spatial[idx] = spat
        hw_h, hw_w = spatial.shape[-2], spatial.shape[-1]
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
