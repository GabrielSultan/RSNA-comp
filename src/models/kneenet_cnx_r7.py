# -*- coding: utf-8 -*-
"""KneeNet-CNX-r7 — tete R7 (Heliosli) sur l'entree hybrid025.

Meme tenseur que CNX-a : (B, 6, 24, 224, 224) grayscale [0, 1], slots
v20b. Le cache px384_v4 n'est pas reconstruit : c'est leur rendu
precalcule (384 px, vues 100 mm, meta DICOM). Ici chaque coupe valide
est une fenetre 2.5D (k-1, k, k+1), comme les 3 canaux de R7.

Graphe, repris de compact_train.Net (spatial + extent + syn_decomp) :
  - ConvNeXt-Small stride 32, drop_path 0.1, carte brute (sans head.norm).
  - Attention spatiale par pathologie (requete nulle a l'init = moyenne).
  - LayerNorm de tete apres le pooling, par classe.
  - Attention par pathologie sur les coupes du study.
  - Etendue OA : gamma * log1p(aire/100), gamma nul a l'init.
    Aire = produit sagittal x coronal (OA TF) ou axial x sagittal (PF).
    Epaisseur de dalle : 3.5 mm x nombre de coupes valides du slot
    (le zip crop130 n'a pas le spacing DICOM).
  - Synovite = part specifique + beta * stopgrad(logit epanchement).

Pas de tetes d'etat ni d'auxiliaire fastMRI : les cibles sont v3b (12
colonnes). Ces tetes ne changent pas les logits tant que leur porte
vaut 0, et il n'y a pas de labels d'etat pour les apprendre.
"""
from __future__ import annotations

import math
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

N_TARGETS = 12
TIMM_NAME = "convnext_small.fb_in22k_ft_in1k"
DEFAULT_WEIGHTS = "weights/convnext_small-fb_in22k_ft_in1k"
N_PROTOCOL_SLOTS = 6
HIDDEN = 384
OA_MED, OA_LAT, OA_PF = 4, 5, 6
SYN, EFF = 8, 7
# slots v20b : SAG_FS, COR_FS, AX_FS, SAG_T1, COR_T1, AX_T1
SLOT_PLANE = (0, 1, 2, 0, 1, 2)
SLOT_FLUID = (1, 1, 1, 0, 0, 0)
SLAB_MM = 3.5
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
    """Carte stride 32 avant head.norm. drop_path 0.1, comme R7."""

    def __init__(self, name: str = TIMM_NAME,
                 weights_path: str = DEFAULT_WEIGHTS,
                 pretrained: bool = True) -> None:
        super().__init__()
        kwargs = dict(
            num_classes=0, global_pool="", output_stride=32,
            drop_path_rate=0.1,
        )
        if pretrained:
            ckpt = _weights_file(weights_path)
            self.backbone = timm.create_model(
                name, pretrained=True,
                pretrained_cfg_overlay={"file": str(ckpt)}, **kwargs,
            )
            print(f"ConvNeXt charge : {name} | {ckpt} | stride 32 | drop_path 0.1",
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
        return self.backbone.forward_features(x)


class KneeNet(nn.Module):
    arch_id = "kneenet_cnx_r7"

    def __init__(self, backbone: str = TIMM_NAME,
                 n_series: int = N_PROTOCOL_SLOTS,
                 n_targets: int = N_TARGETS, hidden: int = HIDDEN,
                 dropout: float = 0.1, pretrained: bool = True,
                 encode_chunk: int = 4, max_depth: int = 64,
                 backbone_weights: str = DEFAULT_WEIGHTS) -> None:
        super().__init__()
        self.n_series = n_series
        self.n_targets = n_targets
        self.encode_chunk = int(encode_chunk)
        self.backbone_name = str(backbone)
        self.hidden = int(hidden)
        self.max_depth = int(max_depth)

        self.encoder = _ConvNextEncoder(
            name=self.backbone_name, weights_path=backbone_weights,
            pretrained=pretrained,
        )
        d = self.encoder.num_features
        norm = self.encoder.backbone.head.norm
        if isinstance(norm, nn.Identity):
            raise RuntimeError("ConvNeXt sans head.norm : le pooling R7 en a besoin")
        self.sp_head_norm = norm

        self.sp_key = nn.Linear(d, 128)
        self.sp_pos = nn.Linear(16, 128, bias=False)
        nn.init.zeros_(self.sp_pos.weight)
        self.sp_q = nn.Parameter(torch.zeros(n_targets, 128))

        self.proj = nn.Sequential(nn.Linear(d, self.hidden), nn.LayerNorm(self.hidden))
        self.slot = nn.Embedding(10, self.hidden)
        self.view = nn.Embedding(6, self.hidden)
        self.geom = nn.Linear(4, self.hidden, bias=False)
        nn.init.zeros_(self.slot.weight)
        nn.init.zeros_(self.view.weight)
        nn.init.zeros_(self.geom.weight)
        self.drop = nn.Dropout(dropout)
        self.win_proj = nn.Linear(self.hidden, self.hidden)
        self.q_win = nn.Parameter(torch.randn(n_targets, self.hidden) * 0.02)
        self.out_norm = nn.LayerNorm(self.hidden)
        self.cls_w = nn.Parameter(torch.randn(n_targets, self.hidden) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_targets))

        self.ext_u = nn.Parameter(torch.zeros(n_targets, self.hidden))
        self.ext_b = nn.Parameter(torch.full((n_targets,), -4.0))
        self.ext_gamma = nn.Parameter(torch.zeros(n_targets))
        self.ext_view = nn.Parameter(torch.zeros(n_targets, 6))
        em = torch.zeros(n_targets)
        em[OA_MED] = em[OA_LAT] = em[OA_PF] = 1.0
        self.register_buffer("ext_mask", em)
        self.syn_beta = nn.Parameter(torch.zeros(()))

        self.register_buffer(
            "img_mean", torch.tensor(self.encoder.mean).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "img_std", torch.tensor(self.encoder.std).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer("slot_plane", torch.tensor(SLOT_PLANE, dtype=torch.long))
        self.register_buffer("slot_fluid", torch.tensor(SLOT_FLUID, dtype=torch.float32))
        print(
            f"KneeNet-CNX-r7 | {self.backbone_name} | hidden {self.hidden} "
            f"| spatial + fenetres + etendue OA + syn_beta "
            f"| slots {self.n_series} | dim {d}",
            flush=True,
        )

    def set_encoder_frozen(self, flag: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not flag

    def encoder_parameters(self):
        yield from self.encoder.parameters()

    def head_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith("encoder."):
                continue
            yield p

    @torch.no_grad()
    def extra_log(self) -> dict[str, float]:
        return {
            "syn_beta": float(self.syn_beta.detach().cpu()),
            "ext_gamma_abs": float(self.ext_gamma.detach().abs().mean()),
            "sp_q_abs": float(self.sp_q.detach().abs().mean()),
        }

    def _stem_rgb(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Trois coupes adjacentes -> 3 canaux. x (B,S,D,H,W)."""
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
        return (rgb - self.img_mean) / self.img_std

    def _encode_indexed(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
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

    def _pos_enc(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing="ij",
        )
        fr = torch.tensor([1.0, 2.0, 4.0, 8.0], device=device) * (math.pi / 2)
        yy = yy.reshape(-1, 1) * fr
        xx = xx.reshape(-1, 1) * fr
        return torch.cat([yy.sin(), yy.cos(), xx.sin(), xx.cos()], 1).to(dtype)

    def spatial_pool(self, maps: torch.Tensor) -> torch.Tensor:
        """maps (N, d, h, w) -> (N, C, d). Requete nulle = moyenne des positions."""
        _n, _d, h, w = maps.shape
        f = maps.flatten(2).transpose(1, 2)
        pe = self._pos_enc(h, w, f.device, f.dtype)
        kin = F.layer_norm(f.float(), (f.shape[-1],)).to(f.dtype)
        k = torch.tanh(self.sp_key(kin) + self.sp_pos(pe)[None])
        att = torch.einsum("npk,ck->npc", k, self.sp_q.to(k.dtype)).float().softmax(1).to(f.dtype)
        g = torch.einsum("npc,npd->ncd", att, f)
        nrm = self.sp_head_norm
        g = F.layer_norm(g.float(), (g.shape[-1],), nrm.weight, nrm.bias, nrm.eps).to(f.dtype)
        return g

    def _window_meta(self, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """valid (B,S,D) -> mask (B,W), meta (B,W,9)."""
        b, s, d = valid.shape
        device = valid.device
        rel = torch.linspace(0, 1, d, device=device)
        n_valid = valid.flatten(2).sum(-1).clamp(min=1).to(torch.float32)
        slot = torch.arange(s, device=device)
        plane = self.slot_plane[:s].to(device)
        fluid = self.slot_fluid[:s].to(device)
        meta = valid.new_zeros(b, s, d, 9, dtype=torch.float32)
        meta[..., 0] = slot.view(1, s, 1).float()
        meta[..., 2] = rel.view(1, 1, d)
        meta[..., 3] = math.log1p(SLAB_MM)
        meta[..., 4] = 1.0
        meta[..., 5] = fluid.view(1, s, 1)
        meta[..., 6] = slot.view(1, s, 1).float()
        meta[..., 7] = plane.view(1, s, 1).float()
        meta[..., 8] = n_valid.view(b, s, 1)
        return valid.reshape(b, s * d), meta.reshape(b, s * d, 9)

    def extent_area(self, e: torch.Tensor, meta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """e (B,W,C) -> aire proxy (B,C), mm^2. Meme combinaison de plans que R7."""
        v = 6
        sid = meta[..., 6].long()
        s_max = max(16, int(sid.max()) + 1)
        vid = meta[..., 1].long().clamp(0, v - 1)
        q = torch.sigmoid(e.float()) * mask[..., None]
        ohv = F.one_hot(sid * v + vid, s_max * v).float() * mask[..., None]
        cntv = ohv.sum(1)
        mean_v = (
            torch.einsum("bwk,bwc->bkc", ohv, q) / cntv.clamp_min(1)[..., None]
        ).view(e.shape[0], s_max, v, -1)
        present = (cntv > 0).view(e.shape[0], s_max, v, 1)
        wv = self.ext_view.float().t()[None, None].masked_fill(~present, -1e4).softmax(2) * present
        mean = (wv * mean_v).sum(2)
        oh = F.one_hot(sid, s_max).float() * mask[..., None]
        cnt = oh.sum(1)
        span = torch.expm1(meta[..., 3].float()).clamp_min(0) * meta[..., 8].float()
        span_s = torch.einsum("bws,bw->bs", oh, span) / cnt.clamp_min(1)
        plane_s = (torch.einsum("bws,bw->bs", oh, meta[..., 7].float()) / cnt.clamp_min(1)).round()
        energy = mean * span_s[..., None]
        per_plane = []
        for pl in (0, 1, 2):
            ok = ((plane_s == pl) & (cnt > 0))[..., None]
            per_plane.append(energy.masked_fill(~ok, 0.0).amax(1))
        sag, cor, ax = per_plane
        area = torch.zeros_like(sag)
        area[:, OA_MED] = sag[:, OA_MED] * cor[:, OA_MED]
        area[:, OA_LAT] = sag[:, OA_LAT] * cor[:, OA_LAT]
        area[:, OA_PF] = ax[:, OA_PF] * sag[:, OA_PF]
        return area

    def _decompose_syn(self, z: torch.Tensor) -> torch.Tensor:
        ze = z[:, EFF].detach()
        onehot = z.new_zeros(z.shape[1])
        onehot[SYN] = 1
        return z + (self.syn_beta * ze)[:, None] * onehot[None, :]

    def head(self, tok: torch.Tensor, meta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tok (N,C,d) des fenetres valides, dans l'ordre du masque. -> (B,C)."""
        b, w = mask.shape
        f = self.proj(tok.float())
        emb = (
            self.slot(meta[..., 0].long().clamp(0, 9))
            + self.view(meta[..., 1].long().clamp(0, 5))
            + self.geom(meta[..., 2:6])
        )
        t = f.new_zeros((b, w) + tuple(f.shape[1:]))
        t[mask] = f
        t = self.drop(t + emb[:, :, None, :])
        s2 = torch.einsum("bwch,ch->bwc", torch.tanh(self.win_proj(t)), self.q_win).float()
        s2 = s2.masked_fill(~mask[:, :, None], float("-inf"))
        a2 = s2.softmax(1).to(t.dtype)
        study = self.out_norm(torch.einsum("bwc,bwch->bch", a2, t))
        e_win = (t.float() * self.ext_u[None, None]).sum(-1) + self.ext_b
        z = (study * self.cls_w).sum(-1).float() + self.bias
        area = self.extent_area(e_win, meta, mask)
        z = z + self.ext_gamma * self.ext_mask * torch.log1p(area / 100.0)
        return self._decompose_syn(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, S, D, H, W) dans [0, 1] -> logits (B, 12)."""
        _b, s, d, _h, _w = x.shape
        if s != self.n_series:
            raise ValueError(f"attendu {self.n_series} series, recu {s}")
        if d > self.max_depth:
            raise ValueError(f"D={d} > max_depth={self.max_depth}")
        valid = x.amax(dim=(-2, -1)) > 1e-6
        mask, meta = self._window_meta(valid)
        idx = mask.reshape(-1).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return x.new_zeros(x.shape[0], self.n_targets)
        maps = self._encode_indexed(x, idx)
        tok = self.spatial_pool(maps)
        return self.head(tok, meta, mask)
