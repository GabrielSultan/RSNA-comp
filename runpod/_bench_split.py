"""Decoupe un step identique au train pour trouver le trou GPU."""
import time
import torch
from src.train.train import load_config, masked_bce
from src.models.registry import build_model
from src.train.param_groups import build_optimizer, set_group_lrs
from src.data.dataset_npz import make_loaders
from src.data.transforms import augment_batch

cfg = load_config("configs/kneenet_vit_k_llrd08_320_v3b_oofmask_v20h_full.yaml")
ld, _, targets = make_loaders(
    labels_csv=cfg["labels_csv"],
    series_csv=cfg["series_csv"],
    zip_path=cfg["zip_path"],
    batch_size=int(cfg["batch_size"]),
    num_workers=int(cfg["num_workers"]),
    n_series=int(cfg["n_series"]),
    train_frac=float(cfg.get("train_frac", 1)),
    img_size=int(cfg["img_size"]),
    include_gold=bool(cfg.get("include_gold")),
    val_frac=float(cfg.get("val_frac", 0)),
    seed=int(cfg.get("seed", 42)),
)
model = build_model(cfg).cuda().train()
optim = build_optimizer(model, cfg)
print("groups", len(optim.param_groups), flush=True)
bb = model.encoder.backbone
print("grad_ckpt", getattr(bb, "grad_checkpointing", None), flush=True)
pos = torch.ones(len(targets), device="cuda")
it = iter(ld)
gold = set()

def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000, out

# warmup
for _ in range(2):
    x, y, uid = next(it)
    x = x.cuda()
    y = y.cuda()
    optim.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(augment_batch(x))
        loss = masked_bce(logits, y, pos)
    loss.backward()
    loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
print("warmup ok", flush=True)

for i in range(4):
    ms_next, batch = sync_ms(lambda: next(it))
    x, y, uid = batch
    ms_h2d, _ = sync_ms(lambda: (x.cuda(non_blocking=True), y.cuda(non_blocking=True)))
    x = x.cuda()
    y = y.cuda()
    def body():
        optim.zero_grad(set_to_none=True)
        xa = augment_batch(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xa)
            sw = torch.tensor([4.0 if u in gold else 1.0 for u in uid], device="cuda")
            return masked_bce(logits, y, pos, sample_weight=sw)
    ms_fwd, loss = sync_ms(body)
    ms_bwd, _ = sync_ms(lambda: loss.backward())
    ms_item, _ = sync_ms(lambda: loss.item())
    ms_clip, _ = sync_ms(lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    ms_opt, _ = sync_ms(lambda: optim.step())
    print(
        f"next {ms_next:.0f} h2d {ms_h2d:.0f} fwd {ms_fwd:.0f} "
        f"bwd {ms_bwd:.0f} item {ms_item:.0f} clip {ms_clip:.0f} opt {ms_opt:.0f}",
        flush=True,
    )
