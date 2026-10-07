"""Step train complet, sans sync artificielle, avec releve GPU."""
import subprocess
import threading
import time
import torch
from src.train.train import load_config, masked_bce, lr_at
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
pos = torch.ones(len(targets), device="cuda")
it = iter(ld)
gold = set()
samples = []
stop = False

def poll():
    while not stop:
        samples.append(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
             "--format=csv,noheader"],
            text=True,
        ).strip())
        time.sleep(0.3)

def one(step):
    x, y, uid = next(it)
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)
    set_group_lrs(optim, step, 30000, 0, None, 0, lr_at)
    sw = torch.tensor([4.0 if u in gold else 1.0 for u in uid], device="cuda")
    optim.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(augment_batch(x))
        loss = masked_bce(logits, y, pos, smoothing=0.0, sample_weight=sw)
    loss.backward()
    loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

for i in range(2):
    one(i)
torch.cuda.synchronize()
stop = False
th = threading.Thread(target=poll, daemon=True)
th.start()
t0 = time.perf_counter()
n = 10
for i in range(n):
    one(i + 2)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / n * 1000
stop = True
th.join(timeout=2)
print(f"PIPE {dt:.0f} ms", flush=True)
pows = []
for line in samples:
    print("smi", line, flush=True)
    pows.append(line)

def one_simple():
    x, y, uid = next(it)
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)
    optim.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(augment_batch(x))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y.clamp(0, 1))
    loss.backward()
    loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

for _ in range(2):
    one_simple()
torch.cuda.synchronize()
samples.clear()
stop = False
th = threading.Thread(target=poll, daemon=True)
th.start()
t0 = time.perf_counter()
for _ in range(n):
    one_simple()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / n * 1000
stop = True
th.join(timeout=2)
print(f"SIMPLE {dt:.0f} ms", flush=True)
for line in samples[-8:]:
    print("smi", line, flush=True)
