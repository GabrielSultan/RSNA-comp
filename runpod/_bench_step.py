"""Boucle serree bf16, comme le train, et releve puissance / util."""
import subprocess
import threading
import time
import torch
from src.models.kneenet_vit_k320 import KneeNet
from src.data.transforms import augment_batch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)
samples = []
stop = False

def poll():
    while not stop:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used,clocks.sm",
             "--format=csv,noheader"],
            text=True,
        ).strip()
        samples.append(out)
        time.sleep(0.4)
model = KneeNet(
    pretrained=True,
    backbone_weights="weights/dinov3-vitb16-lvd1689m",
    encode_chunk=288,
    head_dim=512,
    dropout=0.4,
    n_series=6,
    lstm_hidden=256,
).cuda().train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
x = torch.rand(2, 6, 24, 320, 320, device="cuda")
y = torch.rand(2, 12, device="cuda")

def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(augment_batch(x))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

for _ in range(2):
    step()
torch.cuda.synchronize()
print("warmup ok", round(torch.cuda.max_memory_allocated() / 2**20), "MiB", flush=True)
th = threading.Thread(target=poll, daemon=True)
th.start()
t0 = time.perf_counter()
n = 12
for _ in range(n):
    step()
torch.cuda.synchronize()
dt = time.perf_counter() - t0
stop = True
th.join(timeout=2)
print(f"step {dt / n * 1000:.0f} ms", flush=True)
for line in samples:
    print("smi", line, flush=True)
