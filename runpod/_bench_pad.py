"""Compare un step reel : coupes valides seules, ou batch ViT padde a 288."""
import threading
import time
import subprocess
import torch
from src.data.dataset_npz import make_loaders
from src.data.transforms import augment_batch
from src.models.kneenet_vit_k320 import KneeNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

ld, _, _ = make_loaders(
    labels_csv="data/labels/coolin/targets_v3b_oofmask.csv",
    series_csv="data/csv/train_series.csv",
    zip_path="data/crop130_320",
    batch_size=2,
    num_workers=8,
    n_series=6,
    train_frac=1.0,
    img_size=320,
    include_gold=True,
    val_frac=0,
    seed=42,
)
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
it = iter(ld)
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

def one():
    x, y, _uid = next(it)
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(augment_batch(x))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

def bench(tag, n=8):
    global stop, samples
    for _ in range(2):
        one()
    torch.cuda.synchronize()
    samples = []
    stop = False
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.perf_counter()
    for _ in range(n):
        one()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n * 1000
    stop = True
    th.join(timeout=2)
    print(f"{tag} {dt:.0f} ms", flush=True)
    for line in samples:
        print(" smi", line, flush=True)

print("subset", flush=True)
bench("SUBSET")

def padded(self, rgb):
    n = rgb.size(0)
    c = self.encoder.num_features
    if n == 0:
        h = rgb.size(-2) // self.patch_size
        w = rgb.size(-1) // self.patch_size
        return rgb.new_zeros(0, c, h, w)
    chunk = max(1, self.encode_chunk)
    outs = []
    for i in range(0, n, chunk):
        part = rgb[i:i + chunk]
        n_real = part.size(0)
        if n_real < chunk:
            part = torch.cat(
                [part, part.new_zeros(chunk - n_real, *part.shape[1:])], 0)
        outs.append(self.encoder(part)[:n_real])
    return torch.cat(outs, 0)

model._encode_chunks = padded.__get__(model, type(model))
print("padded", flush=True)
bench("PADDED")
print("done", flush=True)
