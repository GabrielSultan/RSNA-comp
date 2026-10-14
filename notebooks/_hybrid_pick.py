"""One-off: per-target stack vs mean for the 6-model ensemble."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

nb = json.loads(Path("notebooks/13_stack_logreg.ipynb").read_text(encoding="utf-8"))
ns = {"__name__": "__main__"}
for i in (1, 2):
    src = "".join(nb["cells"][i]["source"])
    src = "\n".join(ln for ln in src.splitlines() if not ln.startswith("%"))
    exec(src, ns)

TARGETS = ns["TARGETS"]
macro_auc = ns["macro_auc"]
idxs = tuple(range(len(ns["MODELS"])))
fit_m = ns["fit_if_distil"]

model_logits = ns["model_logits"]
empty = ns["empty"]
z = np.stack([model_logits[i] for i in idxs], axis=1)
m = z.shape[1]
s = empty.shape[1]
raw = np.empty((len(TARGETS), z.shape[0], m + s + m * s), np.float32)
raw[:, :, :m] = np.transpose(z, (2, 0, 1))
raw[:, :, m : m + s] = empty[None]
inter = z[:, :, None, :] * empty[:, None, :, None]
raw[:, :, m + s :] = np.transpose(inter, (3, 0, 1, 2)).reshape(
    len(TARGETS), z.shape[0], m * s
)

mu = raw[:, fit_m].mean(1, keepdims=True)
sd = raw[:, fit_m].std(1, keepdims=True)
keep = sd > 1e-4
sd = np.where(keep, sd, 1.0)
x = (raw - mu) / sd * keep

torch.manual_seed(ns["SEED"])
xt = torch.tensor(x[:, fit_m])
yt = torch.tensor(ns["y_hard"][fit_m].T, dtype=torch.float32)
weight = torch.zeros(len(TARGETS), xt.shape[-1], requires_grad=True)
bias = torch.zeros(len(TARGETS), requires_grad=True)
opt = torch.optim.Adam([weight, bias], lr=ns["LR"], weight_decay=ns["WEIGHT_DECAY"])
bce = nn.BCEWithLogitsLoss()
for _ in range(ns["EPOCHS"]):
    opt.zero_grad()
    logits = torch.einsum("tnd,td->tn", xt, weight) + bias[:, None]
    loss = bce(logits, yt)
    loss.backward()
    opt.step()

val_m = ns["val_m"]
gold_m = ns["gold_m"]
y_val = ns["y_hard"][val_m]
y_gold = ns["y_hard"][gold_m]
with torch.no_grad():
    xv = torch.tensor(x[:, val_m])
    xg = torch.tensor(x[:, gold_m])
    stack_val = torch.sigmoid(
        torch.einsum("tnd,td->tn", xv, weight) + bias[:, None]
    ).numpy().T
    stack_gold = torch.sigmoid(
        torch.einsum("tnd,td->tn", xg, weight) + bias[:, None]
    ).numpy().T
mean_probs = np.mean([ns["model_probs"][i] for i in idxs], axis=0)
mean_val = mean_probs[val_m]
mean_gold = mean_probs[gold_m]

_, st_v = macro_auc(y_val, stack_val, TARGETS)
_, mn_v = macro_auc(y_val, mean_val, TARGETS)
_, st_g = macro_auc(y_gold, stack_gold, TARGETS)
_, mn_g = macro_auc(y_gold, mean_gold, TARGETS)

hybrid_val = np.empty_like(stack_val)
hybrid_gold = np.empty_like(stack_gold)
print(f"{'finding':22} {'stack':8} {'moyenne':8} {'choix':8} {'g_stack':8} {'g_mean':8}")
switches = []
for c, t in enumerate(TARGETS):
    use_mean = mn_v[t] > st_v[t]
    src = "moyenne" if use_mean else "stack"
    if use_mean:
        switches.append(t)
    hybrid_val[:, c] = mean_val[:, c] if use_mean else stack_val[:, c]
    hybrid_gold[:, c] = mean_gold[:, c] if use_mean else stack_gold[:, c]
    print(
        f"{t:22} {st_v[t]:8.4f} {mn_v[t]:8.4f} {src:8} {st_g[t]:8.4f} {mn_g[t]:8.4f}"
    )

hv = macro_auc(y_val, hybrid_val, TARGETS)[0]
hg = macro_auc(y_gold, hybrid_gold, TARGETS)[0]
sv = macro_auc(y_val, stack_val, TARGETS)[0]
mv = macro_auc(y_val, mean_val, TARGETS)[0]
sg = macro_auc(y_gold, stack_gold, TARGETS)[0]
mg = macro_auc(y_gold, mean_gold, TARGETS)[0]
print("VAL stack", f"{sv:.6f}", "moyenne", f"{mv:.6f}", "hybride", f"{hv:.6f}")
print("GOLD stack", f"{sg:.6f}", "moyenne", f"{mg:.6f}", "hybride", f"{hg:.6f}")
print("switches", switches)
