import pandas as pd
import numpy as np

df = pd.read_csv("data/labels/coolin/targets_v3b_hybrid025.csv")
targets = [c for c in df.columns if c not in ("StudyInstanceUID", "split")]
df["oai"] = df.StudyInstanceUID.astype(str).str.startswith("OAI_")
for split in ("train", "val"):
    sub = df[df.split == split]
    print("==", split, "n", len(sub), "oai", int(sub.oai.sum()))
    for t in targets:
        x = sub[t].dropna()
        pos = float((x >= 0.5).mean())
        print(f"  {t:20} mean {x.mean():.3f} min {x.min():.3f} max {x.max():.3f} pos>=0.5 {pos:.3f}")
print("oai train max", df.loc[df.oai & (df.split=="train"), targets].max().max())
print("rsna train max", df.loc[~df.oai & (df.split=="train"), targets].max().max())
