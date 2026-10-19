# -*- coding: utf-8 -*-
import json
from pathlib import Path

root = Path(r"c:\Users\sulta\Documents\RSNA-comp\kaggle_kernel_ensemble")
out = root / "_loops_extract.txt"
parts = []
for name, needles in (
    ("_push_ens_cnx_pub60_tta/submit.ipynb", ("p_v20h_id", "model_vit = load_vit", "apply_affine")),
    ("_push_ens_cnx_tta3b/submit.ipynb", ("DEV_CNX", "def _cnx_job", "ThreadPoolExecutor")),
):
    nb = json.loads((root / name).read_text(encoding="utf-8"))
    parts.append("\n\n######## " + name + "\n")
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if any(n in src for n in needles) and "%%writefile" not in src[:20]:
            if len(src) > 12000:
                # keep head and the inference tail
                src = src[:2500] + "\n\n...[snip]...\n\n" + src[-8000:]
            parts.append(f"\n----- cell {i} {cell['cell_type']} len {len(''.join(cell['source']))} -----\n")
            parts.append(src)
out.write_text("".join(parts), encoding="utf-8")
print("wrote", out, "chars", out.stat().st_size)
