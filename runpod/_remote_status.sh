#!/bin/bash
set -euo pipefail
cd /workspace
python - <<'PY'
import numpy as np
import pandas as pd
from src.data.dataset_npz import PROTOCOL_SLOT_NAMES, PROTOCOL_SLOTS, load_volume, make_series_index

labels = pd.read_csv("data/labels/coolin/targets_v3b_hybrid025.csv")
labels["StudyInstanceUID"] = labels.StudyInstanceUID.astype(str)
index, vol = make_series_index("data/oai/derived/train_series_oai.csv", "data/crop130_320")

def audit(uids, tag):
    n_slot = n_none = n_badshape = n_blank = n_flat = 0
    means, stds = [], []
    worst = []
    for uid in uids:
        for name, (plane, fluid) in zip(PROTOCOL_SLOT_NAMES, PROTOCOL_SLOTS):
            n_slot += 1
            sid = index.pick_protocol(uid, plane, fluid, None, which="first")
            if sid is None:
                n_none += 1
                continue
            arr = load_volume(vol[sid], sid)
            ok_shape = arr.shape == (24, 320, 320) and arr.dtype == np.uint8
            sl_std = arr.std(axis=(1, 2))
            vol_std = float(arr.std())
            vol_mean = float(arr.mean())
            n_flat_sl = int((sl_std < 1.0).sum())
            blank = vol_std < 2.0 or float(arr.max()) < 20
            if not ok_shape:
                n_badshape += 1
            if blank:
                n_blank += 1
            if n_flat_sl >= 12:
                n_flat += 1
            if (not ok_shape) or blank or n_flat_sl >= 12:
                worst.append((vol_std, tag, uid[:22], name, arr.shape, arr.dtype, vol_mean, n_flat_sl, sid))
            else:
                means.append(vol_mean)
                stds.append(vol_std)
    print(f"{tag} etudes {len(uids)} slots {n_slot} vides {n_none} "
          f"mauvaise forme {n_badshape} quasi vides {n_blank} trop de coupes plates {n_flat}")
    if means:
        print(f"  volumes sains mean {np.mean(means):.1f} (med {np.median(means):.1f}) "
              f"std {np.mean(stds):.1f} (med {np.median(stds):.1f}) n={len(means)}")
    worst.sort()
    for row in worst[:8]:
        print("  BAD", row)

oai = labels.loc[labels.StudyInstanceUID.str.startswith("OAI_") & (labels.split=="train"), "StudyInstanceUID"].tolist()
rsna = labels.loc[~labels.StudyInstanceUID.str.startswith("OAI_") & (labels.split=="train"), "StudyInstanceUID"].tolist()
val = labels.loc[labels.split=="val", "StudyInstanceUID"].tolist()
audit(oai, "OAI train")
audit(rsna, "RSNA train")
audit(val, "RSNA val")
PY
