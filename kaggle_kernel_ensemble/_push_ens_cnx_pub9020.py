# -*- coding: utf-8 -*-
"""pub70, fusion par rangs 90/20 (renormalisee). Sans TTA. Soumission."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_ensemble"
SRC = KERNEL / "_push_ens_cnx_pub70" / "submit.ipynb"
SLUG = "rsna-knee-ens-cnx-pub9020"
PUSH = KERNEL / "_push_ens_cnx_pub9020"
OUT = KERNEL / "_out_ens_cnx_pub9020"
KAGGLE_ID = f"sultent/{SLUG}"
MSG = (
    "ens ViT-k320 v20h_full OAI3 SWA 1-3 + v20b OAI SWA + "
    "v20h distil crop145 SWA 7-9 + CoAt OAI SWA + CNX-r7 SWA 7-10, "
    "mix lineaire, rang 90/20 renormalise (81.8% mix, 18.2% public degnonguidi), "
    "sans TTA, pas de flip, mix sur 2 T4"
)


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")]


def _replace(nb: dict, old: str, new: str, expect: int) -> None:
    found = 0
    for cell in nb["cells"]:
        src = _text(cell)
        n = src.count(old)
        if not n:
            continue
        found += n
        _set(cell, src.replace(old, new))
    if found != expect:
        raise SystemExit(f"remplacement x{found}, attendu {expect}: {old[:80]!r}")


def write_nb() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    _replace(nb, "rsna-knee-ens-cnx-pub70", SLUG, 1)
    _replace(
        nb,
        "3. Fusion par rangs, **70 %** le mix des 5 modeles et **30 %** la solution publique.\n",
        "3. Fusion par rangs, **90/20** renormalise : **81,8 %** le mix des 5 modeles "
        "et **18,2 %** la solution publique.\n",
        1,
    )
    _replace(
        nb,
        "# Rang 70/30 : 70 % mix5, 30 % submission publique.\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "\n"
        "W_MIX5, W_PUBLIC = 0.70, 0.30\n",
        "# Rang 90/20 renormalise : 90 parts mix5, 20 parts submission publique.\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "\n"
        "W_MIX5, W_PUBLIC = 90 / 110, 20 / 110\n",
        1,
    )
    _replace(
        nb,
        "print('BLEND70 mix5', len(scored), 'public', len(pub), 'poids mix5 0.70 public 0.30', flush=True)\n",
        "print('BLEND9020 mix5', len(scored), 'public', len(pub), "
        "'poids mix5 %.4f public %.4f ratio 90/20' % (W_MIX5, W_PUBLIC), flush=True)\n",
        1,
    )
    _replace(
        nb,
        "print('BLEND70_OK', dest, 'lignes', len(out), flush=True)\n",
        "print('BLEND9020_OK', dest, 'lignes', len(out), flush=True)\n",
        1,
    )
    for cell in nb["cells"]:
        src = _text(cell)
        if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
            ast.parse(src)
    blob = "\n".join(_text(c) for c in nb["cells"])
    for needle in (
        SLUG,
        "W_MIX5, W_PUBLIC = 90 / 110, 20 / 110",
        "ratio 90/20",
        "BLEND9020_OK",
        "gpus mix 2",
        "MIX5_BUDGET_S = 5.0 * 3600",
        "echeance publique h 8.9",
        "sans TTA",
        "cuda:0",
        "cuda:1",
        "submission.csv",
    ):
        if needle not in blob:
            raise SystemExit(f"manque {needle}")
    for banned in ("0.70, 0.30", "BLEND70", "70 %", "70/30"):
        if banned in blob:
            raise SystemExit(f"residu {banned}")
    if abs((90 / 110) + (20 / 110) - 1.0) > 1e-12:
        raise SystemExit("poids hors simplexe")
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / "submit.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((KERNEL / "_push_ens_cnx_pub70" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KAGGLE_ID
    meta["title"] = SLUG
    meta["machine_shape"] = "NvidiaTeslaT4"
    meta["enable_gpu"] = True
    meta["enable_internet"] = False
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PUSH / "_message.txt").write_text(MSG + "\n", encoding="utf-8")
    print("ecrit", PUSH, "cells", len(nb["cells"]), flush=True)


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "push_pub70", KERNEL / "_push_ens_cnx_pub70.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_and_submit(base, version: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(base.kaggle("kernels", "output", KAGGLE_ID, "-p", str(OUT)), flush=True)
    log = next(p for p in OUT.glob("*.log") if "cufile" not in p.name.lower() and SLUG in p.name)
    text = log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "vit vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3 epoch swa-1-2-3",
        "gold 0.9903",
        "v20b_crop130_dino_ep11_oof_v3b_full_oai3",
        "v20h_crop145_dino_ep11_v3b_rsna_distil_main_full epoch swa-7-8-9 lat False gold 0.9695",
        "coat_a_llrd08_v3b_oofmask_full_oai3",
        "cnx cnx_r7_in22k_v3b_oofmask_full epoch swa-7-8-9-10 lat False gold 0.976",
        "cnx forward fp32",
        "lat False",
        "v20h 145 mm",
        "cnx 130 mm",
        "MIX5_SCORED",
        "GPU mix5 libere",
        "RAPTOR224_STAGE_DONE",
        "NOTEBOOK_COMPLETE",
        "ratio 90/20",
        "poids mix5 0.8182 public 0.1818",
        "BLEND9020_OK",
        "sans TTA",
        "gpus mix 2 | cuda:0 cuda:1",
        "MIX5_COVERAGE 3/3",
        "RAPTOR_SEQUENTIEL",
        "lignes 3",
    ):
        if needle not in text:
            raise SystemExit(f"log sans {needle}")
    if "TTA x" in text or "lat True" in text or "public 0.30" in text or "BLEND70" in text:
        raise SystemExit("ancien blend ou TTA")
    print("LOG_OK", log.name, flush=True)
    import time
    from datetime import datetime, timedelta, timezone
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    waited = False
    while True:
        print("submit", KAGGLE_ID, "version", version, flush=True)
        try:
            resp = api.competition_submit_code(
                "submission.csv", MSG, base.COMPETITION, KAGGLE_ID, int(version), quiet=True,
            )
        except Exception as e:
            http = getattr(e, "response", None)
            body = http.text if http is not None else str(e)
            print(body, flush=True)
            if base._daily_limit(body) and not waited:
                waited = True
                now = datetime.now(timezone.utc)
                target = now.replace(hour=0, minute=5, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)
                delay = (target - now).total_seconds()
                print(f"ATTENTE_QUOTA {delay:.0f}s", flush=True)
                time.sleep(delay + 5)
                continue
            if base._transient(body):
                time.sleep(20)
                continue
            raise
        print(resp, flush=True)
        print("SOUMIS", KAGGLE_ID, "version", version, flush=True)
        return


def main() -> None:
    write_nb()
    if "--write" in sys.argv:
        return
    base = _load_base()
    base.SLUG = SLUG
    base.PUSH = PUSH
    base.OUT = OUT
    base.KAGGLE_ID = KAGGLE_ID
    base.MSG = MSG
    version = base.push()
    base.wait()
    check_and_submit(base, version)


if __name__ == "__main__":
    main()
