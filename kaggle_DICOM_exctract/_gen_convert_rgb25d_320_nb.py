# -*- coding: utf-8 -*-
"""Kernels Kaggle : memes series que crop130_rgb25d, resize 320.

ViT-e_d lit (24, 3, H, W) avec R=k-1, G=k, B=k+1 dans le volume DICOM.
data/crop130_320 est en gris (linspace seul) : inutilisable pour ce run.
data/crop130_rgb25d est le bon format, mais en 224.

Les 21332 series ok des manifests proto sont reconverties, UID figes.
Le subset 25 % est tire au lancement depuis l'index complet : un zip
qui n'aurait que les 924 etudes changerait le split.

12 shards, ~11 Go estimes chacun (plafond sortie Kaggle 20 Go).

  python -X utf8 kaggle_DICOM_exctract/_gen_convert_rgb25d_320_nb.py
  kaggle kernels push -p kaggle_DICOM_exctract/convert_rgb25d_320_part0
  ...
  python -X utf8 kaggle_DICOM_exctract/download_rgb25d_320.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
_spec = importlib.util.spec_from_file_location(
    "_gen_convert_rgb25d_nb", ROOT / "_gen_convert_rgb25d_nb.py",
)
_rgb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_rgb)

IMG_SIZE = 320
N_SHARDS = 12
DIR_PREFIX = "convert_rgb25d_320"
ZIP_PREFIX = "volumes_crop130_rgb25d_320"
KERNEL_SLUG = "rsna-knee-convert-rgb-2-5d-320"
KERNEL_TITLE = "RSNA Knee Convert RGB 2.5D 320"
SCALE = (IMG_SIZE / 224) ** 2


def load_jobs() -> pd.DataFrame:
    frames = [
        pd.read_csv(p, dtype=str)
        for p in sorted((REPO / "data" / "crop130_rgb25d").glob(
            "manifest_rgb25d_proto_part*.csv"))
    ]
    if len(frames) != 6:
        raise SystemExit(f"attendu 6 manifests proto, trouve {len(frames)}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df.ok == "1"].copy()
    df["nbytes"] = pd.to_numeric(df["nbytes"], errors="coerce").fillna(0)
    df = df.drop_duplicates("SeriesInstanceUID")
    df = df.sort_values(
        ["StudyInstanceUID", "slot", "SeriesInstanceUID"],
    ).reset_index(drop=True)
    if len(df) < 20000:
        raise SystemExit(f"seulement {len(df)} series ok, proto incomplet")
    return df


def assign_shards(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Round-robin apres tri : les gros volumes ne tombent pas tous ensemble."""
    buckets = [[] for _ in range(N_SHARDS)]
    for i, rec in enumerate(df.to_dict(orient="records")):
        buckets[i % N_SHARDS].append(rec)
    return [pd.DataFrame(rows) for rows in buckets]


def _jobs_cell(df: pd.DataFrame) -> str:
    payload = []
    for rec in df.to_dict(orient="records"):
        payload.append({
            "study": rec["StudyInstanceUID"],
            "series": rec["SeriesInstanceUID"],
            "plane": rec["plane"],
            "fluid": int(float(rec["fluid"])),
            "slot": rec["slot"],
        })
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "import json\n"
        f"jobs = json.loads({blob!r})\n"
        "print('series 320 rgb, UID figes depuis crop130_rgb25d :', len(jobs))\n"
        f"assert len(jobs) == {len(payload)}\n"
        "assert jobs and {'study', 'series', 'plane', 'fluid', 'slot'} <= set(jobs[0])\n"
    )


def _run_cell() -> str:
    src = _rgb.CODE_RUN
    src = src.replace(
        'f"volumes_crop130_rgb25d_part{SHARD_ID}.zip"',
        f'f"{ZIP_PREFIX}_part{{SHARD_ID}}.zip"',
    )
    src = src.replace(
        'f"manifest_part{SHARD_ID}.csv"',
        'f"manifest_rgb25d_320_part{SHARD_ID}.csv"',
    )
    src = src.replace("Resize 224, uint8.", f"Resize {IMG_SIZE}, uint8.")
    src = src.replace(
        '"fluid": job["fluid"],',
        '"fluid": job["fluid"],\n            "slot": job.get("slot", ""),',
    )
    return src


def make_notebook(shard: int, jobs: pd.DataFrame):
    import nbformat as nbf

    est_gb = float(pd.to_numeric(jobs["nbytes"], errors="coerce").sum()) * SCALE / 1e9
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    convert = _rgb.CODE_CONVERT.replace(
        "IMG_SIZE = 224", f"IMG_SIZE = {IMG_SIZE}", 1,
    )
    verify = _rgb.CODE_VERIFY.replace("(24, 3, 224, 224)", f"(24, 3, {IMG_SIZE}, {IMG_SIZE})")
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# RSNA Knee — RGB 2.5D voisins DICOM, 320\n\n"
            f"**Shard {shard}/{N_SHARDS}** (CPU, internet off). "
            f"**{len(jobs)} series**, ~{est_gb:.1f} Go.\n\n"
            "UID **figés** depuis `data/crop130_rgb25d/manifest_rgb25d_proto_part*.csv` "
            "(6 slots v20b, contraste exact). Même recette que le proto 224, "
            f"seul le resize passe à **{IMG_SIZE}** :\n\n"
            "1. crop physique **130 mm** (`PixelSpacing`) ou crop centre 70 %\n"
            "2. bande profondeur **20–80 %**\n"
            "3. linspace **24** centres dans la bande\n"
            "4. pour chaque centre `k`, **`k-1, k, k+1` dans le volume DICOM**\n"
            f"5. resize **{IMG_SIZE}×{IMG_SIZE}**, uint8, percentiles 1–99\n\n"
            f"Shape `(24, 3, {IMG_SIZE}, {IMG_SIZE})`, canaux R=k-1, G=k, B=k+1.\n\n"
            "Fichiers dans `/kaggle/working/` :\n"
            f"- `{ZIP_PREFIX}_part{shard}.zip`\n"
            f"- `manifest_rgb25d_320_part{shard}.csv`\n"
            "- `RECIPE.json`\n\n"
            "Download : `python -X utf8 kaggle_DICOM_exctract/download_rgb25d_320.py` "
            "→ `data/crop130_rgb25d_320/`."
        ),
        nbf.v4.new_code_cell(
            f"SHARD_ID = {shard}\nN_SHARDS = {N_SHARDS}\n" + convert
        ),
        nbf.v4.new_code_cell(_rgb.CODE_FIND),
        nbf.v4.new_code_cell(_jobs_cell(jobs)),
        nbf.v4.new_code_cell(_rgb.CODE_FUNCS),
        nbf.v4.new_code_cell(_run_cell()),
        nbf.v4.new_code_cell(verify),
    ]
    return nb


def main() -> None:
    import nbformat as nbf

    df = load_jobs()
    shards = assign_shards(df)
    print(
        f"{len(df)} series | est. {df.nbytes.sum() * SCALE / 1e9:.1f} Go a {IMG_SIZE}",
        flush=True,
    )
    for shard, jobs in enumerate(shards):
        d = ROOT / f"{DIR_PREFIX}_part{shard}"
        d.mkdir(parents=True, exist_ok=True)
        nb = make_notebook(shard, jobs)
        path = d / "convert.ipynb"
        nbf.write(nb, path)
        meta = {
            "id": f"sultent/{KERNEL_SLUG}-part{shard}",
            "title": f"{KERNEL_TITLE} part{shard}",
            "code_file": "convert.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "kernel_sources": [],
            "model_sources": [],
        }
        (d / "kernel-metadata.json").write_text(
            _rgb.json_dumps(meta), encoding="utf-8",
        )
        est = float(pd.to_numeric(jobs["nbytes"]).sum()) * SCALE / 1e9
        print(f"wrote {path} ({len(jobs)} series, ~{est:.1f} Go)", flush=True)


if __name__ == "__main__":
    main()
