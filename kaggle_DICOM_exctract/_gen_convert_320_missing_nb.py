# -*- coding: utf-8 -*-
"""Kernels Kaggle : series T1 + proto encore absentes de crop130_320.

Les 13219 series fluid de data/crop130_320 sont deja les memes UID que
data/crop130 (band, crop, n_raw, src_hw identiques). On ne les refait pas.

Ce script fige les UID ok==1 de :

  data/crop130/manifest_t1_part{0,1}.csv      (4406)
  data/crop130/manifest_proto_part{0,1}.csv   (3969)

et genere 4 notebooks, meme recette spatiale, resize 320.

  python -X utf8 kaggle_DICOM_exctract/_gen_convert_320_missing_nb.py
  kaggle kernels push -p kaggle_DICOM_exctract/convert_crop130_320_t1_part0
  kaggle kernels push -p kaggle_DICOM_exctract/convert_crop130_320_t1_part1
  kaggle kernels push -p kaggle_DICOM_exctract/convert_crop130_320_proto_part0
  kaggle kernels push -p kaggle_DICOM_exctract/convert_crop130_320_proto_part1
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
_spec = importlib.util.spec_from_file_location(
    "_gen_convert_nb", ROOT / "_gen_convert_nb.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

IMG_SIZE = 320
N_SHARDS = 2
PLANE_SLOT = {
    "Sagittal": "SAG_T1",
    "Coronal": "COR_T1",
    "Axial": "AX_T1",
}


def _load(folder: Path, pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str) for p in sorted(folder.glob(pattern))]
    if not frames:
        raise SystemExit(f"aucun manifest {folder / pattern}")
    return pd.concat(frames, ignore_index=True)


def _ok_uids(df: pd.DataFrame) -> set[str]:
    return set(df.loc[df.ok == "1", "SeriesInstanceUID"].astype(str))


def assert_keep_fluid() -> None:
    """Refuse de generer si le fluid 320 n'est plus un clone du fluid 224."""
    c224 = _load(REPO / "data" / "crop130", "manifest_part*.csv")
    c320 = _load(REPO / "data" / "crop130_320", "manifest_part*.csv")
    t1 = _load(REPO / "data" / "crop130", "manifest_t1_part*.csv")
    proto = _load(REPO / "data" / "crop130", "manifest_proto_part*.csv")
    s224, s320 = _ok_uids(c224), _ok_uids(c320)
    if s224 != s320:
        raise SystemExit(
            f"fluid 224 et 320 divergent "
            f"(seulement 224 : {len(s224 - s320)}, "
            f"seulement 320 : {len(s320 - s224)}). "
            "Ne pas completer : refaire les series fluid."
        )
    extra = (_ok_uids(t1) | _ok_uids(proto)) & s320
    if extra:
        raise SystemExit(f"{len(extra)} series T1/proto deja dans le 320")
    print(
        f"fluid 320 = fluid 224 ({len(s320)} series). "
        f"A convertir : t1 {len(_ok_uids(t1))} + proto {len(_ok_uids(proto))}",
        flush=True,
    )


def _jobs_frame(kind: str, shard: int) -> pd.DataFrame:
    name = f"manifest_{kind}_part{shard}.csv"
    df = pd.read_csv(REPO / "data" / "crop130" / name, dtype=str)
    df = df[df.ok == "1"].copy()
    if df.empty:
        raise SystemExit(f"{name} : aucune serie ok")
    if "slot" not in df.columns:
        df["slot"] = df.plane.map(PLANE_SLOT)
    missing = set(df.plane) - set(PLANE_SLOT)
    if missing:
        raise SystemExit(f"{name} : plans inconnus {sorted(missing)}")
    df["slot"] = df.slot.fillna(df.plane.map(PLANE_SLOT))
    keep = ["StudyInstanceUID", "SeriesInstanceUID", "plane", "fluid", "slot"]
    return df[keep]


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
        "print('series a convertir (UID figes depuis crop130 224) :', len(jobs))\n"
        f"assert len(jobs) == {len(payload)}\n"
        "assert jobs and {'study', 'series', 'plane', 'fluid', 'slot'} <= set(jobs[0])\n"
    )


def _run_cell(img_size: int, zip_prefix: str, manifest_name: str, note: str) -> str:
    src = _mod.CODE_RUN.replace(
        'f"manifest_part{SHARD_ID}.csv"',
        f'"{manifest_name}"',
    ).replace(
        "1 serie fluid-prioritaire par plan. ",
        note,
    ).replace(
        '"fluid": job["fluid"],',
        '"fluid": job["fluid"],\n            "slot": job.get("slot", ""),',
    )
    return _mod._fill(src, img_size, zip_prefix)


def make_notebook(kind: str, shard: int, jobs: pd.DataFrame):
    import nbformat as nbf

    zip_prefix = f"volumes_crop130_320_{kind}"
    manifest_name = f"manifest_{kind}_part{shard}.csv"
    if kind == "t1":
        note = (
            "1 serie T1/non-fluide extra, UID figes depuis crop130 224. "
        )
        title_md = (
            "UID **figés** depuis `manifest_t1_part` du crop 130 mm 224 "
            "(1 série non-fluide extra par étude, hors des 3 picks fluid)."
        )
    else:
        note = (
            "1 COR_T1 + 1 AX_T1, UID figes depuis crop130 224. "
        )
        title_md = (
            "UID **figés** depuis `manifest_proto_part` du crop 130 mm 224 "
            "(T1 coronal et axial, hors picks fluid et hors T1 sagittal)."
        )
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# RSNA Knee — complément 320 (séries manquantes)\n\n"
            f"**Shard {shard}/{N_SHARDS}** `{kind}` (CPU, internet off). "
            f"**{IMG_SIZE}×{IMG_SIZE}** uint8.\n\n"
            "Les séries fluid 320 sont déjà le clone du 224 "
            "(mêmes UID, même crop, mêmes coupes). Ce kernel ne les refait pas.\n\n"
            f"{title_md} Même recette spatiale :\n\n"
            "1. crop physique **130 mm** (`PixelSpacing`) ou crop centre 70 % si spacing absent\n"
            "2. bande profondeur **20–80 %**\n"
            f"3. resample **24** coupes, resize **{IMG_SIZE}×{IMG_SIZE}**, uint8, percentiles 1–99\n\n"
            "Sortie dans `/kaggle/working/` :\n"
            f"- `{zip_prefix}_part{shard}.zip`\n"
            f"- `{manifest_name}`\n"
            "- `RECIPE.json`\n\n"
            "A déposer dans `data/crop130_320/` à côté des zip fluid."
        ),
        nbf.v4.new_code_cell(
            f"SHARD_ID = {shard}\nN_SHARDS = {N_SHARDS}\n"
            + _mod._fill(_mod.CODE_CONVERT, IMG_SIZE, zip_prefix)
        ),
        nbf.v4.new_code_cell(_mod.CODE_FIND),
        nbf.v4.new_code_cell(_jobs_cell(jobs)),
        nbf.v4.new_code_cell(_mod.CODE_FUNCS),
        nbf.v4.new_code_cell(_run_cell(IMG_SIZE, zip_prefix, manifest_name, note)),
        nbf.v4.new_code_cell(_mod.CODE_VERIFY),
    ]
    return nb


def main() -> None:
    import nbformat as nbf

    assert_keep_fluid()
    specs = [
        ("t1", "rsna-knee-convert-crop130-320-t1", "RSNA Knee Convert crop130 320 T1"),
        ("proto", "rsna-knee-convert-crop130-320-proto", "RSNA Knee Convert crop130 320 proto"),
    ]
    for kind, slug, title in specs:
        for shard in range(N_SHARDS):
            jobs = _jobs_frame(kind, shard)
            d = ROOT / f"convert_crop130_320_{kind}_part{shard}"
            d.mkdir(parents=True, exist_ok=True)
            nb = make_notebook(kind, shard, jobs)
            path = d / "convert.ipynb"
            nbf.write(nb, path)
            meta = {
                "id": f"sultent/{slug}-part{shard}",
                "title": f"{title} part{shard}",
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
                _mod.json_dumps(meta), encoding="utf-8",
            )
            print(f"wrote {path} ({len(jobs)} series)", flush=True)


if __name__ == "__main__":
    main()
