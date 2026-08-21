# -*- coding: utf-8 -*-
"""Patch synovite : silence du rapport + epanchement -> cible molle.

Ne touche PAS au CSV source. Defaut : labels_llm_gpt56sol.csv ->
data/labels/labels_llm_gpt56sol_syn_from_eff.csv

Regle (declenchee sur l'absence du MOT dans le rapport, jamais sur
synovite==0 tout seul) :
  - le lexique trouve « synovite » (9 langues, y compris niee) -> on garde
    la synovite GPT-sol ;
  - sinon, si epanchement >= seuil -> synovite = --soft (defaut 0.60) ;
  - sinon -> 0 (ou NaN si --mask-silent).

Usage :
  python -X utf8 -m src.data.patch_synovitis_from_effusion
  python -X utf8 -m src.data.patch_synovitis_from_effusion --mask-silent
  python -X utf8 -m src.data.patch_synovitis_from_effusion --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nlp.extractor import normalize
from src.nlp.lexicon import LANGS, LESION_TERMS

SYN_COL = "Synovitis"
EFF_COL = "Effusion"


def _compile_syn_pats() -> dict[str, list[re.Pattern]]:
    out: dict[str, list[re.Pattern]] = {}
    for lang in LANGS:
        pats = LESION_TERMS["Synovitis"].get(lang, [])
        out[lang] = [re.compile(r"(?<!\w)" + p) for p in pats]
    return out


SYN_PATS = _compile_syn_pats()


def synovitis_written(report: str) -> bool:
    """True si un terme synovite du lexique apparait, meme nie."""
    if report is None:
        return False
    text = str(report).strip()
    if not text:
        return False
    for lang, pats in SYN_PATS.items():
        if not pats:
            continue
        norm = normalize(text, lang)
        if any(p.search(norm) for p in pats):
            return True
    return False


def patch_synovitis(
    labels: pd.DataFrame,
    reports: pd.Series,
    *,
    soft: float = 0.60,
    effusion_min: float = 0.50,
    mask_silent: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Nouvelle copie : seule la colonne Synovitis peut changer."""
    out = labels.copy()
    counts = {
        "n": 0,
        "written_keep": 0,
        "silent_eff_soft": 0,
        "silent_no_eff": 0,
        "missing_report": 0,
    }
    new_syn = out[SYN_COL].astype(float).copy()
    for idx, row in out.iterrows():
        counts["n"] += 1
        uid = row["StudyInstanceUID"]
        report = reports.get(uid)
        if report is None or (isinstance(report, float) and pd.isna(report)):
            counts["missing_report"] += 1
            continue
        if synovitis_written(report):
            counts["written_keep"] += 1
            continue
        eff = float(row[EFF_COL]) if pd.notna(row[EFF_COL]) else 0.0
        if eff >= effusion_min:
            new_syn.at[idx] = soft
            counts["silent_eff_soft"] += 1
        else:
            new_syn.at[idx] = float("nan") if mask_silent else 0.0
            counts["silent_no_eff"] += 1
    out[SYN_COL] = new_syn
    return out, counts


def _print_counts(counts: dict[str, int], labels: pd.DataFrame, patched: pd.DataFrame) -> None:
    n = max(counts["n"], 1)
    print("=== patch synovite <- epanchement ===")
    for key, label in (
        ("written_keep", "mot present, GPT-sol conserve"),
        ("silent_eff_soft", "silence + epanchement -> molle"),
        ("silent_no_eff", "silence sans epanchement"),
        ("missing_report", "rapport manquant (inchange)"),
    ):
        k = counts[key]
        print(f"  {label:40s} {k:5d}  ({100 * k / n:5.1f}%)")
    before = labels[SYN_COL].astype(float)
    after = patched[SYN_COL].astype(float)
    print(f"  prevalence synovite avant {before.mean():.3f}  apres {after.mean():.3f}")
    print(f"  couverture synovite avant {before.notna().mean():.3f}  "
          f"apres {after.notna().mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cree un CSV de labels ou la synovite silencieuse "
                    "est adoucie via l'epanchement. N'ecrase jamais la source."
    )
    parser.add_argument(
        "--labels",
        default="data/labels/labels_llm_gpt56sol.csv",
        help="CSV source (non modifie).",
    )
    parser.add_argument(
        "--train-csv",
        default="data/csv/train.csv",
        help="CSV competition avec la colonne Report.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="CSV de sortie. Defaut : <stem>_syn_from_eff.csv a cote de la source.",
    )
    parser.add_argument(
        "--soft", type=float, default=0.60,
        help="Cible molle si silence + epanchement (defaut 0.60).",
    )
    parser.add_argument(
        "--effusion-min", type=float, default=0.50,
        help="Seuil d'epanchement 'clairement positif' (defaut 0.50).",
    )
    parser.add_argument(
        "--mask-silent", action="store_true",
        help="Sans epanchement et sans mot : NaN (masque en loss) au lieu de 0.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compte les cas, n'ecrit pas de fichier.",
    )
    args = parser.parse_args()

    labels_path = (ROOT / args.labels).resolve()
    train_path = (ROOT / args.train_csv).resolve()
    if args.out:
        out_path = (ROOT / args.out).resolve()
    else:
        out_path = labels_path.with_name(labels_path.stem + "_syn_from_eff.csv")

    if out_path == labels_path:
        raise SystemExit(
            "Refuse d'ecraser le CSV source. Passe un --out different."
        )

    labels = pd.read_csv(labels_path)
    for col in ("StudyInstanceUID", SYN_COL, EFF_COL):
        if col not in labels.columns:
            raise SystemExit(f"colonne manquante dans {labels_path}: {col}")

    train = pd.read_csv(train_path, usecols=["StudyInstanceUID", "Report"])
    reports = train.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")["Report"]

    patched, counts = patch_synovitis(
        labels, reports,
        soft=args.soft,
        effusion_min=args.effusion_min,
        mask_silent=args.mask_silent,
    )
    _print_counts(counts, labels, patched)
    n_changed = int((patched[SYN_COL].fillna(-1) != labels[SYN_COL].astype(float).fillna(-1)).sum())
    print(f"  cellules synovite modifiees : {n_changed}")

    if args.dry_run:
        print("dry-run : aucun fichier ecrit")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    patched.to_csv(out_path, index=False)
    print("sauvegarde:", out_path)
    print("Pour entrainer : labels_csv:", out_path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
