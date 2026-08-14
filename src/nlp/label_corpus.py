# -*- coding: utf-8 -*-
"""Pseudo-labels LLM sur les 4 349 etudes non-gold + split train/val.

UN seul modele (defaut : production_model du yaml). Ne pas mixer plusieurs modeles.

Usage :
  python -X utf8 -m src.nlp.label_corpus
  python -X utf8 -m src.nlp.label_corpus --model gemini-3.1-pro-preview --workers 2
  python -X utf8 -m src.nlp.label_corpus --from-cache   # rebuild CSV sans API
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd
from tqdm import tqdm
import yaml

from src.data.build_llm_labels import assemble_train_val
from src.nlp.gemini_labeler import (
    MODEL_PRODUCTION,
    cache_dir_for,
    is_model_unavailable,
    label_study,
    load_cached,
    make_client,
    payload_to_flat,
)
from src.nlp.lang_detect import detect_lang
from src.nlp.lexicon import TARGETS


def load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def unlabeled_studies(train_csv: Path) -> pd.DataFrame:
    train = pd.read_csv(train_csv)
    df = train[train[TARGETS].isna().any(axis=1)].copy()
    df["lang"] = df["Report"].map(detect_lang)
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Labellise les 4349 rapports")
    parser.add_argument("--config", default="configs/llm_labels.yaml")
    parser.add_argument("--model", default=None, help="Override production_model")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from-cache", action="store_true",
                        help="Ne fait aucun appel : reconstruit les CSV depuis le cache")
    parser.add_argument("--no-lexicon-net", action="store_true")
    parser.add_argument("--no-cap", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(ROOT / args.config)
    model = args.model or cfg.get("production_model") or MODEL_PRODUCTION
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 2))
    unlabeled = unlabeled_studies(ROOT / "data" / "csv" / "train.csv")
    if args.limit:
        unlabeled = unlabeled.iloc[: args.limit].copy()
        print(f"limit={args.limit}")
    print(f"modele : {model}")
    print(f"etudes non-gold : {len(unlabeled)}")

    rows: list[dict] = []
    missing: list[dict] = []
    for r in unlabeled.to_dict("records"):
        cached = None if args.force else load_cached(model, r["StudyInstanceUID"])
        if cached is not None and "labels" in cached:
            flat = payload_to_flat(cached)
            flat["lang"] = r["lang"]
            rows.append(flat)
        else:
            missing.append(r)

    print(f"cache hit : {len(rows)} | a appeler : {len(missing)}")
    if missing and args.from_cache:
        print("Mode --from-cache : les etudes absentes du cache restent hors CSV.")
    elif missing:
        client = make_client(timeout_ms=int(cfg.get("request_timeout_ms", 180_000)))
        gen_kw = dict(
            force=args.force,
            llm_conf=float(cfg.get("llm_conf", 0.90)),
            temperature=float(cfg.get("temperature", 0.0)),
            seed=cfg.get("seed", 0),
            thinking_budget=cfg.get("thinking_budget"),
            max_output_tokens=int(cfg.get("max_output_tokens", 8192)),
            max_retries=int(cfg.get("max_retries", 6)),
        )
        errors = []

        def job(r):
            return r, label_study(
                client, model, r["StudyInstanceUID"], str(r["Report"]), **gen_kw
            )

        probe = missing[0]
        try:
            _, payload = job(probe)
            flat = payload_to_flat(payload)
            flat["lang"] = probe["lang"]
            rows.append(flat)
            missing = missing[1:]
        except Exception as exc:  # noqa: BLE001
            if is_model_unavailable(exc):
                raise SystemExit(
                    f"{model} indisponible ({exc}).\n"
                    "gemini-2.5-pro est ferme aux nouvelles cles. "
                    "Utilise gemini-3.1-pro-preview (billing AI Studio)."
                ) from exc
            errors.append((probe["StudyInstanceUID"], str(exc)))
            missing = missing[1:]

        if workers <= 1:
            for r in tqdm(missing, desc=model):
                try:
                    _, payload = job(r)
                    flat = payload_to_flat(payload)
                    flat["lang"] = r["lang"]
                    rows.append(flat)
                except Exception as exc:  # noqa: BLE001
                    errors.append((r["StudyInstanceUID"], str(exc)))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(job, r): r for r in missing}
                for fut in tqdm(as_completed(futs), total=len(futs), desc=model):
                    r = futs[fut]
                    try:
                        _, payload = fut.result()
                        flat = payload_to_flat(payload)
                        flat["lang"] = r["lang"]
                        rows.append(flat)
                    except Exception as exc:  # noqa: BLE001
                        errors.append((r["StudyInstanceUID"], str(exc)))
        if errors:
            print(f"{len(errors)} echecs (relance le meme script : le cache reprend) :")
            for uid, msg in errors[:8]:
                print(f"  {uid}: {msg[:200]}")

    if not rows:
        raise SystemExit("Aucune etude labellisee.")

    out_dir = ROOT / "data" / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_df = pd.DataFrame(rows)
    raw_path = out_dir / "llm_pseudo_labels.csv"
    llm_df.to_csv(raw_path, index=False)
    print("sauvegarde:", raw_path)
    print(f"cache : {cache_dir_for(model)}")

    use_net = bool(cfg.get("lexicon_net", True)) and not args.no_lexicon_net
    cap = None if args.no_cap else cfg.get("prevalence_cap")
    assemble_train_val(
        llm_df,
        out_csv=out_dir / "train_val_labels_llm.csv",
        lexicon_csv=ROOT / cfg.get("lexicon_csv", "data/labels/pseudo_labels.csv") if use_net else None,
        lexicon_conf=float(cfg.get("lexicon_conf", 0.75)),
        prevalence_cap=float(cap) if cap is not None else None,
        val_frac=float(cfg.get("val_frac", 0.15)),
        seed=int(cfg.get("split_seed", 42)),
    )
    print("\nPour entrainer le CNN : dans configs/baseline.yaml")
    print("  labels_csv: data/labels/train_val_labels_llm.csv")


if __name__ == "__main__":
    main()
