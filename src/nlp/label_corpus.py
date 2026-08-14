# -*- coding: utf-8 -*-
"""Pseudo-labels LLM sur les 4 349 etudes non-gold + split train/val.

UN seul modele (defaut : production_model du yaml). Ne pas mixer plusieurs modeles.

Usage :
  python -X utf8 -m src.nlp.label_corpus
  python -X utf8 -m src.nlp.label_corpus --model gemini-3.7-flash --workers 2
  python -X utf8 -m src.nlp.label_corpus --from-cache   # rebuild CSV sans API
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event, Lock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd
from tqdm import tqdm
import yaml

from src.data.build_llm_labels import assemble_train_val
from src.nlp.gemini_labeler import (
    MODEL_PRODUCTION,
    GeminiOverloadError,
    cache_dir_for,
    gen_kwargs_from_cfg,
    is_model_unavailable,
    is_overload,
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
        gen_kw = gen_kwargs_from_cfg(cfg, force=args.force)
        errors = []
        abort = Event()
        streak_lock = Lock()
        overload_streak = 0
        circuit_n = int(cfg.get("circuit_breaker", 2))
        aborted = False

        def job(r):
            nonlocal overload_streak
            if abort.is_set():
                raise GeminiOverloadError("coupe-circuit : API surchargee, stop")
            try:
                payload = label_study(
                    client, model, r["StudyInstanceUID"], str(r["Report"]), **gen_kw
                )
            except Exception as exc:  # noqa: BLE001
                if is_overload(exc):
                    with streak_lock:
                        overload_streak += 1
                        if overload_streak >= circuit_n:
                            abort.set()
                raise
            with streak_lock:
                overload_streak = 0
            return r, payload

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
                    "Modele vise : gemini-3.7-flash. "
                    "python -X utf8 -m src.nlp.gold_bakeoff --list-models"
                ) from exc
            errors.append((probe["StudyInstanceUID"], str(exc)))
            missing = missing[1:]
            if abort.is_set() or is_overload(exc):
                aborted = True

        if not aborted and workers <= 1:
            for r in tqdm(missing, desc=model):
                if abort.is_set():
                    aborted = True
                    break
                try:
                    _, payload = job(r)
                    flat = payload_to_flat(payload)
                    flat["lang"] = r["lang"]
                    rows.append(flat)
                except Exception as exc:  # noqa: BLE001
                    errors.append((r["StudyInstanceUID"], str(exc)))
                    if abort.is_set() or is_overload(exc):
                        aborted = True
                        break
        elif not aborted:
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
                        if abort.is_set() or is_overload(exc):
                            aborted = True
                            for pending in futs:
                                pending.cancel()
                            break
        if aborted:
            print(
                "Coupe-circuit : Google repond 503 / tokens=0. "
                "On arrete au lieu de marteler l'API."
            )
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

    n_expected = len(unlabeled)
    if len(rows) < n_expected:
        print(
            f"Corpus incomplet ({len(rows)}/{n_expected}). "
            "train_val_labels_llm.csv NON ecrit (evite un split partiel).\n"
            "Le cache est a jour. Relance quand AI Studio n'affiche plus de 503 :\n"
            "  python -X utf8 -m src.nlp.label_corpus"
        )
        return

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
