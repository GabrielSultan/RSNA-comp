# -*- coding: utf-8 -*-
"""Bake-off gold : lexique vs Gemini 3.7 Flash.

Les 58 gold ne servent QU'a comparer. On ne retune pas le prompt dessus.

Usage :
  python -X utf8 -m src.nlp.gold_bakeoff
  python -X utf8 -m src.nlp.gold_bakeoff --dry-run
  python -X utf8 -m src.nlp.gold_bakeoff --list-models
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd
from tqdm import tqdm
import yaml

from src.nlp.eval_gold import gold_metrics, recommend_production, summarize
from src.nlp.extractor import extract_text
from src.nlp.gemini_labeler import (
    MODEL_GOLD_PREVIEW,
    MODEL_PRODUCTION,
    SYSTEM_PROMPT,
    KneeReportLabels,
    RESPONSE_JSON_SCHEMA,
    get_api_key,
    gen_kwargs_from_cfg,
    is_model_unavailable,
    label_study,
    list_generate_models,
    make_client,
    payload_to_flat,
)
from src.nlp.lang_detect import detect_lang
from src.nlp.lexicon import TARGETS


def load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gold(train_csv: Path) -> pd.DataFrame:
    train = pd.read_csv(train_csv)
    gold = train[train[TARGETS].notna().all(axis=1)].copy()
    gold["lang"] = gold["Report"].map(detect_lang)
    return gold.reset_index(drop=True)


def lexicon_on_gold(gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in gold.iterrows():
        pred = extract_text(str(r["Report"]), r["lang"])
        row = {"StudyInstanceUID": r["StudyInstanceUID"], "lang": r["lang"]}
        for t in TARGETS:
            lab, conf = pred[t]
            row[t] = None if lab is None else float(lab)
            row[t + "_conf"] = float(conf)
        rows.append(row)
    return pd.DataFrame(rows)


def _label_one(client, model, uid, report, cfg, force):
    return label_study(
        client,
        model,
        uid,
        report,
        **gen_kwargs_from_cfg(cfg, force=force),
    )


def _die_unavailable(model: str, exc: BaseException) -> None:
    raise SystemExit(
        f"{model} indisponible ({exc}).\n"
        "Modele vise : gemini-3.7-flash. "
        "Liste les modeles de ta cle : python -X utf8 -m src.nlp.gold_bakeoff --list-models"
    ) from exc


def run_model_on_gold(
    gold: pd.DataFrame,
    model: str,
    cfg: dict,
    *,
    force: bool,
    workers: int,
) -> pd.DataFrame:
    client = make_client(timeout_ms=int(cfg.get("request_timeout_ms", 180_000)))
    probe = gold.iloc[0]
    try:
        _label_one(client, model, probe["StudyInstanceUID"], str(probe["Report"]), cfg, force)
    except Exception as exc:  # noqa: BLE001
        if is_model_unavailable(exc):
            _die_unavailable(model, exc)
        raise

    rows: list[dict] = []
    errors: list[tuple[str, str]] = []

    def job(r):
        uid = r["StudyInstanceUID"]
        return uid, _label_one(client, model, uid, str(r["Report"]), cfg, force)

    records = gold.to_dict("records")
    if workers <= 1:
        for r in tqdm(records, desc=model):
            try:
                uid, payload = job(r)
                flat = payload_to_flat(payload)
                flat["lang"] = r["lang"]
                rows.append(flat)
            except Exception as exc:  # noqa: BLE001
                errors.append((r["StudyInstanceUID"], str(exc)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(job, r): r for r in records}
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
        print(f"\n{len(errors)} echecs {model} (cache partiel conserve) :")
        for uid, msg in errors[:8]:
            print(f"  {uid}: {msg[:200]}")
    if not rows:
        raise SystemExit(f"Aucune prediction pour {model}.")
    df = pd.DataFrame(rows)
    # Reordonner comme gold
    order = {u: i for i, u in enumerate(gold["StudyInstanceUID"])}
    df["_ord"] = df["StudyInstanceUID"].map(order)
    df = df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return df


def print_table(name: str, stats: pd.DataFrame) -> None:
    s = summarize(stats)
    print(f"\n=== {name} ===")
    print(f"macro-AUC {s['macro_AUC']:.3f} | macro-F1 {s['macro_F1']:.3f} "
          f"| fidelite {s['macro_fidelite']:.3f} | couverture {s['couverture_moy']:.2f}")
    show = stats.set_index("cible")[["AUC", "F1", "fidelite_couverte", "couverture"]]
    print(show.round(3).to_string())


def dry_run(gold: pd.DataFrame) -> None:
    print("DRY-RUN — aucun appel API")
    print(f"etudes gold : {len(gold)}")
    print("langues :", gold["lang"].value_counts().to_dict())
    print(f"prompt systeme : {len(SYSTEM_PROMPT)} caracteres")
    schema = KneeReportLabels.model_json_schema()
    print(f"schema pydantic : {len(schema.get('properties', {}))} findings")
    print(f"schema Gemini   : {len(RESPONSE_JSON_SCHEMA['properties'])} cles")
    key = get_api_key()
    print("GEMINI_API_KEY :", "presente" if key else "ABSENTE (remplis .env)")
    sample = gold.iloc[0]
    from src.nlp.gemini_labeler import user_prompt
    up = user_prompt(str(sample["Report"]))
    print(f"exemple user prompt : {len(up)} caracteres, uid={sample['StudyInstanceUID']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake-off LLM sur les 58 gold")
    parser.add_argument("--config", default="configs/llm_labels.yaml")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Defaut : gold_models du yaml")
    parser.add_argument("--limit", type=int, default=0, help="Sous-echantillon (test)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Ignore le cache")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-models", action="store_true",
                        help="Liste les modeles generateContent de la cle")
    args = parser.parse_args()

    cfg = load_cfg(ROOT / args.config)

    if args.list_models:
        client = make_client(timeout_ms=int(cfg.get("request_timeout_ms", 180_000)))
        names = list_generate_models(client)
        print(f"{len(names)} modeles :")
        for n in names:
            mark = " <--" if "pro" in n.lower() and "tts" not in n.lower() else ""
            print(f"  {n}{mark}")
        return

    gold = load_gold(ROOT / "data" / "csv" / "train.csv")
    if args.limit and args.limit < len(gold):
        gold = gold.iloc[: args.limit].copy()
        print(f"limit={args.limit} etudes")

    if args.dry_run:
        dry_run(gold)
        return

    out_dir = ROOT / "data" / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    lex = lexicon_on_gold(gold)
    lex.to_csv(out_dir / "gold_nlp_predictions.csv", index=False)
    lex_stats = gold_metrics(gold, lex)
    print_table("lexique v1 (baseline)", lex_stats)

    models = args.models or list(cfg.get("gold_models") or [MODEL_PRODUCTION, MODEL_GOLD_PREVIEW])
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 2))
    stats_by_model: dict[str, pd.DataFrame] = {"lexicon": lex_stats}

    for model in models:
        print(f"\n--- {model} ---")
        pred = run_model_on_gold(gold, model, cfg, force=args.force, workers=workers)
        slug = model.replace("/", "_")
        pred_path = out_dir / f"llm_gold_{slug}.csv"
        pred.to_csv(pred_path, index=False)
        print("sauvegarde:", pred_path)
        aligned = gold[["StudyInstanceUID"]].merge(pred, on="StudyInstanceUID", how="left")
        st = gold_metrics(gold, aligned)
        stats_by_model[model] = st
        print_table(model, st)

    production = str(cfg.get("production_model", MODEL_PRODUCTION))
    llm_names = [m for m in stats_by_model if m != "lexicon"]
    if len(llm_names) == 1:
        rec = llm_names[0]
    elif production == MODEL_GOLD_PREVIEW:
        rec = production
    else:
        rec = recommend_production(
            stats_by_model, production=production, preview=MODEL_GOLD_PREVIEW,
        )
    print("\n=== Recommandation production (corpus 4349) ===")
    print(f"modele : {rec}")
    lex_s = summarize(lex_stats)
    if rec in stats_by_model:
        llm_s = summarize(stats_by_model[rec])
        print(f"lexique  AUC={lex_s['macro_AUC']:.3f} F1={lex_s['macro_F1']:.3f}")
        print(f"LLM      AUC={llm_s['macro_AUC']:.3f} F1={llm_s['macro_F1']:.3f}")
        if llm_s["macro_AUC"] > lex_s["macro_AUC"]:
            print("Le LLM bat le lexique sur l'or -> python -X utf8 -m src.nlp.label_corpus")
        else:
            print("Le LLM ne bat pas le lexique : ne pas remplacer train_val_labels.csv.")

    dump = {
        "n_gold": int(len(gold)),
        "recommendation": rec,
        "summaries": {k: summarize(v) for k, v in stats_by_model.items()},
        "per_target": {k: v.to_dict(orient="records") for k, v in stats_by_model.items()},
    }
    bakeoff_path = out_dir / "llm_gold_bakeoff.json"
    bakeoff_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print("sauvegarde:", bakeoff_path)
    print("\nEnsuite : python -X utf8 -m src.nlp.label_corpus")


if __name__ == "__main__":
    main()
