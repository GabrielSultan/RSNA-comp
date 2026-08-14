# -*- coding: utf-8 -*-
"""Labellisation des 12 cibles RSNA Knee a partir d'un rapport, via Gemini.

Offline uniquement (jamais au submit). JSON structure, temperature 0.
Les cles JSON n'ont pas d'apostrophe (Bakers) ; mapping vers TARGETS ensuite.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.nlp.lexicon import TARGETS

ROOT = Path(__file__).resolve().parents[2]

# gemini-2.5-pro est ferme aux nouvelles cles (404 "no longer available to
# new users"). Remplacement officiel : gemini-3.1-pro-preview (payant, pas de free tier).
MODEL_PRODUCTION = "gemini-3.1-pro-preview"
MODEL_GOLD_PREVIEW = "gemini-3.1-pro-preview"

# Cles JSON (ASCII, pas d'apostrophe) -> colonnes competition
JSON_TO_TARGET = {
    "ACL": "ACL",
    "MCL": "MCL",
    "MedialMeniscus": "Medial Meniscus",
    "LateralMeniscus": "Lateral Meniscus",
    "MedialOA": "Medial OA",
    "LateralOA": "Lateral OA",
    "PFOA": "PF OA",
    "Effusion": "Effusion",
    "Synovitis": "Synovitis",
    "Bakers": "Baker's",
    "Contusion": "Contusion",
    "Fracture": "Fracture",
}
JSON_KEYS = list(JSON_TO_TARGET.keys())

Present = Literal["yes", "no", "unknown"]
PRESENT_TO_LABEL = {"yes": 1, "no": 0, "unknown": None}

LLM_CONF_DEFAULT = 0.90
MAX_REPORT_CHARS = 20_000


class Finding(BaseModel):
    present: Present = Field(
        description="yes = finding present, no = explicitly absent/normal, "
                    "unknown = report does not address this finding"
    )
    evidence: str = Field(
        default="",
        description="Short quote copied from the report in its original language",
    )


class KneeReportLabels(BaseModel):
    ACL: Finding
    MCL: Finding
    MedialMeniscus: Finding
    LateralMeniscus: Finding
    MedialOA: Finding
    LateralOA: Finding
    PFOA: Finding
    Effusion: Finding
    Synovitis: Finding
    Bakers: Finding
    Contusion: Finding
    Fracture: Finding


FINDING_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "present": {"type": "STRING", "enum": ["yes", "no", "unknown"]},
        "evidence": {"type": "STRING"},
    },
    "required": ["present", "evidence"],
}

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {k: FINDING_SCHEMA for k in JSON_KEYS},
    "required": JSON_KEYS,
}


# Prompt fige : ne pas iterer sur les 58 gold.
SYSTEM_PROMPT = """You are a musculoskeletal radiologist extracting binary labels from a knee MRI report.

Task: for each of 12 findings, decide if it is present in THIS report.
Read the report in its original language. Do NOT translate it. Do NOT use information that is not in the text.

Which text to use
- Use Findings, Impression, Conclusion, Description of lesions.
- Ignore Indication / Clinical history / Technique / Protocol / Comparison, unless that is the only text available.
- If the whole exam is described as normal / unremarkable / no significant abnormality, label every finding "no".

How to fill "present"
- "yes" : the finding is described as present (any severity: trace / mild / small counts as yes).
- "no"  : the finding is explicitly absent, or the relevant structure is called normal / intact / preserved.
- "unknown" : the report does not address this finding. Do not guess from anatomy that was never mentioned.

"evidence" : copy a short quote from the report (original language). Empty string only if present="unknown".

Definitions (binary, competition labels)

ACL
  yes = tear (partial or complete) OR sprain / injury of the anterior cruciate ligament.
  no  = intact / normal ACL. Mucoid degeneration or ganglion WITHOUT tear or sprain = no.

MCL
  yes = injury of the medial collateral ligament (sprain, partial or complete tear, thickening from injury).
  no  = normal / intact MCL. Do not confuse with LCL / lateral collateral.

MedialMeniscus / LateralMeniscus
  yes = tear, rupture, maceration, or displaced fragment of that meniscus (horn, body, or root).
  no  = intact meniscus, or degenerative/mucoid signal that does NOT reach the articular surface and is not called a tear.
  Grade I or II intra-meniscal signal without tear = no.

MedialOA
  yes = osteoarthritis / chondropathy / cartilage loss / chondromalacia / osteophytes / joint-space narrowing of the MEDIAL tibiofemoral compartment.
  no  = medial cartilage described as preserved, or OA explicitly denied medially.

LateralOA
  Same as MedialOA but LATERAL tibiofemoral compartment only.
  Tricompartmental / generalized knee OA = yes for every involved compartment that is named or clearly included. If the report only says "medial OA", do not mark LateralOA yes.

PFOA
  yes = patellofemoral osteoarthritis, patellar / trochlear chondromalacia or cartilage loss, PF chondropathy.
  Isolated plica or chondromalacia of another compartment does not count.

Effusion
  yes = any joint effusion / increased intra-articular fluid / hemarthrosis, including trace or small.
  Physiological / "normal amount of fluid" = no.

Synovitis
  yes = synovitis, synovial hypertrophy / proliferation / thickening.
  Joint effusion alone is NOT synovitis.

Bakers  (Baker's cyst / popliteal cyst)
  yes = Baker's cyst or popliteal cyst, any size.
  Other cysts (parameniscal, ganglion, bursitis that is not popliteal) = no for this label.

Contusion
  yes = bone contusion / bone bruise / traumatic bone-marrow edema.
  no  = subchondral / degenerative bone-marrow edema clearly attributed to osteoarthritis, chondral wear, or chronic overload, WITHOUT wording of contusion / bruise / trauma.
  If the report only says "bone marrow edema" with no traumatic or degenerative context, use "yes" only if it reads as an acute osseous injury; otherwise "unknown".

Fracture
  yes = any fracture (acute, occult, insufficiency, stress, osteochondral fracture).
  Bone bruise without a fracture line = no for Fracture (it may still be Contusion yes).
  Post-surgical changes without fracture = no.

Rules of thumb
- A sentence that lists several ligaments as normal applies to each named ligament.
- "Menisci normal" applies to BOTH menisci.
- Laterality of the KNEE (left/right) is irrelevant; laterality of COMPARTMENT (medial/lateral) is essential.
- If two sentences conflict, prefer Impression / Conclusion over the body.
- Output MUST match the JSON schema. Every key required. present in {yes, no, unknown} only.
"""


def user_prompt(report: str) -> str:
    text = (report or "").strip()
    if len(text) > MAX_REPORT_CHARS:
        text = text[:MAX_REPORT_CHARS] + "\n[TRUNCATED]"
    return (
        "Knee MRI report below. Extract the 12 findings.\n\n"
        "<<<REPORT\n"
        f"{text}\n"
        "REPORT"
    )


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as e:
        raise SystemExit(
            "python-dotenv manquant. pip install python-dotenv"
        ) from e
    load_dotenv(ROOT / ".env", override=False)


def get_api_key() -> str | None:
    load_env()
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key or key.startswith("YOUR_") or key.lower() in {"xxx", "changeme", "todo"}:
        return None
    return key


def require_api_key() -> str:
    key = get_api_key()
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY absente. Ouvre .env a la racine du repo et colle la cle "
            "(https://aistudio.google.com/apikey)."
        )
    return key


def cache_dir_for(model: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model)
    return ROOT / "data" / "labels" / "llm_cache" / slug


def cache_path(model: str, study_uid: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", study_uid)
    return cache_dir_for(model) / f"{safe}.json"


def load_cached(model: str, study_uid: str) -> dict[str, Any] | None:
    path = cache_path(model, study_uid)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_cached(model: str, study_uid: str, payload: dict[str, Any]) -> None:
    path = cache_path(model, study_uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def parsed_to_row(parsed: KneeReportLabels, conf: float = LLM_CONF_DEFAULT) -> dict[str, Any]:
    """Colonnes TARGETS + TARGETS_conf + evidence JSON."""
    row: dict[str, Any] = {}
    evidence: dict[str, str] = {}
    data = parsed.model_dump()
    for jkey, target in JSON_TO_TARGET.items():
        finding = data[jkey]
        present = finding["present"]
        label = PRESENT_TO_LABEL.get(present)
        row[target] = None if label is None else float(label)
        row[target + "_conf"] = 0.0 if label is None else float(conf)
        evidence[target] = finding.get("evidence") or ""
    row["_evidence"] = evidence
    return row


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_response_text(text: str) -> KneeReportLabels:
    return KneeReportLabels.model_validate_json(_strip_fences(text))


def http_status(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and 100 <= v <= 599:
            return v
    msg = str(exc)
    m = re.match(r"^(\d{3})\b", msg)
    if m:
        return int(m.group(1))
    return None


def is_model_unavailable(exc: BaseException) -> bool:
    if http_status(exc) == 404:
        return True
    msg = str(exc).lower()
    return "no longer available" in msg or "not_found" in msg


def _is_retryable(exc: BaseException) -> bool:
    if is_model_unavailable(exc):
        return False
    status = http_status(exc)
    if status in {400, 401, 403, 404}:
        return False
    try:
        from pydantic import ValidationError
        if isinstance(exc, (json.JSONDecodeError, ValidationError)):
            return True
    except ImportError:
        if isinstance(exc, json.JSONDecodeError):
            return True
    if status in {408, 429, 500, 502, 503, 504}:
        return True
    msg = str(exc).lower()
    needles = (
        "resource_exhausted", "unavailable", "deadline", "timeout",
        "429", "503", "500", "rate limit", "temporarily",
        "connection", "reset", "json", "empty gemini",
    )
    return any(n in msg for n in needles)


def list_generate_models(client) -> list[str]:
    """IDs utilisables en generateContent (sans le prefixe models/)."""
    names: list[str] = []
    for m in client.models.list():
        name = getattr(m, "name", "") or ""
        slug = name.split("/")[-1]
        methods = set(getattr(m, "supported_actions", None) or [])
        if methods and "generateContent" not in methods:
            continue
        if slug:
            names.append(slug)
    return sorted(set(names))


def make_client(timeout_ms: int = 180_000):
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise SystemExit(
            "google-genai manquant. pip install 'google-genai>=1.40' pydantic python-dotenv"
        ) from e
    key = require_api_key()
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )


def generate_labels(
    client,
    model: str,
    report: str,
    *,
    temperature: float = 0.0,
    seed: int | None = 0,
    thinking_budget: int | None = None,
    max_output_tokens: int = 8192,
    max_retries: int = 6,
) -> tuple[KneeReportLabels, dict[str, Any]]:
    """Un appel Gemini -> schema valide + meta (usage, retries)."""
    from google.genai import types

    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "response_mime_type": "application/json",
        "response_json_schema": RESPONSE_JSON_SCHEMA,
        "system_instruction": SYSTEM_PROMPT,
    }
    if seed is not None:
        config_kwargs["seed"] = seed
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=False,
        )
    config = types.GenerateContentConfig(**config_kwargs)
    contents = user_prompt(report)

    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            text = response.text or ""
            if not text.strip():
                fr = None
                if getattr(response, "candidates", None):
                    fr = getattr(response.candidates[0], "finish_reason", None)
                raise ValueError(f"empty Gemini response (finish_reason={fr})")
            parsed = parse_response_text(text)
            usage = {}
            um = getattr(response, "usage_metadata", None)
            if um is not None:
                usage = {
                    "prompt_tokens": getattr(um, "prompt_token_count", None),
                    "candidates_tokens": getattr(um, "candidates_token_count", None),
                    "total_tokens": getattr(um, "total_token_count", None),
                    "thoughts_tokens": getattr(um, "thoughts_token_count", None),
                }
            return parsed, {"usage": usage, "attempts": attempt + 1, "raw_text": text}
        except Exception as exc:  # noqa: BLE001 — retry reseau / parse
            last_exc = exc
            if attempt + 1 >= max_retries or not _is_retryable(exc):
                raise
            sleep = (2 ** attempt) + random.random()
            time.sleep(min(sleep, 60.0))
    raise RuntimeError(f"Gemini a echoue apres {max_retries} essais: {last_exc}")


def label_study(
    client,
    model: str,
    study_uid: str,
    report: str,
    *,
    force: bool = False,
    llm_conf: float = LLM_CONF_DEFAULT,
    **gen_kwargs: Any,
) -> dict[str, Any]:
    """Cache-first. Retourne un dict serialisable (labels + conf + evidence + meta)."""
    if not force:
        cached = load_cached(model, study_uid)
        if cached is not None and "labels" in cached:
            return cached
    parsed, meta = generate_labels(client, model, report, **gen_kwargs)
    row = parsed_to_row(parsed, conf=llm_conf)
    payload = {
        "model": model,
        "StudyInstanceUID": study_uid,
        "labels": {t: row[t] for t in TARGETS},
        "conf": {t: row[t + "_conf"] for t in TARGETS},
        "evidence": row["_evidence"],
        "usage": meta.get("usage"),
        "attempts": meta.get("attempts"),
    }
    save_cached(model, study_uid, payload)
    return payload


def payload_to_flat(payload: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {"StudyInstanceUID": payload["StudyInstanceUID"]}
    labels = payload.get("labels") or {}
    conf = payload.get("conf") or {}
    for t in TARGETS:
        v = labels.get(t)
        flat[t] = None if v is None else float(v)
        flat[t + "_conf"] = float(conf.get(t, 0.0) or 0.0)
    return flat
