# -*- coding: utf-8 -*-
"""Extracteur de labels : rapport radiologique -> 12 cibles binaires + confiance.

Logique :
1. Normalisation dependante de la langue.
2. Decoupage en clauses (la negation ne traverse pas la ponctuation).
3. Par clause et par cible :
   - cible "lesion pure" : match -> vote 1, sauf si negateur dans la fenetre
     directionnelle (gauche, ou droite pour le turc) ou mot de normalite proche.
   - cible "anatomique" (ligaments/menisques/OA) : vote 1 si un mot-lesion est
     attribue a cette anatomie (plus proche anatomie gagne), vote 0 si
     normalite/negation proche, rien sinon.
4. Agregation des votes de clauses -> label (0/1/None) + confiance [0, 1].
   None = pas de signal -> la cible sera masquee dans la loss vision.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .lexicon import (
    ANATOMY_TARGETS,
    ANATOMY_TERMS,
    LANGS,
    LESION_TARGETS,
    LESION_TERMS,
    NEGATORS,
    NORMAL_WORDS,
    OA_ANATOMY,
    OA_LESION,
    OA_TARGETS,
    RIGHT_NEGATION,
    SECTION_RESUME,
    SECTION_SKIP,
    TARGETS,
    TEAR_WORDS,
    UNCERTAIN_WORDS,
)

NEG_WINDOW = 4        # mots (directionnel)
NORMAL_WINDOW = 6     # mots (bidirectionnel)
UNCERTAIN_WINDOW = 5  # mots (bidirectionnel)
TEAR_MAX_DIST = 80    # caracteres max anatomie <-> lesion

CLAUSE_RE = re.compile(r"[\n\r.!?;•|]+")
WORD_RE = re.compile(r"\w+")


def normalize(text: str, lang: str) -> str:
    """NFKC + casse + gestion des diacritiques (conserves en turc)."""
    text = unicodedata.normalize("NFKC", text or "")
    if lang == "tr":
        text = text.replace("İ", "i").replace("I", "ı").lower()
    else:
        text = text.casefold()
        text = "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
        if lang == "el":
            # casefold() unifie le sigma final -> sigma median ; aligner le texte
            text = text.replace("ς", "σ")
    return text


def _compile(patterns: list[str]) -> list[re.Pattern]:
    # (?<!\w) evite les matchs en sous-mot (ex. "αρθρικη" dans "ενδαρθρικη")
    return [re.compile(r"(?<!\w)" + p) for p in patterns]


# Compilation unique au chargement du module
_COMPILED: dict = {}


def _get(kind: str, key: str, lang: str) -> list[re.Pattern]:
    k = (kind, key, lang)
    if k not in _COMPILED:
        table = {
            "neg": NEGATORS,
            "normal": NORMAL_WORDS,
            "uncertain": UNCERTAIN_WORDS,
            "tear": TEAR_WORDS,
            "oa_lesion": OA_LESION,
            "lesion": LESION_TERMS.get(key, {}),
            "anatomy": ANATOMY_TERMS.get(key, {}),
            "oa_anatomy": OA_ANATOMY.get(key, {}),
        }[kind]
        pats = table.get(lang, []) if isinstance(table, dict) else []
        _COMPILED[k] = _compile(pats)
    return _COMPILED[k]


@dataclass
class Vote:
    value: int        # 0 ou 1
    weight: float     # 1.0, reduit si incertain


def _word_index(words: list[tuple[int, int]], pos: int) -> int:
    """Index du mot couvrant la position caractere `pos` (ou le plus proche)."""
    for i, (s, e) in enumerate(words):
        if s <= pos < e:
            return i
        if s > pos:
            return i
    return len(words) - 1


def _window_text(clause: str, words: list[tuple[int, int]], span: tuple[int, int],
                 n_before: int, n_after: int) -> tuple[str, str]:
    """Texte des fenetres avant/apres le match, en nombre de mots."""
    i0 = _word_index(words, span[0])
    i1 = _word_index(words, span[1] - 1)
    left_start = words[max(0, i0 - n_before)][0] if words else 0
    right_end = words[min(len(words) - 1, i1 + n_after)][1] if words else len(clause)
    return clause[left_start:span[0]], clause[span[1]:right_end]


def _any_match(patterns: list[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _is_negated(clause: str, words: list[tuple[int, int]], span: tuple[int, int],
                lang: str) -> bool:
    """Negateur directionnel OU mot de normalite a proximite du match."""
    left, right = _window_text(clause, words, span, NEG_WINDOW, NEG_WINDOW)
    negators = _get("neg", "", lang)
    if lang in RIGHT_NEGATION:
        if _any_match(negators, right):
            return True
    else:
        if _any_match(negators, left):
            return True
    left_n, right_n = _window_text(clause, words, span, NORMAL_WINDOW, NORMAL_WINDOW)
    normals = _get("normal", "", lang)
    if _any_match(normals, left_n) or _any_match(normals, right_n):
        return True
    return False


def _uncertainty_factor(clause: str, words: list[tuple[int, int]],
                        span: tuple[int, int], lang: str) -> float:
    left, right = _window_text(clause, words, span, UNCERTAIN_WINDOW, UNCERTAIN_WINDOW)
    uncertain = _get("uncertain", "", lang)
    if _any_match(uncertain, left) or _any_match(uncertain, right):
        return 0.6
    return 1.0


def _votes_lesion(clause: str, words: list[tuple[int, int]], label: str,
                  lang: str) -> list[Vote]:
    votes = []
    for pat in _get("lesion", label, lang):
        for m in pat.finditer(clause):
            if _is_negated(clause, words, m.span(), lang):
                votes.append(Vote(0, 1.0))
            else:
                w = _uncertainty_factor(clause, words, m.span(), lang)
                votes.append(Vote(1, w))
    return votes


def _votes_anatomy(clause: str, words: list[tuple[int, int]], label: str,
                   lang: str) -> list[Vote]:
    is_oa = label in OA_TARGETS
    anatomy_pats = _get("oa_anatomy" if is_oa else "anatomy", label, lang)
    lesion_pats = _get("oa_lesion" if is_oa else "tear", "", lang)

    anatomy_matches = []
    for pat in anatomy_pats:
        anatomy_matches.extend(pat.finditer(clause))
    lesion_matches = []
    for pat in lesion_pats:
        lesion_matches.extend(pat.finditer(clause))

    votes = []
    for am in anatomy_matches:
        if _is_negated(clause, words, am.span(), lang):
            votes.append(Vote(0, 1.0))
            continue
        # lesion la plus proche de cette anatomie
        best, best_d = None, TEAR_MAX_DIST + 1
        for lm in lesion_matches:
            d = min(abs(lm.start() - am.end()), abs(am.start() - lm.end()))
            if d < best_d:
                best, best_d = lm, d
        if best is not None and best_d <= TEAR_MAX_DIST:
            # attribution : une autre anatomie est-elle plus proche de cette lesion ?
            closer = False
            for other in anatomy_matches:
                if other is am:
                    continue
                d2 = min(abs(best.start() - other.end()), abs(other.start() - best.end()))
                if d2 < best_d:
                    closer = True
                    break
            if not closer:
                # la lesion elle-meme ne doit pas etre niee
                if not _is_negated(clause, words, best.span(), lang):
                    w = _uncertainty_factor(clause, words, best.span(), lang)
                    votes.append(Vote(1, w))
    return votes


def _aggregate(votes: list[Vote]) -> tuple[int | None, float]:
    if not votes:
        return None, 0.0
    pos = sum(v.weight for v in votes if v.value == 1)
    neg = sum(v.weight for v in votes if v.value == 0)
    if pos > 0 and neg == 0:
        return 1, min(0.95, 0.75 + 0.1 * pos)
    if neg > 0 and pos == 0:
        return 0, min(0.95, 0.75 + 0.1 * neg)
    label = 1 if pos >= neg else 0
    return label, 0.5


def _strip_history_sections(norm_text: str, lang: str) -> str:
    """Retire les lignes des sections antecedents/indication/technique.

    Le saut demarre sur un en-tete SECTION_SKIP en debut de ligne et s'arrete
    sur un en-tete SECTION_RESUME. Sans en-tetes recognisables, rien n'est
    retire (safe pour les rapports en texte libre).
    """
    skip_pats = [re.compile(r"^\s*" + p) for p in SECTION_SKIP.get(lang, [])]
    resume_pats = [re.compile(r"^\s*" + p) for p in SECTION_RESUME.get(lang, [])]
    if not skip_pats:
        return norm_text
    kept, skipping, skip_count = [], False, 0
    for line in norm_text.split("\n"):
        if skipping:
            # reprise sur en-tete de findings, ligne vide, ou apres 3 lignes
            if (any(p.search(line) for p in resume_pats)
                    or not line.strip() or skip_count >= 3):
                skipping = False
            else:
                skip_count += 1
                continue
        elif any(p.search(line) for p in skip_pats):
            skipping = True
            skip_count = 0
            continue
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


# Exclusion des sections antecedents : testee (commit precedent), DEGRADE le
# gold-AUC (0.734 vs 0.762) — trop de rapports courts perdent leurs findings.
# Desactivee par defaut, code garde pour reference.
STRIP_SECTIONS = False


def extract_text(text: str, lang: str) -> dict[str, tuple[int | None, float]]:
    """Un rapport deja detecte -> {cible: (label ou None, confiance)}."""
    out: dict[str, tuple[int | None, float]] = {}
    if lang not in LANGS:
        return {t: (None, 0.0) for t in TARGETS}

    norm = normalize(text, lang)
    if STRIP_SECTIONS:
        norm = _strip_history_sections(norm, lang)
    clauses = [c for c in CLAUSE_RE.split(norm) if c.strip()]

    for label in TARGETS:
        votes: list[Vote] = []
        for clause in clauses:
            words = [(m.start(), m.end()) for m in WORD_RE.finditer(clause)]
            if not words:
                continue
            if label in LESION_TARGETS:
                votes.extend(_votes_lesion(clause, words, label, lang))
            elif label in ANATOMY_TARGETS or label in OA_TARGETS:
                votes.extend(_votes_anatomy(clause, words, label, lang))
        out[label] = _aggregate(votes)
    return out


def extract_study(text: str, lang: str) -> dict[str, tuple[int | None, float]]:
    """Alias de haut niveau (un rapport = une etude)."""
    return extract_text(text, lang)
