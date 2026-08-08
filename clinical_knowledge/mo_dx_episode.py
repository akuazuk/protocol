"""Эпизод диагноза из истории визитов → query для подбора КП.

План: docs/plans/2026-08-08-mo-kp-history-episode-suggest-v1.md

Текущий визит первичен. История только обогащает тот же эпизод
(близкий текст / stem МКБ / алиасы). Чужой прошлый Dx не подменяет текущий.
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping

ENGINE = "mo_dx_episode_v1"
_ICD_STEM_RE = re.compile(r"\b([A-TV-ZА-Яа-я]\d{2})", re.IGNORECASE)


def episode_from_history_enabled() -> bool:
    raw = (os.environ.get("MO_KP_EPISODE_FROM_HISTORY") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _norm_stem(code: str) -> str:
    c = (code or "").strip().upper().replace(",", ".")
    if len(c) >= 3 and c[0].isalpha() and c[1:3].isdigit():
        return c[:3]
    m = _ICD_STEM_RE.search(c)
    return m.group(1).upper() if m else ""


def _visit_text(visit: Mapping[str, Any]) -> str:
    for key in (
        "diagnosis_text",
        "clinical_diagnosis",
        "diagnosis_main_text",
        "diagnosis_short",
        "mis_diagnos",
        "text",
    ):
        val = str(visit.get(key) or "").strip()
        if val:
            return val
    return ""


def _visit_code(visit: Mapping[str, Any]) -> str:
    for key in ("diagnosis_code", "mis_diagnos", "icd10", "mkb_code_main"):
        raw = visit.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        stem = _norm_stem(str(raw or ""))
        if stem:
            return stem
        # код может сидеть в тексте
        stem = _norm_stem(str(raw or ""))
        if stem:
            return stem
    return _norm_stem(_visit_text(visit))


def _current_text(clinical: Mapping[str, Any] | None) -> str:
    clinical = clinical if isinstance(clinical, Mapping) else {}
    from clinical_knowledge.dx_query_expand import strip_icd_tokens

    parts = [
        str(clinical.get(key) or "").strip()
        for key in ("clinical_diagnosis", "diagnosis_main_text", "diagnosis_short", "diagnosis_text")
        if clinical.get(key)
    ]
    joined = " ".join(parts).strip()
    cleaned = strip_icd_tokens(joined)
    return (cleaned or joined).strip()


def _current_stem(clinical: Mapping[str, Any] | None) -> str:
    clinical = clinical if isinstance(clinical, Mapping) else {}
    for key in ("mis_diagnos", "diagnosis_code", "mkb_code_main"):
        stem = _norm_stem(str(clinical.get(key) or ""))
        if stem:
            return stem
    return _norm_stem(_current_text(clinical))


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        from clinical_knowledge.clinical_text_similarity import combined_score

        return float(combined_score(a, b))
    except Exception:  # noqa: BLE001
        pass
    from clinical_knowledge.dx_query_expand import diagnosis_tokens

    ta = set(diagnosis_tokens(a, min_len=4, limit=20))
    tb = set(diagnosis_tokens(b, min_len=4, limit=20))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _alias_overlap(a: str, b: str) -> bool:
    from clinical_knowledge.dx_query_expand import expand_diagnosis_query, matched_alias_phrases

    ea = expand_diagnosis_query(a).lower()
    eb = expand_diagnosis_query(b).lower()
    if not ea or not eb:
        return False
    # общие alias-фразы или сильные маркеры
    if matched_alias_phrases(a) and matched_alias_phrases(b):
        pa = {p.lower() for p in matched_alias_phrases(a)}
        pb = {p.lower() for p in matched_alias_phrases(b)}
        if pa & pb:
            return True
    markers = ("вальгус", "плоско", "стоп", "planus", "пвус", "рефлюкс", "орви")
    return any(m in ea and m in eb for m in markers)


def _iter_history_visits(bundle: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(bundle, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for key in ("same_doctor", "same_specialty", "other"):
        rows = bundle.get(key) or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                out.append(dict(row))
    return out


def resolve_dx_episode_for_suggest(
    *,
    clinical: Mapping[str, Any] | None,
    history_bundle: Mapping[str, Any] | None = None,
    history_visits: list[Mapping[str, Any]] | None = None,
    min_sim: float = 0.42,
) -> dict[str, Any]:
    """Собрать query эпизода для suggest.

    Returns:
      engine, query, current_text, episode_texts, matched_visits, mode
      mode: current_only | enriched | history_fallback | empty
    """
    current = _current_text(clinical)
    cur_stem = _current_stem(clinical)
    visits: list[dict[str, Any]] = []
    if history_visits:
        visits.extend(dict(v) for v in history_visits if isinstance(v, Mapping))
    visits.extend(_iter_history_visits(history_bundle))

    if not episode_from_history_enabled() or not visits:
        mode = "current_only" if current else "empty"
        return {
            "engine": ENGINE,
            "query": current,
            "current_text": current,
            "current_stem": cur_stem,
            "episode_texts": [],
            "matched_visits": [],
            "mode": mode,
        }

    matched: list[dict[str, Any]] = []
    episode_texts: list[str] = []
    for visit in visits:
        text = _visit_text(visit)
        stem = _visit_code(visit)
        same_stem = bool(cur_stem and stem and cur_stem == stem)
        sim = _similarity(current, text) if current and text else 0.0
        alias_ok = _alias_overlap(current, text) if current and text else False
        # пустой current - берём prior с текстом (fallback), иначе только same episode
        if current:
            if not (same_stem or sim >= min_sim or alias_ok):
                continue
        elif not text:
            continue
        matched.append(
            {
                "visit_id": str(visit.get("visit_id") or visit.get("mis_id") or ""),
                "visit_date": str(visit.get("visit_date") or visit.get("date") or "")[:10],
                "diagnosis_text": text[:240],
                "diagnosis_stem": stem,
                "similarity": round(sim, 3),
                "same_stem": same_stem,
                "alias_overlap": alias_ok,
            }
        )
        if text and text not in episode_texts:
            episode_texts.append(text)

    if current and episode_texts:
        # обогащение: current + отличимые формулировки эпизода
        extras = [t for t in episode_texts if _similarity(current, t) < 0.92][:3]
        query = " ".join([current] + extras).strip()
        mode = "enriched"
    elif current:
        query = current
        mode = "current_only"
    elif episode_texts:
        query = episode_texts[0]
        mode = "history_fallback"
    else:
        query = ""
        mode = "empty"

    return {
        "engine": ENGINE,
        "query": query,
        "current_text": current,
        "current_stem": cur_stem,
        "episode_texts": episode_texts[:8],
        "matched_visits": matched[:12],
        "mode": mode,
    }
