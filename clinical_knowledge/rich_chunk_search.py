"""Rich-chunk search: фильтрация, intent-boost, hybrid merge, навигация по разделам."""
from __future__ import annotations

import re
from typing import Any

_PREAMBLE_MARKERS = (
    "постановление министерства",
    "об утверждении",
    "постановляет:",
    "на основании абзаца",
    "министр здравоохранения",
)

_LOW_SIGNAL_TYPES = frozenset({"body", "terms"})

_CHUNK_TYPE_LABELS: dict[str, str] = {
    "diagnostics": "Диагностика",
    "treatment": "Лечение",
    "prevention": "Профилактика",
    "rehabilitation": "Реабилитация",
    "dispensary": "Диспансерное наблюдение",
    "classification": "МКБ / классификация",
    "routing": "Маршрутизация",
    "pharmacotherapy": "Фармакотерапия",
    "algorithm": "Алгоритм",
    "criteria_block": "Показания / критерии",
    "drug_list": "Лекарственные средства",
    "table": "Таблица",
    "appendix": "Приложение",
    "terms": "Термины",
    "body": "Текст",
    "protocol_overview": "Описание протокола",
}

_POP_TO_AUDIENCE: dict[str, frozenset[str]] = {
    "adult": frozenset({"взрослые", "мужчины", "женщины", "пожилые"}),
    "child": frozenset({"дети", "подростки", "новорождённые"}),
    "pediatric": frozenset({"дети", "подростки", "новорождённые"}),
    "pregnant": frozenset({"беременные"}),
}


def is_rich_chunk_row(row: dict[str, Any]) -> bool:
    return bool(row.get("rich_chunk") or row.get("doc_id"))


def should_skip_rich_chunk_row(row: dict[str, Any]) -> bool:
    """Не индексировать служебный мусор из rich_chunks."""
    if not is_rich_chunk_row(row):
        return False
    try:
        from clinical_knowledge.chunk_tags import chunk_usable_for_retrieval

        if not chunk_usable_for_retrieval(row, ambulatory=True):
            return True
    except Exception:
        pass
    if row.get("chunk_is_empty"):
        return True
    text = (row.get("text") or "").strip()
    ctype = (row.get("chunk_type") or "body").strip().lower()
    if ctype != "protocol_overview" and len(text) < 80:
        return True
    if len(text) < 40:
        return True
    head = text[:120].lower()
    if any(m in head for m in _PREAMBLE_MARKERS):
        return True
    if ctype == "terms" and not (row.get("icd10_codes") or row.get("icd10_weights")):
        return True
    if ctype in _LOW_SIGNAL_TYPES and not (row.get("icd10_codes") or []):
        if row.get("is_preamble_filtered") or "постановление" in head:
            return True
    return False


def enrich_lex_source(ch: dict[str, Any]) -> str:
    """Расширенный текст для лексического скоринга."""
    cached = (ch.get("_lex_search") or "").strip()
    if cached:
        return cached
    parts: list[str] = [
        (ch.get("lex_text") or ch.get("text") or "").strip(),
        (ch.get("title") or "").strip(),
    ]
    for key in ("icd10_codes", "population", "lab_tests", "imaging", "keywords"):
        vals = ch.get(key) or []
        if isinstance(vals, list) and vals:
            parts.append(" ".join(str(v) for v in vals[:12]))
    weights = ch.get("icd10_weights") or {}
    if isinstance(weights, dict) and weights:
        parts.append(
            " ".join(f"{k}{int(v)}%" for k, v in list(weights.items())[:10])
        )
    sec = (ch.get("section_title") or "").strip()
    if sec:
        parts.append(sec)
    out = " ".join(p for p in parts if p)
    try:
        import os

        cap = int(os.environ.get("RAG_LEXICAL_MAX_CHARS", "0") or "0")
        if cap > 0 and len(out) > cap:
            out = out[:cap]
    except (TypeError, ValueError):
        pass
    if out:
        ch["_lex_search"] = out
    return out


def detect_query_intent(query: str, icd_codes: list[str] | None = None) -> set[str]:
    """Намерение запроса для буста chunk_type."""
    ql = (query or "").lower()
    intents: set[str] = set()
    if icd_codes:
        intents.add("classification")
    if re.search(r"лечен|терапи|назнач|препарат|операц|хирург", ql):
        intents.add("treatment")
    if re.search(
        r"диагноз|обслед|жалоб|симптом|кашел|температ|бол[ьи]т|одыш|тошн|рвот",
        ql,
    ):
        intents.add("diagnostics")
    if re.search(r"доз|мг|мкг|таблиц|режим|сут\b", ql):
        intents.add("table")
    if re.search(r"показан|противопоказ|критери", ql):
        intents.add("criteria_block")
    if re.search(r"профилакт", ql):
        intents.add("prevention")
    if not intents:
        intents.add("diagnostics")
    return intents


def chunk_type_multiplier(
    query: str,
    ch: dict[str, Any],
    *,
    icd_codes: list[str] | None = None,
) -> float:
    """Множитель score по типу чанка и intent запроса."""
    if not is_rich_chunk_row(ch) and not ch.get("rich_chunk_meta"):
        kind = (ch.get("kind") or ch.get("chunk_type") or "body").strip().lower()
        if kind == "table_block":
            return 1.0
        return 1.0
    ctype = (ch.get("kind") or ch.get("chunk_type") or "body").strip().lower()
    intents = detect_query_intent(query, icd_codes)
    mult = 1.0
    boosts = {
        "diagnostics": ("diagnostics", "criteria_block", "classification", "protocol_overview"),
        "treatment": ("treatment", "pharmacotherapy", "drug_list", "protocol_overview"),
        "table": ("table", "drug_list"),
        "classification": ("classification", "protocol_overview"),
        "criteria_block": ("criteria_block", "diagnostics", "protocol_overview"),
        "prevention": ("prevention", "protocol_overview"),
    }
    for intent in intents:
        if ctype in boosts.get(intent, ()):
            mult *= 1.35
    if ctype == "terms":
        mult *= 0.35
    elif ctype in _LOW_SIGNAL_TYPES and "classification" not in intents:
        mult *= 0.55
    if ctype == "appendix":
        mult *= 0.7
    if ctype == "table" and "table" not in intents:
        mult *= 0.85
    if ctype == "protocol_overview":
        if icd_codes:
            weights = ch.get("icd10_weights") or {}
            icd_set = {str(c).upper() for c in icd_codes}
            overlap = bool(weights) and any(
                str(k).upper() in icd_set
                or any(c.startswith(str(k).upper()[:3]) for c in icd_set if len(str(k)) >= 3)
                for k in weights
            )
            mult *= 2.0 if overlap else 1.48
        else:
            mult *= 1.18
    return mult


def chunk_population_penalty(audience: str | None, ch: dict[str, Any]) -> float:
    """Штраф если population чанка явно не совпадает с аудиторией воронки."""
    if not audience:
        return 1.0
    pops = ch.get("chunk_population") or ch.get("population") or []
    if not isinstance(pops, list) or not pops:
        return 1.0
    pop_set = {str(p).lower() for p in pops}
    aud = audience.strip().lower()
    allowed = _POP_TO_AUDIENCE.get(aud)
    if not allowed:
        return 1.0
    if pop_set & allowed:
        return 1.0
    child_markers = _POP_TO_AUDIENCE["child"]
    adult_markers = _POP_TO_AUDIENCE["adult"]
    if aud in ("adult",) and pop_set & child_markers and not pop_set & adult_markers:
        return 0.35
    if aud in ("child", "pediatric") and pop_set & adult_markers and not pop_set & child_markers:
        return 0.35
    if aud == "pregnant" and "беременные" not in pop_set and pop_set:
        return 0.5
    return 1.0


def build_chunk_match_reason(row: dict[str, Any], icd_codes: list[str] | None = None) -> str:
    """Короткая причина совпадения для UI (≤70 символов)."""
    parts: list[str] = []
    ctype = (row.get("kind") or row.get("chunk_type") or "").strip().lower()
    label = _CHUNK_TYPE_LABELS.get(ctype, "")
    if label and ctype not in ("body",):
        parts.append(label)
    sec = (row.get("section_title") or "").strip()
    if sec and len(sec) < 40 and sec.upper() != sec:
        parts.append(sec[:35])
    elif sec and len(sec) < 40:
        parts.append(sec[:35])
    chunk_icd = row.get("icd10_codes") or []
    if icd_codes and chunk_icd:
        overlap = [c for c in chunk_icd if c.upper() in {x.upper() for x in icd_codes}]
        if overlap:
            parts.append("МКБ " + overlap[0])
    pf = int(row.get("page_from") or 0)
    if pf:
        parts.append(f"стр. {pf}")
    if not parts:
        return "Совпадение по тексту протокола"
    reason = ", ".join(parts)
    return reason[:70]


def aggregate_retrieval_by_path(
    rows: list[dict[str, Any]],
    *,
    icd_codes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Один лучший чанк на path + match_reason."""
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        p = str(row.get("path") or row.get("catalog_source_path") or "").replace("\\", "/")
        if not p:
            continue
        sc = float(row.get("score") or row.get("lexical_score") or 0.0)
        prev = by_path.get(p)
        if prev is None or sc > float(prev.get("score") or 0):
            enriched = dict(row)
            enriched["path"] = p
            enriched["score"] = sc
            enriched["match_reason"] = build_chunk_match_reason(row, icd_codes)
            by_path[p] = enriched
    out = list(by_path.values())
    out.sort(key=lambda r: -float(r.get("score") or 0))
    return out


def hybrid_merge_protocols(
    icd_protocols: list[dict[str, Any]],
    rag_protocols: list[dict[str, Any]],
    *,
    icd_weight: float = 0.4,
    rag_weight: float = 0.6,
) -> list[dict[str, Any]]:
    """Объединить ICD fast lookup и RAG ranking по path."""
    by_path: dict[str, dict[str, Any]] = {}

    def _conf(pr: dict[str, Any]) -> float:
        try:
            return float(pr.get("confidence_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    for pr in icd_protocols:
        p = str(pr.get("path") or "").replace("\\", "/")
        if not p:
            continue
        by_path[p] = dict(pr)
        by_path[p]["icd_score"] = _conf(pr)
        by_path[p]["rag_score"] = 0.0

    for pr in rag_protocols:
        p = str(pr.get("path") or "").replace("\\", "/")
        if not p:
            continue
        prev = by_path.get(p)
        rag_sc = _conf(pr)
        if prev:
            prev["rag_score"] = rag_sc
            if pr.get("match_reason") and not str(prev.get("match_reason") or "").startswith("МКБ"):
                prev["match_reason"] = pr.get("match_reason")
            if pr.get("section_title"):
                prev["section_title"] = pr.get("section_title")
            if pr.get("page_from"):
                prev["page_from"] = pr.get("page_from")
            icd_sc = float(prev.get("icd_score") or 0.0)
            merged = icd_weight * icd_sc + rag_weight * rag_sc
            prev["confidence_score"] = round(min(0.97, max(0.35, merged)), 4)
            prev["hybrid"] = True
        else:
            entry = dict(pr)
            entry["icd_score"] = 0.0
            entry["rag_score"] = rag_sc
            entry["hybrid"] = True
            by_path[p] = entry

    out = list(by_path.values())
    for pr in out:
        if "confidence_score" not in pr or not pr.get("hybrid"):
            icd_sc = float(pr.get("icd_score") or _conf(pr))
            rag_sc = float(pr.get("rag_score") or 0.0)
            if icd_sc and rag_sc:
                pr["confidence_score"] = round(
                    min(0.97, max(0.35, icd_weight * icd_sc + rag_weight * rag_sc)), 4
                )
            elif icd_sc:
                pr["confidence_score"] = round(icd_sc, 4)
            elif rag_sc:
                pr["confidence_score"] = round(rag_sc, 4)

    out.sort(key=lambda x: -float(x.get("confidence_score") or 0))
    return out


def hybrid_pin_trusted_icd_top1(
    merged: list[dict[str, Any]],
    icd_protocols: list[dict[str, Any]],
    *,
    query: str,
    ambiguous: bool = False,
    icd_codes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Не отдавать RAG top-1 поверх доверенного ICD fast lookup."""
    if ambiguous or not merged or not icd_protocols:
        return merged
    from clinical_knowledge.protocol_icd_index import icd_fast_lookup_trusted

    lookup_result = {"protocols": icd_protocols}
    if not icd_fast_lookup_trusted(query, lookup_result, icd_codes=icd_codes):
        return merged
    icd_path = str(icd_protocols[0].get("path") or "").replace("\\", "/")
    if not icd_path:
        return merged
    idx = next(
        (i for i, pr in enumerate(merged) if str(pr.get("path") or "").replace("\\", "/") == icd_path),
        None,
    )
    if idx is None or idx == 0:
        return merged
    out = list(merged)
    pinned = dict(out[idx])
    pinned["hybrid_icd_pinned"] = True
    out.pop(idx)
    return [pinned] + out


def query_wants_tables(query: str) -> bool:
    ql = (query or "").lower()
    return bool(re.search(r"доз|мг|мкг|таблиц|режим|сут\b|мл\b", ql))


def filter_table_chunks_for_paths(
    chunks: list[dict[str, Any]],
    paths: list[str],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Дополнительные table-чанки для top протоколов."""
    path_set = {p.replace("\\", "/") for p in paths if p}
    out: list[dict[str, Any]] = []
    for ch in chunks:
        p = str(ch.get("path") or "").replace("\\", "/")
        if p not in path_set:
            continue
        kind = (ch.get("kind") or ch.get("chunk_type") or "").strip().lower()
        if kind not in ("table", "table_block"):
            continue
        out.append(ch)
        if len(out) >= limit:
            break
    return out


_SECTION_TYPE_ORDER = [
    "classification",
    "diagnostics",
    "criteria_block",
    "treatment",
    "pharmacotherapy",
    "drug_list",
    "table",
    "prevention",
    "dispensary",
    "rehabilitation",
    "routing",
    "algorithm",
    "appendix",
]


def build_rich_protocol_nav(
    chunks_for_path: list[dict[str, Any]],
    *,
    path: str,
    query: str = "",
    icd_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Навигация по rich-чанкам (fallback если нет Summary YAML)."""
    if not chunks_for_path:
        return {"available": False, "source": "rich_chunks", "path": path}

    by_section: dict[str, dict[str, Any]] = {}
    for ch in chunks_for_path:
        ctype = (ch.get("kind") or ch.get("chunk_type") or "body").strip().lower()
        if ctype in _LOW_SIGNAL_TYPES:
            continue
        sec = (ch.get("section_title") or "").strip()
        if not sec or len(sec) > 120:
            sec = _CHUNK_TYPE_LABELS.get(ctype, ctype)
        key = f"{ctype}::{sec}"
        bucket = by_section.setdefault(
            key,
            {
                "id": key,
                "label": sec,
                "chunk_type": ctype,
                "count": 0,
                "items": [],
            },
        )
        bucket["count"] += 1
        text = (ch.get("text") or "")[:1200]
        if not text:
            continue
        bucket["items"].append(
            {
                "chunk_id": ch.get("chunk_id"),
                "text": text,
                "page_from": ch.get("page_from"),
                "page_to": ch.get("page_to"),
                "section_title": sec,
            }
        )

    sections = list(by_section.values())
    sections.sort(
        key=lambda s: (
            _SECTION_TYPE_ORDER.index(s["chunk_type"])
            if s["chunk_type"] in _SECTION_TYPE_ORDER
            else 99,
            s["label"],
        )
    )

    if query or icd_codes:
        ql = (query or "").lower()
        icd_set = {c.upper() for c in (icd_codes or [])}

        def _sec_score(sec: dict[str, Any]) -> float:
            sc = 0.0
            ct = sec.get("chunk_type") or ""
            intents = detect_query_intent(query, icd_codes)
            if ct in intents:
                sc += 2.0
            for item in sec.get("items") or []:
                t = (item.get("text") or "").lower()
                if ql and any(w in t for w in ql.split()[:6] if len(w) > 4):
                    sc += 0.5
            return sc

        sections.sort(key=lambda s: (-_sec_score(s), s["label"]))

    if not sections:
        return {"available": False, "source": "rich_chunks", "path": path}

    condition_id = "rich_default"
    return {
        "available": True,
        "source": "rich_chunks",
        "path": path,
        "conditions": [
            {
                "condition_id": condition_id,
                "name": "Разделы протокола",
                "sections": [
                    {
                        "id": s["id"],
                        "label": s["label"],
                        "count": s["count"],
                        "chunk_type": s["chunk_type"],
                    }
                    for s in sections[:20]
                ],
            }
        ],
        "_sections_full": sections,
    }


def build_rich_section_excerpt(
    nav: dict[str, Any],
    *,
    condition_id: str,
    section_id: str,
) -> dict[str, Any]:
    """Выдержка для шага 7 из rich-nav."""
    sections = nav.get("_sections_full") or []
    sec = next((s for s in sections if s.get("id") == section_id), None)
    if not sec:
        return {"items": [], "text": ""}
    items = sec.get("items") or []
    merged = "\n\n".join((it.get("text") or "").strip() for it in items[:5] if it.get("text"))
    first = items[0] if items else {}
    return {
        "items": items[:8],
        "text": merged,
        "source_ref": {
            "page_from": first.get("page_from"),
            "page_to": first.get("page_to"),
            "section_title": first.get("section_title") or sec.get("label"),
        },
    }
