"""Подбор протоколов МЗ РБ к случаю МО (не L1-балл оформления).

См. docs/plans/2026-08-05-mo-case-protocol-suggest-v1.md и
docs/plans/2026-08-07-mo-dx-text-suggest-icd-directory-eval-v1.md.

Suggest ищет КП по установленному диагнозу (текст), не по МКБ.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from clinical_knowledge.protocol_links import protocol_display_name, protocol_nav_api_path

ENGINE = "case_protocol_suggest_v3"
MATCH_KIND_LABELS = {
    "clinical": "Клиника",
    "ddx": "Дифдиагноз",
    "specialty": "Специальность",
}

# Замечания, которые не должны попадать в reasons / gaps suggest
# (safety/doc gaps тащат чужие КП, напр. ЧЛХ по C_nsaid_dup).
_GAP_SKIP_PREFIXES = (
    "D_reg55",
    "E_template",
    "A_missing_",
    "C_nsaid",
    "C_ddi",
    "C_high_alert",
    "C_drug",
    "B_icd",
    "MED_",
    "completeness_ru",
    "stage_a_ru",
    "stage_b_ru",
)

# Жёсткий блок путей каталога по специальности случая.
_SPECIALTY_PATH_BLOCK: dict[str, tuple[str, ...]] = {
    "urolog": (
        "stomatolog",
        "chelust",
        "челюст",
        "zabolevaniya_chelust",
        "zub",
        "области рта",
        "область рта",
        "члх",
    ),
    "уролог": (
        "stomatolog",
        "chelust",
        "челюст",
        "zabolevaniya_chelust",
        "zub",
        "области рта",
        "область рта",
        "члх",
    ),
    "neurolog": ("stomatolog", "chelust", "akusher", "ginekolog", "челюст", "члх"),
    "невролог": ("stomatolog", "chelust", "akusher", "ginekolog", "челюст", "члх"),
}


def suggest_enabled() -> bool:
    raw = (os.environ.get("CASE_PROTOCOL_SUGGEST") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _gap_allowed(code: str) -> bool:
    cid = str(code or "").strip()
    if not cid:
        return False
    return not any(cid.startswith(prefix) for prefix in _GAP_SKIP_PREFIXES)


def _suggest_title(source_path: str | None, registry_title: str | None) -> str:
    return protocol_display_name(
        source_path,
        fallback=str(registry_title or "") or "Протокол МЗ",
        registry_title=registry_title,
        prefer_filename_if_truncated=True,
    )


def _diagnosis_text(clinical: dict[str, Any]) -> str:
    return " ".join(
        str(clinical.get(key) or "")
        for key in ("clinical_diagnosis", "mis_diagnos", "diagnosis_main_text", "diagnosis_short")
        if clinical.get(key)
    ).strip()


def _search_query(graph: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in graph.get("diagnoses") or []:
        text = str(item.get("text") or "").strip()
        if text:
            parts.append(text[:160])
    for complaint in (graph.get("complaints") or [])[:2]:
        if complaint and not parts:
            parts.append(str(complaint)[:80])
    specialty = str((graph.get("specialty") or {}).get("label") or "").strip()
    if specialty and not parts:
        parts.append(specialty)
    query = " ".join(parts).strip()
    return query[:160] or "клинический протокол"


def _search_url(query: str) -> str:
    return "/doctor/search?q=" + quote(query, safe="")


def build_case_fact_graph(
    *,
    clinical: dict[str, Any] | None,
    record: dict[str, Any] | None = None,
    findings: list[dict[str, Any]] | None = None,
    llm_judge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Нормализованные факты случая для suggest (без сырого result).

    МКБ в diagnoses не сеем - подбор КП только по тексту диагноза.
    """
    clinical = clinical if isinstance(clinical, dict) else {}
    record = record if isinstance(record, dict) else {}
    findings = findings if isinstance(findings, list) else []
    _ = llm_judge  # reserved for future DDx hints

    diag_text = _diagnosis_text(clinical)
    complaints_raw = str(clinical.get("complaints") or "")
    complaints = [part.strip() for part in re.split(r"[;\n]+", complaints_raw) if part.strip()][:12]
    specialty = str(
        record.get("specialty")
        or record.get("specialization")
        or record.get("doctor_specialty")
        or record.get("doctor_specialization")
        or clinical.get("doctor_specialization")
        or clinical.get("specialty")
        or clinical.get("specialization")
        or ""
    ).strip()
    specialty_slug = None
    try:
        from clinical_knowledge.rubric_extractors import specialty_to_rubric

        specialty_slug = specialty_to_rubric(specialty)
    except Exception:  # noqa: BLE001
        specialty_slug = None

    gaps: list[dict[str, str]] = []
    for finding in findings[:20]:
        if not isinstance(finding, dict):
            continue
        code = str(finding.get("code") or finding.get("finding_code") or "").strip()
        if not _gap_allowed(code):
            continue
        title = str(finding.get("title_ru") or finding.get("title") or finding.get("detail_ru") or "").strip()
        if code or title:
            gaps.append({"code": code or "finding", "detail": title[:240]})

    diagnoses: list[dict[str, Any]] = []
    if diag_text:
        diagnoses.append({"text": diag_text, "role": "primary"})

    return {
        "case_id": str(record.get("visit_id") or record.get("case_id") or record.get("mis_id") or ""),
        "audience": "unknown",
        "specialty": {"label": specialty, "slug": specialty_slug},
        "complaints": complaints,
        "diagnoses": diagnoses,
        "plan": {
            "exam": str(clinical.get("exam_recommendations") or "")[:500],
            "treatment": str(clinical.get("treatment_recommendations") or "")[:500],
        },
        "gaps": gaps[:15],
        "objective_status": str(clinical.get("objective_status") or "")[:800],
        "anamnesis": str(clinical.get("anamnesis_doctor") or clinical.get("anamnesis_auto") or "")[:800],
    }


def _diag_overlap(item: dict[str, Any], graph: dict[str, Any]) -> float:
    diag = " ".join(str(d.get("text") or "") for d in (graph.get("diagnoses") or [])).lower()
    if not diag.strip():
        return 0.0
    blob = (
        str(item.get("title") or "")
        + " "
        + str(item.get("source_path") or "")
        + " "
        + str(item.get("matched_condition") or "")
    ).lower()
    tokens = [t for t in re.findall(r"[а-яa-z]{4,}", diag) if len(t) >= 4][:12]
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in blob)
    return min(1.0, hits / max(3, len(tokens) * 0.4))


def _match_kind(item: dict[str, Any], graph: dict[str, Any]) -> str:
    score = float(item.get("match_score") or 0)
    overlap = _diag_overlap(item, graph)
    if overlap >= 0.35 or (overlap > 0 and score >= 50):
        return "clinical"
    if graph.get("gaps") and score >= 55:
        return "ddx"
    return "specialty"


def _specialty_tokens(graph: dict[str, Any]) -> list[str]:
    label = str((graph.get("specialty") or {}).get("label") or "").lower()
    slug = str((graph.get("specialty") or {}).get("slug") or "").lower()
    return [token for token in (label, slug) if token]


def _path_blocked_for_specialty(row: dict[str, Any], graph: dict[str, Any]) -> bool:
    blob = (
        str(row.get("source_path") or "")
        + " "
        + str(row.get("title") or "")
        + " "
        + str(row.get("specialty_slug") or "")
    ).lower()
    for token in _specialty_tokens(graph):
        for key, blocked in _SPECIALTY_PATH_BLOCK.items():
            if key in token:
                if any(part in blob for part in blocked):
                    return True
    return False


def _rank_rows(matched: list[dict[str, Any]], graph: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Предпочесть clinical по тексту Dx; specialty - только добивка."""
    filtered = [row for row in matched if not _path_blocked_for_specialty(row, graph)]
    decorated: list[tuple[int, float, dict[str, Any]]] = []
    for row in filtered:
        kind = _match_kind(row, graph)
        tier = {"clinical": 0, "ddx": 1, "specialty": 2}.get(kind, 3)
        decorated.append((tier, -float(row.get("match_score") or 0), row))
    decorated.sort(key=lambda item: (item[0], item[1]))
    strong = [row for tier, _, row in decorated if tier == 0]
    if len(strong) >= limit:
        return strong[:limit]
    out = strong[:]
    for _, _, row in decorated:
        if row in out:
            continue
        out.append(row)
        if len(out) >= limit:
            break
    return out


def suggest_protocols_for_case(
    *,
    clinical: dict[str, Any] | None,
    record: dict[str, Any] | None = None,
    findings: list[dict[str, Any]] | None = None,
    llm_judge: dict[str, Any] | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """Top-K протоколов МЗ по тексту диагноза (детерминированно, без LLM и без МКБ)."""
    if not suggest_enabled():
        return {
            "ok": True,
            "available": False,
            "reason": "Подбор протоколов выключен (CASE_PROTOCOL_SUGGEST=0)",
            "engine": ENGINE,
            "items": [],
            "gaps": [],
        }

    from .protocol_match import match_protocol_cards_by_diagnosis_text

    graph = build_case_fact_graph(
        clinical=clinical,
        record=record,
        findings=findings,
        llm_judge=llm_judge,
    )
    diag_text = " ".join(str(item.get("text") or "") for item in (graph.get("diagnoses") or [])).strip()
    facts = {
        "patient_context": {"adult_or_child": graph.get("audience") or "unknown"},
        "consultation": {
            "icd10": [],  # намеренно пусто: КП не ищем по МКБ
            "diagnosis_text": diag_text,
            "complaints": list(graph.get("complaints") or []),
            "conditions_hint": [diag_text] if diag_text else [],
            "performed_exams": [],
        },
    }
    specialty_label = str((graph.get("specialty") or {}).get("label") or "")
    specialty_slug = (graph.get("specialty") or {}).get("slug")
    matched: list[dict[str, Any]] = []
    if specialty_slug:
        matched = match_protocol_cards_by_diagnosis_text(
            facts, specialty_slug=str(specialty_slug), limit=max(12, limit * 4)
        )
    if len(matched) < limit:
        extra = match_protocol_cards_by_diagnosis_text(
            facts, specialty_slug=None, limit=max(12, limit * 4)
        )
        seen_ids = {str(row.get("protocol_id") or row.get("source_path") or "") for row in matched}
        for row in extra:
            pid = str(row.get("protocol_id") or row.get("source_path") or "")
            if pid in seen_ids or _path_blocked_for_specialty(row, graph):
                continue
            matched.append(row)
            seen_ids.add(pid)

    ranked = _rank_rows(matched, graph, limit=limit)
    search_query = _search_query(graph)
    search_url = _search_url(search_query)
    items: list[dict[str, Any]] = []
    for row in ranked:
        kind = _match_kind(row, graph)
        title = _suggest_title(row.get("source_path"), row.get("title"))
        reasons: list[dict[str, str]] = []
        if diag_text and kind == "clinical":
            short = diag_text[:120] + ("…" if len(diag_text) > 120 else "")
            reasons.append({"code": "diagnosis_fit", "text": f"Совпадение с диагнозом: {short}"})
        if specialty_label and kind == "specialty":
            reasons.append({"code": "specialty", "text": f"Специальность случая: {specialty_label}"})
        elif specialty_label and specialty_slug and str(row.get("specialty_slug") or "") == specialty_slug:
            reasons.append({"code": "specialty", "text": f"Рубрика: {specialty_label}"})
        for gap in (graph.get("gaps") or [])[:2]:
            if gap.get("detail") and kind == "ddx":
                reasons.append(
                    {
                        "code": f"gap_{(gap.get('code') or 'x')[:40]}",
                        "text": f"Клинический разрыв: {gap['detail'][:160]}",
                    }
                )
        if not reasons:
            reasons.append(
                {
                    "code": "lexical",
                    "text": "Совпадение по тексту диагноза или специальности",
                }
            )
        source_path = str(row.get("source_path") or "")
        items.append(
            {
                "protocol_id": row.get("protocol_id"),
                "title": title,
                "source_path": source_path,
                "score": round(float(row.get("match_score") or 0), 1),
                "match_kind": kind,
                "match_kind_label": MATCH_KIND_LABELS.get(kind, kind),
                "reasons": reasons[:4],
                "covered_gaps": [g.get("code") for g in (graph.get("gaps") or [])[:3] if g.get("code")],
                "warnings": [],
                "viewer_url": protocol_nav_api_path(source_path) if source_path else None,
                "search_query": search_query,
                "search_url": search_url,
            }
        )
    return {
        "ok": True,
        "available": bool(items),
        "engine": ENGINE,
        "case_id": graph.get("case_id"),
        "gaps": graph.get("gaps") or [],
        "search_query": search_query,
        "search_url": search_url,
        "items": items,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "reason": None if items else "Не удалось подобрать протоколы по тексту диагноза",
    }
