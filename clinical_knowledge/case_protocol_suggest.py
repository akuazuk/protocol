"""Подбор протоколов МЗ РБ к случаю МО (не L1-балл оформления).

См. docs/plans/2026-08-05-mo-case-protocol-suggest-v1.md и
docs/plans/2026-08-06-mo-protocol-suggest-titles-search-v1.md.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from clinical_knowledge.protocol_links import protocol_display_name, protocol_nav_api_path

ENGINE = "case_protocol_suggest_v2"
MATCH_KIND_LABELS = {
    "clinical": "Клиника",
    "code_only": "Только код",
    "ddx": "Дифдиагноз",
    "specialty": "Специальность",
}

# Замечания, которые не должны попадать в reasons «Учитывает замечание».
_GAP_SKIP_PREFIXES = (
    "D_reg55",
    "E_template",
    "A_missing_",
    "completeness_ru",
    "stage_a_ru",
    "stage_b_ru",
)


def suggest_enabled() -> bool:
    raw = (os.environ.get("CASE_PROTOCOL_SUGGEST") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _icd_root(code: str) -> str:
    text = (code or "").upper().strip()
    return text[:3] if len(text) >= 3 else text


def _extract_icd(text: str) -> list[str]:
    found = re.findall(r"\b([A-ZА-Я]\d{2}(?:\.\d{1,2})?)\b", text or "", flags=re.I)
    out: list[str] = []
    for item in found:
        code = item.upper().replace("А", "A").replace("В", "B").replace("С", "C")
        if code not in out:
            out.append(code)
    return out


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


def _search_query(graph: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in graph.get("diagnoses") or []:
        text = str(item.get("text") or "").strip()
        icd = str(item.get("icd") or "").strip()
        if text:
            parts.append(text[:120])
        elif icd:
            parts.append(icd)
    for complaint in (graph.get("complaints") or [])[:2]:
        if complaint:
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
    """Нормализованные факты случая для suggest (без сырого result)."""
    clinical = clinical if isinstance(clinical, dict) else {}
    record = record if isinstance(record, dict) else {}
    findings = findings if isinstance(findings, list) else []
    llm_judge = llm_judge if isinstance(llm_judge, dict) else {}

    diag_text = " ".join(
        str(clinical.get(key) or "")
        for key in ("clinical_diagnosis", "mis_diagnos")
        if clinical.get(key)
    ).strip()
    # МКБ по всему МО (не только графа диагноза) - plan mo-icd-full-document-search.
    from clinical_knowledge.mo_icd_resolve import resolve_icd_codes_from_mo

    icd_blob = {**record, **{k: v for k, v in clinical.items() if v}}
    icd = list(resolve_icd_codes_from_mo(icd_blob).get("all") or [])
    for code in _extract_icd(diag_text):
        if code not in icd:
            icd.append(code)
    for key in ("diagnosis_code", "mkb_code_main", "icd10"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            for code in _extract_icd(value):
                if code not in icd:
                    icd.append(code)
        elif isinstance(value, list):
            for item in value:
                for code in _extract_icd(str(item)):
                    if code not in icd:
                        icd.append(code)

    complaints_raw = str(clinical.get("complaints") or "")
    complaints = [part.strip() for part in re.split(r"[;\n]+", complaints_raw) if part.strip()][:12]
    specialty = str(
        record.get("specialty")
        or record.get("specialization")
        or record.get("doctor_specialty")
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

    return {
        "case_id": str(record.get("visit_id") or record.get("case_id") or record.get("mis_id") or ""),
        "audience": "unknown",
        "specialty": {"label": specialty, "slug": specialty_slug},
        "complaints": complaints,
        "diagnoses": [
            {"icd": code, "text": diag_text, "role": "primary" if i == 0 else "secondary"}
            for i, code in enumerate(icd[:6])
        ],
        "plan": {
            "exam": str(clinical.get("exam_recommendations") or "")[:500],
            "treatment": str(clinical.get("treatment_recommendations") or "")[:500],
        },
        "gaps": gaps[:15],
        "objective_status": str(clinical.get("objective_status") or "")[:800],
        "anamnesis": str(clinical.get("anamnesis_doctor") or clinical.get("anamnesis_auto") or "")[:800],
    }


def _match_kind(item: dict[str, Any], graph: dict[str, Any]) -> str:
    fit = item.get("icd_fit") or []
    score = float(item.get("match_score") or 0)
    has_icd = bool(fit)
    complaints = " ".join(graph.get("complaints") or []).lower()
    title = str(item.get("title") or "").lower()
    path = str(item.get("source_path") or "").lower()
    blob = title + " " + path
    clinical_overlap = bool(complaints) and any(
        token and token in blob for token in re.findall(r"[а-яa-z]{4,}", complaints)[:8]
    )
    if has_icd and clinical_overlap:
        return "clinical"
    if has_icd and score < 45:
        return "code_only"
    if has_icd:
        return "clinical" if score >= 50 else "code_only"
    if graph.get("gaps") and score >= 55:
        return "ddx"
    return "specialty"


def _rank_rows(matched: list[dict[str, Any]], graph: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Предпочесть ICD/clinical матчи; specialty-only - только добивка."""
    decorated: list[tuple[int, float, dict[str, Any]]] = []
    for row in matched:
        kind = _match_kind(row, graph)
        tier = {"clinical": 0, "code_only": 1, "ddx": 2, "specialty": 3}.get(kind, 4)
        decorated.append((tier, -float(row.get("match_score") or 0), row))
    decorated.sort(key=lambda item: (item[0], item[1]))
    strong = [row for tier, _, row in decorated if tier <= 1]
    if len(strong) >= limit:
        return strong[:limit]
    out = strong[:]
    for tier, _, row in decorated:
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
    """Top-K протоколов МЗ с reasons (детерминированно, без LLM)."""
    if not suggest_enabled():
        return {
            "ok": True,
            "available": False,
            "reason": "Подбор протоколов выключен (CASE_PROTOCOL_SUGGEST=0)",
            "engine": ENGINE,
            "items": [],
            "gaps": [],
        }

    from .protocol_match import match_protocol_cards

    graph = build_case_fact_graph(
        clinical=clinical,
        record=record,
        findings=findings,
        llm_judge=llm_judge,
    )
    icd_list = [
        str(item.get("icd") or "").upper()
        for item in (graph.get("diagnoses") or [])
        if item.get("icd")
    ]
    facts = {
        "patient_context": {"adult_or_child": graph.get("audience") or "unknown"},
        "consultation": {
            "icd10": icd_list,
            "diagnosis_text": " ".join(
                str(item.get("text") or "") for item in (graph.get("diagnoses") or [])
            ),
            "complaints": list(graph.get("complaints") or []),
            "conditions_hint": [],
            "performed_exams": [],
        },
    }
    specialty_label = str((graph.get("specialty") or {}).get("label") or "")
    specialty_slug = (graph.get("specialty") or {}).get("slug")
    # Сначала с рубрикой специальности (hard filter), при пусто - без фильтра.
    matched: list[dict[str, Any]] = []
    if specialty_slug:
        matched = match_protocol_cards(
            facts, specialty_slug=str(specialty_slug), limit=max(12, limit * 4)
        )
    if len(matched) < limit:
        matched = match_protocol_cards(facts, specialty_slug=None, limit=max(12, limit * 4))

    ranked = _rank_rows(matched, graph, limit=limit)
    search_query = _search_query(graph)
    search_url = _search_url(search_query)
    items: list[dict[str, Any]] = []
    for row in ranked:
        kind = _match_kind(row, graph)
        title = _suggest_title(row.get("source_path"), row.get("title"))
        reasons: list[dict[str, str]] = []
        if row.get("icd_fit_label"):
            reasons.append({"code": "icd_fit", "text": f"Совпадение МКБ: {row['icd_fit_label']}"})
        if specialty_label and kind == "specialty":
            reasons.append({"code": "specialty", "text": f"Специальность случая: {specialty_label}"})
        elif specialty_label and specialty_slug and str(row.get("specialty_slug") or "") == specialty_slug:
            reasons.append({"code": "specialty", "text": f"Рубрика: {specialty_label}"})
        for gap in (graph.get("gaps") or [])[:2]:
            if gap.get("detail"):
                reasons.append(
                    {
                        "code": f"gap_{(gap.get('code') or 'x')[:40]}",
                        "text": f"Клинический разрыв: {gap['detail'][:160]}",
                    }
                )
        if not reasons:
            reasons.append({"code": "lexical", "text": "Совпадение по тексту диагноза или жалоб"})
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
        "reason": None if items else "Не удалось подобрать протоколы по МКБ и тексту случая",
    }
