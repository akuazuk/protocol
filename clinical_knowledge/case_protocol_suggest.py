"""Подбор протоколов МЗ РБ к случаю МО (не L1-балл оформления).

См. docs/plans/2026-08-05-mo-case-protocol-suggest-v1.md и
docs/plans/2026-08-05-mo-case-review-workspace-v2.md.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

ENGINE = "case_protocol_suggest_v1"
MATCH_KIND_LABELS = {
    "clinical": "Клиника",
    "code_only": "Только код",
    "ddx": "Дифдиагноз",
    "specialty": "Специальность",
}


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
    icd = _extract_icd(diag_text)
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

    gaps: list[dict[str, str]] = []
    for finding in findings[:20]:
        if not isinstance(finding, dict):
            continue
        code = str(finding.get("code") or finding.get("finding_code") or "").strip()
        title = str(finding.get("title_ru") or finding.get("title") or finding.get("detail_ru") or "").strip()
        if code or title:
            gaps.append(
                {
                    "code": code or "finding",
                    "detail": title[:240],
                }
            )
    conclusions = llm_judge.get("conclusions") if isinstance(llm_judge.get("conclusions"), dict) else {}
    for key in ("completeness_ru", "stage_a_ru", "stage_b_ru"):
        text = str(conclusions.get(key) or "").strip()
        if text:
            gaps.append({"code": key, "detail": text[:240]})

    return {
        "case_id": str(record.get("visit_id") or record.get("case_id") or record.get("mis_id") or ""),
        "audience": "unknown",
        "specialty": {"label": specialty, "slug": None},
        "complaints": complaints,
        "diagnoses": [{"icd": code, "text": diag_text, "role": "primary" if i == 0 else "secondary"} for i, code in enumerate(icd[:6])],
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
    clinical_overlap = bool(complaints) and any(
        token and token in title for token in re.findall(r"[а-яa-z]{4,}", complaints)[:8]
    )
    if has_icd and clinical_overlap:
        return "clinical"
    if has_icd and score < 45:
        return "code_only"
    if graph.get("gaps") and score >= 45:
        return "ddx"
    if not has_icd:
        return "specialty"
    return "clinical" if score >= 50 else "code_only"


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
    matched = match_protocol_cards(facts, specialty_slug=None, limit=max(8, limit * 3))
    items: list[dict[str, Any]] = []
    for row in matched[:limit]:
        kind = _match_kind(row, graph)
        reasons: list[dict[str, str]] = []
        if row.get("icd_fit_label"):
            reasons.append({"code": "icd_fit", "text": f"Совпадение МКБ: {row['icd_fit_label']}"})
        if specialty_label:
            reasons.append({"code": "specialty", "text": f"Специальность случая: {specialty_label}"})
        for gap in (graph.get("gaps") or [])[:2]:
            if gap.get("detail"):
                reasons.append(
                    {
                        "code": f"gap_{(gap.get('code') or 'x')[:40]}",
                        "text": f"Учитывает замечание: {gap['detail'][:160]}",
                    }
                )
        if not reasons:
            reasons.append({"code": "lexical", "text": "Совпадение по тексту диагноза или жалоб"})
        items.append(
            {
                "protocol_id": row.get("protocol_id"),
                "title": row.get("title") or "Протокол МЗ",
                "source_path": row.get("source_path") or "",
                "score": round(float(row.get("match_score") or 0), 1),
                "match_kind": kind,
                "match_kind_label": MATCH_KIND_LABELS.get(kind, kind),
                "reasons": reasons[:4],
                "covered_gaps": [g.get("code") for g in (graph.get("gaps") or [])[:3] if g.get("code")],
                "warnings": [],
                "viewer_url": (
                    "/proto?path=" + str(row.get("source_path") or "")
                    if row.get("source_path")
                    else None
                ),
            }
        )
    return {
        "ok": True,
        "available": bool(items),
        "engine": ENGINE,
        "case_id": graph.get("case_id"),
        "gaps": graph.get("gaps") or [],
        "items": items,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "reason": None if items else "Не удалось подобрать протоколы по МКБ и тексту случая",
    }
