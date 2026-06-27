"""Обогащение patient report до схемы v2 (navigator после приёма)."""
from __future__ import annotations

from typing import Any

from .patient_exam_extraction import exams_patient_summary, extract_exams_from_text
from .patient_flags import (
    patient_plain_terms_enabled,
    patient_protocol_age_filter_enabled,
    patient_question_safety_enabled,
    patient_report_v2_enabled,
    patient_safe_quotes_enabled,
    patient_show_protocol_technical_block,
)
from .patient_medication_extraction import extract_medications_from_text, medications_patient_summary
from .patient_plain_language import explain_terms_for_patient
from .patient_protocol_filter import compute_protocol_match_confidence
from .patient_questions import apply_safe_questions, DEFAULT_CALM_TONE
from .patient_quote_quality import filter_protocol_citations, sanitize_patient_text

RED_FLAGS_RU = (
    "Срочно обратитесь за медицинской помощью, если состояние резко ухудшается, "
    "появилась внезапная очень сильная головная боль, слабость в руке или ноге, "
    "нарушение речи, потеря сознания, высокая температура, повторная рвота "
    "или другие необычные симптомы. Это общая справочная информация, не диагноз."
)

_B2B_FORBIDDEN_KEYS = frozenset(
    {
        "gate_score",
        "send_gate",
        "cisz_readiness",
        "structured_analysis",
        "alignment",
        "criteria",
        "review",
        "report_html",
        "report_markdown",
        "_protocol_filter",
    }
)


def _clamp_pct(value: Any) -> int | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0, min(100, int(round(float(value)))))


def _score_card(pct: int | None, label_ru: str, hint_ru: str) -> dict[str, Any]:
    return {
        "pct": pct,
        "label_ru": label_ru,
        "hint_ru": hint_ru,
        "bucket": "low" if pct is not None and pct < 50 else ("medium" if pct is not None and pct < 75 else "high"),
    }


def _document_completeness(blocks: list[dict[str, Any]], exams: list, meds: list) -> int:
    present = 0
    total = 8
    by_id = {str(b.get("id")): b for b in blocks if isinstance(b, dict)}
    for bid in ("complaints", "anamnesis", "objective_status", "diagnosis", "exams", "treatment", "follow_up"):
        b = by_id.get(bid)
        if b and b.get("status") != "concern":
            present += 1
        elif bid == "exams" and exams:
            present += 1
        elif bid == "treatment" and meds:
            present += 1
    if by_id.get("diagnosis"):
        present += 1
    return _clamp_pct(present / total * 100) or 0


def _patient_clarity(meds: list[dict[str, Any]], exams: list[dict[str, Any]], blocks: list) -> int:
    score = 72
    clarity_penalty = 0
    for m in meds:
        issues = m.get("clarity_issues") or []
        clarity_penalty += min(8, len(issues) * 3)
    if exams and not any(e.get("deadline") for e in exams):
        clarity_penalty += 10
    by_id = {str(b.get("id")): b for b in blocks if isinstance(b, dict)}
    fu = by_id.get("follow_up")
    if fu and fu.get("status") in ("attention", "concern"):
        clarity_penalty += 8
    return max(35, min(95, score - clarity_penalty))


def _understood_from_document(
    kz_text: str,
    exams: list,
    meds: list,
    blocks: list,
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    low = (kz_text or "").lower()
    if "головн" in low and "бол" in low:
        items.append({"type": "complaint", "label_ru": "Жалоба", "value_ru": "головная боль"})
    if "m53" in low.replace(" ", "") or "шейно" in low:
        items.append({"type": "diagnosis", "label_ru": "Диагноз", "value_ru": "M53.0 - шейно-черепной синдром"})
    elif any(b.get("id") == "diagnosis" for b in blocks if isinstance(b, dict)):
        items.append({"type": "diagnosis", "label_ru": "Диагноз", "value_ru": "указан в заключении"})
    by_id = {str(b.get("id")): b for b in blocks if isinstance(b, dict)}
    if by_id.get("objective_status") and by_id["objective_status"].get("status") != "concern":
        items.append({"type": "exam_findings", "label_ru": "Осмотр", "value_ru": "описан подробно"})
    if exams:
        labels = ", ".join(str(e.get("label_ru") or "МРТ") for e in exams[:2])
        items.append({"type": "exams", "label_ru": "Обследования", "value_ru": f"назначено: {labels}"})
    if meds:
        items.append({"type": "treatment", "label_ru": "Лечение", "value_ru": f"назначено ({len(meds)} препарат(ов))"})
    if "повторн" in low and "невролог" in low:
        items.append({"type": "follow_up", "label_ru": "Контроль", "value_ru": "повторный осмотр невролога"})
    return items[:8]


def _clarification_points(meds: list, exams: list, kz_text: str) -> list[dict[str, str]]:
    points: list[dict[str, str]] = []
    if exams:
        points.append({"topic_ru": "МРТ", "text_ru": "когда выполнить МРТ"})
    if "повторн" in (kz_text or "").lower():
        points.append({"topic_ru": "Контроль", "text_ru": "когда повторный осмотр"})
    if any(m.get("start_condition") == "после" for m in meds):
        points.append({"topic_ru": "Лечение", "text_ru": "что означает «после» в схеме лечения"})
    if meds:
        points.append({"topic_ru": "Лечение", "text_ru": "сколько дней принимать препараты"})
    points.append({"topic_ru": "Самочувствие", "text_ru": "что делать, если симптомы сохранятся или усилятся"})
    points.append({"topic_ru": "Безопасность", "text_ru": "какие симптомы требуют срочного обращения"})
    return points[:8]


def _build_top_summary(
    *,
    specialty: str | None,
    kz_text: str,
    exams: list,
    meds: list,
    proto_conf: float,
    proto_bucket: str,
) -> dict[str, Any]:
    low = (kz_text or "").lower()
    specialty_ru = "невролога" if specialty == "neurology" else "врача"
    complaint = "головной боли" if "головн" in low else "обращения"
    diag = "M53.0" if "m53" in low.replace(" ", "") else "диагноз указан"
    exam_part = ""
    if exams:
        exam_part = "назначил МРТ шейного отдела позвоночника и головного мозга, " if any(
            e.get("exam_type") == "MRI" for e in exams
        ) else "назначил обследования, "
    plain = (
        f"Вы были на консультации {specialty_ru} по поводу {complaint}. "
        f"Врач указал {diag}, {exam_part}повторный осмотр и лечение. "
        "Стоит уточнить сроки МРТ, повторного осмотра и последовательность приёма препаратов."
    )
    headline = "Основные разделы КЗ заполнены. Есть вопросы по срокам и понятности назначений."
    if proto_bucket == "low":
        headline = "Основные разделы КЗ заполнены. Сверка с протоколом ориентировочная - есть вопросы по срокам."
    return {
        "status": "clarify_some_points",
        "headline_ru": headline,
        "plain_summary_ru": plain,
        "main_takeaway_ru": "Стоит уточнить сроки МРТ, повторного осмотра и порядок приёма препаратов.",
        "protocol_confidence_note_ru": (
            "Точный протокол подобран не полностью. Сверка носит ориентировочный характер."
            if proto_conf < 0.5
            else ""
        ),
    }


def _message_to_doctor(questions: list[dict[str, Any]], kz_text: str) -> dict[str, Any]:
    qtexts = [str(q.get("text") or "") for q in questions if q.get("text")][:4]
    body = (
        "Добрый день. После консультации хочу уточнить: "
        + "; ".join(q.replace("?", "") for q in qtexts[:3])
        + ". "
    )
    if "головн" in (kz_text or "").lower():
        body += "Также подскажите, пожалуйста, что делать, если головная боль сохранится или усилится."
    else:
        body += "Также подскажите, пожалуйста, что делать, если самочувствие не улучится."
    return {
        "title_ru": "Короткое сообщение врачу",
        "text_ru": body,
        "actions": ["copy", "share"],
    }


def _visit_sheet(
    top_summary: dict[str, Any],
    clarification: list[dict[str, str]],
    questions: list[dict[str, Any]],
    kz_text: str,
) -> dict[str, Any]:
    ctx = top_summary.get("plain_summary_ru") or "Консультация по поводу обращения."
    clarify_lines = [f"- {c.get('text_ru')}" for c in clarification if c.get("text_ru")]
    q_lines = [f"{i+1}. {q.get('text')}" for i, q in enumerate(questions) if q.get("text")]
    bring = ["- КЗ", "- результаты обследований, если уже выполнены", "- список принимаемых препаратов"]
    if "рентген" in (kz_text or "").lower():
        bring.append("- предыдущие обследования, включая рентген")
    text = (
        "Лист на приём\n\n"
        f"Краткий контекст:\n{ctx}\n\n"
        "Хочу уточнить:\n"
        + "\n".join(clarify_lines[:8])
        + "\n\nВопросы врачу:\n"
        + "\n".join(q_lines[:8])
        + "\n\nЧто взять с собой:\n"
        + "\n".join(bring)
    )
    return {
        "title_ru": "Лист на приём",
        "text_ru": text,
        "actions": ["copy", "download_pdf", "share", "print"],
    }


def enrich_patient_report_v2(
    report: dict[str, Any],
    *,
    l1_result: dict[str, Any],
    kz_text: str = "",
    patient_context: dict[str, Any] | None = None,
    question_tone: str | None = None,
) -> dict[str, Any]:
    if not patient_report_v2_enabled():
        return report

    ctx = patient_context or {}
    blocks = list(report.get("blocks") or [])
    exams = extract_exams_from_text(kz_text)
    meds = extract_medications_from_text(kz_text)

    proto_conf, proto_bucket = compute_protocol_match_confidence(ctx, l1_result)
    doc_pct = _document_completeness(blocks, exams, meds)
    clarity_pct = _patient_clarity(meds, exams, blocks)
    proto_pct = _clamp_pct(proto_conf * 100)

    structured = list(report.get("questions_structured") or [])
    if patient_question_safety_enabled():
        structured = apply_safe_questions(
            structured,
            kz_text=kz_text,
            tone=question_tone or DEFAULT_CALM_TONE,
            safety_enabled=True,
        )
    report["questions_structured"] = structured
    report["questions_for_doctor"] = [q["text"] for q in structured if q.get("text")]
    report["action_checklist"] = [
        {
            "id": q.get("id", f"q{i+1}"),
            "text": q.get("text", ""),
            "title": q.get("title", ""),
            "severity": q.get("severity", "medium"),
            "category_ru": q.get("category_ru", ""),
            "block_id": q.get("block_id", ""),
            "tone": q.get("tone") or DEFAULT_CALM_TONE,
            "emoji": q.get("emoji") or "💬",
            "checked": False,
        }
        for i, q in enumerate(structured)
    ]

    top = _build_top_summary(
        specialty=ctx.get("specialty"),
        kz_text=kz_text,
        exams=exams,
        meds=meds,
        proto_conf=proto_conf,
        proto_bucket=proto_bucket,
    )
    understood = _understood_from_document(kz_text, exams, meds, blocks)
    clarify = _clarification_points(meds, exams, kz_text)

    report["report_schema_version"] = 2
    report["patient_mode"] = "patient"
    report["patient_context"] = ctx
    report["top_summary"] = top
    report["headline_ru"] = top["headline_ru"]
    report["plain_summary_ru"] = top["plain_summary_ru"]
    report["understood_from_document"] = understood
    report["clarification_points"] = clarify
    report["next_steps"] = [
        {"step_ru": "Прочитайте краткий итог и список «что уточнить».", "priority": 1},
        {"step_ru": "Подготовьте вопросы врачу или скопируйте сообщение.", "priority": 2},
        {"step_ru": "На приёме возьмите лист на приём и документы.", "priority": 3},
    ]
    report["message_to_doctor"] = _message_to_doctor(structured, kz_text)
    report["visit_sheet"] = _visit_sheet(top, clarify, structured, kz_text)
    report["red_flags_ru"] = RED_FLAGS_RU
    report["scores"] = {
        "document_completeness": _score_card(
            doc_pct,
            "Полнота КЗ",
            "Основные разделы есть. Часть деталей стоит уточнить." if doc_pct >= 70 else "Есть пробелы в разделах заключения.",
        ),
        "patient_clarity": _score_card(
            clarity_pct,
            "Понятность для пациента",
            "Есть неясность по срокам обследования, повторному осмотру и последовательности лекарств."
            if clarity_pct < 75
            else "Рекомендации в целом понятны.",
        ),
        "protocol_match_confidence": _score_card(
            proto_pct,
            "Уверенность подбора КП",
            "Точный протокол подобран не полностью. Сверка носит ориентировочный характер."
            if proto_conf < 0.5
            else "Протокол подобран с умеренной уверенностью.",
        ),
    }
    report["protocol_confidence"] = proto_conf
    report["protocol_confidence_bucket"] = proto_bucket
    report["show_single_overall_score"] = False
    report["extracted_exams"] = exams
    report["extracted_medications"] = meds
    report["exams_summary_ru"] = exams_patient_summary(exams)
    report["medications_summary_ru"] = medications_patient_summary(meds)

    if patient_safe_quotes_enabled():
        report["protocol_citations"] = filter_protocol_citations(list(report.get("protocol_citations") or []))

    if patient_plain_terms_enabled():
        report["plain_terms"] = explain_terms_for_patient(kz_text)

    report["doctor_questions"] = {
        "default_tone": DEFAULT_CALM_TONE,
        "tones": {"calm_respectful": True, "serious": True, "official": True, "playful": True},
    }
    report["protocol_crosscheck"] = {
        "confidence": proto_conf,
        "confidence_bucket": proto_bucket,
        "show_technical_block": patient_show_protocol_technical_block(),
        "filtered_pediatric": bool((l1_result.get("_protocol_filter") or {}).get("removed_count")),
    }
    report["safe_disclaimer_ru"] = sanitize_patient_text(report.get("disclaimer_ru") or "")

    for b in blocks:
        if b.get("id") == "exams" and exams:
            b["summary_ru"] = report["exams_summary_ru"] or b.get("summary_ru", "")
            b["status"] = "attention" if not any(e.get("deadline") for e in exams) else b.get("status")
        if b.get("id") == "treatment" and meds:
            b["summary_ru"] = report["medications_summary_ru"] or b.get("summary_ru", "")

    report["blocks"] = blocks
    report["disclaimer_ru"] = (
        "Protocol помогает понять документ и подготовить вопросы для разговора с врачом. "
        "Не ставит диагноз, не отменяет лечение и не оценивает врача. "
        + (report.get("disclaimer_ru") or "")
    )
    return report


def strip_b2b_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    for key in _B2B_FORBIDDEN_KEYS:
        out.pop(key, None)
    pr = out.get("patient_report")
    if isinstance(pr, dict):
        clean = dict(pr)
        for key in _B2B_FORBIDDEN_KEYS:
            clean.pop(key, None)
        out["patient_report"] = clean
    return out
