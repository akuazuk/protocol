"""Единый пайплайн «Вопросы врачу» для B2C: кандидаты → tone → checklist → сообщения."""
from __future__ import annotations

import re
from typing import Any

from .patient_exam_extraction import extract_exams_from_text
from .patient_flags import patient_question_safety_enabled
from .patient_question_builder import build_useful_patient_questions
from .patient_question_tone import (
    category_emoji,
    normalize_question_tone,
    questions_etiquette_ru,
    questions_panel_intro_ru,
    tone_meta,
)
from .patient_questions import apply_safe_questions, sanitize_question_text


def _why_ru_for_tone(base: str, tone: str, *, intent: str = "") -> str:
    tid = normalize_question_tone(tone)
    b = (base or "").strip().rstrip(".")
    if not b:
        b = "Это стоит уточнить на приёме."
    if tid == "playful":
        if intent == "labs_missing_in_kz":
            return "Не для претензии - хочу понять, учтены ли цифры из бланка."
        if intent in ("treatment_duration", "treatment_dose", "treatment_order"):
            return "Чтобы не перепутать схему дома - прошу пояснить простыми словами."
        return f"Не для претензии - хочу понять: {b.lower()}."
    if tid == "official":
        return f"Прошу разъяснить: {b.lower()}."
    return b + "."


def _prepare_candidates_for_tone(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Добавить source_comment/intent для apply_tone, сохранив plain_context."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if not item.get("source_gap") and not item.get("source_comment"):
            hint = (item.get("why_ru") or item.get("plain_context") or "").strip()
            if hint:
                item["source_comment"] = hint
            elif item.get("text"):
                item["source_comment"] = str(item["text"]).rstrip("?").strip()
        if not item.get("intent") and item.get("block_id"):
            item["intent"] = str(item.get("block_id") or "")
        out.append(item)
    return out


def _dedupe_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = re.sub(r"\s+", " ", str(row.get("text") or "").lower())[:72]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _mark_discuss_first(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    marked = 0
    for row in rows:
        item = dict(row)
        sev = str(item.get("severity") or "")
        pri = int(item.get("priority") or 99)
        if marked < 2 and (sev == "high" or pri <= 15):
            item["discuss_first"] = True
            marked += 1
        else:
            item["discuss_first"] = False
        out.append(item)
    return out


def _action_checklist_from_questions(
    questions: list[dict[str, Any]],
    *,
    tone: str,
) -> list[dict[str, Any]]:
    tid = normalize_question_tone(tone)
    out: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        if not q.get("text"):
            continue
        icon = q.get("icon") or category_emoji(str(q.get("category_ru") or ""))
        out.append(
            {
                "id": q.get("id", f"q{i+1}"),
                "text": q.get("text", ""),
                "title": q.get("title") or (str(q.get("text", "")).split("?")[0].strip()[:72] + "?"),
                "severity": q.get("severity", "medium"),
                "category_ru": q.get("category_ru", ""),
                "block_id": q.get("block_id", ""),
                "tone": q.get("tone") or tid,
                "emoji": q.get("emoji") or icon,
                "icon": icon,
                "why_ru": q.get("why_ru") or "",
                "plain_context": q.get("plain_context") or "",
                "intent": q.get("intent") or "",
                "discuss_first": bool(q.get("discuss_first")),
                "checked": False,
            }
        )
    return out


def maybe_llm_rephrase_questions(
    questions: list[dict[str, Any]],
    *,
    tone: str,
    kz_text: str,
) -> list[dict[str, Any]]:
    """Опциональный LLM-слой (P2): по умолчанию выключен, возвращает как есть."""
    from .patient_flags import patient_question_llm_rephrase_enabled

    if not patient_question_llm_rephrase_enabled():
        return questions
    # Hook для будущего LLM-rephrase; без ключа/API не меняем текст.
    return questions


def build_patient_doctor_questions(
    *,
    kz_text: str,
    clarification_points: list[dict[str, str]] | None = None,
    exams: list[dict[str, Any]] | None = None,
    meds: list[dict[str, Any]] | None = None,
    lab_crosscheck: dict[str, Any] | None = None,
    structured_gaps: list[dict[str, Any]] | None = None,
    extra_candidates: list[dict[str, Any]] | None = None,
    question_tone: str | None = None,
    limit: int = 5,
    age_group: str | None = None,
) -> list[dict[str, Any]]:
    """Собрать финальные вопросы: факты → tone → sanitize → dedupe."""
    tone = normalize_question_tone(question_tone)
    raw = build_useful_patient_questions(
        kz_text=kz_text,
        clarification_points=clarification_points,
        exams=exams,
        meds=meds,
        lab_crosscheck=lab_crosscheck,
        structured_gaps=structured_gaps,
        limit=limit + 2,
        age_group=age_group,
    )
    merged = list(raw) + list(extra_candidates or [])
    merged = _dedupe_questions(merged)
    prepared = _prepare_candidates_for_tone(merged)
    styled = apply_safe_questions(
        prepared,
        kz_text=kz_text,
        exams=exams,
        tone=tone,
        safety_enabled=patient_question_safety_enabled(),
    )
    for row in styled:
        intent = str(row.get("intent") or "")
        base_why = str(row.get("why_ru") or row.get("source_comment") or row.get("source_gap") or "")
        row["why_ru"] = _why_ru_for_tone(base_why, tone, intent=intent)
        text = sanitize_question_text(str(row.get("text") or ""))
        if text:
            row["text"] = text
            row["title"] = text.split("?")[0].strip()[:72] + ("?" if "?" in text else "")
        row["tone"] = tone
        icon = category_emoji(str(row.get("category_ru") or ""))
        row["icon"] = icon
        row["emoji"] = icon
    styled = _mark_discuss_first(styled[:limit])
    return maybe_llm_rephrase_questions(styled, tone=tone, kz_text=kz_text)


def attach_questions_to_report(
    report: dict[str, Any],
    questions: list[dict[str, Any]],
    *,
    question_tone: str | None = None,
) -> dict[str, Any]:
    """Записать вопросы в отчёт + мета тона."""
    tone = normalize_question_tone(question_tone)
    report = dict(report)
    checklist = _action_checklist_from_questions(questions, tone=tone)
    report["questions_structured"] = questions
    report["questions_for_doctor"] = [q["text"] for q in questions if q.get("text")]
    report["action_checklist"] = checklist
    report["question_tone"] = tone
    report["question_tone_meta"] = tone_meta(tone)
    report["questions_intro_ru"] = questions_panel_intro_ru(tone)
    report["questions_etiquette_ru"] = questions_etiquette_ru(tone)
    high = sum(1 for q in questions if q.get("discuss_first"))
    report["questions_discuss_first_count"] = high
    return report


def sync_report_questions_from_checklist(
    report: dict[str, Any],
    *,
    kz_text: str = "",
    question_tone: str | None = None,
) -> dict[str, Any]:
    """После RAG-дополнения: dedupe, tone, пересборка checklist и производных."""
    from .patient_medication_extraction import extract_medications_from_text
    from .patient_report_v2 import _message_to_doctor, _visit_sheet

    tone = normalize_question_tone(question_tone or report.get("question_tone"))
    exams = extract_exams_from_text(kz_text)
    raw = list(report.get("questions_structured") or report.get("action_checklist") or [])
    if not raw:
        return report
    prepared = _prepare_candidates_for_tone(raw)
    styled = apply_safe_questions(
        prepared,
        kz_text=kz_text,
        exams=exams,
        tone=tone,
        safety_enabled=patient_question_safety_enabled(),
    )
    for row in styled:
        intent = str(row.get("intent") or "")
        base_why = str(row.get("why_ru") or row.get("source_comment") or "")
        row["why_ru"] = _why_ru_for_tone(base_why, tone, intent=intent)
        row["tone"] = tone
    styled = _mark_discuss_first(_dedupe_questions(styled)[:5])
    report = attach_questions_to_report(report, styled, question_tone=tone)
    top = report.get("top_summary") if isinstance(report.get("top_summary"), dict) else {}
    clarify = list(report.get("clarification_points") or [])
    report["message_to_doctor"] = _message_to_doctor(styled, kz_text)
    if top:
        report["visit_sheet"] = _visit_sheet(top, clarify, styled, kz_text)
    return report
