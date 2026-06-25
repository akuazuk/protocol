"""Отчёт проверки КЗ для пациента (B2C / tier P1)."""
from __future__ import annotations

import re
from typing import Any, Literal

TrafficLight = Literal["green", "yellow", "red"]
BlockStatus = Literal["ok", "attention", "concern"]

PATIENT_DISCLAIMER_RU = (
    "Ориентировочная сверка с клиническими протоколами Минздрава РБ. "
    "Не является диагнозом, медицинским заключением или заменой очного приёма. "
    "При сомнениях обратитесь к лечащему врачу или методслужбе клиники."
)

PATIENT_BLOCK_ORDER = (
    "complaints",
    "anamnesis",
    "objective_status",
    "diagnosis",
    "exams",
    "treatment",
    "follow_up",
    "limitations",
)

_QUESTION_PREFIXES = (
    "уточните",
    "проверьте",
    "обсудите",
    "спросите",
    "не указан",
    "не описан",
    "не отраж",
    "отсутств",
)


def _clamp_pct(value: Any) -> int | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0, min(100, int(round(float(value)))))


def traffic_light_for_pct(pct: int | None) -> tuple[TrafficLight, str]:
    if pct is None:
        return "yellow", "Недостаточно данных для уверенной оценки"
    if pct >= 75:
        return "green", "В целом соответствует стандарту Минздрава"
    if pct >= 50:
        return "yellow", "Есть пробелы - задайте врачу вопросы из списка"
    return "red", "Много неучтённого по стандарту - рекомендуем обсудить с врачом"


def block_status_for_score(score_pct: int | None) -> BlockStatus:
    if score_pct is None:
        return "attention"
    if score_pct >= 75:
        return "ok"
    if score_pct >= 50:
        return "attention"
    return "concern"


def _gap_to_question(gap: str, block_name: str) -> str:
    g = (gap or "").strip()
    if not g:
        return ""
    if g.endswith("?"):
        return g
    low = g.lower()
    if any(low.startswith(p) for p in _QUESTION_PREFIXES):
        return g[0].upper() + g[1:] if g else g
    prefix = f"По разделу «{block_name}»: " if block_name else ""
    return f"{prefix}уточните у врача - {g.rstrip('.')}"


def _question_title(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "Вопрос врачу"
    if "?" in t[:80]:
        return t.split("?")[0].strip()[:60] + "?"
    words = t.split()
    return " ".join(words[:6]) + ("…" if len(words) > 6 else "")


def _collect_structured_questions(cards: list[dict[str, Any]], limit: int = 8) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for card in cards:
        if not isinstance(card, dict):
            continue
        if card.get("block_id") == "limitations":
            continue
        name = str(card.get("name_ru") or "").strip()
        score = _clamp_pct(card.get("score_pct"))
        severity = "low"
        if score is not None and score < 50:
            severity = "high"
        elif score is not None and score < 75:
            severity = "medium"
        items: list[str] = []
        if score is not None and score < 75:
            comment = str(card.get("comment_ru") or "").strip()
            if comment and len(comment) > 12:
                items.append(comment)
        for g in card.get("gaps_ru") or []:
            txt = str(g).strip()
            if txt:
                items.append(txt)
        for raw in items:
            q = _gap_to_question(raw, name)
            key = re.sub(r"\s+", " ", q.lower())[:100]
            if not q or key in seen:
                continue
            seen.add(key)
            out.append({"id": f"q{len(out)+1}", "title": _question_title(q), "text": q, "severity": severity})
            if len(out) >= limit:
                return out
    return out


def _collect_citations(cards: list[dict[str, Any]], limit: int = 5) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for card in cards:
        if not isinstance(card, dict):
            continue
        excerpt = str(card.get("protocol_excerpt") or "").strip()
        if len(excerpt) < 24:
            continue
        key = excerpt[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        title = str(card.get("protocol_title") or card.get("name_ru") or "Клинический протокол").strip()
        section = str(card.get("protocol_section") or "").strip()
        out.append(
            {
                "protocol_title": title[:200],
                "section": section[:120],
                "excerpt": excerpt[:420],
            }
        )
        if len(out) >= limit:
            break
    return out


def _patient_blocks(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {
        str(c.get("block_id")): c
        for c in cards
        if isinstance(c, dict) and c.get("block_id")
    }
    blocks: list[dict[str, Any]] = []
    for bid in PATIENT_BLOCK_ORDER:
        card = by_id.get(bid)
        if not card or bid == "limitations":
            continue
        score = _clamp_pct(card.get("score_pct"))
        status = block_status_for_score(score)
        summary = str(card.get("comment_ru") or "").strip()
        if not summary and card.get("findings_ru"):
            findings = [str(x).strip() for x in card.get("findings_ru") or [] if str(x).strip()]
            if findings:
                summary = findings[0]
        gaps = [str(g).strip() for g in card.get("gaps_ru") or [] if str(g).strip()][:3]
        blocks.append(
            {
                "id": bid,
                "title": str(card.get("name_ru") or bid),
                "score_pct": score,
                "status": status,
                "summary_ru": summary,
                "gaps": gaps,
            }
        )
    return blocks


def resolve_patient_overall_pct(l1_result: dict[str, Any]) -> int | None:
    align = l1_result.get("alignment")
    if isinstance(align, dict):
        mean = _clamp_pct(align.get("alignment_mean_score"))
        if mean is not None:
            return mean
    comp = (l1_result.get("structured_analysis") or {}).get("compliance") or {}
    return _clamp_pct(comp.get("overall_score") or l1_result.get("overall_score"))


def _document_read_back(l1_result: dict[str, Any]) -> list[str]:
    doc = (l1_result.get("structured_analysis") or {}).get("document") or {}
    if not isinstance(doc, dict):
        return []
    sections = doc.get("sections") if isinstance(doc.get("sections"), dict) else {}
    lines: list[str] = []
    mapping = (
        ("complaints", "Жалобы"),
        ("diagnosis_text", "Диагноз"),
        ("recommendations_treatment", "Лечение"),
        ("follow_up_text", "Контроль"),
    )
    for key, label in mapping:
        val = str(sections.get(key) or "").strip()
        if val and len(val) > 8:
            lines.append(f"{label}: {val[:160]}{'…' if len(val) > 160 else ''}")
    diags = doc.get("diagnoses") if isinstance(doc.get("diagnoses"), list) else []
    if not any(l.startswith("Диагноз:") for l in lines) and diags:
        d0 = diags[0] if isinstance(diags[0], dict) else {}
        txt = str(d0.get("text_ru") or d0.get("icd10_code") or "").strip()
        if txt:
            lines.insert(1, f"Диагноз: {txt[:160]}")
    return lines[:5]


def _document_quality(conf: int | None, limitations: str) -> dict[str, Any]:
    hint = "Документ читается хорошо."
    level = "good"
    if conf is not None and conf < 55:
        hint = "Качество распознавания низкое - переснимите при хорошем свете или загрузите PDF."
        level = "low"
    elif conf is not None and conf < 75:
        hint = "Часть текста распознана не полностью - проверьте, что все страницы загружены."
        level = "medium"
    elif limitations:
        hint = limitations[:200]
        level = "medium"
    return {"confidence_pct": conf, "level": level, "hint_ru": hint}


def _priority_topics(
    blocks: list[dict[str, Any]],
    protocol_context: dict[str, Any] | None,
) -> list[dict[str, str]]:
    topics: list[dict[str, str]] = []
    for b in blocks:
        if b.get("status") == "concern":
            topics.append(
                {
                    "topic": str(b.get("title") or b.get("id") or "Раздел"),
                    "why_ru": b.get("summary_ru") or "; ".join(b.get("gaps") or []) or "Есть замечания.",
                    "severity": "high",
                }
            )
        elif b.get("status") == "attention" and len(topics) < 5:
            why = b.get("summary_ru") or (b.get("gaps") or [""])[0]
            if why:
                topics.append({"topic": str(b.get("title") or b.get("id")), "why_ru": why, "severity": "medium"})
    if protocol_context:
        for m in protocol_context.get("missing_recommended_exams") or []:
            if not isinstance(m, dict):
                continue
            topics.insert(
                0,
                {
                    "topic": "Протокол Минздрава",
                    "why_ru": str(m.get("patient_note_ru") or m.get("exam_name") or ""),
                    "severity": str(m.get("severity") or "high"),
                },
            )
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for t in topics:
        k = t.get("topic", "") + t.get("why_ru", "")[:40]
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= 3:
            break
    return out


def _plain_summary(
    light: TrafficLight,
    blocks: list[dict[str, Any]],
    questions: list[dict[str, str]],
    protocol_context: dict[str, Any] | None,
) -> str:
    if light == "green" and not questions:
        return (
            "По основным разделам заключение в целом согласуется с клиническими протоколами Минздрава. "
            "Критичных пробелов для обсуждения с врачом не найдено."
        )
    weak = [b for b in blocks if b.get("status") in ("concern", "attention")]
    names = [str(b.get("title") or "") for b in weak[:3] if b.get("title")]
    parts: list[str] = []
    if names:
        parts.append("Обратите внимание на разделы: " + ", ".join(names) + ".")
    if questions:
        parts.append(f"Подготовлено {len(questions)} вопрос(ов) для разговора с врачом.")
    if protocol_context and protocol_context.get("missing_recommended_exams"):
        parts.append("Есть расхождения с рекомендациями протокола по обследованиям.")
    if light == "red":
        parts.insert(0, "В заключении много неучтённого по стандарту Минздрава - обсудите выписку с врачом.")
    elif light == "yellow" and not parts:
        parts.append("Есть отдельные пробелы - используйте чек-лист вопросов ниже.")
    return " ".join(parts) if parts else traffic_light_for_pct(75 if light == "green" else 60)[1]


def build_patient_report(
    l1_result: dict[str, Any],
    *,
    lab_crosscheck: dict[str, Any] | None = None,
    protocol_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Преобразует результат L1 structured в отчёт для пациента."""
    align = l1_result.get("alignment") if isinstance(l1_result.get("alignment"), dict) else {}
    cards = list(align.get("alignment_cards") or [])
    overall = resolve_patient_overall_pct(l1_result)
    light, overall_label = traffic_light_for_pct(overall)

    conf = _clamp_pct(l1_result.get("confidence_score"))
    limitations = str(
        align.get("limitations_ru")
        or (l1_result.get("review") or {}).get("limitations_ru")
        or ""
    ).strip()

    structured_questions = _collect_structured_questions(cards)
    if lab_crosscheck:
        for note in lab_crosscheck.get("notes_ru") or []:
            if note and not any(q.get("text") == note for q in structured_questions):
                structured_questions.insert(
                    0,
                    {"id": f"q{len(structured_questions)+1}", "title": _question_title(note), "text": note, "severity": "medium"},
                )
    if protocol_context:
        for m in protocol_context.get("missing_recommended_exams") or []:
            if not isinstance(m, dict):
                continue
            note = str(m.get("patient_note_ru") or "").strip()
            if note and not any(q.get("text") == note for q in structured_questions):
                structured_questions.insert(
                    0,
                    {"id": f"q{len(structured_questions)+1}", "title": _question_title(note), "text": note, "severity": "high"},
                )
    structured_questions = structured_questions[:8]

    blocks = _patient_blocks(cards)
    action_checklist = [
        {"id": q["id"], "text": q["text"], "title": q["title"], "severity": q.get("severity", "medium"), "checked": False}
        for q in structured_questions
    ]

    if conf is not None and conf < 55 and limitations:
        warn = f"Качество распознавания документа низкое ({conf}%). {limitations}"
        structured_questions.insert(0, {"id": "q0", "title": "Качество документа", "text": warn, "severity": "high"})
        action_checklist.insert(0, {"id": "q0", "text": warn, "title": "Качество документа", "severity": "high", "checked": False})

    report = {
        "overall_pct": overall,
        "overall_label_ru": overall_label,
        "traffic_light": light,
        "plain_summary_ru": _plain_summary(light, blocks, structured_questions, protocol_context),
        "document_read_back_ru": _document_read_back(l1_result),
        "document_quality": _document_quality(conf, limitations),
        "priority_topics": _priority_topics(blocks, protocol_context),
        "blocks": blocks,
        "questions_for_doctor": [q["text"] for q in structured_questions],
        "questions_structured": structured_questions,
        "action_checklist": action_checklist,
        "protocol_citations": _collect_citations(cards),
        "limitations_ru": limitations,
        "confidence_score": conf,
        "disclaimer_ru": PATIENT_DISCLAIMER_RU,
        "matched_protocols_count": int(l1_result.get("matched_protocols_count") or 0),
        "next_steps_ru": [
            "Прочитайте краткий итог и вопросы ниже.",
            "Отметьте, что уже обсудили с врачом.",
            "На приёме покажите список или сохраните PDF.",
        ],
    }
    if lab_crosscheck:
        report["lab_crosscheck"] = lab_crosscheck
    if protocol_context:
        report["protocol_context"] = protocol_context
    return report


def sanitize_patient_api_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Убирает B2B-поля (ЦИСЗ, send_gate, сырой structured)."""
    out = dict(payload)
    for key in (
        "send_gate",
        "cisz_readiness",
        "structured_analysis",
        "alignment",
        "report_html",
        "report_markdown",
        "review",
    ):
        out.pop(key, None)
    return out
