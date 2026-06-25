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


def _collect_questions(cards: list[dict[str, Any]], limit: int = 6) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for card in cards:
        if not isinstance(card, dict):
            continue
        if card.get("block_id") == "limitations":
            continue
        name = str(card.get("name_ru") or "").strip()
        score = _clamp_pct(card.get("score_pct"))
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
            out.append(q)
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


def build_patient_report(l1_result: dict[str, Any]) -> dict[str, Any]:
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

    questions = _collect_questions(cards)
    if conf is not None and conf < 55 and limitations:
        questions = [f"Качество распознавания документа низкое ({conf}%). {limitations}"] + questions
        questions = questions[:6]

    return {
        "overall_pct": overall,
        "overall_label_ru": overall_label,
        "traffic_light": light,
        "blocks": _patient_blocks(cards),
        "questions_for_doctor": questions,
        "protocol_citations": _collect_citations(cards),
        "limitations_ru": limitations,
        "confidence_score": conf,
        "disclaimer_ru": PATIENT_DISCLAIMER_RU,
        "matched_protocols_count": int(l1_result.get("matched_protocols_count") or 0),
    }


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
