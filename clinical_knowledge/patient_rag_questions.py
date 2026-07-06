"""Вопросы врачу из RAG-чанков (target-agnostic дополнение к patient_question_builder)."""
from __future__ import annotations

import re
from typing import Any

from .patient_exam_extraction import extract_exams_from_text
from .patient_medication_extraction import extract_medications_from_text
from .patient_questions import sanitize_question_text

_ANCHOR_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "exams",
        re.compile(r"(?:узи|кт|мрт|рентген|обследован|анализ|лабор)", re.I),
        "Когда нужно пройти назначенное обследование и куда записаться?",
    ),
    (
        "treatment",
        re.compile(r"(?:парацетамол|ибупрофен|антибиот|препарат|лечен|мг\b|таблет)", re.I),
        "Как правильно принимать назначенные препараты и сколько дней?",
    ),
    (
        "follow_up",
        re.compile(r"(?:контрол|повторн|явк|через\s+\d+)", re.I),
        "Когда записаться на повторный осмотр и при каких симптомах обращаться раньше?",
    ),
    (
        "diagnosis",
        re.compile(r"(?:орви|грип|диагноз|\?)", re.I),
        "Какие обследования подтверждают диагноз и когда ждать улучшения?",
    ),
]

_CHUNK_ACTION_RE = re.compile(
    r"(?:рекоменду(?:ется|ют)|назнача(?:ется|ют)|показан|следует|необходим)",
    re.I,
)


def _dedupe_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())[:72]


def _anchor_in_kz(kz_text: str, block_id: str) -> bool:
    low = (kz_text or "").lower()
    for bid, pat, _ in _ANCHOR_PATTERNS:
        if bid == block_id and pat.search(low):
            return True
    return False


def _question_from_chunk(
    *,
    chunk_text: str,
    chunk_title: str,
    kz_text: str,
    qid: str,
) -> dict[str, Any] | None:
    excerpt = re.sub(r"\s+", " ", (chunk_text or "").strip())
    if len(excerpt) < 50 or not _CHUNK_ACTION_RE.search(excerpt):
        return None

    block_id = "clarify"
    question = ""
    for bid, pat, template in _ANCHOR_PATTERNS:
        if pat.search(kz_text or "") or pat.search(excerpt):
            block_id = bid
            question = template
            break
    if not question:
        question = "Что из рекомендаций протокола относится к моему случаю и что уточнить?"

    plain = excerpt[:220].rstrip() + ("…" if len(excerpt) > 220 else "")
    title = (chunk_title or "Протокол").strip()[:80]
    text = sanitize_question_text(question)
    if not text:
        return None
    return {
        "id": qid,
        "text": text,
        "title": text.split("?")[0].strip()[:72] + "?",
        "why_ru": f"В протоколах МЗ ({title}) обычно уточняют подобные пункты.",
        "plain_context": plain,
        "severity": "medium",
        "category_ru": "Протокол",
        "block_id": block_id,
        "intent": "rag_chunk",
        "priority": 55,
        "source_gap": "",
        "source_comment": "",
        "tone": "serious",
        "emoji": "💬",
    }


def augment_questions_from_retrieved(
    report: dict[str, Any],
    *,
    retrieved: list[dict[str, Any]],
    kz_text: str,
    limit: int = 3,
) -> dict[str, Any]:
    """Добавить до `limit` вопросов из top RAG-чанков, не дублируя существующие."""
    if not isinstance(report, dict) or not retrieved:
        return report

    existing = list(report.get("questions_structured") or [])
    seen = {_dedupe_key(str(q.get("text") or "")) for q in existing if q.get("text")}

    added: list[dict[str, Any]] = []
    for i, row in enumerate(retrieved):
        if not isinstance(row, dict):
            continue
        chunk_text = str(row.get("text") or "")
        if not _anchor_in_kz(kz_text, "exams") and not _anchor_in_kz(kz_text, "treatment"):
            if not _ANCHOR_PATTERNS[0][1].search(chunk_text) and not _ANCHOR_PATTERNS[1][1].search(chunk_text):
                if len((kz_text or "").strip()) < 120:
                    pass
                elif not _CHUNK_ACTION_RE.search(chunk_text):
                    continue
        qrow = _question_from_chunk(
            chunk_text=chunk_text,
            chunk_title=str(row.get("section_title") or row.get("title") or ""),
            kz_text=kz_text,
            qid=f"q-rag-{i}",
        )
        if not qrow:
            continue
        key = _dedupe_key(qrow["text"])
        if key in seen:
            continue
        seen.add(key)
        added.append(qrow)
        if len(added) >= limit:
            break

    if not added:
        return report

    merged = existing + added
    merged.sort(key=lambda x: int(x.get("priority") or 99))
    report = dict(report)
    report["questions_structured"] = merged[: max(len(existing), 5)]
    report["questions_for_doctor"] = [q["text"] for q in report["questions_structured"] if q.get("text")]
    report["action_checklist"] = [
        {
            "id": q.get("id", f"q{i+1}"),
            "text": q.get("text", ""),
            "title": q.get("title", ""),
            "severity": q.get("severity", "medium"),
            "category_ru": q.get("category_ru", ""),
            "block_id": q.get("block_id", ""),
            "tone": q.get("tone") or "serious",
            "emoji": q.get("emoji") or "💬",
            "why_ru": q.get("why_ru") or "",
            "plain_context": q.get("plain_context") or "",
            "checked": False,
        }
        for i, q in enumerate(report["questions_structured"])
    ]
    return report
