"""Полезные вопросы врачу для пациента - из фактов КЗ, не из SOP-пробелов методиста."""
from __future__ import annotations

import re
from typing import Any

from .patient_exam_extraction import imaging_exams, lab_exams
from .patient_medication_extraction import extract_medications_from_text
from .patient_narrative import extract_follow_up_phrase, follow_up_has_deadline
from .patient_questions import sanitize_question_text

# Пробелы L1 для методиста - не превращаем в вопросы пациенту
_SKIP_GAP_RE = re.compile(
    r"аллергоанамнез|локальный статус не выделен|"
    r"в жалобах нет давности|в жалобах нет|не отражён \(в соп|"
    r"объективном статусе не описан важный признак|"
    r"в анамнезе не указаны важные сведения",
    re.I,
)


def _norm_q(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if not t.endswith("?"):
        t = t.rstrip(".") + "?"
    if t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def _dedupe_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())[:72]


def _question_row(
    *,
    qid: str,
    text: str,
    why_ru: str,
    category_ru: str,
    block_id: str,
    severity: str = "medium",
    intent: str = "",
    priority: int = 50,
) -> dict[str, Any]:
    text = sanitize_question_text(_norm_q(text))
    if not text:
        return {}
    return {
        "id": qid,
        "text": text,
        "title": text.split("?")[0].strip()[:72] + "?",
        "why_ru": why_ru.strip(),
        "severity": severity,
        "category_ru": category_ru,
        "block_id": block_id,
        "intent": intent or block_id,
        "priority": priority,
        "source_gap": "",
        "source_comment": "",
    }


def build_useful_patient_questions(
    *,
    kz_text: str,
    clarification_points: list[dict[str, str]] | None = None,
    exams: list[dict[str, Any]] | None = None,
    meds: list[dict[str, Any]] | None = None,
    lab_crosscheck: dict[str, Any] | None = None,
    structured_gaps: list[dict[str, Any]] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Собрать 3-5 вопросов, которые пациент реально задаст на приёме."""
    exams = exams or []
    meds = meds or list(extract_medications_from_text(kz_text))
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        if not row:
            return
        key = _dedupe_key(row.get("text") or "")
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append(row)

    # 1. Анализы vs заключение (высокий приоритет)
    if lab_crosscheck:
        miss = lab_crosscheck.get("missing_in_kz_lines") or []
        panels = lab_crosscheck.get("panels_ru") or []
        if miss:
            sample = ", ".join(str(x) for x in miss[:3])
            panel = panels[0] if panels else "анализах"
            add(
                _question_row(
                    qid="q-labs",
                    text=f"В {panel.lower()} есть показатели ({sample}) - учли ли вы их при назначении лечения?",
                    why_ru="Результаты анализов не отражены в тексте заключения.",
                    category_ru="Анализы",
                    block_id="labs",
                    severity="high",
                    intent="labs_missing_in_kz",
                    priority=10,
                )
            )

    # 2. Сроки обследований и анализов
    imaging = imaging_exams(exams)
    labs = lab_exams(exams)
    if imaging:
        labels = ", ".join(str(e.get("label_ru") or "обследование") for e in imaging[:2])
        add(
            _question_row(
                qid="q-imaging",
                text=f"Когда нужно пройти {labels.lower()} и куда записаться?",
                why_ru="В заключении назначено обследование, но срок не указан.",
                category_ru="Обследования",
                block_id="exams",
                severity="medium",
                intent="exams_timing",
                priority=20,
            )
        )
    if labs:
        labels = ", ".join(str(e.get("label_ru") or "анализ") for e in labs[:2])
        add(
            _question_row(
                qid="q-labs-timing",
                text=f"Когда сдать {labels.lower()} и нужна ли подготовка?",
                why_ru="Назначены анализы без сроков или правил подготовки.",
                category_ru="Анализы",
                block_id="exams",
                severity="medium",
                intent="labs_plan",
                priority=22,
            )
        )

    # 3. Лечение
    if meds:
        if any(m.get("start_condition") == "после" for m in meds):
            add(
                _question_row(
                    qid="q-after",
                    text="Как правильно понимать слово «после» в схеме лечения - с какого дня начинать?",
                    why_ru="В назначениях есть этап «после», который не расшифрован.",
                    category_ru="Лечение",
                    block_id="treatment",
                    severity="high",
                    intent="treatment_after",
                    priority=25,
                )
            )
        if any("duration_missing" in (m.get("clarity_issues") or []) for m in meds):
            names = ", ".join(str(m.get("name") or "") for m in meds[:2] if m.get("name"))
            add(
                _question_row(
                    qid="q-duration",
                    text=f"На сколько дней назначены {names or 'препараты'} и что делать после окончания курса?",
                    why_ru="Не указана длительность приёма лекарств.",
                    category_ru="Лечение",
                    block_id="treatment",
                    severity="medium",
                    intent="treatment_duration",
                    priority=28,
                )
            )
        if len(meds) >= 2:
            add(
                _question_row(
                    qid="q-meds-order",
                    text="В какой последовательности принимать назначенные препараты - можно ли вместе или в разное время?",
                    why_ru="Несколько препаратов - важно не перепутать схему.",
                    category_ru="Лечение",
                    block_id="treatment",
                    severity="medium",
                    intent="treatment_order",
                    priority=35,
                )
            )

    # 4. Контрольный визит
    fu = extract_follow_up_phrase(kz_text)
    if fu and not follow_up_has_deadline(kz_text):
        add(
            _question_row(
                qid="q-follow",
                text="Когда записаться на повторный осмотр и что взять с собой?",
                why_ru="Контрольный визит упомянут без точной даты.",
                category_ru="Контроль",
                block_id="follow_up",
                severity="medium",
                intent="follow_up",
                priority=30,
            )
        )

    # 5. Из clarification_points (уже patient-facing)
    for i, cp in enumerate(clarification_points or []):
        topic = str(cp.get("topic_ru") or "").strip()
        hint = str(cp.get("text_ru") or "").strip()
        if not hint:
            continue
        if topic in ("Безопасность", "Самочувствие"):
            continue
        text = hint[0].upper() + hint[1:] if hint else ""
        if not text.endswith("?"):
            text = "Подскажите, пожалуйста: " + text + "?"
        add(
            _question_row(
                qid=f"q-cl-{i}",
                text=text,
                why_ru=f"Стоит уточнить: {hint.rstrip('.')}.",
                category_ru=topic or "Уточнение",
                block_id="clarify",
                severity="medium",
                intent="clarify",
                priority=40 + i,
            )
        )

    # 6. Диагноз с «?»
    if "?" in (kz_text or "") and re.search(r"диагноз", kz_text or "", re.I):
        add(
            _question_row(
                qid="q-dx",
                text="Диагноз указан предварительно - какие обследования подтвердят или опровергнут его?",
                why_ru="В заключении диагноз сформулирован с сомнением.",
                category_ru="Диагноз",
                block_id="diagnosis",
                severity="medium",
                intent="diagnosis_uncertain",
                priority=15,
            )
        )

    # 7. Самочувствие (один универсальный)
    low = (kz_text or "").lower()
    if "головн" in low:
        worse = "если головная боль не уменьшится или усилится"
    elif "высыпан" in low or "кож" in low:
        worse = "если высыпания распространятся или появится температура"
    else:
        worse = "если самочувствие не улучится"
    add(
        _question_row(
            qid="q-worse",
            text=f"Что делать, {worse}, и когда обращаться срочно?",
            why_ru="Важно заранее знать план действий при ухудшении.",
            category_ru="Самочувствие",
            block_id="follow_up",
            severity="low",
            intent="symptoms_worse",
            priority=90,
        )
    )

    # Низкоприоритетные L1 gaps - только если мало вопросов и gap actionable
    if len(candidates) < 3 and structured_gaps:
        for g in structured_gaps:
            raw = str(g.get("source_gap") or g.get("source_comment") or "").strip()
            if not raw or _SKIP_GAP_RE.search(raw):
                continue
            bid = str(g.get("block_id") or "")
            if bid in ("anamnesis", "objective_status", "complaints"):
                continue
            add(
                _question_row(
                    qid=f"q-gap-{len(candidates)}",
                    text=f"Поясните, пожалуйста: {raw.rstrip('.')}?",
                    why_ru="Этот пункт в заключении описан неполно.",
                    category_ru=str(g.get("category_ru") or "Уточнение"),
                    block_id=bid,
                    severity=str(g.get("severity") or "medium"),
                    intent="gap",
                    priority=70,
                )
            )
            if len(candidates) >= limit:
                break

    candidates.sort(key=lambda x: int(x.get("priority") or 99))
    return candidates[:limit]
