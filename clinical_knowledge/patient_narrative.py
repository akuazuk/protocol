"""Фактологические тексты отчёта B2C из содержимого КЗ (не шаблоны неврологии)."""
from __future__ import annotations

import re
from typing import Any

from .patient_exam_extraction import imaging_exams, lab_exams


def _section_after(text: str, label: str, *, max_len: int = 220) -> str:
    pat = re.compile(rf"{re.escape(label)}\s*:?\s*(.+?)(?:\n\s*\n|\n[A-ZА-ЯЁ][^\n]{{3,}}:|\Z)", re.I | re.S)
    m = pat.search(text or "")
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()[:max_len]


def extract_complaint_phrase(kz_text: str) -> str:
    low = (kz_text or "").lower()
    sec = _section_after(kz_text, "Жалобы")
    if sec and len(sec) > 6:
        sec = re.sub(r"\s+", " ", sec).strip()
        sec = re.sub(r"^на\s+", "", sec, flags=re.I)
        sec = sec.replace(" а коже", " на коже")
        return sec[:120]
    if "высыпан" in low:
        return "высыпания на коже"
    if "головн" in low and "бол" in low:
        return "головная боль"
    if "бол" in low and "кож" in low:
        return "боль и изменения на коже"
    return "обращения"


def extract_diagnosis_phrase(kz_text: str) -> str:
    sec = _section_after(kz_text, "Диагноз")
    if sec:
        sec = sec.replace("?", "").strip()
        if len(sec) > 8:
            return sec[:140]
    low = (kz_text or "").lower()
    m = re.search(r"\b([A-Z]\d{2}(?:\.\d)?)\b", kz_text or "")
    code = m.group(1) if m else ""
    if "l93" in low.replace(" ", "") or "волчан" in low:
        return f"{code} - дискоидная красная волчанка".strip(" -")
    if code:
        return code
    return ""


def extract_specialty_phrase(kz_text: str, specialty: str | None) -> str:
    low = (kz_text or "").lower()
    if specialty == "neurology" or "невролог" in low:
        return "невролога"
    if specialty == "dermatology" or "дерматовенеролог" in low or "дерматолог" in low:
        return "дерматовенеролога"
    return "врача"


def extract_follow_up_phrase(kz_text: str) -> str:
    sec = _section_after(kz_text, "Повторный осмотр", max_len=120)
    if sec:
        return sec
    m = re.search(r"повторн\w*\s+осмотр[^.\n]{0,80}", kz_text or "", re.I)
    return m.group(0).strip() if m else ""


def follow_up_has_deadline(kz_text: str) -> bool:
    fu = extract_follow_up_phrase(kz_text).lower()
    if re.search(r"\d+\s*(?:дн|дней|нед|недель|мес)", fu):
        return True
    return bool(re.search(r"через\s+\d+", (kz_text or "").lower()))


def build_clarification_points(
    *,
    meds: list[dict[str, Any]],
    exams: list[dict[str, Any]],
    kz_text: str,
) -> list[dict[str, str]]:
    points: list[dict[str, str]] = []
    imaging = imaging_exams(exams)
    labs = lab_exams(exams)

    for ex in imaging:
        label = str(ex.get("label_ru") or ex.get("exam_type") or "обследование")
        if ex.get("exam_type") == "MRI":
            points.append({"topic_ru": "МРТ", "text_ru": f"когда выполнить {label.lower()}"})
        else:
            points.append({"topic_ru": "Обследования", "text_ru": f"нужно ли и когда пройти {label.lower()}"})

    if labs:
        labels = ", ".join(str(e.get("label_ru") or "анализ") for e in labs[:3])
        points.append({"topic_ru": "Анализы", "text_ru": f"когда сдать {labels.lower()}"})

    fu = extract_follow_up_phrase(kz_text)
    if fu:
        if follow_up_has_deadline(kz_text):
            points.append({"topic_ru": "Контроль", "text_ru": f"что подготовить к визиту ({fu.lower()})"})
        else:
            points.append({"topic_ru": "Контроль", "text_ru": "когда повторный осмотр"})
    elif "контрол" in (kz_text or "").lower() or "повторн" in (kz_text or "").lower():
        points.append({"topic_ru": "Контроль", "text_ru": "когда повторный осмотр"})

    if any(m.get("start_condition") == "после" for m in meds):
        points.append({"topic_ru": "Лечение", "text_ru": "что означает «после» в схеме лечения"})
    if meds and any("duration_missing" in (m.get("clarity_issues") or []) for m in meds):
        points.append({"topic_ru": "Лечение", "text_ru": "сколько дней принимать препараты"})

    low = (kz_text or "").lower()
    if "головн" in low:
        worse = "если головная боль сохранится или усилится"
    elif "высыпан" in low or "кож" in low:
        worse = "если высыпания распространятся или появится температура"
    else:
        worse = "если самочувствие не улучится"
    points.append({"topic_ru": "Самочувствие", "text_ru": f"что делать, {worse}"})
    points.append({"topic_ru": "Безопасность", "text_ru": "какие симптомы требуют срочного обращения"})
    return points[:8]


def build_main_takeaway(
    *,
    exams: list[dict[str, Any]],
    meds: list[dict[str, Any]],
    kz_text: str,
) -> str:
    parts: list[str] = []
    imaging = imaging_exams(exams)
    labs = lab_exams(exams)
    if imaging:
        parts.append("сроки обследований")
    if labs:
        parts.append("сроки анализов")
    if not follow_up_has_deadline(kz_text) and ("повторн" in (kz_text or "").lower() or "контрол" in (kz_text or "").lower()):
        parts.append("дату повторного осмотра")
    if meds and any("duration_missing" in (m.get("clarity_issues") or []) for m in meds):
        parts.append("длительность лечения")
    if not parts:
        return "Стоит уточнить детали назначений, которые неясны в заключении."
    return "Стоит уточнить " + ", ".join(parts) + "."


def build_top_summary_plain(
    *,
    specialty: str | None,
    kz_text: str,
    exams: list[dict[str, Any]],
    meds: list[dict[str, Any]],
) -> str:
    who = extract_specialty_phrase(kz_text, specialty)
    complaint = extract_complaint_phrase(kz_text)
    diag = extract_diagnosis_phrase(kz_text)
    imaging = imaging_exams(exams)
    labs = lab_exams(exams)

    parts: list[str] = [f"Вы были на консультации {who} по поводу {complaint}."]
    if diag:
        parts.append(f"Врач указал диагноз: {diag}.")
    else:
        parts.append("Диагноз указан в заключении.")

    action_bits: list[str] = []
    if imaging:
        labels = ", ".join(str(e.get("label_ru") or "") for e in imaging[:2])
        action_bits.append(f"назначены обследования ({labels.lower()})")
    if labs:
        labels = ", ".join(str(e.get("label_ru") or "") for e in labs[:2])
        action_bits.append(f"рекомендованы анализы ({labels.lower()})")
    if meds:
        action_bits.append("назначено лечение")
    fu = extract_follow_up_phrase(kz_text)
    if fu:
        action_bits.append(fu)

    if action_bits:
        parts.append("В заключении: " + ", ".join(action_bits) + ".")
    parts.append(build_main_takeaway(exams=exams, meds=meds, kz_text=kz_text))
    return " ".join(parts)


def red_flags_for_context(kz_text: str, specialty: str | None) -> str:
    low = (kz_text or "").lower()
    base = (
        "Срочно обратитесь за медицинской помощью, если состояние резко ухудшается, "
        "появилась высокая температура, сильная слабость, потеря сознания "
        "или другие необычные симптомы."
    )
    if "головн" in low or specialty == "neurology":
        base += (
            " При неврологических жалобах - внезапная очень сильная головная боль, "
            "слабость в руке или ноге, нарушение речи."
        )
    if "высыпан" in low or "кож" in low or specialty == "dermatology" or "l93" in low.replace(" ", ""):
        base += " При кожных проблемах - быстрое распространение высыпаний, отёк лица, одышка."
    base += " Это общая справочная информация, не диагноз."
    return base
