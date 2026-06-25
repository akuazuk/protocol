"""Требования СОП Кравира к амбулаторной карте (doc_kravira, СОП № 2).

Используются как эталон для колонки «Эталон / источник» и для findings/gaps
в детерминированных карточках согласования КЗ.
"""
from __future__ import annotations

import re
from typing import Any

from clinical_knowledge.consult_schema import ConsultationDocument
from clinical_knowledge.kz_clinical_context import split_anamnesis_parts
from clinical_knowledge.meaningful_excerpt import meaningful_excerpt

KRAVIRA_SOP_SOURCE = "СОП № 2 Кравира (амбулаторная карта)"

_BLOCK_REFERENCES: dict[str, str] = {
    "complaints": (
        "Жалобы с детализацией: характер, давность, динамика "
        "(СОП № 2, первичный осмотр, п. 1)."
    ),
    "anamnesis": (
        "Анамнез заболевания и жизни, аллергоанамнез, наследственный анамнез, "
        "перенесённые заболевания (СОП № 2, п. 1)."
    ),
    "objective_status": (
        "Общий осмотр: сознание, антропометрия, температура, АД, пульс, ЧД; "
        "локальный статус в полном объёме, согласованный с диагнозом "
        "(СОП № 2, п. 2; NB: локальный статус обязателен)."
    ),
    "diagnosis": (
        "Диагноз по МКБ-10; при отсутствии патологии - код профилактического "
        "осмотра (например Z01.4), формулировки «здоров» не используют "
        "(СОП № 2, п. 3)."
    ),
    "exams": "Обследование и назначения - по клиническому протоколу Минздрава (СОП № 2, п. 4).",
    "treatment": "Лечение, рекомендации и режим - по клиническому протоколу (СОП № 2, п. 4).",
    "follow_up": (
        "Сроки контроля, динамика состояния при повторных приёмах "
        "(СОП № 2, повторный осмотр)."
    ),
}

_DETAIL_HINTS = re.compile(
    r"дн[еяюё]|недел|месяц|лет\b|год\b|сутк|час|внезап|постепен|усил|слаб|"
    r"локализ|справа|слева|верх|низ|ноч|утр|после|до\b|при\b|без\b|характер|"
    r"интенсив|умерен|сильн|период|эпизод|обостр",
    re.I,
)
_VITALS = re.compile(
    r"ад\b|артериальн\w*\s+давлен|давлен\w*\s+\d|мм\s*рт|пульс|чсс\b|"
    r"чд\b|частот\w*\s+дыхан|температур|°|сатурац|spo2|антропометр|рост|вес\b|имт",
    re.I,
)
_DATE_TIME = re.compile(
    r"дата\s+консультац|дата\s+осмотр|время\s+осмотр|\d{1,2}[./]\d{1,2}[./]\d{2,4}",
    re.I,
)
_ALLERGY = re.compile(r"аллерг|непереносим|гиперчувств", re.I)
_HEALTHY_BAD = re.compile(r"\bздоров\w*\b|\bбез\s+патолог", re.I)


def sop_reference_for_block(block_id: str) -> str:
    return _BLOCK_REFERENCES.get(block_id, KRAVIRA_SOP_SOURCE)


def _blob(*parts: str | None) -> str:
    return "\n".join(p.strip() for p in parts if (p or "").strip())


def evaluate_sop_block(doc: ConsultationDocument, block_id: str) -> dict[str, Any]:
    """Findings/gaps по СОП Кравira для блока alignment."""
    findings: list[str] = []
    gaps: list[str] = []
    reference = sop_reference_for_block(block_id)
    s = doc.sections
    anam = split_anamnesis_parts(doc)
    penalty = 0

    if block_id == "complaints":
        text = (s.complaints or "").strip()
        if not text:
            gaps.append("Жалобы не заполнены.")
            penalty += 25
        elif len(text) < 20:
            gaps.append("Жалобы указаны слишком кратко - нужна детализация.")
            penalty += 12
        elif not _DETAIL_HINTS.search(text):
            gaps.append("В жалобах нет давности, локализации или динамики.")
            penalty += 8
        else:
            findings.append("Жалобы описаны развёрнуто.")

    elif block_id == "anamnesis":
        disease = anam.get("disease") or ""
        life = anam.get("life") or (s.life_history or "")
        allergy = (s.allergy_history or "").strip()
        if disease:
            findings.append("Анамнез заболевания отражён.")
        else:
            gaps.append("Анамнез заболевания не описан.")
            penalty += 15
        if life:
            findings.append("Анамнез жизни отражён.")
        else:
            gaps.append("Анамнез жизни не указан.")
            penalty += 10
        if allergy or _ALLERGY.search(_blob(disease, life, s.anamnesis or "")):
            findings.append("Аллергоанамнез учтён.")
        else:
            gaps.append("Аллергоанамнез не отражён (в СОП - обязательный пункт).")
            penalty += 8

    elif block_id == "objective_status":
        obj = (s.objective_status or "").strip()
        local = (s.local_status or "").strip()
        combined = _blob(obj, local)
        if not obj:
            gaps.append("Объективный статус не описан.")
            penalty += 20
        elif len(obj) < 25:
            gaps.append("Объективный статус слишком краткий.")
            penalty += 10
        else:
            findings.append("Объективный статус заполнен.")
        if _VITALS.search(combined):
            findings.append("В статусе есть витальные параметры (АД, пульс, температура и т.п.).")
        else:
            gaps.append("Не отражены витальные параметры (АД, пульс, ЧД, температура).")
            penalty += 10
        if local:
            findings.append("Локальный статус описан отдельно.")
        elif len(obj) >= 80 and not re.search(r"локальн|статус|осмотр", obj, re.I):
            gaps.append("Локальный статус не выделен - по СОП описывается полностью.")
            penalty += 8

    elif block_id == "diagnosis":
        diag_text = (s.diagnosis_text or "").strip()
        if _HEALTHY_BAD.search(diag_text) and not re.search(r"\bZ\d", diag_text, re.I):
            gaps.append("Формулировка «здоров» без кода Z - по СОП Кравira не допускается.")
            penalty += 15
        if doc.diagnoses or re.search(r"\b[A-TV-Z]\d{2}", diag_text, re.I):
            findings.append("Диагноз с кодом МКБ-10 указан.")
        else:
            gaps.append("Код МКБ-10 в диагнозе не найден.")
            penalty += 12

    elif block_id == "follow_up":
        follow = _blob(s.follow_up_text, s.general_recommendations)
        if follow and re.search(r"контрол|повтор|через\s+\d|явк|наблюден|диспанс", follow, re.I):
            findings.append("Сроки наблюдения или контроля указаны.")
        elif s.recommendations_treatment or doc.medications:
            gaps.append("Лечение назначено, но срок контрольного визита не описан явно.")
            penalty += 8

    header_blob = _blob(s.header, doc.raw_text[:400] if getattr(doc, "raw_text", None) else "")
    if block_id in ("complaints", "anamnesis", "objective_status") and _DATE_TIME.search(header_blob):
        findings.append("Дата или время осмотра указаны.")

    return {
        "findings_ru": findings,
        "gaps_ru": gaps,
        "reference_ru": reference,
        "score_penalty": min(penalty, 35),
    }


def merge_sop_into_card(card: dict[str, Any], sop: dict[str, Any]) -> None:
    """Дополнить карточку findings/gaps и эталоном СОП (in-place)."""
    if card.get("source_kind") in ("kp", "mkb") and card.get("protocol_excerpt"):
        ref = sop.get("reference_ru") or ""
        if ref and card.get("source_kind") == "kp":
            card.setdefault("reference_ru", ref)
    elif card.get("source_kind") in ("completeness", "regulation"):
        card["reference_ru"] = sop.get("reference_ru") or card.get("reference_ru") or ""
        if not (card.get("protocol_excerpt") or "").strip():
            card["protocol_excerpt"] = card["reference_ru"]
        if not (card.get("protocol_section") or "").strip():
            card["protocol_section"] = KRAVIRA_SOP_SOURCE

    existing_f = list(card.get("findings_ru") or [])
    existing_g = list(card.get("gaps_ru") or [])
    for line in sop.get("findings_ru") or []:
        if line not in existing_f:
            existing_f.append(line)
    for line in sop.get("gaps_ru") or []:
        if line not in existing_g:
            existing_g.append(line)
    card["findings_ru"] = existing_f[:8]
    card["gaps_ru"] = existing_g[:8]

    pen = int(sop.get("score_penalty") or 0)
    if pen and card.get("block_id") in (
        "complaints", "anamnesis", "objective_status", "diagnosis", "follow_up",
    ):
        card["score_pct"] = max(0, int(card.get("score_pct") or 0) - pen)

