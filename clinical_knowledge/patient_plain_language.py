"""Пояснение медицинских терминов простым языком (B2C)."""
from __future__ import annotations

import re
from typing import Any

_TERM_DICT: dict[str, str] = {
    "цервикокраниалгия": "Боль, связанная с шейным отделом и отдающая в область головы.",
    "шейно-черепной синдром": "Сочетание болей и напряжения в шее с симптомами в голове.",
    "вертеброгенная": "Связанная с позвоночником (позвонками).",
    "мышечно-тонический синдром": "Болезненное напряжение мышц, которое может поддерживать боль.",
    "симптом ласега": "Неврологический тест, который врач использует при осмотре.",
    "менingeальные знаки": "Признаки, которые врач проверяет для исключения опасных состояний.",
    "менингеальные знаки": "Признаки, которые врач проверяет для исключения опасных состояний.",
    "радикулопатия": "Раздражение или сдавление нервного корешка, часто с болью и онемением.",
    "ишиас": "Боль, отдающая по ходу нерва, часто в ногу.",
    "флеботромбоз": "Сгусток крови в вене.",
    "мигрень": "Приступообразная головная боль, иногда с тошнотой и чувствительностью к свету.",
}

_TERM_RE = re.compile(
    r"\b([A-Za-zА-Яа-яЁё\-]{5,}(?:\s+[A-Za-zА-Яа-яЁё\-]+)?)\b",
)


def extract_medical_terms(text: str) -> list[str]:
    low = (text or "").lower()
    found: list[str] = []
    seen: set[str] = set()
    for key in _TERM_DICT:
        if key in low and key not in seen:
            seen.add(key)
            found.append(key)
    return found[:12]


def explain_terms_for_patient(text: str) -> list[dict[str, Any]]:
    terms = extract_medical_terms(text)
    out: list[dict[str, Any]] = []
    for t in terms:
        expl = _TERM_DICT.get(t.lower(), "")
        if not expl:
            continue
        out.append(
            {
                "term": t[:80].capitalize() if t.islower() else t[:80],
                "explanation_ru": expl,
                "disclaimer_ru": "Справочное объяснение, не диагноз.",
            }
        )
    return out
