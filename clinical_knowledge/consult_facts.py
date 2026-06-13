"""Эвристическое извлечение фактов из консультативного заключения (без LLM)."""
from __future__ import annotations

import re
from typing import Any

from corpus_pipeline.entities_extract import extract_icd10

from .condition_registry import infer_conditions_hints

RE_DIAG_BLOCK = re.compile(
    r"(?:диагноз|заключительный\s+диагноз|клинический\s+диагноз)\s*[:\- - ]?\s*([^\n]{5,400})",
    re.I,
)
RE_COMPLAINT = re.compile(
    r"(?:жалоб[ыа]?)\s*[:\- - ]?\s*([^\n]{5,500})",
    re.I,
)
RE_SEX_F = re.compile(r"\b(женский|жен\.?\s*пол|пол\s*[:\-]?\s*ж)\b", re.I)
RE_SEX_M = re.compile(r"\b(мужской|муж\.?\s*пол|пол\s*[:\-]?\s*м)\b", re.I)
RE_PREG = re.compile(r"\b(беременн|гестаци)\w*", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def extract_consult_facts_heuristic(
    text: str,
    *,
    demographics_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Структурированные факты из КЗ для rule checker."""
    raw = text or ""
    low = _norm(raw)

    diagnosis_text = ""
    m = RE_DIAG_BLOCK.search(raw)
    if m:
        diagnosis_text = m.group(1).strip()
    elif "диагноз" in low:
        for line in raw.split("\n"):
            if "диагноз" in line.lower() and len(line.strip()) > 12:
                diagnosis_text = line.strip()[:400]
                break

    complaints: list[str] = []
    for m in RE_COMPLAINT.finditer(raw):
        c = m.group(1).strip()
        if c and c not in complaints:
            complaints.append(c[:300])
        if len(complaints) >= 5:
            break

    icd = extract_icd10(raw[:120_000])
    if diagnosis_text:
        icd = list(dict.fromkeys(icd + extract_icd10(diagnosis_text)))

    sex = None
    if RE_SEX_F.search(raw):
        sex = "female"
    elif RE_SEX_M.search(raw):
        sex = "male"

    pregnancy = bool(RE_PREG.search(raw))
    demo = demographics_meta or {}
    audience = demo.get("audience")
    age_years = demo.get("age_years")

    adult_or_child = audience
    if not adult_or_child and age_years is not None:
        try:
            adult_or_child = "adult" if int(age_years) >= 18 else "child"
        except (TypeError, ValueError):
            pass

    conditions_hint = infer_conditions_hints(low, icd)

    return {
        "patient_context": {
            "age_years": age_years,
            "sex": sex,
            "adult_or_child": adult_or_child,
            "pregnancy": pregnancy if pregnancy else None,
        },
        "consultation": {
            "complaints": complaints,
            "diagnosis_text": diagnosis_text,
            "icd10": icd,
            "conditions_hint": conditions_hint,
            "text_sample": raw[:2000],
        },
        "extraction_method": "heuristic",
    }
