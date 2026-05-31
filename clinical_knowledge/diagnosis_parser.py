"""Разбор диагнозов из КЗ: несколько диагнозов, ICD-10, роль, степень достоверности.

ТЗ раздел 9 (ConsultationDiagnosis) и раздел 24 (test_diagnosis_parser).
"""
from __future__ import annotations

import re

from corpus_pipeline.entities_extract import extract_icd10

from .consult_schema import ConsultationDiagnosis

# Код МКБ в начале строки диагноза: «K30. Диспепсия», «L93.0 Дискоидная...»
RE_LEADING_ICD = re.compile(r"^\s*([A-ZА-Я]\d{2}(?:\.\d{1,2})?)\.?\s*(.*)$")

SUSPECTED_MARKERS = (
    "?", "подозрени", "нельзя исключить", "вероятно",
    "под вопросом", "susp", "не исключен",
)
EXCLUDED_MARKERS = ("исключен", "снят диагноз", "данных за ... не получено")
PRIMARY_MARKERS = ("основн",)
SECONDARY_MARKERS = ("сопутств", "фон", "осложнени")
MALIGNANCY_MARKERS = (
    "нельзя исключить инвазию", "злокачествен", "образование кишки",
    "опухолев", "c-r", "сr ", "новообразование",
)


def _certainty(raw_low: str) -> str:
    if any(mk in raw_low for mk in EXCLUDED_MARKERS):
        return "excluded"
    if any(mk in raw_low for mk in SUSPECTED_MARKERS):
        return "suspected"
    return "confirmed"


def _role(raw_low: str, idx: int) -> str:
    if any(mk in raw_low for mk in PRIMARY_MARKERS):
        return "primary"
    if any(mk in raw_low for mk in SECONDARY_MARKERS):
        return "secondary"
    return "primary" if idx == 0 else "secondary"


def _split_diagnosis_lines(diagnosis_block: str) -> list[str]:
    if not diagnosis_block:
        return []
    parts: list[str] = []
    for line in re.split(r"[\n;]+", diagnosis_block):
        line = line.strip(" -—\t")
        if len(line) >= 3:
            parts.append(line)
    return parts


def parse_diagnoses(
    diagnosis_block: str,
    *,
    source_section: str | None = "diagnosis_text",
) -> list[ConsultationDiagnosis]:
    """Разбирает блок диагноза(ов) в список ConsultationDiagnosis."""
    out: list[ConsultationDiagnosis] = []
    lines = _split_diagnosis_lines(diagnosis_block)
    for i, line in enumerate(lines):
        low = line.lower()
        icd = None
        name = line
        m = RE_LEADING_ICD.match(line)
        if m:
            icd = m.group(1).upper().replace(" ", "")
            name = (m.group(2) or "").strip() or line
        else:
            codes = extract_icd10(line)
            if codes:
                icd = codes[0]
        certainty = _certainty(low)
        role = _role(low, i)
        if certainty == "suspected" and role not in ("primary", "secondary"):
            role = "suspected"
        out.append(
            ConsultationDiagnosis(
                diagnosis_id=f"dx{i + 1}",
                raw_text=line,
                icd10_code=icd,
                diagnosis_name=name,
                diagnosis_role=role,
                certainty=certainty,
                is_protocol_relevant=certainty != "excluded",
                source_section=source_section,
            )
        )
    return out


def has_malignancy_flag(text: str) -> bool:
    low = (text or "").lower()
    return any(mk in low for mk in MALIGNANCY_MARKERS)
