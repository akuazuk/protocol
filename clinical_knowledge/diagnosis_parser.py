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
SECONDARY_MARKERS = ("сопутств", "соп.", "соп:", "фон", "осложнени")
_JUNK_DIAG_LINE_RE = re.compile(
    r"^(?:соп\.?\s*:?\s*)?(?:мкб[-\s]?10?)\.?\s*$",
    re.IGNORECASE,
)
MALIGNANCY_MARKERS = (
    "нельзя исключить инвазию", "злокачествен", "образование кишки",
    "опухолев", "c-r", "сr ", "новообразование",
)

# После «;» без кода МКБ - отдельный диагноз, если строка начинается с новой нозологии.
_NEW_DX_ENTITY_STARTERS = (
    "флеботромб", "флебит", "тромбофлеб", "тромбоз", "гэрб", "гастрит", "язв",
    "пневмон", "орви", "гипертон", "диабет", "артрит",
)


def _looks_like_new_dx_entity(line: str) -> bool:
    low = (line or "").strip().lower()
    return any(low.startswith(s) for s in _NEW_DX_ENTITY_STARTERS)


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


def _starts_new_diagnosis(s: str) -> bool:
    """Строка начинает НОВЫЙ диагноз (код МКБ в начале или прописная без переноса)."""
    m = RE_LEADING_ICD.match(s)
    if m and m.group(1):
        return True
    first = s[:1]
    if first.isdigit() or first.islower():
        # дата/«от 11.09.2024…» или продолжение со строчной - это перенос, не новый диагноз
        return False
    return True


def _split_diagnosis_lines(diagnosis_block: str) -> list[str]:
    """Разбивает блок на диагнозы, склеивая перенесённые строки (даты/уточнения).

    Жёсткие разделители - `;` и перенос строки; но строка-продолжение
    (без кода МКБ, начинается со строчной буквы или с цифры-даты) приклеивается
    к предыдущему диагнозу, а не образует отдельный.
    """
    if not diagnosis_block:
        return []
    parts: list[str] = []
    for raw in re.split(r"[\n;]+", diagnosis_block):
        line = raw.strip(" \t-\u2013\u2014")
        if len(line) < 3:
            # слишком короткий фрагмент - приклеиваем к предыдущему, если это хвост
            if line and parts:
                parts[-1] = (parts[-1] + " " + line).strip()
            continue
        if parts and not _starts_new_diagnosis(line):
            parts[-1] = (parts[-1] + " " + line).strip()
        else:
            parts.append(line)
    return parts


def _is_junk_diagnosis_line(line: str) -> bool:
    t = (line or "").strip()
    if len(t) < 3:
        return True
    if _JUNK_DIAG_LINE_RE.match(t):
        return True
    if t.upper() in ("МКБ", "МКБ.", "МКБ-10"):
        return True
    if re.match(r"^соп\.?\s*:", t, re.IGNORECASE):
        rest = re.sub(r"^соп\.?\s*:\s*", "", t, count=1, flags=re.IGNORECASE).strip()
        if not RE_LEADING_ICD.match(rest) and not extract_icd10(rest):
            return True
    return False


def _merge_clinical_continuations(parts: list[str]) -> list[str]:
    """Склеивает клиническое уточнение после «;» с предыдущим кодом МКБ (не отдельный диагноз)."""
    if not parts:
        return parts
    out: list[str] = []
    for line in parts:
        if _is_junk_diagnosis_line(line):
            continue
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        prev_has_icd = bool(RE_LEADING_ICD.match(prev.strip()))
        cur_has_icd = bool(RE_LEADING_ICD.match(line.strip()))
        low = line.lower()
        is_secondary = any(mk in low for mk in SECONDARY_MARKERS)
        is_primary_labeled = any(mk in low for mk in PRIMARY_MARKERS)
        if prev_has_icd and not cur_has_icd and not is_secondary and not is_primary_labeled:
            head = line[:1]
            is_continuation = head.islower() or head.isdigit() or not _looks_like_new_dx_entity(line)
            if is_continuation:
                out[-1] = (prev + "; " + line).strip()
            else:
                out.append(line)
        else:
            out.append(line)
    return out


def parse_diagnoses(
    diagnosis_block: str,
    *,
    source_section: str | None = "diagnosis_text",
) -> list[ConsultationDiagnosis]:
    """Разбирает блок диагноза(ов) в список ConsultationDiagnosis."""
    out: list[ConsultationDiagnosis] = []
    lines = _merge_clinical_continuations(_split_diagnosis_lines(diagnosis_block))
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
        safety_flags: list[str] = []
        if any(mk in low for mk in MALIGNANCY_MARKERS):
            safety_flags.append("possible_malignancy")
            if role not in ("primary", "secondary"):
                role = "red_flag_finding"
        if certainty == "suspected" and role not in ("primary", "secondary", "red_flag_finding"):
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
                safety_flags=safety_flags,
                source_section=source_section,
                source_text=line[:300],
            )
        )
    return out


def has_malignancy_flag(text: str) -> bool:
    low = (text or "").lower()
    return any(mk in low for mk in MALIGNANCY_MARKERS)
