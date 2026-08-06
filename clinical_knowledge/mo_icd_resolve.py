"""Разрешение кодов МКБ-10 по всему тексту МО/КЗ (не только графа «Диагноз»).

Правило: docs/plans/2026-08-06-mo-icd-full-document-search-v1.md
stdlib-only - можно импортировать из reg55_criteria.
"""
from __future__ import annotations

import re
from typing import Any

# Формат МКБ-10 (латиница; U исключён как в extract_mkb_code / reg55).
_ICD_RE = re.compile(r"\b([A-TV-Z][0-9]{2}(?:\.[0-9]{1,2})?)\b", re.I)
_ICD_FORM_RE = re.compile(r"^[A-TV-Z][0-9]{2}(?:\.[0-9]{1,2})?$", re.I)

# Явные плейсхолдеры / мусор рядом с «кодом».
_PLACEHOLDER_LINE = re.compile(
    r"(?:"
    r"код\s*мкб\s*[:：]?\s*$|"
    r"мкб\s*[-:]?\s*(?:10)?\s*[:：]?\s*$|"
    r"см\.?\s*мкб|"
    r"\bXXX(?:\.\d+)?\b|"
    r"\bMK[BВ]\s*\.\.\."
    r")",
    re.I,
)

# Клинические слоты МО + запасные имена из export / deep / review pack.
MO_ICD_TEXT_KEYS: tuple[str, ...] = (
    "clinical_diagnosis",
    "diagnosis_main_text",
    "diagnosis_short",
    "diagnosis_text",
    "complaints",
    "anamnesis_doctor",
    "anamnesis_auto",
    "anamnesis",
    "objective_status",
    "exam_data",
    "exam_recommendations",
    "treatment_recommendations",
    "recommendations_exam",
    "recommendations_treatment",
    "manipulations",
    "service_names",
    "result",
    "raw_text",
    "full_text",
    "document_text",
)

_EXPLICIT_CODE_KEYS: tuple[str, ...] = (
    "mkb_code_main",
    "diagnosis_code",
    "icd10",
    "mkb_code",
)


def is_valid_icd_format(code: str | None) -> bool:
    return bool(code and _ICD_FORM_RE.match(str(code).strip()))


def _normalize_code(raw: str) -> str:
    return str(raw or "").strip().upper().replace(" ", "")


def _codes_in_text(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    if _PLACEHOLDER_LINE.search(text) and not _ICD_RE.search(text):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _ICD_RE.finditer(text):
        # отрезок строки вокруг матча - отсечь «см. МКБ …»
        start = max(0, match.start() - 24)
        end = min(len(text), match.end() + 12)
        window = text[start:end]
        if _PLACEHOLDER_LINE.search(window) and "см." in window.lower():
            continue
        code = _normalize_code(match.group(1))
        if not is_valid_icd_format(code) or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def _iter_text_fields(case: dict[str, Any]) -> list[tuple[str, str]]:
    """(source_key, text) по кейсу и вложенному clinical*."""
    blobs: list[tuple[str, str]] = []
    seen_keys: set[str] = set()

    def add(prefix: str, key: str, value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return
        full_key = f"{prefix}{key}" if prefix else key
        if full_key in seen_keys:
            return
        seen_keys.add(full_key)
        blobs.append((full_key, text))

    for key in MO_ICD_TEXT_KEYS:
        add("", key, case.get(key))

    for nested_key in ("clinical", "clinical_json", "fields", "document"):
        nested = case.get(nested_key)
        if isinstance(nested, dict):
            for key in MO_ICD_TEXT_KEYS:
                add(f"{nested_key}.", key, nested.get(key))

    return blobs


def resolve_icd_codes_from_mo(case: dict[str, Any] | None) -> dict[str, Any]:
    """Найти коды МКБ по всему МО.

    Returns:
        main: str - основной код (слот диагноза / явный mkb_code_main, иначе первый
              валидный по документу)
        all: list[str] - уникальные коды в порядке обнаружения
        sources: list[{code, field}] - откуда взяли
        present: bool - есть ли хотя бы один валидный код
    """
    case = case if isinstance(case, dict) else {}
    sources: list[dict[str, str]] = []
    all_codes: list[str] = []
    seen: set[str] = set()

    def add_code(code: str, field: str) -> None:
        code = _normalize_code(code)
        if not is_valid_icd_format(code):
            return
        sources.append({"code": code, "field": field})
        if code not in seen:
            seen.add(code)
            all_codes.append(code)

    # 1) явные колонки экспорта
    for key in _EXPLICIT_CODE_KEYS:
        raw = case.get(key)
        if isinstance(raw, (list, tuple)):
            for item in raw:
                add_code(str(item), key)
        elif raw:
            add_code(str(raw), key)

    # 2) полный текст по слотам (диагнозные ключи раньше остальных за счёт порядка)
    for field, text in _iter_text_fields(case):
        for code in _codes_in_text(text):
            add_code(code, field)

    main = ""
    for key in _EXPLICIT_CODE_KEYS:
        raw = case.get(key)
        candidate = ""
        if isinstance(raw, (list, tuple)) and raw:
            candidate = _normalize_code(str(raw[0]))
        elif raw:
            candidate = _normalize_code(str(raw))
        if is_valid_icd_format(candidate):
            main = candidate
            break
    if not main:
        for field, text in _iter_text_fields(case):
            if not any(
                field == k or field.endswith(f".{k}")
                for k in (
                    "clinical_diagnosis",
                    "diagnosis_main_text",
                    "diagnosis_short",
                    "diagnosis_text",
                )
            ):
                continue
            found = _codes_in_text(text)
            if found:
                main = found[0]
                break
    if not main and all_codes:
        main = all_codes[0]

    return {
        "main": main,
        "all": all_codes,
        "sources": sources,
        "present": bool(all_codes),
    }
