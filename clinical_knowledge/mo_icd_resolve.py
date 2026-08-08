"""Разрешение кода МКБ и текста диагноза только из слотов диагноза МО.

Правило: docs/plans/2026-08-08-mo-icd-diag-slots-only-v1.md

Ищем в «Клинический диагноз» / «Диагноз МИС» (+ явные колонки кода).
Не сканируем жалобы, анамнез, статус, план - оттуда подтягивались чужие коды.
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

# Только слоты, которые UI показывает как «Клинический диагноз» / «Диагноз МИС»
# (+ короткие алиасы витрины, которые пишутся из этих же слотов).
MO_DIAGNOSIS_SLOT_KEYS: tuple[str, ...] = (
    "clinical_diagnosis",
    "mis_diagnos",
    "mis_diagnosis",
    "diagnosis_mis",
    "diagnosis_main_text",
    "diagnosis_short",
    "diagnosis_text",
)

# Совместимость со старым именем (раньше - полный документ).
MO_ICD_TEXT_KEYS: tuple[str, ...] = MO_DIAGNOSIS_SLOT_KEYS

_EXPLICIT_CODE_KEYS: tuple[str, ...] = (
    "mkb_code_main",
    "diagnosis_code",
    "icd10",
    "mkb_code",
)

_DIAG_SLOT_KEYS: tuple[str, ...] = MO_DIAGNOSIS_SLOT_KEYS

_DIAG_LABEL_LINE = re.compile(
    r"(?i)(?:клинический\s+)?диагноз\s*[:：\-–—]?\s*(.+)",
)


def is_valid_icd_format(code: str | None) -> bool:
    return bool(code and _ICD_FORM_RE.match(str(code).strip()))


def _normalize_code(raw: str) -> str:
    """К латинице H52.1: кириллица Н/К/…, пробелы («Н 52.1»), запятая."""
    token = str(raw or "").strip()
    if not token:
        return ""
    try:
        import icd_mkb

        canon = icd_mkb._canonicalize_icd_like_token(token)
        if canon:
            return canon
        scanned = icd_mkb.normalize_text_for_icd_scan(token)
        match = icd_mkb.ICD10_CODE_RE.search(scanned)
        if match:
            return icd_mkb.normalize_icd_code(match.group(1))
        return icd_mkb.normalize_icd_code(scanned)
    except Exception:  # noqa: BLE001
        return token.upper().replace(" ", "")


def _codes_in_text(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    raw = str(text)
    try:
        import icd_mkb

        scanned = icd_mkb.normalize_text_for_icd_scan(raw)
    except Exception:  # noqa: BLE001
        scanned = raw
    if _PLACEHOLDER_LINE.search(scanned) and not _ICD_RE.search(scanned):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _ICD_RE.finditer(scanned):
        start = max(0, match.start() - 24)
        end = min(len(scanned), match.end() + 12)
        window = scanned[start:end]
        if _PLACEHOLDER_LINE.search(window) and "см." in window.lower():
            continue
        code = _normalize_code(match.group(1))
        if not is_valid_icd_format(code) or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def _iter_diagnosis_fields(case: dict[str, Any]) -> list[tuple[str, str]]:
    """(source_key, text) только по слотам диагноза и вложенному clinical*."""
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

    for key in MO_DIAGNOSIS_SLOT_KEYS:
        add("", key, case.get(key))

    for nested_key in ("clinical", "clinical_json", "fields", "document"):
        nested = case.get(nested_key)
        if isinstance(nested, dict):
            for key in MO_DIAGNOSIS_SLOT_KEYS:
                add(f"{nested_key}.", key, nested.get(key))

    return blobs


# Alias для старых импортов / тестов.
def _iter_text_fields(case: dict[str, Any]) -> list[tuple[str, str]]:
    return _iter_diagnosis_fields(case)


def _diagnosis_slots_text(case: dict[str, Any]) -> str:
    """Клинический диагноз предпочтительнее Диагноза МИС."""
    preferred = (
        "clinical_diagnosis",
        "diagnosis_main_text",
        "diagnosis_short",
        "diagnosis_text",
        "mis_diagnos",
        "mis_diagnosis",
        "diagnosis_mis",
    )
    parts: list[str] = []
    seen: set[str] = set()

    def take(key: str, value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return
        if text in seen:
            return
        seen.add(text)
        parts.append(text)

    for key in preferred:
        take(key, case.get(key))
    for nested_key in ("clinical", "clinical_json", "fields", "document"):
        nested = case.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in preferred:
            take(key, nested.get(key))
    return " ".join(parts).strip()


def _snippet_near_code(text: str, code: str, *, before: int = 80, after: int = 120) -> str:
    """Короткий фрагмент вокруг кода МКБ внутри слота диагноза."""
    want = _normalize_code(code)
    if not text or not want:
        return ""
    for match in _ICD_RE.finditer(text):
        if _normalize_code(match.group(1)) != want:
            continue
        start = max(0, match.start() - before)
        end = min(len(text), match.end() + after)
        chunk = text[start:end]
        chunk = re.sub(r"[\r\n]+", " ", chunk)
        chunk = re.sub(r"\s+", " ", chunk).strip()
        chunk = _ICD_RE.sub(" ", chunk)
        chunk = re.sub(r"\s+", " ", chunk).strip(" .;,:|-")
        return chunk[:220]
    return ""


def _labeled_diagnosis_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in re.split(r"[\r\n]+", text or ""):
        m = _DIAG_LABEL_LINE.search(line.strip())
        if not m:
            continue
        frag = m.group(1).strip()
        frag = _ICD_RE.sub(" ", frag)
        frag = re.sub(r"\s+", " ", frag).strip(" .;,:|-")
        if len(frag) >= 3:
            out.append(frag[:220])
    return out


def resolve_diagnosis_text_from_mo(case: dict[str, Any] | None) -> dict[str, Any]:
    """Текст диагноза только из слотов clinical_diagnosis / mis_diagnos.

    Returns:
        text, source (slots|label_line|near_code|empty), used_fallback, codes
    """
    case = case if isinstance(case, dict) else {}
    slots = _diagnosis_slots_text(case)
    resolved = resolve_icd_codes_from_mo(case)
    codes = list(resolved.get("all") or [])
    if slots:
        return {
            "text": slots,
            "source": "slots",
            "used_fallback": False,
            "codes": codes,
            "main": resolved.get("main") or "",
        }

    # Только внутри слотов диагноза (если слот пуст как join, но label внутри поля).
    for field, text in _iter_diagnosis_fields(case):
        labeled = _labeled_diagnosis_lines(text)
        if labeled:
            return {
                "text": labeled[0],
                "source": f"label_line:{field}",
                "used_fallback": True,
                "codes": codes,
                "main": resolved.get("main") or "",
            }

    main = str(resolved.get("main") or "")
    probe_codes = [main] + [c for c in codes if c != main] if main else codes
    for code in probe_codes:
        if not code:
            continue
        for field, text in _iter_diagnosis_fields(case):
            snippet = _snippet_near_code(text, code)
            if len(snippet) >= 3:
                return {
                    "text": snippet,
                    "source": f"near_code:{field}:{code}",
                    "used_fallback": True,
                    "codes": codes,
                    "main": main or code,
                }

    return {
        "text": "",
        "source": "empty",
        "used_fallback": False,
        "codes": codes,
        "main": main,
    }


def has_mo_diagnosis_text(case: dict[str, Any] | None) -> bool:
    """Есть ли осмысленная формулировка диагноза (не только токены кода МКБ)."""
    case = case if isinstance(case, dict) else {}
    try:
        text = str(resolve_diagnosis_text_from_mo(case).get("text") or "").strip()
    except Exception:  # noqa: BLE001
        text = _diagnosis_slots_text(case)
    if not text:
        return False
    try:
        from clinical_knowledge.mo_icd_directory_eval import free_text_is_substantive

        return bool(free_text_is_substantive(text))
    except Exception:  # noqa: BLE001
        cleaned = _ICD_RE.sub(" ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;,:|-")
        return len(cleaned) >= 3


def _explicit_raw_icd_token(case: dict[str, Any]) -> str:
    """Сырой токен из явных колонок кода (может быть битым форматом)."""
    for key in ("mkb_code_main", "diagnosis_code", "icd10", "mkb_code"):
        raw = case.get(key)
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else ""
        token = str(raw or "").strip()
        if token and token.lower() not in {"nan", "none", "null", "-", "—"}:
            return token
    return ""


def assess_icd_code_requirement(case: dict[str, Any] | None) -> dict[str, Any]:
    """Правило кодирования МКБ для findings / штрафа.

    - валидный код в слотах диагноза / явной колонке → ok;
    - кода нет, но есть формулировка диагноза → ok (не дефект);
    - указан невалидный код → defect (format);
    - нет ни кода, ни диагноза → defect (missing_both).
    """
    case = case if isinstance(case, dict) else {}
    resolved = resolve_icd_codes_from_mo(case)
    code = str(resolved.get("main") or "").strip()
    if not code and resolved.get("all"):
        code = str(resolved["all"][0]).strip()
    has_valid = bool(code) and is_valid_icd_format(code)
    has_dx = has_mo_diagnosis_text(case)
    raw_explicit = _explicit_raw_icd_token(case)
    raw_norm = _normalize_code(raw_explicit) if raw_explicit else ""
    malformed_explicit = bool(raw_explicit) and not is_valid_icd_format(raw_norm)

    if has_valid:
        return {
            "ok": True,
            "status": "ok",
            "code": code,
            "has_diagnosis_text": has_dx,
            "has_valid_code": True,
            "reason_ru": "",
            "title_ru": "",
        }
    if malformed_explicit:
        return {
            "ok": False,
            "status": "invalid_format",
            "code": raw_norm or raw_explicit,
            "has_diagnosis_text": has_dx,
            "has_valid_code": False,
            "reason_ru": f"код «{raw_explicit}» не соответствует формату МКБ-10",
            "title_ru": "Код МКБ не соответствует формату",
        }
    if has_dx:
        return {
            "ok": True,
            "status": "diagnosis_without_code",
            "code": "",
            "has_diagnosis_text": True,
            "has_valid_code": False,
            "reason_ru": "",
            "title_ru": "",
        }
    return {
        "ok": False,
        "status": "missing_both",
        "code": "",
        "has_diagnosis_text": False,
        "has_valid_code": False,
        "reason_ru": "нет формулировки диагноза и кода МКБ",
        "title_ru": "Нет диагноза и кода МКБ",
    }


def resolve_icd_codes_from_mo(case: dict[str, Any] | None) -> dict[str, Any]:
    """Найти коды МКБ только в слотах диагноза + явных колонках кода.

    Returns:
        main: str - mkb_code_main / код из clinical_diagnosis / mis_diagnos
        all: list[str] - уникальные коды в порядке обнаружения
        sources: list[{code, field}]
        present: bool
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

    # 1) явные колонки экспорта кода
    for key in _EXPLICIT_CODE_KEYS:
        raw = case.get(key)
        if isinstance(raw, (list, tuple)):
            for item in raw:
                add_code(str(item), key)
        elif raw:
            add_code(str(raw), key)

    # 2) только слоты диагноза (порядок: clinical → mis)
    for field, text in _iter_diagnosis_fields(case):
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
        for field, text in _iter_diagnosis_fields(case):
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


# Warehouse / KPI soft-fill. Не трогает mkb_code_agreement (слот экспорта).
SOURCE_SLOT = "slot"
SOURCE_SOFT_FILL_DIAG_SLOTS = "soft_fill_diag_slots"
# Совместимость со старым именем в тестах / отчётах.
SOURCE_SOFT_FILL_FULL_DOC = SOURCE_SOFT_FILL_DIAG_SLOTS
SOURCE_EMPTY = "empty"


def _slot_mkb_codes(case: dict[str, Any]) -> list[str]:
    """Коды только из слотов экспорта mkb_codes / mkb_code_main."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in (case.get("mkb_codes"), case.get("mkb_code_main")):
        if raw is None or raw == "":
            continue
        if isinstance(raw, (list, tuple)):
            parts = [str(x) for x in raw]
        else:
            parts = str(raw).replace(",", "|").split("|")
        for part in parts:
            code = _normalize_code(part)
            if not is_valid_icd_format(code) or code in seen:
                continue
            seen.add(code)
            out.append(code)
    return out


def soft_fill_mkb_for_warehouse(case: dict[str, Any] | None) -> dict[str, Any]:
    """Код для KPI/UI витрины: явный слот кода, иначе код из слотов диагноза.

    Returns:
        code, codes, source (slot|soft_fill_diag_slots|empty), slot_code
    Не мутирует case и не предназначен для mkb_code_agreement.
    """
    case = case if isinstance(case, dict) else {}
    slot_codes = _slot_mkb_codes(case)
    slot_main = slot_codes[0] if slot_codes else ""
    if slot_main:
        return {
            "code": slot_main,
            "codes": slot_codes,
            "source": SOURCE_SLOT,
            "slot_code": slot_main,
        }

    resolved = resolve_icd_codes_from_mo(case)
    fill = str(resolved.get("main") or "").strip()
    if fill and is_valid_icd_format(fill):
        all_codes = list(resolved.get("all") or [])
        if fill not in all_codes:
            all_codes.insert(0, fill)
        return {
            "code": fill,
            "codes": all_codes,
            "source": SOURCE_SOFT_FILL_DIAG_SLOTS,
            "slot_code": "",
        }
    return {
        "code": "",
        "codes": [],
        "source": SOURCE_EMPTY,
        "slot_code": "",
    }
