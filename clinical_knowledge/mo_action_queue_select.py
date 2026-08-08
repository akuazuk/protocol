"""Отбор МО в очередь разбора по точным сигналам оценки.

Не используем «любой P0/P1 finding» и не поднимаем тикет из одного №55.
Каталог сигналов и правила - docs/plans/2026-08-08-mo-action-queue-precise-signals-v2.md.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

# Полосы для методиста (в UI только эти слова, не P0…P3).
BAND_CRITICAL = "critical"
BAND_IMPORTANT = "important"

BAND_LABEL_RU = {
    BAND_CRITICAL: "Критично",
    BAND_IMPORTANT: "Важно",
}

BAND_TO_INTERNAL = {
    BAND_CRITICAL: "P0",
    BAND_IMPORTANT: "P1",
}

# Точные детерминированные сигналы «МО не выглядит хорошо».
# trust: A = правило с цитатой/справочником; B = полезно, но мягче.
QUEUE_SIGNALS: dict[str, dict[str, Any]] = {
    "C_red_flag": {
        "band": BAND_CRITICAL,
        "trust": "A",
        "title_ru": "Красный флаг без маршрутизации",
    },
    "C_red_flag_unrouted": {
        "band": BAND_CRITICAL,
        "trust": "A",
        "title_ru": "Красный флаг без маршрутизации",
    },
    "C_ddi": {
        "band": BAND_IMPORTANT,
        "trust": "A",
        "title_ru": "Лекарственное взаимодействие",
        "major_lifts_to": BAND_CRITICAL,
    },
    "C_high_alert_no_dose": {
        "band": BAND_IMPORTANT,
        "trust": "A",
        "title_ru": "Препарат высокого риска без дозы",
    },
    "C_nsaid_dup": {
        "band": BAND_IMPORTANT,
        "trust": "A",
        "title_ru": "Дублирование НПВС",
    },
    "C_uncertainty_unrouted": {
        "band": BAND_IMPORTANT,
        "trust": "B",
        "title_ru": "Клиническая неопределённость без маршрутизации",
    },
    "B_dx_no_support": {
        "band": BAND_IMPORTANT,
        "trust": "A",
        "title_ru": "Диагноз не подкреплён клиникой",
    },
    "B_dx_absent": {
        "band": BAND_IMPORTANT,
        "trust": "A",
        "title_ru": "Диагноз отсутствует в МО",
    },
}

QUEUE_INCLUDE_CODES = frozenset(QUEUE_SIGNALS)

# Явно вне очереди (даже если severity P0/P1 в витрине).
QUEUE_EXCLUDE_CODES = frozenset(
    {
        "D_reg55_p0",
        "D_reg55_gap",
        "B_icd_invalid",
        "B_icd_mismatch_mis",
        "B_icd_dir_no_match",
        "B_icd_dir_code_unknown",
        "B_icd_dir_text_mismatch",
        "B_icd_name_no_match",
        "B_icd_name_weak_match",
        "B_icd_llm_review_yes",
        "B_icd_llm_review_partial",
        "B_icd_llm_review_no",
        "E_template_copy",
        "C_drug_unresolved",
    }
)

QUEUE_EXCLUDE_PREFIXES = (
    "D_reg55",
    "A_missing_",
    "B_icd_",
)

_MAJOR_DDI_RE = re.compile(
    r"(?i)\b(major|contraindicat|противопоказ|критичн|опасн\w*\s+взаимодейств)"
)
_PN_TOKEN_RE = re.compile(r"\bP[0-3]\b")


def sql_finding_code_in_clause(alias: str = "f") -> str:
    """Фрагмент SQL: finding_code IN (...whitelist...)."""
    codes = ", ".join(f"'{code}'" for code in sorted(QUEUE_INCLUDE_CODES))
    return f"{alias}.finding_code IN ({codes})"


def is_excluded_queue_code(code: str | None) -> bool:
    cid = str(code or "").strip()
    if not cid:
        return True
    if cid in QUEUE_EXCLUDE_CODES:
        return True
    return any(cid.startswith(prefix) for prefix in QUEUE_EXCLUDE_PREFIXES)


def _blob(*parts: Any) -> str:
    return " ".join(str(p or "") for p in parts)


def ddi_is_major(row: Mapping[str, Any] | None) -> bool:
    """Major/contraindicated DDI (по тексту finding или severity P0)."""
    row = row or {}
    sev = str(row.get("severity") or row.get("finding_severity") or "").strip().upper()
    if sev == "P0":
        return True
    blob = _blob(
        row.get("finding_title"),
        row.get("title_ru"),
        row.get("detail_ru"),
        row.get("evidence"),
        row.get("reason"),
    )
    return bool(_MAJOR_DDI_RE.search(blob))


def signal_band_for_finding(row: Mapping[str, Any] | None) -> str | None:
    """Полоса очереди для finding или None, если не берём в разбор."""
    row = row or {}
    code = str(row.get("finding_code") or row.get("code") or "").strip()
    if not code or is_excluded_queue_code(code):
        return None
    meta = QUEUE_SIGNALS.get(code)
    if not meta:
        return None
    if bool(row.get("is_shadow")):
        return None
    if code == "C_ddi":
        # Moderate (обычно P2) - не тикет; Major / P0-P1 - да.
        sev = str(row.get("severity") or "").strip().upper()
        if ddi_is_major(row):
            return str(meta.get("major_lifts_to") or BAND_CRITICAL)
        if sev in {"P0", "P1"}:
            return BAND_IMPORTANT
        return None
    return str(meta.get("band") or BAND_IMPORTANT)


def finding_eligible_for_action_queue(row: Mapping[str, Any] | None) -> bool:
    return signal_band_for_finding(row) is not None


def strip_pn_tokens(text: str | None) -> str:
    """Убрать сырые P0…P3 из пользовательских строк."""
    raw = str(text or "")
    cleaned = _PN_TOKEN_RE.sub("", raw)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip(" -·|")


def queue_reason_ru(*, band: str, finding_title: str | None, finding_code: str | None = None) -> str:
    label = BAND_LABEL_RU.get(band, "Важно")
    title = strip_pn_tokens(finding_title) or str(
        (QUEUE_SIGNALS.get(str(finding_code or "")) or {}).get("title_ru") or "замечание по оценке МО"
    )
    # Не дублировать полосу, если title уже начинается с неё.
    if title.lower().startswith(label.lower()):
        return title
    return f"{label}: {title}"


def pick_primary_queue_finding(findings: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Из нескольких eligible findings выбрать самый тяжёлый/точный."""
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    trust_rank = {"A": 0, "B": 1}
    band_rank = {BAND_CRITICAL: 0, BAND_IMPORTANT: 1}
    for raw in findings:
        if not isinstance(raw, Mapping):
            continue
        band = signal_band_for_finding(raw)
        if not band:
            continue
        code = str(raw.get("finding_code") or raw.get("code") or "")
        trust = str((QUEUE_SIGNALS.get(code) or {}).get("trust") or "B")
        item = dict(raw)
        item["_queue_band"] = band
        ranked.append((band_rank.get(band, 9), trust_rank.get(trust, 9), item))
    if not ranked:
        return None
    ranked.sort(key=lambda t: (t[0], t[1], str(t[2].get("finding_code") or "")))
    return ranked[0][2]


def catalog_for_docs() -> list[dict[str, Any]]:
    """Список сигналов для отладки / документации."""
    rows = []
    for code, meta in sorted(QUEUE_SIGNALS.items()):
        rows.append(
            {
                "code": code,
                "band": meta.get("band"),
                "trust": meta.get("trust"),
                "title_ru": meta.get("title_ru"),
                "in_queue": True,
            }
        )
    for code in sorted(QUEUE_EXCLUDE_CODES):
        rows.append(
            {
                "code": code,
                "band": None,
                "trust": "exclude",
                "title_ru": "вне очереди",
                "in_queue": False,
            }
        )
    return rows
