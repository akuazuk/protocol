"""Клинические специальности врача для отчёта методиста по КЗ.

Охват: все врачебные специальности из выгрузки; исключаем только неклинические
роли (медсестра, стоматология, логопед, лаборатория, пустая/нераспознанная).
См. docs/plans/2026-07-22-kz-scoring-methodology-v1.md §9.
"""
from __future__ import annotations

import re
from typing import Any

from clinical_knowledge.mis_protocol_parse import (
    KZ_SCORED_KINDS,
    classify_kz_kind,
    is_diagnostic_specialty,
)

__all__ = [
    "normalize_specialty",
    "is_clinical_specialty",
    "is_diagnostic_specialty",
    "filter_clinical_rows",
    "filter_clinical_doctors",
    "filter_clinical_visits",
    "filter_kz_rows",
]

# Подстроки / точные имена неклинических ролей (lower).
# Прочерки/пустые - через unicode-коды, чтобы normalize_ui_dashes не схлопнул их в "-".
_NON_CLINICAL_EXACT = {
    "",
    "-",
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\u2212",  # minus
    "медицинская сестра",
    "медсестра",
    "мед. сестра",
    "логопед",
    "лаборатория",
    "лаборант",
}

_NON_CLINICAL_RE = re.compile(
    r"(стоматолог|зубн|медсестр|медицинск\w*\s+сестр|"
    r"логопед|лаборатор|фельдшер|регистратор)",
    re.IGNORECASE,
)

# Только пунктуация / пробелы / прочерки - не специальность.
_ONLY_DASH_RE = re.compile(r"^[\s\-\u2013\u2014\u2212\u2011._]+$")


def normalize_specialty(name: str | None) -> str:
    return (name or "").strip()


def is_clinical_specialty(name: str | None) -> bool:
    """True, если специальность врачебная клиническая (для отчёта качества КЗ).

    Исключаются неклинические роли (медсестра/стоматология/логопед/лаборатория) И
    диагностические специальности (УЗИ, рентген, функц. диагностика, эндоскопия) -
    их протоколы не являются консультативными заключениями.
    """
    s = normalize_specialty(name)
    low = s.lower()
    if not s or low in _NON_CLINICAL_EXACT or _ONLY_DASH_RE.match(s):
        return False
    if _NON_CLINICAL_RE.search(low):
        return False
    if is_diagnostic_specialty(s):
        return False
    return True


def filter_clinical_rows(
    rows: list[dict[str, Any]] | None,
    *,
    key: str = "specialization",
) -> list[dict[str, Any]]:
    """Оставляет только клинические специальности в агрегатах summary."""
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = row.get(key) or row.get("doctor_specialization") or row.get("specialty")
        if is_clinical_specialty(str(name) if name is not None else None):
            out.append(row)
    return out


def filter_clinical_doctors(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return filter_clinical_rows(rows, key="specialization")


def filter_clinical_visits(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [
        r
        for r in (rows or [])
        if isinstance(r, dict)
        and is_clinical_specialty(r.get("doctor_specialization") or r.get("specialization"))
    ]


def filter_kz_rows(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Оставить только строки-КЗ, идущие в оценку (kz / certificate).

    Предпочитает готовый столбец `kz_kind` (из экспортёра); если его нет -
    классифицирует строку на месте через classify_kz_kind.
    """
    out: list[dict[str, Any]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        kind = str(r.get("kz_kind") or "").strip() or classify_kz_kind(r)[0]
        if kind in KZ_SCORED_KINDS:
            out.append(r)
    return out
