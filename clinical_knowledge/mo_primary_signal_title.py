"""Заголовок главного замечания для списка и блока «Почему так».

Оценку записи не меняет и в склад ничего не пишет. Один порядок для
страницы «Найти МО» и карточки случая.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from clinical_knowledge.mo_action_queue_select import QUEUE_INCLUDE_CODES, QUEUE_SIGNALS, ddi_is_major

STALE_TITLE_RU = "Оценка старше замечания: живого сигнала P0/P1 нет"
TITLE_LIMIT = 140
TITLE_FULL_LIMIT = 400

# Красный флаг, Major DDI, high-alert без дозы, дубль НПВП, неясный диагноз.
_NAMED_RANK = {
    "C_red_flag": 0,
    "C_red_flag_unrouted": 0,
    "C_high_alert_no_dose": 2,
    "C_nsaid_dup": 3,
    "C_uncertainty_unrouted": 4,
}
_LAYER_RU = {
    "safety": "Риск",
    "zone1": "Оформление",
    "zone2a": "Диагноз",
    "zone2b": "План",
}


def clip_signal_title(text: str | None, limit: int = TITLE_LIMIT) -> str:
    """Обрезать по границе слова. Середина названия препарата не остаётся обрубком."""
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(raw) <= limit:
        return raw
    cut = raw[:limit]
    space = cut.rfind(" ")
    if space >= 80:
        cut = cut[:space]
    return cut.rstrip(".,;:- ") + "..."


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value) == 1
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _open_safety_finding(finding: Mapping[str, Any]) -> bool:
    if _truthy_flag(finding.get("passed")):
        return False
    if _truthy_flag(finding.get("is_shadow")) or _truthy_flag(finding.get("shadow")):
        return False
    code = str(finding.get("finding_code") or finding.get("code") or "").strip()
    if not code.startswith("C_") or code not in QUEUE_INCLUDE_CODES:
        return False
    severity = str(finding.get("severity") or "").strip().upper()
    return severity in {"P0", "P1"}


def _rank(finding: Mapping[str, Any]) -> tuple[int, int]:
    code = str(finding.get("finding_code") or finding.get("code") or "").strip()
    severity = str(finding.get("severity") or "").strip().upper()
    severity_rank = 0 if severity == "P0" else 1
    if code == "C_ddi" and ddi_is_major(finding):
        return (1, severity_rank)
    if code == "C_ddi":
        return (6, severity_rank)
    if code in _NAMED_RANK:
        return (_NAMED_RANK[code], severity_rank)
    return (5 if severity == "P0" else 7, severity_rank)


def _finding_title(finding: Mapping[str, Any]) -> str:
    title = str(finding.get("title_ru") or "").strip()
    if title:
        return title
    code = str(finding.get("finding_code") or finding.get("code") or "").strip()
    catalog = QUEUE_SIGNALS.get(code) or {}
    return str(catalog.get("title_ru") or "").strip()


def _blank(*, layer: str = "") -> dict[str, Any]:
    return {
        "title_ru": "",
        "title_full_ru": "",
        "code": "",
        "severity": "",
        "severity_label_ru": "",
        "layer_ru": layer,
        "stale": False,
    }


def primary_signal_title(
    *,
    attention_primary: str | None,
    attention_reason_ru: str | None,
    findings: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Короткий заголовок строки.

    zone2a остаётся фразой про диагноз. Слой риска берёт title_ru живого
    C_ P0/P1. Если такого замечания нет, это устаревшая оценка, не новая.
    """
    primary = str(attention_primary or "").strip()
    reason = re.sub(r"\s+", " ", str(attention_reason_ru or "")).strip()
    if primary == "zone2a":
        shown = clip_signal_title(reason)
        return {
            "title_ru": shown,
            "title_full_ru": reason[:TITLE_FULL_LIMIT],
            "code": "",
            "severity": "",
            "severity_label_ru": "",
            "layer_ru": "Диагноз",
            "stale": False,
        }
    if primary != "safety":
        return _blank(layer=_LAYER_RU.get(primary, ""))
    if findings is None:
        return _blank(layer="Риск")
    live = [item for item in findings if isinstance(item, Mapping) and _open_safety_finding(item)]
    if not live:
        return {
            "title_ru": STALE_TITLE_RU,
            "title_full_ru": STALE_TITLE_RU,
            "code": "",
            "severity": "",
            "severity_label_ru": "",
            "layer_ru": "Риск",
            "stale": True,
        }
    best = min(enumerate(live), key=lambda pair: (_rank(pair[1]), pair[0]))[1]
    full = _finding_title(best)[:TITLE_FULL_LIMIT]
    code = str(best.get("finding_code") or best.get("code") or "").strip()
    severity = str(best.get("severity") or "").strip().upper()
    return {
        "title_ru": clip_signal_title(full),
        "title_full_ru": full,
        "code": code,
        "severity": severity,
        "severity_label_ru": "Критично" if severity == "P0" else "Важно",
        "layer_ru": "Риск",
        "stale": False,
    }


def apply_primary_signal_titles(
    records: Sequence[dict[str, Any]],
    findings_by_mis: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    *,
    safety_queried: bool,
) -> None:
    """Дописать заголовок в строки страницы списка. Склад не трогает."""
    grouped = findings_by_mis or {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        primary = str(rec.get("attention_primary") or "")
        if primary == "safety" and not safety_queried:
            continue
        if primary not in {"safety", "zone2a"}:
            continue
        findings = grouped.get(str(rec.get("mis_id") or ""), [])
        signal = primary_signal_title(
            attention_primary=primary,
            attention_reason_ru=str(rec.get("attention_reason_ru") or ""),
            findings=findings if primary == "safety" else [],
        )
        title = str(signal.get("title_ru") or "")
        if not title:
            continue
        rec["primary_finding_title_ru"] = title
        rec["primary_finding_title_full_ru"] = str(signal.get("title_full_ru") or title)
        rec["primary_finding_code"] = str(signal.get("code") or "") or None
        rec["primary_finding_layer_ru"] = str(signal.get("layer_ru") or "") or None
        rec["primary_finding_severity_label_ru"] = (
            str(signal.get("severity_label_ru") or "") or None
        )
        rec["primary_finding_stale"] = bool(signal.get("stale"))
