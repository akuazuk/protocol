"""Срок действия КП на дату визита. Пустые даты не выдумываем."""
from __future__ import annotations

from datetime import date
from typing import Any

from clinical_knowledge.patient_age import parse_iso_date

_LEGAL_CANCEL = frozenset(
    {"cancelled", "canceled", "withdrawn", "repealed", "obsolete", "deprecated", "expired"}
)
_SYNC_MISSING = frozenset({"superseded", "outdated", "sync_missing"})


def _approval(card: dict[str, Any]) -> dict[str, Any]:
    raw = card.get("approval")
    return raw if isinstance(raw, dict) else {}


def validity_from_card(card: dict[str, Any] | None) -> dict[str, Any]:
    """Даты силы из карточки. Не заполняем пробелы догадками об отмене."""
    card = card if isinstance(card, dict) else {}
    appr = _approval(card)
    valid_from = parse_iso_date(card.get("valid_from") or appr.get("valid_from") or appr.get("date"))
    valid_to = parse_iso_date(card.get("valid_to") or appr.get("valid_to"))
    superseded_on = parse_iso_date(
        card.get("superseded_on") or appr.get("superseded_on") or card.get("cancelled_on")
    )
    replaced_by = str(card.get("replaced_by") or appr.get("replaced_by") or card.get("superseded_by") or "").strip()
    status = str(card.get("status") or card.get("doc_status") or "active").strip().lower()
    legal_end = valid_to or superseded_on
    kind = "active"
    if legal_end is not None:
        kind = "legal"
    elif status in _LEGAL_CANCEL:
        kind = "legal_undated"
    elif status in _SYNC_MISSING or card.get("superseded_by"):
        kind = "sync_missing"
    return {
        "valid_from": valid_from.isoformat() if valid_from else None,
        "valid_to": valid_to.isoformat() if valid_to else None,
        "superseded_on": superseded_on.isoformat() if superseded_on else None,
        "replaced_by": replaced_by or None,
        "status": status or "active",
        "kind": kind,
    }


def card_in_force_on(card: dict[str, Any] | None, visit: date | None) -> bool:
    """True, если КП можно искать на дату визита.

    Отмена до визита - нет. Визит до отмены - да.
    Нет даты отмены, только sync_missing - не hard-drop.
    """
    card = card if isinstance(card, dict) else {}
    meta = validity_from_card(card)
    day = visit
    valid_from = parse_iso_date(meta.get("valid_from"))
    legal_end = parse_iso_date(meta.get("valid_to")) or parse_iso_date(meta.get("superseded_on"))
    if day is None:
        # Живой поиск без даты визита: не берём КП с уже наступившим valid_to.
        # Нет даты отмены - не hard-drop (формула плана: дата_отмены < V).
        return not (legal_end and legal_end < date.today())
    if valid_from and day < valid_from:
        return False
    if legal_end and legal_end < day:
        return False
    return True


def looks_omnibus(card: dict[str, Any] | None) -> bool:
    """Широкий «все болезни специальности» без узкой нозологии в названии/пути."""
    card = card if isinstance(card, dict) else {}
    blob = (
        str(card.get("title") or "")
        + " "
        + str(card.get("source_path") or "")
        + " "
        + str(card.get("condition_label") or "")
    ).lower()
    if any(n in blob for n in ("диспансер", "дн (", "дн_", "мед_осмотр", "профосмотр")):
        return True
    broad = (
        "урологическими заболеваниями",
        "оториноларингологическими заболеваниями",
        "болезнями кожи",
        "кардиологическими заболеваниями",
        "с заболеваниями нервной системы",
    )
    return any(n in blob for n in broad)


def attach_validity_fields(card: dict[str, Any] | None) -> dict[str, Any]:
    """Нормализовать даты силы на карточке из уже известных полей. Пустое не выдумывать."""
    if not isinstance(card, dict):
        return {}
    meta = validity_from_card(card)
    if not card.get("valid_from") and meta.get("valid_from"):
        card["valid_from"] = meta["valid_from"]
    if not card.get("valid_to") and meta.get("valid_to"):
        card["valid_to"] = meta["valid_to"]
    if not card.get("superseded_on") and meta.get("superseded_on"):
        card["superseded_on"] = meta["superseded_on"]
    if not card.get("replaced_by") and meta.get("replaced_by"):
        card["replaced_by"] = meta["replaced_by"]
    return card
