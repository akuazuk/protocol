"""White-label конфигурация клиник и тарифы B2C (B2B2C)."""
from __future__ import annotations

from typing import Any

# clinic query param → branding + default tier
CLINIC_PRESETS: dict[str, dict[str, Any]] = {
    "kravira": {
        "clinic_id": "kravira",
        "name_ru": "МЦ «Кравира»",
        "tagline_ru": "Проверка заключения по протоколам Минздрава",
        "primary_color": "#1a8a72",
        "rev_share_pct": 30,
        "default_tier": "basic",
        "footer_ru": "Проверено для МЦ «Кравира» · технология Protocol",
    },
    "demo": {
        "clinic_id": "demo",
        "name_ru": "Клиника-партнёр (демо)",
        "tagline_ru": "Сервис качества заключений",
        "primary_color": "#0d5c4a",
        "rev_share_pct": 30,
        "default_tier": "promo",
        "footer_ru": "Проверено для вашей клиники · Protocol",
    },
}

TIER_CATALOG: dict[str, dict[str, Any]] = {
    "promo": {
        "tier_id": "promo",
        "label_ru": "Промо-проверка",
        "price_byn": 2.99,
        "review_tier": "P1",
        "hint_ru": "Светофор, блоки КЗ и вопросы врачу - вход по ссылке клиники.",
        "includes": ["traffic_light", "blocks", "questions"],
    },
    "basic": {
        "tier_id": "basic",
        "label_ru": "Базовая проверка",
        "price_byn": 4.99,
        "review_tier": "P1",
        "hint_ru": "Полный разбор 8 блоков, вопросы врачу и цитаты из протоколов Минздрава.",
        "includes": ["traffic_light", "blocks", "questions", "citations"],
    },
    "plus": {
        "tier_id": "plus",
        "label_ru": "С анализами",
        "price_byn": 6.99,
        "review_tier": "P1",
        "hint_ru": "Всё из базового + сверка бланков анализов с заключением и протоколом.",
        "includes": ["basic", "lab_crosscheck", "protocol_crosscheck"],
    },
    "detailed": {
        "tier_id": "detailed",
        "label_ru": "Подробная проверка",
        "price_byn": 9.99,
        "review_tier": "P2",
        "hint_ru": "Расширенный отчёт простым языком и пакет доказательств по протоколу.",
        "includes": ["plus", "plain_narrative", "evidence_pack"],
    },
    "onco": {
        "tier_id": "onco",
        "label_ru": "Онкология",
        "price_byn": 14.99,
        "review_tier": "P2",
        "hint_ru": "Углублённый разбор при онкологическом контексте и приоритеты безопасности лечения.",
        "includes": ["detailed", "priority_treatment_safety"],
    },
}


def resolve_clinic(clinic_id: str | None) -> dict[str, Any] | None:
    key = (clinic_id or "").strip().lower()
    if not key:
        return None
    return CLINIC_PRESETS.get(key)


def resolve_tier(tier_id: str | None, clinic: dict[str, Any] | None = None) -> dict[str, Any]:
    key = (tier_id or "").strip().lower()
    if not key and clinic:
        key = str(clinic.get("default_tier") or "basic")
    if key not in TIER_CATALOG:
        key = "basic"
    return TIER_CATALOG[key]


def clinic_public_view(clinic: dict[str, Any]) -> dict[str, Any]:
    return {
        "clinic_id": clinic.get("clinic_id"),
        "name_ru": clinic.get("name_ru"),
        "tagline_ru": clinic.get("tagline_ru"),
        "primary_color": clinic.get("primary_color"),
        "footer_ru": clinic.get("footer_ru"),
    }
