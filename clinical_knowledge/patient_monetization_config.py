"""Настройки монетизации B2C - управление из кабинета методиста."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .patient_clinic_config import TIER_CATALOG

ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = ROOT / "data" / "patient_monetization.json"

_ALL_TIER_IDS = list(TIER_CATALOG.keys())

DEFAULT_PATIENT_MONETIZATION: dict[str, Any] = {
    "monetization_enabled": False,
    "payment_required": False,
    "show_tier_picker": True,
    "show_prices": True,
    "default_tier_id": "basic",
    "enabled_tier_ids": list(_ALL_TIER_IDS),
    "demo_note_ru": (
        "Проверка бесплатна в демо-режиме. Результат - ориентир для разговора с врачом, не диагноз."
    ),
    "paid_note_ru": (
        "Перед проверкой нужна оплата выбранного тарифа. После оплаты вы вернётесь на эту страницу автоматически."
    ),
    "value_banner_ru": (
        "Вы получите светофор по 8 блокам КЗ, вопросы врачу своими словами и ссылки на протоколы Минздрава."
    ),
}


def _config_path() -> Path:
    raw = (os.environ.get("PATIENT_MONETIZATION_CONFIG") or "").strip()
    return Path(raw) if raw else _DEFAULT_CONFIG_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_tier_ids(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return list(_ALL_TIER_IDS)
    out: list[str] = []
    for item in raw:
        tid = str(item or "").strip().lower()
        if tid in TIER_CATALOG and tid not in out:
            out.append(tid)
    return out or list(_ALL_TIER_IDS)


def _merge_defaults(data: dict[str, Any] | None) -> dict[str, Any]:
    base = deepcopy(DEFAULT_PATIENT_MONETIZATION)
    if not isinstance(data, dict):
        return base
    for key in (
        "monetization_enabled",
        "payment_required",
        "show_tier_picker",
        "show_prices",
    ):
        if key in data:
            base[key] = bool(data[key])
    if data.get("default_tier_id"):
        tid = str(data["default_tier_id"]).strip().lower()
        if tid in TIER_CATALOG:
            base["default_tier_id"] = tid
    base["enabled_tier_ids"] = _normalize_tier_ids(data.get("enabled_tier_ids"))
    if base["default_tier_id"] not in base["enabled_tier_ids"]:
        base["default_tier_id"] = base["enabled_tier_ids"][0]
    for note_key in ("demo_note_ru", "paid_note_ru", "value_banner_ru"):
        if data.get(note_key):
            base[note_key] = str(data[note_key]).strip()[:500]
    if data.get("updated_at"):
        base["updated_at"] = str(data["updated_at"])
    if data.get("updated_by"):
        base["updated_by"] = str(data["updated_by"])
    return base


def load_patient_monetization_config() -> dict[str, Any]:
    path = _config_path()
    if not path.is_file():
        return _merge_defaults(None)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _merge_defaults(None)
    return _merge_defaults(raw if isinstance(raw, dict) else None)


def save_patient_monetization_config(
    patch: dict[str, Any],
    *,
    reviewer: str = "",
) -> dict[str, Any]:
    current = load_patient_monetization_config()
    merged = _merge_defaults({**current, **(patch or {})})
    merged["updated_at"] = _utc_now()
    merged["updated_by"] = (reviewer or merged.get("updated_by") or "").strip()[:80]
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return merged


def payment_required_effective() -> bool:
    """Требовать оплату: конфиг методиста → env PATIENT_PAYMENT_REQUIRED."""
    cfg = load_patient_monetization_config()
    if not cfg.get("monetization_enabled"):
        return False
    if cfg.get("payment_required"):
        return True
    return os.environ.get("PATIENT_PAYMENT_REQUIRED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def enabled_tier_ids(cfg: dict[str, Any] | None = None) -> list[str]:
    data = cfg or load_patient_monetization_config()
    return _normalize_tier_ids(data.get("enabled_tier_ids"))


def tier_catalog_for_patient(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = cfg or load_patient_monetization_config()
    ids = enabled_tier_ids(data) if data.get("monetization_enabled") else [data.get("default_tier_id") or "basic"]
    if data.get("default_tier_id") not in ids:
        ids = [str(data.get("default_tier_id") or "basic"), *ids]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for tid in ids:
        if tid in seen or tid not in TIER_CATALOG:
            continue
        seen.add(tid)
        t = TIER_CATALOG[tid]
        row = {
            "tier_id": t["tier_id"],
            "label_ru": t["label_ru"],
            "price_byn": t["price_byn"],
            "review_tier": t["review_tier"],
            "hint_ru": t.get("hint_ru") or "",
            "includes": list(t.get("includes") or []),
        }
        if not data.get("show_prices"):
            row.pop("price_byn", None)
        out.append(row)
    return out


def monetization_public_view() -> dict[str, Any]:
    cfg = load_patient_monetization_config()
    pay_req = payment_required_effective()
    note = cfg.get("paid_note_ru") if pay_req else cfg.get("demo_note_ru")
    return {
        "monetization_enabled": bool(cfg.get("monetization_enabled")),
        "payment_required": pay_req,
        "show_tier_picker": bool(cfg.get("show_tier_picker")) and bool(cfg.get("monetization_enabled")),
        "show_prices": bool(cfg.get("show_prices")),
        "default_tier_id": cfg.get("default_tier_id") or "basic",
        "demo_note_ru": cfg.get("demo_note_ru") or "",
        "paid_note_ru": cfg.get("paid_note_ru") or "",
        "value_banner_ru": cfg.get("value_banner_ru") or "",
        "payment_note_ru": note or "",
        "tiers": tier_catalog_for_patient(cfg),
        "updated_at": cfg.get("updated_at"),
    }


def monetization_admin_view() -> dict[str, Any]:
    cfg = load_patient_monetization_config()
    return {
        **cfg,
        "payment_required_effective": payment_required_effective(),
        "env_payment_required": os.environ.get("PATIENT_PAYMENT_REQUIRED", "0"),
        "tier_catalog_all": [
            {
                "tier_id": t["tier_id"],
                "label_ru": t["label_ru"],
                "price_byn": t["price_byn"],
                "review_tier": t["review_tier"],
                "hint_ru": t.get("hint_ru") or "",
            }
            for t in TIER_CATALOG.values()
        ],
        "config_path": str(_config_path()),
    }
