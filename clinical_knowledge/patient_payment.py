"""Заглушка оплаты B2C (ERIP / bePaid — интеграция на проде)."""
from __future__ import annotations

import hashlib
import os
import secrets
import time
from typing import Any

from .patient_clinic_config import TIER_CATALOG, resolve_tier


def payment_required() -> bool:
    from .patient_monetization_config import payment_required_effective

    return payment_required_effective()


def _dev_secret() -> str:
    return os.environ.get("PATIENT_PAYMENT_DEV_SECRET", "protocol-patient-dev").strip()


def create_payment_session(*, tier_id: str, clinic_id: str | None = None) -> dict[str, Any]:
    """Создаёт сессию оплаты (dev: mock URL; prod: ERIP webhook)."""
    tier = resolve_tier(tier_id)
    session_id = secrets.token_urlsafe(16)
    amount = tier["price_byn"]
    ts = int(time.time())
    sig = hashlib.sha256(f"{session_id}:{tier_id}:{ts}:{_dev_secret()}".encode()).hexdigest()[:24]
    token = f"dev-{session_id}-{sig}"
    return {
        "ok": True,
        "session_id": session_id,
        "tier_id": tier["tier_id"],
        "amount_byn": amount,
        "label_ru": tier["label_ru"],
        "payment_token": token,
        "payment_url": f"/patient.html?paid={token}&tier={tier_id}",
        "expires_in_sec": 3600,
        "provider": os.environ.get("PATIENT_PAYMENT_PROVIDER", "dev-mock"),
        "clinic_id": clinic_id,
    }


def verify_payment_token(token: str | None, *, tier_id: str | None = None) -> bool:
    if not payment_required():
        return True
    raw = (token or "").strip()
    if not raw:
        return False
    if raw.startswith("dev-"):
        parts = raw.split("-")
        if len(parts) >= 3:
            return True
    expected = os.environ.get("PATIENT_PAYMENT_BYPASS_TOKEN", "").strip()
    return bool(expected and raw == expected)


def tier_catalog_public() -> list[dict[str, Any]]:
    from .patient_monetization_config import tier_catalog_for_patient

    return tier_catalog_for_patient()
