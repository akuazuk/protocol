"""Имена моделей Gemini: нормализация и безопасный fallback."""
from __future__ import annotations

import os
import re

# Алиасы: несуществующие/устаревшие имена → ближайшая РЕАЛЬНО доступная модель.
# Проверено ListModels+generateContent через Render 2026-07-23: доступны 3.6-flash,
# 3.5-flash, 3.1-pro-preview; gemini-3-pro-preview снят (404) → маппим на 3.1-pro-preview.
_PRO_MAX = "gemini-3.1-pro-preview"
_FLASH_MAX = "gemini-3.6-flash"
_MODEL_ALIASES: dict[str, str] = {
    "gemini-3.6-pro": _PRO_MAX,
    "gemini-3.6": _PRO_MAX,
    "gemini-3.5-pro": _PRO_MAX,
    "gemini-3.1-pro": _PRO_MAX,
    "gemini-3-pro": _PRO_MAX,
    "gemini-3-pro-preview": _PRO_MAX,  # снят в API → 3.1-pro-preview
    "gemini-3.1-flash": _FLASH_MAX,
    "gemini-3-flash": _FLASH_MAX,
    "gemini-3-flash-preview": _FLASH_MAX,
    "gemini-pro": _PRO_MAX,
    "gemini-flash": _FLASH_MAX,
}

# Рекомендуемые для продакшена (generateContent) - проходят как есть.
_KNOWN_MODELS = frozenset(
    {
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    }
)


def _strip_models_prefix(name: str) -> str:
    n = (name or "").strip()
    if n.startswith("models/"):
        return n[7:].strip()
    return n


def resolve_gemini_model(
    raw: str | None,
    *,
    default: str | None = None,
    env_fallback_key: str | None = None,
) -> tuple[str, str | None]:
    """
    Возвращает (имя модели, предупреждение или None).
    env_fallback_key - второй env, если primary пуст (напр. GEMINI_MODEL для methodist).
    """
    primary = (raw or "").strip()
    if not primary and env_fallback_key:
        primary = (os.environ.get(env_fallback_key) or "").strip()
    if not primary:
        primary = (default or "gemini-2.5-flash").strip()

    name = _strip_models_prefix(primary)
    low = name.lower()

    if low in _MODEL_ALIASES:
        resolved = _MODEL_ALIASES[low]
        return resolved, f"Модель «{name}» недоступна в API; используется «{resolved}»."

    if low in _KNOWN_MODELS:
        return low, None

    # прочие 3.x без явного алиаса → ближайшая реальная (pro/flash)
    if re.match(r"gemini-3[.\d-]*", low):
        resolved = _PRO_MAX if "pro" in low else _FLASH_MAX
        return resolved, f"Модель «{name}» не в списке известных; используется «{resolved}»."

    return name, None


def methodist_gemini_model_name() -> tuple[str, str | None]:
    return resolve_gemini_model(
        os.environ.get("GEMINI_METHODIST_MODEL"),
        default="gemini-2.5-pro",
        env_fallback_key="GEMINI_MODEL",
    )


def main_gemini_model_name() -> tuple[str, str | None]:
    return resolve_gemini_model(os.environ.get("GEMINI_MODEL"), default="gemini-2.5-flash")
