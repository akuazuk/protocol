"""Единые пороги сверки Dx↔МКБ (фаза 3 калибровки).

Значения по умолчанию - из v2/v3 плана. Переопределение через env для A/B на GCE
без релиза кода. Primary-флаги - отдельно (см. icd_*_primary_enabled).
"""
from __future__ import annotations

import os


def _float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# name_only (Dx ↔ title_ru)
NAME_OK_DEFAULT = 0.42
NAME_REVIEW_DEFAULT = 0.28
SUGGEST_MIN_DEFAULT = 0.08

# directory text_rubric_fit (coverage title)
TEXT_FIT_OK_DEFAULT = 0.35
TEXT_FIT_REVIEW_DEFAULT = 0.25
DIR_HIT_SCORE_MIN_DEFAULT = 0.12


def name_ok() -> float:
    return _float_env("MO_ICD_NAME_OK", NAME_OK_DEFAULT)


def name_review() -> float:
    return _float_env("MO_ICD_NAME_REVIEW", NAME_REVIEW_DEFAULT)


def suggest_min() -> float:
    return _float_env("MO_ICD_SUGGEST_MIN", SUGGEST_MIN_DEFAULT)


def text_fit_ok() -> float:
    return _float_env("MO_ICD_TEXT_FIT_OK", TEXT_FIT_OK_DEFAULT)


def text_fit_review() -> float:
    return _float_env("MO_ICD_TEXT_FIT_REVIEW", TEXT_FIT_REVIEW_DEFAULT)


def dir_hit_score_min() -> float:
    return _float_env("MO_ICD_DIR_HIT_SCORE_MIN", DIR_HIT_SCORE_MIN_DEFAULT)


def snapshot() -> dict[str, float]:
    return {
        "name_ok": name_ok(),
        "name_review": name_review(),
        "suggest_min": suggest_min(),
        "text_fit_ok": text_fit_ok(),
        "text_fit_review": text_fit_review(),
        "dir_hit_score_min": dir_hit_score_min(),
    }


def _flag_on(name: str, *, default: str = "0") -> bool:
    raw = (os.environ.get(name) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def pipeline_in_primary_enabled() -> bool:
    """Общий флаг: оба набора findings (directory + name) в primary."""
    return _flag_on("MO_ICD_PIPELINE_IN_PRIMARY", default="0")


# Обратная совместимость: константы-алиасы для импортов в тестах
NAME_OK = NAME_OK_DEFAULT
NAME_REVIEW = NAME_REVIEW_DEFAULT
SUGGEST_MIN = SUGGEST_MIN_DEFAULT
TEXT_FIT_OK = TEXT_FIT_OK_DEFAULT
TEXT_FIT_REVIEW = TEXT_FIT_REVIEW_DEFAULT
DIR_HIT_SCORE_MIN = DIR_HIT_SCORE_MIN_DEFAULT
