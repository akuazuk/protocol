"""Условия оказания помощи по протоколу: стационар / амбулаторно / скорая.

Определяется по имени файла и заголовку (быстрый первичный источник), с возможностью
уточнения по тегам чанков. Возвращает нормализованный код, читаемую метку и оценку
уверенности, чтобы в выдаче было понятно: протокол для стационарного или амбулаторного
лечения.
"""
from __future__ import annotations

import re
from typing import Any, Callable

# Нормализованные коды условий оказания.
CARE_SETTING_LABELS: dict[str, str] = {
    "inpatient": "стационарно",
    "outpatient": "амбулаторно",
    "mixed": "стационар и амбулаторно",
    "emergency": "скорая и неотложная",
}

_INPATIENT_MARKERS = (
    "стационар",
    "стац услов",
    "стац. услов",
    "стац-услов",
    "в стац",
    "круглосуточн",
)
_OUTPATIENT_MARKERS = (
    "амбулатор",
    "амбул",
    "поликлин",
    "дневн стационар",
    "дневного стационара",
)
_EMERGENCY_MARKERS = (
    "скорой медицинской помощи",
    "скорая медицинская помощь",
    "неотложн",
    "экстренн",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("_", " ").replace("-", " ")).strip()


def infer_care_setting_from_filename(path: str, title: str = "") -> str | None:
    """inpatient | outpatient | mixed | emergency | None - по имени файла/заголовку."""
    blob = _norm(f"{path} {title}")
    if not blob:
        return None
    has_in = any(m in blob for m in _INPATIENT_MARKERS)
    has_out = any(m in blob for m in _OUTPATIENT_MARKERS)
    has_emerg = any(m in blob for m in _EMERGENCY_MARKERS)
    if has_in and has_out:
        return "mixed"
    if has_in:
        return "inpatient"
    if has_out:
        return "outpatient"
    if has_emerg:
        return "emergency"
    return None


def _tags_care_setting_counts(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"inpatient": 0, "outpatient": 0}
    for ch in chunks or []:
        tags = ch.get("tags") or {}
        vals = []
        if isinstance(tags, dict):
            cs = tags.get("care_setting")
            if isinstance(cs, list):
                vals.extend(cs)
            elif isinstance(cs, str):
                vals.append(cs)
        raw_cs = ch.get("care_setting")
        if isinstance(raw_cs, list):
            vals.extend(raw_cs)
        for v in vals:
            vn = _norm(str(v))
            if "стационар" in vn or vn == "inpatient":
                counts["inpatient"] += 1
            elif "амбулатор" in vn or vn == "ambulatory" or vn == "outpatient":
                counts["outpatient"] += 1
    return counts


def infer_care_setting_for_path(
    path: str,
    title: str = "",
    chunks_getter: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Полная оценка условий оказания с источником и уверенностью."""
    code = infer_care_setting_from_filename(path, title)
    if code:
        return {
            "care_setting": code,
            "care_setting_label": CARE_SETTING_LABELS.get(code, code),
            "care_setting_source": "filename",
            "care_setting_confidence": 0.9 if code != "mixed" else 0.75,
        }
    if chunks_getter is not None:
        try:
            counts = _tags_care_setting_counts(chunks_getter(path) or [])
        except Exception:
            counts = {"inpatient": 0, "outpatient": 0}
        total = counts["inpatient"] + counts["outpatient"]
        if total >= 2:
            if counts["inpatient"] and counts["outpatient"]:
                dominant = max(counts, key=lambda k: counts[k])
                minor = min(counts, key=lambda k: counts[k])
                if counts[minor] >= max(1, counts[dominant] // 4):
                    code2 = "mixed"
                else:
                    code2 = dominant
            else:
                code2 = "inpatient" if counts["inpatient"] else "outpatient"
            conf = min(0.7, 0.4 + 0.05 * total)
            return {
                "care_setting": code2,
                "care_setting_label": CARE_SETTING_LABELS.get(code2, code2),
                "care_setting_source": "chunk_tags",
                "care_setting_confidence": round(conf, 3),
            }
    return {
        "care_setting": None,
        "care_setting_label": None,
        "care_setting_source": None,
        "care_setting_confidence": 0.0,
    }


def care_setting_label_ru(code: str | None) -> str | None:
    if not code:
        return None
    return CARE_SETTING_LABELS.get(code, code)
