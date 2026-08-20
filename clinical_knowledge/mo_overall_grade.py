"""Итог МО: одна шкала из пяти уровней на уже посчитанных зонах и риске.

Не заменяет зоны / очередь / №55. Не читает Rceth, пока findings не в primary.
"""
from __future__ import annotations

from typing import Any, Mapping

ENGINE = "mo_overall_grade_v1"

GRADE_ORDER = ("critical", "important", "poor", "fair", "good")

GRADE_LABEL_RU = {
    "critical": "Критично",
    "important": "Важно",
    "poor": "Слабо",
    "fair": "С замечанием",
    "good": "Хорошо",
}

GRADE_HINT_RU = {
    "critical": "Есть риск вреда. Разбирать в первую очередь.",
    "important": "Клинически важный дефект или важный риск. В очереди.",
    "poor": "Существенный пробел оформления или план не по подобранному протоколу.",
    "fair": "Клиника в целом держится, есть дырки оформления или слабый диагноз.",
    "good": "Зоны в норме, опасных сигналов нет.",
}


def _band(raw: Any) -> str:
    return str(raw or "na").strip().lower() or "na"


def compute_mo_overall_grade(
    zones: Mapping[str, Any] | None,
    *,
    rceth_primary: bool = False,
) -> dict[str, Any]:
    """Собрать итог из выхода compute_mo_zone_scores (или плоских колонок склада)."""
    z = dict(zones or {})
    if z.get("skipped") or z.get("zone1_band") == "na" and z.get("reason") == "non_clinical":
        return {
            "ok": True,
            "engine": ENGINE,
            "grade": "na",
            "label_ru": "нет данных",
            "hint_ru": "Не клиническое МО, оценку не ставим.",
            "reason_ru": "",
            "rank": None,
        }

    safety = z.get("safety") if isinstance(z.get("safety"), Mapping) else {}
    safety_band = _band(safety.get("band") or z.get("safety_band"))
    zone1 = _band(z.get("zone1_band") or (z.get("zone1") or {}).get("band"))
    zone2a = _band(z.get("zone2a_band") or (z.get("zone2a") or {}).get("band"))
    zone2b = _band(z.get("zone2b_band") or (z.get("zone2b") or {}).get("band"))
    kp = str(z.get("zone2b_kp_status") or (z.get("zone2b") or {}).get("kp_status") or "unmatched")

    if safety_band == "critical":
        grade, reason = "critical", "Критичный риск (безопасность)"
    elif safety_band == "important":
        grade, reason = "important", "Важный сигнал риска"
    elif zone2a == "bad":
        grade, reason = "important", "Диагноз не оформлен или не обоснован"
    elif zone2b == "bad" and kp == "matched":
        grade, reason = "poor", "План не соответствует подобранному протоколу"
    elif zone1 == "bad":
        grade, reason = "poor", "Оформление МО с существенными пробелами"
    elif zone2a == "weak" or zone1 == "weak":
        grade, reason = "fair", (
            "Диагноз слабо оформлен" if zone2a == "weak" else "Оформление с замечаниями"
        )
    elif zone1 == "ok" and zone2a == "ok" and zone2b in {"ok", "na"}:
        grade, reason = "good", "Оформление и диагноз в норме, опасных сигналов нет"
    elif zone2b == "bad" and kp != "matched":
        # Склад 2026-08: 3946 unmatched+bad. Это не «план не по протоколу».
        grade, reason = "fair", "План спорный, протокол не подобран - не штрафуем как клинику"
    else:
        grade, reason = "fair", "Смешанная картина по зонам"

    if rceth_primary:
        rceth_codes = [
            str(c)
            for c in (z.get("rceth_codes") or [])
            if str(c).startswith("C_rceth_")
        ]
        if "C_rceth_contraindication" in rceth_codes and grade in {"good", "fair"}:
            grade, reason = "important", "Противопоказание по инструкции ЛС (rceth)"

    return {
        "ok": True,
        "engine": ENGINE,
        "grade": grade,
        "label_ru": GRADE_LABEL_RU[grade],
        "hint_ru": GRADE_HINT_RU[grade],
        "reason_ru": reason,
        "rank": GRADE_ORDER.index(grade),
    }
