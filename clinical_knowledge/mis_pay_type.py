"""Человекочитаемые ярлыки pay_type из mis_data (Kravira MIS).

Коды проверены по живым строкам июля 2026:
- 3 → почти всегда страховая компания в company
- 12 → contracttype со справками / паспортами здоровья / профосмотрами
- 2 → без company, типичный наличный контур
- 0 → pay_type пустой/нулевой, company пустой
"""
from __future__ import annotations

from typing import Any

# Код → ярлык для UI / агрегатов L1
PAY_TYPE_LABELS_RU: dict[str, str] = {
    "0": "Не указан",
    "2": "Наличный расчёт",
    "3": "Страхование (ДМС)",
    "12": "Справки и профосмотры",
}

PAY_TYPE_NOTES_RU: dict[str, str] = {
    "0": "В mis_data company и contracttype пустые.",
    "2": "Без страховой компании; чаще платный приём.",
    "3": "В company - страховая (Белгосстрах и др.).",
    "12": "В contracttype - справки, паспорта здоровья, школьные осмотры.",
}


def normalize_pay_type_code(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    try:
        # 3.0 → "3"
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def pay_type_label_ru(raw: Any) -> str:
    code = normalize_pay_type_code(raw)
    if not code:
        return "Не указан"
    return PAY_TYPE_LABELS_RU.get(code, f"Код {code}")


def pay_type_meta(raw: Any) -> dict[str, str]:
    code = normalize_pay_type_code(raw)
    return {
        "pay_type": code or "",
        "pay_type_label": pay_type_label_ru(code),
        "pay_type_note": PAY_TYPE_NOTES_RU.get(code, ""),
    }
