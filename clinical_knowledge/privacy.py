"""Обезличивание персональных данных в отчётах и архивах."""
from __future__ import annotations

import re

# ФИО: «Иванов Павел Леонидович», «Иванов П. Л.», «Ф.И.О: Кузавка Павел, 12.07.1976».
_RE_NAME_PARTS = re.compile(
    r"^\s*(?:ф\.?\s*и\.?\s*о\.?\s*:?\s*)?"
    r"([А-ЯЁA-Z][а-яёa-z\-]+)"
    r"(?:\s+([А-ЯЁA-Z][а-яёa-z\-]+|\w\.))?"
    r"(?:\s+([А-ЯЁA-Z][а-яёa-z\-]+|\w\.))?"
    r"(?:\s*,.*)?$",
    re.UNICODE,
)
_RE_INITIAL = re.compile(r"^[А-ЯЁA-Z]\.?$")


def name_to_initials(full_name: str | None) -> str:
    """Преобразует ФИО в инициалы: «Кузавка Павел Леонидович» → «К. П. Л.»."""
    if not full_name or not str(full_name).strip():
        return "—"
    raw = str(full_name).strip()
    # Отрезаем дату/хвост после запятой.
    raw = raw.split(",")[0].strip()
    m = _RE_NAME_PARTS.match(raw)
    if not m:
        parts = [p for p in re.split(r"\s+", raw) if p and not re.match(r"^\d", p)]
    else:
        parts = [g for g in m.groups() if g]
    if not parts:
        return "—"
    initials: list[str] = []
    for p in parts[:3]:
        ch = p[0].upper()
        if ch.isalpha():
            initials.append(ch + ".")
    return " ".join(initials) if initials else "—"
