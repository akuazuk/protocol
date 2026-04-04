"""Нормализация текста: пробелы, типографика, без изменения смысла."""
from __future__ import annotations

import re

_WS = re.compile(r"[ \t\r\f\v]+")
_MULTINL = re.compile(r"\n{3,}")


def normalize_text(s: str) -> str:
    if not s:
        return ""
    t = s.replace("\r\n", "\n").replace("\r", "\n")
    t = _MULTINL.sub("\n\n", t)
    lines = []
    for line in t.split("\n"):
        lines.append(_WS.sub(" ", line).strip())
    t = "\n".join(lines)
    return t.strip()
