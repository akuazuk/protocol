"""Фильтры качества для evidence pack и пробелов L2."""
from __future__ import annotations

import re

_KP_LINE_NOISE = re.compile(
    r"клинический протокол диагностики|"
    r"по медицинскому применению|листке-вкладыш|"
    r"учитываются результаты обследований пациента и клиническая картина|"
    r"как можно раньше обеспечить прием препаратов|"
    r"состоянием пациента\s*,?\s*$",
    re.I,
)
_MONTH_ONLY = re.compile(
    r"^(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|"
    r"октября|ноября|декабря)(?:\s*;\s*(?:января|февраля|марта|апреля|мая|июня|"
    r"июля|августа|сентября|октября|ноября|декабря))*\.?$",
    re.I,
)
_TOC_BULLET = re.compile(r"^[\s—\-–•;.,\d]+$")


def normalize_gap_text(text: str) -> str:
    return re.sub(r"^[—\-–•\s]+", "", (text or "").strip())


def is_kp_checklist_item(text: str) -> bool:
    """Строка похожа на пункт обследования/лечения, а не оглавление PDF."""
    t = normalize_gap_text(text)
    if len(t) < 5 or len(t) > 180:
        return False
    if _TOC_BULLET.match(t):
        return False
    if _MONTH_ONLY.match(t):
        return False
    if _KP_LINE_NOISE.search(t):
        return False
    if t.lower().startswith("кп ") and "диагностика" in t.lower() and len(t) > 80:
        return False
    alpha = sum(1 for c in t if c.isalpha())
    if alpha < 6:
        return False
    return True


def is_usable_evidence_excerpt(text: str) -> bool:
    """Выдержка пригодна для таблицы evidence pack."""
    t = (text or "").strip()
    if len(t) < 12:
        return False
    if not is_kp_checklist_item(t) and _KP_LINE_NOISE.search(t):
        return False
    if _MONTH_ONLY.match(t):
        return False
    if re.match(r"^\[treatment\]", t, re.I):
        return False
    if t.count(";") >= 3 and len(t) < 80:
        return False
    alpha = sum(1 for c in t if c.isalpha())
    return alpha >= 10


def protocol_title_for_path(path: str) -> str:
    from clinical_knowledge.protocol_links import protocol_display_name

    return protocol_display_name(path or None, fallback="", registry_title=None)
