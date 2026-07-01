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

# Организационно-маршрутный / процедурный текст: не является клинической выдержкой
# (кто выполняет, куда направляют, порядок направления, чем определяется).
_ORG_ROUTING = re.compile(
    r"направля(?:ю|е)т(?:ся)?\s+пациент|"
    r"на\s+консультаци\w*\s+к\s+врач|"
    r"порядок\s+направлени|"
    r"определяется\s+министерством|"
    r"выполняют\s+врачи|"
    r"в\s+организаци\w*\s+здравоохранени\w*\s+в\s+амбулаторных|"
    r"устанавливает\s+общие\s+требования\s+к\s+объ[её]му|"
    r"осуществляется\s+в\s+соответствии\s+с\s+клиническим\s+протоколом",
    re.I,
)
# Признак обрезанного слова в конце коротких выдержек (напр. «динамическое наблюден»).
_TRUNCATED_TAIL = re.compile(
    r"\b(?:наблюден|обследован|консультаци|лечени|диагностик|рекомендаци|"
    r"вмешательств|показани)$",
    re.I,
)


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
    """Выдержка пригодна для таблицы evidence pack (клиническая, не мусор)."""
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
    # Административный/нормативный текст (утверждение, подпись министра, портал НПА).
    try:
        from clinical_knowledge.chunk_tags import is_administrative_text

        if is_administrative_text(t):
            return False
    except Exception:
        pass
    # Организационно-маршрутный/процедурный текст - не клиническая выдержка.
    if _ORG_ROUTING.search(t):
        return False
    # Обрывочные короткие выдержки с обрезанным словом в конце.
    tokens = [w for w in re.split(r"\s+", t) if w]
    if len(tokens) <= 2 and len(t) < 25 and not re.search(r"\d", t):
        return False
    stripped = t.rstrip(".!?;:,) ")
    if len(tokens) <= 4 and _TRUNCATED_TAIL.search(stripped):
        return False
    alpha = sum(1 for c in t if c.isalpha())
    return alpha >= 10


def is_usable_summary_excerpt(text: str) -> bool:
    """Проверка для structured-полей сводки (названия обследований, препаратов).

    В отличие от is_usable_evidence_excerpt не требует минимальной длины:
    клинические названия часто короткие («МРТ», «УЗИ», «ОАК»). Но так же
    отсекает административный/организационный/обрывочный шум.
    """
    t = (text or "").strip()
    if len(t) < 2:
        return False
    if _MONTH_ONLY.match(t):
        return False
    if _ORG_ROUTING.search(t):
        return False
    try:
        from clinical_knowledge.chunk_tags import is_administrative_text

        if is_administrative_text(t):
            return False
    except Exception:
        pass
    tokens = [w for w in re.split(r"\s+", t) if w]
    stripped = t.rstrip(".!?;:,) ")
    if len(tokens) <= 4 and _TRUNCATED_TAIL.search(stripped):
        return False
    return any(c.isalpha() for c in t)


def protocol_title_for_path(path: str) -> str:
    from clinical_knowledge.protocol_links import protocol_display_name

    return protocol_display_name(path or None, fallback="", registry_title=None)
