"""Извлечение маркеров лабораторных анализов из OCR/PDF текста (B2C)."""
from __future__ import annotations

import re
from typing import Any

_UNITS = r"г/л|ммоль/л|мкмоль/л|Ед/л|мг/л|мкг/л|МЕ/мл|мм/ч|%|сек\.?|10\^?\d+/л"

# Код на бланке → понятное имя (Кравира / MZ формы).
_CODE_ALIASES: dict[str, str] = {
    "TPROT": "общий белок",
    "UREA": "мочевина",
    "CREA": "креатинин",
    "UR AC": "мочевая кислота",
    "URAC": "мочевая кислота",
    "T-BIL": "билирубин общий",
    "GLUC": "глюкоза",
    "CHOL": "холестерин общий",
    "TRIGLY": "триглицериды",
    "HDL": "ЛПВП",
    "LDL": "ЛПНП",
    "AST": "АСТ",
    "ALT": "АЛТ",
    "AMY": "амилаза",
    "GGT": "ГГТ",
    "ALP": "ЩФ",
    "CRP": "СРБ",
    "FERRITIN": "ферритин",
    "IRON": "железо",
    "CA": "кальций",
    "HGB": "гемоглобин",
    "WBC": "лейкоциты",
    "RBC": "эритроциты",
    "PLT": "тромбоциты",
    "ESR": "СОЭ",
    "INR": "МНО",
    "HBA1C": "HbA1c",
    "PRO-UR": "белок в моче",
    "BIL": "билирубин в моче",
    "KET": "кетоновые тела",
}

# Строка бланка Кравира: обязателен код в скобках «(UREA)».
_KRAVIRA_CODE_ROW = re.compile(
    rf"(?P<name>[A-Za-zА-Яа-яё][A-Za-zА-Яа-яё0-9\s\-.,]+?"
    rf"\([A-Z][A-Z0-9 \-]+\))"
    rf"\s*,?\s*"
    rf"(?P<unit>{_UNITS})?\s*[↑↓]?\s*"
    rf"(?P<val>\d+[.,]\d+|\d+)",
    re.I,
)

# Продолжение названия: «(AST), Ед/л 18,5» (если OCR разбил строку).
_PAREN_CODE_ROW = re.compile(
    rf"\((?P<code>[A-Z][A-Z0-9 \-]*)\)\s*,?\s*(?P<unit>{_UNITS})\s*[↑↓]?\s*(?P<val>\d+[.,]\d+|\d+)",
    re.I,
)
_KRAVIRA_UNIT_ROW = re.compile(
    rf"(?P<name>[A-Za-zА-Яа-яё][A-Za-zА-Яа-яё0-9\s\-]+)"
    rf"\s*,?\s*"
    rf"(?P<unit>{_UNITS})\s*[↑↓]?\s*"
    rf"(?P<val>\d+[.,]\d+|\d+)",
    re.I,
)

# ИНВИТРО: известные шаблоны.
_INVITRO_KNOWN: list[tuple[str, re.Pattern[str]]] = [
    (
        "АТ к нативной (двуспир.) ДНК IgG",
        re.compile(
            r"АТ к нативной \(двуспир\.\) ДНК IgG\s+(?P<val>\d+[.,]\d+)\s+(?P<unit>МЕ/мл)",
            re.I,
        ),
    ),
]

# ИНВИТРО и похожие: короткие названия + число + единица.
_INVITRO_ROW = re.compile(
    rf"(?P<name>[A-Za-zА-Яа-яё][A-Za-zА-Яа-яё0-9\s\-.,()]+?)"
    rf"\s+(?P<val>\d+[.,]\d+|\d+)\s+(?P<unit>{_UNITS})",
    re.I,
)

_OAM_NAMED = re.compile(
    r"(?P<name>Белок|Глюкоза|Кетоновые тела|Билирубин|Уробилиноген|Нитриты|"
    r"Лейкоциты|Эритроциты|Эпителий плоский|Эпителий переходный|Эпителий почечный)"
    r"(?:\s*\([A-Z][A-Z0-9 \-]*\))?"
    r"(?:,\s*[^,]+)?"
    r"\s+[↑↓]?\s*"
    r"(?P<val>отрицательн\w+|не\s+обнаружен\w*|отсутств\w*|[\d+\-–]+)",
    re.I,
)

_PANEL_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("Биохимия крови (БАК)", re.compile(r"биохимич\w*\s+анализ\s+крови|\bбак\b", re.I)),
    ("Общий анализ крови (ОАК)", re.compile(r"общий\s+анализ\s+крови|\bоак\b", re.I)),
    ("Общий анализ мочи (ОАМ)", re.compile(r"анализ\s+мочи\s+общ|общий\s+анализ\s+мочи|\bоам\b", re.I)),
    ("Коагулограмма", re.compile(r"коагулограмм|фibrinogen|фибриноген|мно\b", re.I)),
    ("HbA1c", re.compile(r"\bhba1c\b|гликированн", re.I)),
    ("ИНВИТРО", re.compile(r"\bинвитро\b|независимая\s+лаборатория", re.I)),
]

_JUNK_NAME = re.compile(
    r"^(?:\d+|приложение|к\s+приказу|форма|ф\.?и\.?о|адрес|диагноз|время|дата|"
    r"число,\s*месяц|родился|возраст|пол:|инз|страница|\d+\s+из|\d{1,2}\.\d{1,2}\.\d{2,4}|"
    r"n\s*\d+|у-0|нeman|минск|ул\.|пр-т|медицинский\s+центр|исследование\s+результат|"
    r"референсн|комментар|лицензи|показател|интервал\s+общ)",
    re.I,
)

# Резерв: простые ключевые слова, если структурных строк нет.
_LAB_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("гемоглобин", re.compile(r"гемоглобин|\bhgb\b", re.I)),
    ("лейкоциты", re.compile(r"лейкоцит|\bwbc\b", re.I)),
    ("эритроциты", re.compile(r"эритроцит|\brbc\b", re.I)),
    ("тромбоциты", re.compile(r"тромбоцит|\bplt\b", re.I)),
    ("СОЭ", re.compile(r"\bсоэ\b", re.I)),
    ("глюкоза", re.compile(r"глюкоз|\bgluc\b", re.I)),
    ("СРБ", re.compile(r"\bсрб\b|с-реактивн|\bcrp\b", re.I)),
    ("АЛТ", re.compile(r"\bалт\b|\balt\b", re.I)),
    ("АСТ", re.compile(r"\bast\b", re.I)),
    ("креатинин", re.compile(r"креатинин|\bcrea\b", re.I)),
    ("мочевина", re.compile(r"мочевин|\burea\b", re.I)),
    ("билирубин", re.compile(r"билирубин|\bbil\b", re.I)),
    ("ферритин", re.compile(r"ферритин|\bferritin\b", re.I)),
    ("холестерин", re.compile(r"холестерин|\bchol\b", re.I)),
]


def _norm_val(raw: str) -> str:
    return (raw or "").replace(",", ".").strip()


def _clean_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", (raw or "").strip(" ,."))
    name = re.sub(r"^[↑↓]\s*", "", name)
    return name[:60]


def _name_from_code(code: str, fallback: str) -> str:
    c = re.sub(r"\s+", " ", (code or "").strip().upper())
    if c in _CODE_ALIASES:
        return _CODE_ALIASES[c]
    fb = _clean_name(fallback)
    m = re.search(r"\(([A-Z][A-Z0-9 \-]*)\)", fb)
    if m:
        alias = _CODE_ALIASES.get(m.group(1).strip().upper())
        if alias:
            return alias
    if len(fb) >= 3 and not _JUNK_NAME.search(fb):
        return fb
    if c:
        return c
    return fb or "показатель"


def _is_junk_marker(name: str, value: str | None = None) -> bool:
    n = (name or "").strip()
    if len(n) < 3:
        return True
    if _JUNK_NAME.search(n):
        return True
    if re.match(r"^[а-яёa-z]\s", n, re.I):
        return True
    if re.match(r"^(отрицательн|не\s+обнаруж|отсутств|мл\s*-)", n, re.I):
        return True
    if re.search(r"показател|референсный\s+интервал|результат\s+референс", n, re.I):
        return True
    if re.fullmatch(r"[\d\s.,\-/]+", n):
        return True
    if value and re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{2,4}", _norm_val(value)):
        return True
    return False


def detect_lab_panels(text: str) -> list[str]:
    blob = re.sub(r"\s+", " ", text or "")
    out: list[str] = []
    for label, pat in _PANEL_HINTS:
        if pat.search(blob):
            out.append(label)
    return out


def _append_marker(
    found: list[dict[str, Any]],
    seen: set[str],
    *,
    marker: str,
    value: str | None = None,
    unit: str = "",
    source: str,
    flag: str = "",
) -> None:
    name = _clean_name(marker)
    if _is_junk_marker(name, value):
        return
    key = name.lower()[:50]
    if key in seen:
        return
    seen.add(key)
    item: dict[str, Any] = {"marker": name, "source": source}
    if value is not None:
        item["value"] = _norm_val(value)
    if unit:
        item["unit"] = unit.strip()
    if flag:
        item["flag"] = flag
    found.append(item)


def extract_lab_markers(text: str) -> list[dict[str, Any]]:
    """Список найденных показателей в тексте бланка анализов."""
    if not (text or "").strip():
        return []
    blob = re.sub(r"\s+", " ", text)
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    for m in _KRAVIRA_CODE_ROW.finditer(blob):
        raw_name = m.group("name") or ""
        val = m.group("val") or ""
        unit = (m.group("unit") or "").strip()
        flag = "high" if "↑" in m.group(0) else ""
        marker = _name_from_code("", raw_name)
        _append_marker(found, seen, marker=marker, value=val, unit=unit, source="kravira_row", flag=flag)
        if len(found) >= 35:
            break

    for m in _KRAVIRA_UNIT_ROW.finditer(blob):
        raw_name = m.group("name") or ""
        if _JUNK_NAME.search(raw_name):
            continue
        _append_marker(
            found,
            seen,
            marker=_clean_name(raw_name),
            value=m.group("val"),
            unit=(m.group("unit") or "").strip(),
            source="kravira_unit_row",
            flag="high" if "↑" in m.group(0) else "",
        )

    for m in _PAREN_CODE_ROW.finditer(blob):
        code = m.group("code") or ""
        marker = _name_from_code(code, code)
        _append_marker(
            found,
            seen,
            marker=marker,
            value=m.group("val"),
            unit=(m.group("unit") or "").strip(),
            source="code_row",
        )

    for label, pat in _INVITRO_KNOWN:
        m = pat.search(blob)
        if m:
            _append_marker(
                found,
                seen,
                marker=label,
                value=m.group("val"),
                unit=(m.group("unit") or "").strip(),
                source="invitro_known",
            )

    invitro_hits = list(_INVITRO_ROW.finditer(blob))
    invitro_hits.sort(key=lambda m: len(m.group("name") or ""))
    for m in invitro_hits:
        raw_name = (m.group("name") or "").strip()
        if _JUNK_NAME.search(raw_name) or len(raw_name) > 55 or len(raw_name) < 4:
            continue
        if re.search(r"центр|исследование|референс|лиценз|комментар", raw_name, re.I):
            continue
        _append_marker(
            found,
            seen,
            marker=raw_name,
            value=m.group("val"),
            unit=(m.group("unit") or "").strip(),
            source="invitro_row",
        )

    if detect_lab_panels(blob):
        for m in _OAM_NAMED.finditer(blob):
            raw_name = _clean_name(m.group("name") or "")
            flag = "high" if "↑" in m.group(0) else ""
            _append_marker(
                found,
                seen,
                marker=_name_from_code("", raw_name),
                value=m.group("val"),
                source="oam_row",
                flag=flag,
            )

    if not found:
        for canonical, pat in _LAB_KEYWORDS:
            if pat.search(blob):
                _append_marker(found, seen, marker=canonical, source="keyword")
                if len(found) >= 12:
                    break

    return found


def marker_names(markers: list[dict[str, Any]]) -> list[str]:
    return [str(m.get("marker") or "").strip() for m in markers if m.get("marker")]


def format_marker_line(m: dict[str, Any]) -> str:
    """Человекочитаемая строка для UI: «глюкоза 5.6 ммоль/л»."""
    name = str(m.get("marker") or "").strip()
    val = m.get("value")
    unit = str(m.get("unit") or "").strip()
    if val is None or val == "":
        return name
    line = f"{name} {val}"
    if unit:
        line += f" {unit}"
    flag = str(m.get("flag") or "")
    if flag == "high":
        line += " ↑"
    return line
