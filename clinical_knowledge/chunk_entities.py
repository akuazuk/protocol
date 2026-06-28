"""Извлечение и обогащение сущностей в rich-чанках."""
from __future__ import annotations

import re
from typing import Any

# Базовые regex (дублируем минимально необходимое из build_rich_chunks)
LAB_RE = re.compile(
    r"\b(ОАК|ОАМ|БАК|ПТИ|МНО|СОЭ|ЦИК|ИФА|ПЦР|СРБ|ТТГ|Т3|Т4|PSA|АЛТ|АСТ|ЩФ|ГГТ|АЧТВ|"
    r"HbA1c|D-димер|"
    r"гемоглобин|гематокрит|лейкоцит|тромбоцит|эритроцит|нейтрофил|"
    r"билирубин|креатинин|мочевин|глюкоз|холестерин|триглицерид|фибриноген)\b",
    re.I,
)

IMAGING_RE = re.compile(
    r"\b(КТ|МРТ|УЗИ|НСГ|ЭКГ|ЭЭГ|ЭхоКГ|РОГП|ПЭТ|"
    r"рентген(?:ография)?|сцинтиграфи|ангиографи|"
    r"флюорографи|денситометри|эхокардиографи)\b",
    re.I,
)

DRUG_RE = re.compile(
    r"(?:назнача(?:ют|ется|ть)|применя(?:ют|ется|ть)|терапия|препарат|лечение)\s+"
    r"([А-ЯЁA-Z][а-яёa-z\-]+(?:\s+[а-яёa-z\-]+){0,3})",
    re.I,
)

# Синонимы → каноническое сокращение
_LAB_SYNONYMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"общ(?:ий|его)\s*\(?\s*клиническ(?:ий|ого)\s*\)?\s*анализ\s*крови", re.I), "ОАК"),
    (re.compile(r"биохимическ(?:ий|ого)\s*анализ\s*крови", re.I), "БАК"),
    (re.compile(r"общ(?:ий|его)\s*анализ\s*мочи", re.I), "ОАМ"),
    (re.compile(r"коагулограмм", re.I), "МНО"),
    (re.compile(r"электрокардиограф", re.I), "ЭКГ"),
    (re.compile(r"ультразвуков(?:ое|ого)\s*исследован", re.I), "УЗИ"),
    (re.compile(r"рентгенограф(?:ия|ии)\s+органов\s+груд", re.I), "РОГП"),
    (re.compile(r"с[\-\s]?реактивн(?:ый|ого)\s+белок", re.I), "СРБ"),
    (re.compile(r"аланинаминотransferase|алат", re.I), "АЛТ"),
    (re.compile(r"асpartatаминотransferase|асат", re.I), "АСТ"),
]

_IMAGING_SYNONYMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"компьютерн(?:ая|ой)\s*tomograph|компьютерн(?:ая|ой)\s*tomograph", re.I), "КТ"),
    (re.compile(r"магнитно[\-\s]?резонанс", re.I), "МРТ"),
    (re.compile(r"нейросонograph|нейросонograph", re.I), "НСГ"),
]


def _unique_append(out: list[str], seen: set[str], value: str, *, limit: int = 30) -> None:
    v = value.strip()
    if not v or v in seen:
        return
    seen.add(v)
    out.append(v)
    if len(out) >= limit:
        return


def extract_lab_tests_enriched(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in LAB_RE.finditer(text or ""):
        _unique_append(out, seen, m.group(0).upper() if len(m.group(0)) <= 6 else m.group(0), limit=30)
    for pat, label in _LAB_SYNONYMS:
        if pat.search(text or ""):
            _unique_append(out, seen, label, limit=30)
    return out[:30]


def extract_imaging_enriched(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in IMAGING_RE.finditer(text or ""):
        val = m.group(0)
        _unique_append(out, seen, val.upper() if len(val) <= 4 else val, limit=15)
    for pat, label in _IMAGING_SYNONYMS:
        if pat.search(text or ""):
            _unique_append(out, seen, label, limit=15)
    return out[:15]


def extract_drugs_heuristic(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in DRUG_RE.finditer(text or ""):
        s = m.group(1).strip()
        if len(s) > 4:
            _unique_append(out, seen, s, limit=30)
    return out[:30]


def enrich_chunk_entities(chunk: dict[str, Any]) -> dict[str, Any]:
    """Обновить lab_tests, imaging, drugs на чанке."""
    text = str(chunk.get("text") or "")
    labs = extract_lab_tests_enriched(text)
    imaging = extract_imaging_enriched(text)
    drugs = extract_drugs_heuristic(text)
    if labs:
        chunk["lab_tests"] = labs
    if imaging:
        chunk["imaging"] = imaging
    if drugs:
        chunk["drugs"] = drugs
    return chunk
