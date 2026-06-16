#!/usr/bin/env python3
"""
Глубокий переразбор всех PDF протоколов в rich-чанки.

Выходной файл: output/rich_chunks/rich_chunks.jsonl
Паспорта файлов: output/rich_meta/{doc_id}.json

Запуск:
  cd /Users/pavelkuzauka/Cursor_Folders/Protocol
  .venv/bin/python scripts/build_rich_chunks.py [--rubric NAME] [--resume] [--file PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icd_mkb import extract_icd_codes_raw, normalize_icd_code

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PDF_ROOT = ROOT / "minzdrav_protocols"
OUT_DIR = ROOT / "output" / "rich_chunks"
META_DIR = ROOT / "output" / "rich_meta"
CATALOG_PATH = ROOT / "data" / "protocol_catalog.jsonl"
CHUNKS_FILE = OUT_DIR / "rich_chunks.jsonl"
MANIFEST_FILE = OUT_DIR / "_manifest.json"
ERRORS_FILE = OUT_DIR / "_errors.jsonl"

SPECIALTY_RU: dict[str, str] = {
    "akusherstvo-ginekologiya": "Акушерство и гинекология",
    "allergologiya-immunologiya": "Аллергология и иммунология",
    "anesteziologiya-reanimatologiya": "Анестезиология и реаниматология",
    "bolezni-sistemy-krovoobrashcheniya": "Болезни системы кровообращения",
    "dermatovenerologiya": "Дерматовенерология",
    "endokrinologiya-narusheniya-obmena-veshchestv": "Эндокринология",
    "gastroenterologiya": "Гастроэнтерология",
    "gematologiya": "Гематология",
    "infektsionnye-zabolevaniya": "Инфекционные заболевания",
    "khirurgiya": "Хирургия",
    "nefrologiya": "Нефрология",
    "nevrologiya-neyrokhirurgiya": "Неврология и нейрохирургия",
    "novoobrazovaniya": "Онкология",
    "oftalmologiya": "Офтальмология",
    "otorinolaringologiya": "Оториноларингология",
    "palliativnaya-pomoshch": "Паллиативная помощь",
    "psikhiatriya-narkologiya": "Психиатрия и наркология",
    "pulmonologiya-ftiziatriya": "Пульмонология и фтизиатрия",
    "revmatologiya": "Ревматология",
    "stomatologiya": "Стоматология",
    "transplantatsiya-organov-i-tkaney": "Трансплантация органов и тканей",
    "travmatologiya-ortopediya": "Травматология и ортопедия",
    "urologiya": "Урология",
    "zabolevaniya-perinatalnogo-perioda": "Заболевания перинатального периода",
}

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------
ICD10_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,4})?)\b", re.I)

# Пресамбула постановления: блоки до начала протокола
PREAMBLE_RE = re.compile(
    r"(?:ПОСТАНОВЛЕНИЕ\s+МИНИСТЕРСТВА|МИНИСТЕРСТВО\s+ЗДРАВООХРАНЕНИЯ"
    r"|Об\s+утверждении\s+(?:некоторых\s+)?клинических?\s+протоколов?"
    r"|ПОСТАНОВЛЯЕТ\s*:|На\s+основании\s+абзаца"
    r"|Министр\s+здравоохранения"
    r"|Форма\s+\d+[-/])"
    ,
    re.I,
)

# Заголовки разделов протокола
SECTION_RE = re.compile(
    r"(?m)^(?:ГЛАВА\s+\d+|(?:\d+\.)+\s*[А-ЯЁA-Z]|"
    r"(?:ДИАГНОСТИКА|ЛЕЧЕНИЕ|ПРОФИЛАКТИКА|РЕАБИЛИТАЦИЯ|ДИСПАНСЕРНОЕ|"
    r"МАРШРУТИЗАЦИЯ|ФАРМАКОТЕРАПИЯ|КЛАССИФИКАЦИЯ|ПРИЛОЖЕНИЕ|АЛГОРИТМ"
    r"|ТЕРМИНЫ|ОБЩИЕ\s+ПОЛОЖЕНИЯ|ПОКАЗАНИЯ|КРИТЕРИИ))",
)

SECTION_NUMBER_RE = re.compile(r"^\s*((?:\d+\.)+\d*|\d+)[.)]\s*", re.M)

# Нумерованный пункт внутри раздела
POINT_RE = re.compile(r"(?m)(?:^|\n)\s*(?:\d+(?:\.\d+)*[.)]\s+|[пП]\.?\s*\d+[.)]\s+)")

# Возрастные диапазоны
AGE_RANGE_RE = re.compile(
    r"(\d+)\s*[-–]\s*(\d+)\s*(лет|год(?:а)?|месяц(?:ев|а)?)"
    r"|(?:старше|до|от|менее|более)\s+(\d+)\s*(лет|год(?:а)?|месяц(?:ев|а)?)",
    re.I,
)

# Степени тяжести
SEVERITY_RE = re.compile(r"(лёгк(?:ая|ой|ого)|средн(?:яя|ей)|тяжёл(?:ая|ой)|крайне\s+тяжёл)", re.I)

# Дозировки
DOSAGE_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:мг|мкг|г|ЕД|мл|ммоль|мкмоль|МЕ)\s*/?\s*(?:кг|сут|день|ч|мин)?",
    re.I,
)

# Лабораторные тесты
LAB_RE = re.compile(
    r"\b(ОАК|ОАМ|БАК|ПТИ|МНО|СОЭ|ЦИК|ИФА|ПЦР|КТ|МРТ|УЗИ|ЭКГ|ЭЭГ|"
    r"ЭхоКГ|ФГДС|ФЭГДС|HbA1c|СРБ|ТТГ|Т3|Т4|PSA|АЛТ|АСТ|ЩФ|ГГТ|"
    r"гемоглобин|гематокрит|лейкоцит|тромбоцит|эритроцит|нейтрофил|"
    r"билирубин|креатинин|мочевин|глюкоз|холестерин|триглицерид)\b",
    re.I,
)

# Визуализация
IMAGING_RE = re.compile(
    r"\b(КТ|МРТ|УЗИ|рентген(?:ография)?|ПЭТ|сцинтиграфи|ангиографи|"
    r"флюорографи|денситометри|эхокардиографи)\b",
    re.I,
)

# Упоминания других протоколов
CROSS_PROTOCOL_RE = re.compile(
    r"(?:клинический\s+протокол|КП)[^.]{5,120}?(?:от|пост(?:ановление)?|№)\s*[\d.]+",
    re.I,
)

# Кросс-ссылки через «постановление»
APPROVAL_RE = re.compile(
    r"(?:постановлени(?:е|я|ем)\s+(?:Министерства|МЗ|Минздрав)[^.]{0,60}"
    r"(?:от\s+)?(\d{1,2}[.\s]\d{1,2}[.\s]\d{4}|\d{4})[^.]{0,30}№\s*([\d\-а-яА-Я]+))",
    re.I,
)

POPULATION_MARKERS = [
    ("новорожд", "новорождённые"),
    ("детск", "дети"),
    ("детей", "дети"),
    ("ребён", "дети"),
    ("ребен", "дети"),
    ("подрост", "подростки"),
    ("взросл", "взрослые"),
    ("беремен", "беременные"),
    ("женщин", "женщины"),
    ("мужчин", "мужчины"),
    ("пожил", "пожилые"),
    ("геронт", "пожилые"),
]

CARE_MARKERS = [
    ("амбулатор", "амбулаторно"),
    ("стационар", "стационар"),
    ("скорой", "скорая помощь"),
    ("неотложн", "неотложная помощь"),
    ("I уров", "I уровень"),
    ("II уров", "II уровень"),
    ("III уров", "III уровень"),
    ("поликлиник", "поликлиника"),
    ("реанимац", "реанимация/ОРИТ"),
]

CHUNK_TYPE_MAP = [
    (re.compile(r"диагностик|критерии|обследование|лаборатор", re.I), "diagnostics"),
    (re.compile(r"лечени|терапи|лечения|хирург|операц", re.I), "treatment"),
    (re.compile(r"профилактик", re.I), "prevention"),
    (re.compile(r"реабилитац", re.I), "rehabilitation"),
    (re.compile(r"диспансерн|наблюден", re.I), "dispensary"),
    (re.compile(r"классификац|шифр|МКБ", re.I), "classification"),
    (re.compile(r"маршрутизац|госпитализац|направлени", re.I), "routing"),
    (re.compile(r"фармакотерапи|медикамент", re.I), "pharmacotherapy"),
    (re.compile(r"алгоритм|схем", re.I), "algorithm"),
    (re.compile(r"приложени", re.I), "appendix"),
    (re.compile(r"показан|противопоказ|критери", re.I), "criteria_block"),
    (re.compile(r"лекарств|препарат|доз", re.I), "drug_list"),
    (re.compile(r"термин|определени|понятие", re.I), "terms"),
]

# Чанки, где коды из текста должны явно попадать в icd10_codes и embedding_ready_text.
_ICD_ENRICH_CHUNK_TYPES = frozenset(
    {
        "diagnostics",
        "classification",
        "criteria_block",
        "treatment",
        "prevention",
    }
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def doc_id(rel_path: str) -> str:
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:24]


def _clean_cell(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def table_to_markdown(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    clean = [[_clean_cell(c) for c in row] for row in rows]
    clean = [r for r in clean if any(r)]
    if len(clean) < 2:
        return " | ".join(clean[0]) if clean else ""
    max_w = max(len(r) for r in clean)
    header = clean[0] + [""] * (max_w - len(clean[0]))
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in clean[1:]:
        padded = row + [""] * (max_w - len(row))
        lines.append("| " + " | ".join(padded[:max_w]) + " |")
    return "\n".join(lines)


def extract_icd10(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in extract_icd_codes_raw(text or ""):
        code = normalize_icd_code(raw)
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


@lru_cache(maxsize=1)
def _protocol_catalog_icd_by_path() -> dict[str, dict[str, list[str]]]:
    """МКБ из data/protocol_catalog.jsonl (icd10_primary / icd10_all)."""
    out: dict[str, dict[str, list[str]]] = {}
    if not CATALOG_PATH.is_file():
        return out
    for line in CATALOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = str(row.get("path") or "").replace("\\", "/").strip()
        if not path:
            continue
        primary = [
            normalize_icd_code(str(c))
            for c in (row.get("icd10_primary") or [])
            if normalize_icd_code(str(c))
        ]
        all_codes = [
            normalize_icd_code(str(c))
            for c in (row.get("icd10_all") or [])
            if normalize_icd_code(str(c))
        ]
        out[path] = {"primary": primary, "all": all_codes}
    return out


def catalog_icd_for_path(rel_path: str) -> dict[str, list[str]]:
    key = (rel_path or "").replace("\\", "/").strip()
    return _protocol_catalog_icd_by_path().get(key, {"primary": [], "all": []})


def merge_icd10_code_lists(*sources: list[str] | tuple[str, ...], max_codes: int = 24) -> list[str]:
    """Уникальные коды МКБ в порядке приоритета источников."""
    seen: set[str] = set()
    out: list[str] = []
    for src in sources:
        for raw in src or []:
            code = normalize_icd_code(str(raw))
            if not code or code in seen:
                continue
            seen.add(code)
            out.append(code)
            if len(out) >= max_codes:
                return out
    return out


def build_chunk_icd10_codes(
    *,
    chunk_text: str,
    chunk_type: str,
    protocol_primary: list[str],
    protocol_all: list[str],
    catalog_primary: list[str],
    catalog_all: list[str],
) -> list[str]:
    """icd10_codes чанка: метаданные протокола + коды из текста (для диагнозов/показаний)."""
    text_codes = extract_icd10(chunk_text)
    doc_primary = merge_icd10_code_lists(catalog_primary, protocol_primary, max_codes=12)
    if chunk_type in _ICD_ENRICH_CHUNK_TYPES:
        return merge_icd10_code_lists(
            text_codes,
            doc_primary,
            catalog_all,
            protocol_all,
            max_codes=28,
        )
    return merge_icd10_code_lists(text_codes, doc_primary, max_codes=18)


def build_embedding_ready_text(
    *,
    section_title: str,
    chunk_text: str,
    icd_codes: list[str],
    populations: list[str],
    chunk_type: str,
) -> str:
    """Текст для эмбеддинга/лексики; для диагнозов/показаний дублирует МКБ из текста."""
    emb_text = section_title + "\n" + chunk_text
    if icd_codes:
        icd_line = "МКБ-10: " + ", ".join(icd_codes[:12])
        emb_text = icd_line + "\n" + emb_text
        if chunk_type in _ICD_ENRICH_CHUNK_TYPES:
            text_codes = extract_icd10(chunk_text)
            extra = [c for c in text_codes if c in icd_codes]
            if extra:
                emb_text = "МКБ-10: " + ", ".join(extra[:10]) + "\n" + emb_text
    if populations:
        emb_text = "Популяция: " + ", ".join(populations) + "\n" + emb_text
    return emb_text


def extract_populations(text: str) -> list[str]:
    low = (text or "").lower()
    seen: set[str] = set()
    out: list[str] = []
    for needle, label in POPULATION_MARKERS:
        if needle in low and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def extract_care_settings(text: str) -> list[str]:
    low = (text or "").lower()
    seen: set[str] = set()
    out: list[str] = []
    for needle, label in CARE_MARKERS:
        if needle in low and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def extract_age_ranges(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in AGE_RANGE_RE.finditer(text or ""):
        s = m.group(0).strip()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def extract_severity(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in SEVERITY_RE.finditer(text or ""):
        s = m.group(1).lower()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def extract_lab_tests(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in LAB_RE.finditer(text or ""):
        s = m.group(0)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:30]


def extract_imaging(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in IMAGING_RE.finditer(text or ""):
        s = m.group(0)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:15]


def extract_dosages(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in DOSAGE_RE.finditer(text or ""):
        s = m.group(0).strip()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:20]


def extract_cross_protocols(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in CROSS_PROTOCOL_RE.finditer(text or ""):
        s = m.group(0).strip()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:10]


def extract_drugs_heuristic(text: str) -> list[str]:
    DRUG_RE = re.compile(
        r"(?:назначают?|применяют?|терапия|препарат|лечение)\s+"
        r"([А-ЯЁA-Z][а-яёa-z\-]+(?:\s+[а-яёa-z\-]+){0,3})",
        re.I,
    )
    out: list[str] = []
    seen: set[str] = set()
    for m in DRUG_RE.finditer(text or ""):
        s = m.group(1).strip()
        if len(s) > 4 and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= 30:
            break
    return out


def extract_durations(text: str) -> list[str]:
    DURATION_RE = re.compile(
        r"\d+(?:[.,]\d+)?\s*(?:суток?|сут|дн(?:ей|я)?|недел(?:ь|и|е)|"
        r"мес(?:яц(?:ев|а)?)?|лет|год(?:а)?|час(?:ов|а)?|мин(?:ут)?)",
        re.I,
    )
    out: list[str] = []
    seen: set[str] = set()
    for m in DURATION_RE.finditer(text or ""):
        s = m.group(0).strip()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:15]


def keywords_from_text(text: str, max_kw: int = 40) -> list[str]:
    STOPWORDS = {
        "также", "более", "менее", "после", "перед", "между", "через",
        "которые", "которых", "которого", "которой", "который", "этого",
        "этой", "этот", "этим", "этих", "такое", "такого", "такой",
        "такие", "такая", "данные", "данного", "данной", "данной",
        "пункта", "пунктов", "статьи", "части", "указан", "указанных",
        "следующих", "следующие", "следующем",
    }
    words = re.findall(r"[а-яА-ЯёЁ]{5,}", text or "")
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        low = w.lower()
        if low not in seen and low not in STOPWORDS:
            seen.add(low)
            out.append(low)
        if len(out) >= max_kw:
            break
    return out


def guess_chunk_type(section_title: str, text: str) -> str:
    combined = (section_title + " " + text[:200]).lower()
    for pattern, ctype in CHUNK_TYPE_MAP:
        if pattern.search(combined):
            return ctype
    return "body"


def is_preamble_block(text: str) -> bool:
    """Возвращает True, если блок целиком является служебной шапкой постановления."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    preamble_hits = sum(1 for l in lines if PREAMBLE_RE.search(l))
    return preamble_hits / max(len(lines), 1) >= 0.5


def filter_preamble_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    Убирает страницы/блоки которые целиком состоят из служебной шапки.
    Останавливается, как только нашли содержательный текст.
    """
    result = []
    found_content = False
    for page_no, text in pages:
        if not found_content and is_preamble_block(text):
            continue  # пропускаем служебную страницу
        found_content = True
        # Внутри страницы: убираем строки-колонтитулы в начале
        lines = text.splitlines()
        cleaned: list[str] = []
        skip_head = True
        for line in lines:
            stripped = line.strip()
            if skip_head and (not stripped or PREAMBLE_RE.search(stripped)):
                continue
            skip_head = False
            cleaned.append(line)
        result.append((page_no, "\n".join(cleaned)))
    return result


def split_into_sections(full_text: str) -> list[dict[str, Any]]:
    """Разбивает полный текст документа на разделы."""
    lines = full_text.splitlines()
    sections: list[dict[str, Any]] = []
    current_title = "Документ"
    current_number = ""
    current_lines: list[str] = []

    def flush():
        nonlocal current_title, current_number, current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append({
                "title": current_title,
                "number": current_number,
                "text": text,
            })
        current_lines = []

    for line in lines:
        stripped = line.strip()
        # Detect section heading: short line in CAPS, or numbered like "1.", "1.1."
        is_heading = False
        if 5 < len(stripped) < 120:
            if stripped.isupper() and len(stripped.split()) <= 8:
                is_heading = True
            m = SECTION_NUMBER_RE.match(stripped)
            if m and len(stripped) < 100:
                # Check if it's a section header (few words after number)
                after_num = stripped[m.end():].strip()
                if (
                    after_num
                    and len(after_num.split()) <= 12
                    and not after_num.endswith(".")
                    and (after_num[0].isupper() or after_num[0].isdigit())
                ):
                    is_heading = True
        if is_heading and len(current_lines) > 0:
            flush()
            current_title = stripped
            m2 = SECTION_NUMBER_RE.match(stripped)
            current_number = m2.group(1) if m2 else ""
        else:
            current_lines.append(line)

    flush()
    return sections


def split_section_into_chunks(
    section_text: str,
    max_chars: int = 1200,
    min_chars: int = 80,
) -> list[str]:
    """Нарезает текст раздела на чанки по пунктам/абзацам."""
    if len(section_text) <= max_chars:
        return [section_text] if len(section_text) >= min_chars else []

    # Сначала пробуем нарезку по нумерованным пунктам
    parts: list[str] = []
    last = 0
    for m in POINT_RE.finditer(section_text):
        if m.start() > last + 30:
            piece = section_text[last:m.start()].strip()
            if len(piece) >= min_chars:
                parts.append(piece)
        last = m.start()
    tail = section_text[last:].strip()
    if len(tail) >= min_chars:
        parts.append(tail)

    if not parts:
        # Нарезка по абзацам
        paras = re.split(r"\n{2,}", section_text)
        parts = [p.strip() for p in paras if len(p.strip()) >= min_chars]

    if not parts:
        parts = [section_text]

    # Если кусок > max_chars, дробим по концам предложений
    final: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
        else:
            # Split by sentence boundary
            sentences = re.split(r"(?<=[.!?])\s+", part)
            buf = ""
            for sent in sentences:
                if buf and len(buf) + len(sent) > max_chars:
                    if len(buf) >= min_chars:
                        final.append(buf.strip())
                    buf = sent
                else:
                    buf = (buf + " " + sent).strip() if buf else sent
            if len(buf) >= min_chars:
                final.append(buf.strip())

    return final


# ---------------------------------------------------------------------------
# Approval / metadata extraction from first pages
# ---------------------------------------------------------------------------

DATE_RE = re.compile(
    r"(\d{1,2})[.\s](\d{1,2})[.\s](\d{4})"
    r"|(\d{4})\s*г(?:ода)?\s*[,.]?\s*№\s*([\w\-]+)"
    r"|(\d{1,2})\s+([а-яА-Я]+)\s+(\d{4})\s*г",
    re.I,
)

MONTHS = {
    "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
    "мая": "05", "июня": "06", "июля": "07", "августа": "08",
    "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12",
}

ORDER_NO_RE = re.compile(r"№\s*([\d\-а-яА-ЯёЁ\/]+)", re.I)
TITLE_RE = re.compile(
    r"(?:КЛИНИЧЕСКИЙ\s+ПРОТОКОЛ|КП)\s+"
    r"([^\n]{10,200})",
    re.I,
)
REPLACES_RE = re.compile(
    r"признать\s+утратившим[и]?\s+силу[^.]{0,200}",
    re.I,
)


def parse_approval_date(text: str) -> str:
    for m in DATE_RE.finditer(text):
        d, mo, yr, yr2, no2, d3, mo3s, yr3 = m.groups()
        if d and mo and yr:
            try:
                return f"{yr}-{int(mo):02d}-{int(d):02d}"
            except Exception:
                pass
        if d3 and mo3s and yr3:
            mo_num = MONTHS.get(mo3s.lower(), "01")
            return f"{yr3}-{mo_num}-{int(d3):02d}"
    return ""


def extract_protocol_metadata(first_pages_text: str, file_name: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    # Title from filename heuristic
    name = file_name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\.(pdf|PDF)$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    meta["protocol_title"] = name

    # Approval date
    meta["approval_date"] = parse_approval_date(first_pages_text)

    # Approval number
    m_no = ORDER_NO_RE.search(first_pages_text)
    meta["approval_number"] = m_no.group(1) if m_no else ""

    meta["approval_body"] = "Министерство здравоохранения Республики Беларусь"

    # Replaces
    replaces: list[str] = []
    for m in REPLACES_RE.finditer(first_pages_text):
        s = m.group(0).strip()
        if len(s) < 300:
            replaces.append(s)
    meta["replaces"] = replaces

    # Related protocols (cross-refs in full text)
    meta["related_protocols"] = extract_cross_protocols(first_pages_text)

    return meta


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path, rel_path: str) -> dict[str, Any]:
    """
    Полная обработка одного PDF.
    Returns: {"chunks": [...], "meta": {...}, "errors": [...]}
    """
    import fitz  # PyMuPDF

    errors: list[str] = []
    chunks_out: list[dict[str, Any]] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        return {"chunks": [], "meta": {}, "errors": [str(e)]}

    total_pages = doc.page_count
    specialty_slug = pdf_path.parent.name
    specialty_ru_name = SPECIALTY_RU.get(specialty_slug, specialty_slug)
    did = doc_id(rel_path)
    catalog_icd = catalog_icd_for_path(rel_path)

    # --- Extract text page by page ---
    pages: list[tuple[int, str]] = []
    total_chars = 0
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        total_chars += len(text)
        pages.append((i, text))

    extraction_confidence = min(1.0, total_chars / max(total_pages * 200, 1))
    extraction_confidence = round(extraction_confidence, 2)

    # --- Extract tables via fitz find_tables ---
    table_chunks: list[dict[str, Any]] = []
    has_tables = False
    for i, page in enumerate(doc, start=1):
        try:
            tab_finder = page.find_tables()
            for tidx, tab in enumerate(tab_finder.tables):
                rows = tab.extract()
                if not rows:
                    continue
                md = table_to_markdown(rows)
                if len(md) < 20:
                    continue
                has_tables = True
                # Find caption: text just above the table bbox
                caption = ""
                tb_bbox = tab.bbox  # (x0,y0,x1,y1)
                all_text = page.get_text("blocks")
                for blk in all_text:
                    if blk[1] < tb_bbox[1] and blk[3] < tb_bbox[1] + 30:
                        cap = blk[4].strip()[:200]
                        if cap:
                            caption = cap
                # Split large tables (>15 data rows) into parts
                header_row = rows[0] if rows else []
                data_rows = rows[1:] if len(rows) > 1 else []
                BATCH = 15
                for bi in range(0, max(1, len(data_rows)), BATCH):
                    batch = data_rows[bi:bi + BATCH]
                    if header_row and batch:
                        md_part = table_to_markdown([header_row] + batch)
                    else:
                        md_part = md
                    chunk_idx = len(table_chunks)
                    chunk_id = f"{did}_tbl_p{i}_{tidx}_{bi}_{chunk_idx}"
                    icd = build_chunk_icd10_codes(
                        chunk_text=md_part,
                        chunk_type="table",
                        protocol_primary=[],
                        protocol_all=[],
                        catalog_primary=catalog_icd.get("primary") or [],
                        catalog_all=catalog_icd.get("all") or [],
                    )
                    table_chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": did,
                        "source_path": rel_path,
                        "file_name": pdf_path.name,
                        "specialty_slug": specialty_slug,
                        "specialty_ru": specialty_ru_name,
                        "chunk_type": "table",
                        "section_title": caption or "Таблица",
                        "section_path": ["Таблица"],
                        "section_number": "",
                        "page_from": i,
                        "page_to": i,
                        "text": md_part,
                        "table_caption": caption,
                        "embedding_ready_text": build_embedding_ready_text(
                            section_title=caption or "Таблица",
                            chunk_text=md_part,
                            icd_codes=icd,
                            populations=extract_populations(md_part),
                            chunk_type="table",
                        ),
                        "icd10_codes": icd,
                        "population": extract_populations(md_part),
                        "age_range": extract_age_ranges(md_part),
                        "sex": [],
                        "care_setting": extract_care_settings(md_part),
                        "conditions": [],
                        "drugs": extract_drugs_heuristic(md_part),
                        "procedures": [],
                        "lab_tests": extract_lab_tests(md_part),
                        "imaging": extract_imaging(md_part),
                        "durations": extract_durations(md_part),
                        "dosages": extract_dosages(md_part),
                        "severity": extract_severity(md_part),
                        "keywords": keywords_from_text(md_part, 20),
                        "extraction_confidence": extraction_confidence,
                        "is_preamble_filtered": False,
                        "chunk_has_table": True,
                        "chunk_is_empty": False,
                    })
        except Exception as e:
            errors.append(f"table page {i}: {e}")

    doc.close()

    # --- Filter preamble pages ---
    cleaned_pages = filter_preamble_pages(pages)
    preamble_filtered = len(cleaned_pages) < len(pages) or any(
        len(c) < len(o) for (_, c), (_, o) in zip(cleaned_pages, pages)
    )

    # Build full text with page tracking
    page_starts: list[tuple[int, int]] = []  # (offset, page_no)
    full_text_parts: list[str] = []
    offset = 0
    for page_no, text in cleaned_pages:
        page_starts.append((offset, page_no))
        full_text_parts.append(text)
        offset += len(text) + 1  # +1 for \n

    full_text = "\n".join(full_text_parts)

    # --- Extract metadata from first 3 pages ---
    first_text = "\n".join(t for _, t in cleaned_pages[:3])
    meta = extract_protocol_metadata(first_text, pdf_path.name)
    meta["doc_id"] = did
    meta["source_path"] = rel_path
    meta["file_name"] = pdf_path.name
    meta["specialty_slug"] = specialty_slug
    meta["specialty_ru"] = specialty_ru_name
    meta["has_tables"] = has_tables
    meta["has_algorithms"] = bool(re.search(r"алгоритм|схема\s+(?:лечения|диагностики)", full_text, re.I))
    meta["total_pages"] = total_pages
    meta["language"] = "ru"
    meta["extraction_confidence"] = extraction_confidence
    meta["icd10_catalog_primary"] = catalog_icd.get("primary") or []
    meta["icd10_catalog_all"] = catalog_icd.get("all") or []
    # Global ICD from all text
    meta["icd10_all"] = extract_icd10(full_text)
    meta["icd10_primary"] = extract_icd10(first_text)
    meta["icd10_protocol"] = merge_icd10_code_lists(
        meta["icd10_catalog_primary"],
        meta["icd10_primary"],
        meta["icd10_catalog_all"],
        meta["icd10_all"],
        max_codes=32,
    )
    meta["population"] = extract_populations(full_text)
    meta["related_protocols"] = meta.get("related_protocols", [])

    # --- Section splitting ---
    sections = split_into_sections(full_text)

    # --- Chunk building ---
    chunk_global_idx = 0
    for sec_idx, section in enumerate(sections):
        sec_title = section["title"]
        sec_number = section["number"]
        sec_text = section["text"]

        # Build section_path from number
        if sec_number:
            parts_num = sec_number.split(".")
            path_labels = [sec_title]
        else:
            path_labels = [sec_title]

        sub_chunks = split_section_into_chunks(sec_text)
        if not sub_chunks:
            # Still emit as single body chunk if > 40 chars
            if len(sec_text.strip()) > 40:
                sub_chunks = [sec_text.strip()]
            else:
                continue

        for ci, chunk_text in enumerate(sub_chunks):
            if not chunk_text.strip():
                continue
            chunk_id = f"{did}_s{sec_idx}_c{ci}"
            pops = extract_populations(chunk_text)
            care = extract_care_settings(chunk_text)
            ctype = guess_chunk_type(sec_title, chunk_text)
            icd = build_chunk_icd10_codes(
                chunk_text=chunk_text,
                chunk_type=ctype,
                protocol_primary=meta.get("icd10_primary") or [],
                protocol_all=meta.get("icd10_all") or [],
                catalog_primary=meta.get("icd10_catalog_primary") or [],
                catalog_all=meta.get("icd10_catalog_all") or [],
            )

            # Page mapping: find which page this chunk falls on (approx)
            # We use offset position in full_text
            chunk_offset = full_text.find(chunk_text[:60])
            page_from = cleaned_pages[0][0] if cleaned_pages else 1
            page_to = page_from
            if chunk_offset >= 0 and page_starts:
                for (ps_off, ps_pno) in reversed(page_starts):
                    if chunk_offset >= ps_off:
                        page_from = ps_pno
                        page_to = ps_pno
                        break

            emb_text = build_embedding_ready_text(
                section_title=sec_title,
                chunk_text=chunk_text,
                icd_codes=icd,
                populations=pops,
                chunk_type=ctype,
            )

            chunks_out.append({
                "chunk_id": chunk_id,
                "doc_id": did,
                "source_path": rel_path,
                "file_name": pdf_path.name,
                "specialty_slug": specialty_slug,
                "specialty_ru": specialty_ru_name,
                "chunk_type": ctype,
                "section_title": sec_title,
                "section_path": path_labels,
                "section_number": sec_number,
                "page_from": page_from,
                "page_to": page_to,
                "text": chunk_text,
                "embedding_ready_text": emb_text,
                # --- Protocol-level metadata (repeated per chunk) ---
                "protocol_title": meta["protocol_title"],
                "approval_date": meta["approval_date"],
                "approval_number": meta["approval_number"],
                "approval_body": meta["approval_body"],
                "replaces": meta["replaces"],
                "related_protocols": meta["related_protocols"],
                "has_tables": has_tables,
                "has_algorithms": meta["has_algorithms"],
                "total_pages": total_pages,
                "language": "ru",
                # --- Entities ---
                "icd10_codes": icd,
                "population": pops,
                "age_range": extract_age_ranges(chunk_text),
                "sex": [],
                "care_setting": care,
                "conditions": extract_cross_protocols(chunk_text),
                "drugs": extract_drugs_heuristic(chunk_text),
                "procedures": [],
                "lab_tests": extract_lab_tests(chunk_text),
                "imaging": extract_imaging(chunk_text),
                "durations": extract_durations(chunk_text),
                "dosages": extract_dosages(chunk_text),
                "severity": extract_severity(chunk_text),
                "keywords": keywords_from_text(chunk_text),
                # --- Quality flags ---
                "extraction_confidence": extraction_confidence,
                "is_preamble_filtered": preamble_filtered,
                "chunk_has_table": False,
                "chunk_is_empty": False,
            })
            chunk_global_idx += 1

    # Merge table chunks (re-enrich ICD after protocol-level meta is known)
    for ch in table_chunks:
        md_part = str(ch.get("text") or "")
        caption = str(ch.get("table_caption") or "Таблица")
        icd = build_chunk_icd10_codes(
            chunk_text=md_part,
            chunk_type="table",
            protocol_primary=meta.get("icd10_primary") or [],
            protocol_all=meta.get("icd10_all") or [],
            catalog_primary=meta.get("icd10_catalog_primary") or [],
            catalog_all=meta.get("icd10_catalog_all") or [],
        )
        ch["icd10_codes"] = icd
        ch["embedding_ready_text"] = build_embedding_ready_text(
            section_title=caption,
            chunk_text=md_part,
            icd_codes=icd,
            populations=extract_populations(md_part),
            chunk_type="table",
        )
    chunks_out.extend(table_chunks)

    return {"chunks": chunks_out, "meta": meta, "errors": errors}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build rich chunks from PDF protocols")
    parser.add_argument("--rubric", help="Process only this rubric (folder name)")
    parser.add_argument("--resume", action="store_true", help="Skip already processed files")
    parser.add_argument("--file", help="Process a single PDF file (relative to project root)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest for resume
    processed: set[str] = set()
    manifest: dict[str, Any] = {
        "total_pdfs": 0,
        "total_chunks": 0,
        "total_tables": 0,
        "total_errors": 0,
        "files": {},
    }
    if args.resume and MANIFEST_FILE.exists():
        try:
            manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            processed = set(manifest.get("files", {}).keys())
            print(f"Resume: skipping {len(processed)} already processed files")
        except Exception:
            pass

    # Collect PDFs
    if args.file:
        pdfs = [ROOT / args.file]
    elif args.rubric:
        pdfs = sorted((PDF_ROOT / args.rubric).rglob("*.pdf"))
    else:
        pdfs = sorted(PDF_ROOT.rglob("*.pdf"))

    # If starting fresh, truncate output
    if not args.resume:
        CHUNKS_FILE.write_text("", encoding="utf-8")
        ERRORS_FILE.write_text("", encoding="utf-8")

    chunks_fh = open(CHUNKS_FILE, "a", encoding="utf-8")
    errors_fh = open(ERRORS_FILE, "a", encoding="utf-8")

    t0 = time.time()
    n_chunks = 0
    n_tables = 0
    n_errors = 0

    for pi, pdf_path in enumerate(pdfs, start=1):
        rel_path = str(pdf_path.relative_to(ROOT))

        if rel_path in processed:
            continue

        try:
            result = process_pdf(pdf_path, rel_path)
        except Exception as e:
            result = {"chunks": [], "meta": {}, "errors": [f"FATAL: {e}"]}

        # Write chunks
        for ch in result["chunks"]:
            chunks_fh.write(json.dumps(ch, ensure_ascii=False) + "\n")

        # Write meta
        if result["meta"]:
            did_val = result["meta"].get("doc_id", doc_id(rel_path))
            (META_DIR / f"{did_val}.json").write_text(
                json.dumps(result["meta"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Log errors
        for err in result.get("errors", []):
            errors_fh.write(json.dumps({"path": rel_path, "error": err}, ensure_ascii=False) + "\n")
            n_errors += 1

        nc = len(result["chunks"])
        nt = sum(1 for c in result["chunks"] if c.get("chunk_type") == "table")
        n_chunks += nc
        n_tables += nt

        # Update manifest
        manifest["files"][rel_path] = {
            "chunks": nc,
            "tables": nt,
            "errors": len(result.get("errors", [])),
        }
        manifest["total_pdfs"] = pi
        manifest["total_chunks"] = manifest.get("total_chunks", 0) + nc
        manifest["total_tables"] = manifest.get("total_tables", 0) + nt
        manifest["total_errors"] = manifest.get("total_errors", 0) + len(result.get("errors", []))

        # Progress
        elapsed = time.time() - t0
        print(
            f"\r[{pi:4d}/{len(pdfs)}] {pdf_path.name[:55]:<56} "
            f"chunks={nc:4d} tbl={nt:3d} | total={n_chunks:6d} t={elapsed:.0f}s",
            end="",
            flush=True,
        )

    chunks_fh.close()
    errors_fh.close()

    # Save manifest
    manifest["total_pdfs"] = len(pdfs)
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n\nГотово за {elapsed:.1f}с")
    print(f"  PDFs      : {len(pdfs)}")
    print(f"  Чанков    : {n_chunks}")
    print(f"  Таблиц    : {n_tables}")
    print(f"  Ошибок    : {n_errors}")
    print(f"  Выход     : {CHUNKS_FILE}")
    if n_errors:
        print(f"  Ошибки    : {ERRORS_FILE}")


if __name__ == "__main__":
    main()
