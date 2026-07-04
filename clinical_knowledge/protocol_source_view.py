"""Компактное клиническое представление протокола для UI-навигатора.

Источник - блоки source_text (или rich-чанки), у каждого есть настоящий
`chunk_type` от экстрактора и, для rich-чанков, списки сущностей
(drugs / imaging / lab_tests / procedures). Здесь мы:

1. группируем по клиническому типу, а не по эвристике заголовка;
2. склеиваем разорванный по строкам PDF-текст обратно в связный абзац;
3. отбрасываем административный/нормативный/глоссарный шум;
4. формируем короткий человекочитаемый lead + развёрнутый body;
5. добавляем чипы сущностей (препараты, визуализация, анализы, процедуры).
"""
from __future__ import annotations

import re
from typing import Any

# Клинический тип чанка -> группа навигатора.
_TYPE_TO_GROUP: dict[str, str] = {
    "classification": "diagnosis",
    "criteria_block": "diagnosis",
    "diagnostics": "diagnostics",
    "treatment": "treatment",
    "drug_list": "treatment",
    "pharmacotherapy": "treatment",
    "prevention": "followup",
    "dispensary": "followup",
    "rehabilitation": "followup",
    "routing": "followup",
}

# Типы, которые не несут клинической навигации.
_DROP_TYPES = frozenset(
    {"body", "terms", "definitions", "protocol_overview", "other", "table"}
)

_GROUP_ORDER: tuple[tuple[str, str], ...] = (
    ("diagnosis", "Диагноз и классификация"),
    ("diagnostics", "Диагностика и обследования"),
    ("treatment", "Лечение и препараты"),
    ("followup", "Наблюдение, профилактика, маршрут"),
)

# Только чистые номенклатурные поля: drugs в rich-чанках сильно зашумлён фразами.
_ENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("imaging", "Визуализация"),
    ("lab_tests", "Анализы"),
    ("procedures", "Процедуры"),
)

# Теги намерения для смыслового поиска внутри протокола.
_TYPE_INTENT_TAGS: dict[str, tuple[str, ...]] = {
    "classification": ("diagnosis",),
    "criteria_block": ("diagnosis", "criteria"),
    "diagnostics": ("diagnostics", "exams"),
    "treatment": ("treatment", "drugs"),
    "drug_list": ("treatment", "drugs"),
    "pharmacotherapy": ("treatment", "drugs"),
    "prevention": ("followup", "prevention"),
    "dispensary": ("followup",),
    "rehabilitation": ("followup",),
    "routing": ("followup", "routing"),
}

# Синонимы типа - попадают в search_blob, чтобы «какие лекарства» находило treatment.
_TYPE_SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "classification": ("диагноз", "классификация", "критерии", "мкб"),
    "criteria_block": ("критерии", "показания", "противопоказания", "диагноз"),
    "diagnostics": (
        "обследование",
        "диагностика",
        "анализы",
        "лабораторные",
        "инструментальные",
        "узи",
        "уздс",
    ),
    "treatment": (
        "лечение",
        "препараты",
        "лекарства",
        "назначение",
        "терапия",
        "дозировка",
        "фармакотерапия",
    ),
    "drug_list": ("лекарства", "препараты", "фармакотерапия", "дозировка", "назначение"),
    "pharmacotherapy": ("фармакотерапия", "препараты", "лекарства", "дозировка"),
    "prevention": ("профилактика", "наблюдение"),
    "dispensary": ("диспансерное", "наблюдение", "контроль"),
    "rehabilitation": ("реабилитация",),
    "routing": ("маршрут", "направление", "консультация", "госпитализация"),
}

# Обрывок середины фразы: начинается со строчной буквы или с «сироты» «АЧТВ)» / «ЛПИ) -».
_MIDSENTENCE_LEAD = re.compile(r"^(?:[а-яё]|[А-ЯЁA-Z]{2,}\s*[)\]])")
_TRAILING_COLON_DOT = re.compile(r"\s*:\s*\.\s*$")

_LEADING_NUM = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s*")
_CHAPTER_PREFIX = re.compile(r"^\s*ГЛАВА\s+\d+\s*", re.I)
# Обрывок ALL-CAPS заголовка перед нормальным предложением: «И ЕГО ЭТАПЫ При плановой...»
_CAPS_RUN_PREFIX = re.compile(r"^[А-ЯЁA-Z\s,()«»\d.-]{14,}?\s(?=[А-ЯЁ][а-яё])")
_MARKUP_TAG = re.compile(r"^\s*\[[a-z_]+\]\s*", re.I)
_TABLE_MARK = re.compile(r"\|\s*-{2,}\s*\|")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"[ \t]+")

# Хвост колонтитула НПА посреди/в конце блока.
_PORTAL_TAIL = re.compile(
    r"\s*Национальный\s+правовой\s+Интернет-портал[^\n]*(?:\d+\s*)?$",
    re.I,
)
# Блоки-обёртки, не несущие клиники (глоссарий, шапка, преамбула).
_WRAPPER_HEAD = re.compile(
    r"^(?:термины\s+и\s+определения|описание\s+протокола|клинический\s+протокол\b|"
    r"настоящее\s+постановление|настоящий\s+клинический\s+протокол\s+устанавливает|"
    r"для\s+целей\s+настоящего\s+клинического\s+протокола|"
    r"признать\s+утратившим|№\s*\d+\s)",
    re.I,
)
# Юридический шаблон, повторяющийся в каждом КП.
_LEGAL_BOILERPLATE = re.compile(
    r"в\s+соответствии\s+с\s+настоящим\s+клиническим\s+протоколом\s+с\s+учетом\s+всех\s+"
    r"индивидуальных|среди\s+рекомендованных\s+к\s+применению\s+в\s+настоящем\s+клиническом\s+"
    r"протоколе\s+указаны|клинический\s+протокол\s+[«\"]",
    re.I,
)
# Организационно-маршрутный/процедурный текст (кто выполняет, куда направляют).
_ORG_LINE = re.compile(
    r"направля\w+\s+пациент|на\s+консультаци\w+\s+к\s+врач|порядок\s+направлени|"
    r"определяется\s+министерством|выполня\w+\s+врач\w*[-\s]|"
    r"определяются?\s+инструкцией\s+по\s+медицинскому\s+применению|"
    r"осуществляется\s+в\s+организациях\s+здравоохранения",
    re.I,
)


def _join_pdf_lines(text: str) -> str:
    """Склеивает разорванный по строкам PDF-текст в связные абзацы."""
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # дефисный перенос слова: "заболе-\nвания" -> "заболевания"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # двойной перевод строки = граница абзаца (маркер), одиночный = пробел
    t = t.replace("\n\n", "\u0001")
    t = t.replace("\n", " ")
    t = t.replace("\u0001", "\n")
    t = _WS.sub(" ", t)
    t = _PORTAL_TAIL.sub("", t)
    return t.strip()


def _clean_head(text: str) -> str:
    t = _MARKUP_TAG.sub("", text or "")
    t = _LEADING_NUM.sub("", t)
    t = _CHAPTER_PREFIX.sub("", t)
    t = _CAPS_RUN_PREFIX.sub("", t)
    return t.strip(" \u00b7\u2014\u2013-")


def _combine_title_text(title: str | None, text: str) -> str:
    """Заголовок = начало предложения, text = продолжение. Склеиваем в абзац."""
    body = _join_pdf_lines(text)
    head = _clean_head(_join_pdf_lines(title or ""))
    if not head:
        return body
    # заголовок оканчивается на слове (обрезанное предложение) -> сшиваем
    if head[-1:] not in ".!?:;" and body:
        first = body[0]
        if first.islower() or first.isalpha():
            return _clean_head(f"{head} {body}")
    if body:
        return _clean_head(f"{head}. {body}")
    return head


def _fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())[:180]


def _is_noise(text: str, chunk_type: str) -> bool:
    t = (text or "").strip()
    if len(t) < 28:
        return True
    if _TABLE_MARK.search(t) or t.count("|") >= 6:
        return True
    if _WRAPPER_HEAD.match(t):
        return True
    if _ORG_LINE.search(t):
        return True
    if _LEGAL_BOILERPLATE.search(t[:260]):
        return True
    try:
        from clinical_knowledge.chunk_tags import is_administrative_text
        from clinical_knowledge.consult_evidence_quality import is_reference_noise

        if is_administrative_text(t):
            return True
        if is_reference_noise(t):
            if chunk_type not in _TYPE_TO_GROUP or len(t) < 60:
                return True
    except Exception:
        pass
    alpha = sum(1 for c in t if c.isalpha())
    return alpha < 20


def _title_case_lead(lead: str) -> str:
    """ALL-CAPS заголовок-фраза PDF -> нормальный регистр.

    Не трогаем строки с цифрами/скобками (аббревиатуры и коды вида «ВБНК (С1-С4)»).
    """
    if any(ch.isdigit() for ch in lead) or "(" in lead or ")" in lead:
        return lead
    letters = [c for c in lead if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(lead.split()) >= 2:
        return lead[0] + lead[1:].lower()
    return lead


def _split_lead_body(text: str, *, lead_max: int = 160, body_max: int = 640) -> tuple[str, str | None]:
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        clip = text[:lead_max].rstrip()
        return (_title_case_lead(clip + ("…" if len(text) > lead_max else "")), None)
    lead = sentences[0]
    # заголовок-двоеточие («...являются:») - подтянуть следующее предложение как перечень
    idx = 1
    while lead.rstrip().endswith(":") and idx < len(sentences) and len(lead) < lead_max:
        lead = f"{lead} {sentences[idx]}"
        idx += 1
    lead = _title_case_lead(lead)
    if len(lead) > lead_max:
        return (lead[: lead_max - 1].rstrip() + "…", _clip(text, body_max))
    rest = " ".join(sentences[idx:]).strip()
    if not rest:
        return (lead, None)
    return (lead, _clip(rest, body_max))


def _clip(text: str, limit: int) -> str:
    t = text.strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def _entity_chips(block: dict[str, Any]) -> list[dict[str, Any]]:
    chips: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field, label in _ENTITY_FIELDS:
        vals = block.get(field) or []
        if not isinstance(vals, list):
            continue
        clean: list[str] = []
        for v in vals:
            s = re.sub(r"\s+", " ", str(v or "")).strip(" .,;·—-")
            key = s.lower()
            if len(s) < 3 or len(s) > 70 or key in seen:
                continue
            # мусорные «сущности» вида «пациентов с ...»
            if s.lower().startswith(("пациент", "население", "лечени", "диагностик")):
                continue
            seen.add(key)
            clean.append(s)
            if len(clean) >= 6:
                break
        if clean:
            chips.append({"label": label, "items": clean})
    return chips


def _entity_search_terms(block: dict[str, Any], entities: list[dict[str, Any]]) -> list[str]:
    """Термины для поиска: чипы + сырые сущности (в т.ч. drugs, даже если чип скрыт)."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        s = re.sub(r"\s+", " ", str(raw or "")).strip(" .,;·—-")
        key = s.lower()
        if len(s) < 3 or len(s) > 80 or key in seen:
            return
        if key.startswith(("пациент", "население")):
            return
        seen.add(key)
        terms.append(s)

    for chip in entities:
        add(chip.get("label"))
        for item in chip.get("items") or []:
            add(item)
    for field in ("drugs", "imaging", "lab_tests", "procedures", "dosages"):
        vals = block.get(field) or []
        if isinstance(vals, list):
            for v in vals[:10]:
                add(v)
    return terms


def _item_search_fields(
    *,
    lead: str,
    body: str | None,
    chunk_type: str,
    block: dict[str, Any],
    entities: list[dict[str, Any]],
) -> dict[str, Any]:
    intent_tags = list(_TYPE_INTENT_TAGS.get(chunk_type, ()))
    parts: list[str] = [lead, body or ""]
    parts.extend(_TYPE_SEARCH_TERMS.get(chunk_type, ()))
    entity_terms = _entity_search_terms(block, entities)
    parts.extend(entity_terms)
    blob = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip().lower()
    return {
        "intent_tags": intent_tags,
        "search_blob": blob[:1200],
        "entity_terms": entity_terms[:12],
    }


def _iter_blocks(doc: dict[str, Any]) -> list[dict[str, Any]]:
    sections = doc.get("sections") or {}
    out: list[dict[str, Any]] = []
    for blocks in sections.values():
        if isinstance(blocks, list):
            out.extend(b for b in blocks if isinstance(b, dict))
    return out


def format_rich_chunk_nav_item(block: dict[str, Any]) -> dict[str, Any] | None:
    """Один rich-чанк -> элемент навигатора (или None если шум)."""
    ctype = str(block.get("chunk_type") or "").strip().lower()
    if ctype in _DROP_TYPES or ctype not in _TYPE_TO_GROUP:
        return None
    group_id = _TYPE_TO_GROUP[ctype]
    combined = _combine_title_text(block.get("section_title"), str(block.get("text") or ""))
    if not combined or _is_noise(combined, ctype):
        return None
    lead, body = _split_lead_body(combined)
    lead = _TRAILING_COLON_DOT.sub(":", lead).strip()
    if len(lead) < 16 or _MIDSENTENCE_LEAD.match(lead):
        return None
    page = block.get("page_from") or block.get("page")
    entities = _entity_chips(block)
    search_fields = _item_search_fields(
        lead=lead,
        body=body,
        chunk_type=ctype,
        block=block,
        entities=entities,
    )
    return {
        "id": f"{group_id}-{ctype}-{page or 0}",
        "section_id": group_id,
        "lead": lead,
        "body": body,
        "page": page,
        "chunk_type": ctype,
        "entities": entities,
        "intent_tags": search_fields["intent_tags"],
        "search_blob": search_fields["search_blob"],
        "entity_terms": search_fields["entity_terms"],
        "chunk_index": block.get("chunk_index"),
        "global_index": block.get("_global_index"),
    }


def build_view_from_items(
    items: list[dict[str, Any]],
    *,
    max_per_group: int = 16,
) -> dict[str, Any]:
    """Собрать view (toc/sections) из уже отранжированных элементов."""
    groups: dict[str, list[dict[str, Any]]] = {gid: [] for gid, _ in _GROUP_ORDER}
    for item in items:
        gid = str(item.get("section_id") or "").strip()
        if gid not in groups:
            continue
        if len(groups[gid]) >= max_per_group:
            continue
        groups[gid].append(item)
    toc: list[dict[str, Any]] = []
    view_sections: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for gid, label in _GROUP_ORDER:
        labels[gid] = label
        sec_items = groups[gid]
        if sec_items:
            view_sections[gid] = sec_items
            toc.append({"id": gid, "label": label, "count": len(sec_items)})
    return {
        "toc": toc,
        "sections": view_sections,
        "section_labels": labels,
        "stats": {"shown_blocks": sum(len(v) for v in view_sections.values())},
    }


def prepare_protocol_source_view(doc: dict[str, Any], *, max_per_group: int = 12) -> dict[str, Any]:
    """Строит компактную клиническую навигацию из блоков source_text/rich."""
    groups: dict[str, list[dict[str, Any]]] = {gid: [] for gid, _ in _GROUP_ORDER}
    seen: set[str] = set()
    raw_blocks = 0
    filtered = 0

    for block in _iter_blocks(doc):
        raw_blocks += 1
        ctype = str(block.get("chunk_type") or "").strip().lower()
        if ctype in _DROP_TYPES or ctype not in _TYPE_TO_GROUP:
            filtered += 1
            continue
        group_id = _TYPE_TO_GROUP[ctype]
        combined = _combine_title_text(block.get("section_title"), str(block.get("text") or ""))
        if not combined or _is_noise(combined, ctype):
            filtered += 1
            continue
        fp = _fingerprint(combined)
        if fp in seen:
            filtered += 1
            continue
        seen.add(fp)
        lead, body = _split_lead_body(combined)
        lead = _TRAILING_COLON_DOT.sub(":", lead).strip()
        if len(lead) < 16 or _MIDSENTENCE_LEAD.match(lead):
            filtered += 1
            continue
        page = block.get("page_from") or block.get("page")
        entities = _entity_chips(block)
        search_fields = _item_search_fields(
            lead=lead,
            body=body,
            chunk_type=ctype,
            block=block,
            entities=entities,
        )
        items = groups[group_id]
        items.append(
            {
                "id": f"{group_id}-{len(items)}",
                "lead": lead,
                "body": body,
                "page": page,
                "chunk_type": ctype,
                "entities": entities,
                "intent_tags": search_fields["intent_tags"],
                "search_blob": search_fields["search_blob"],
                "entity_terms": search_fields["entity_terms"],
            }
        )

    for gid in groups:
        groups[gid] = groups[gid][:max_per_group]

    toc: list[dict[str, Any]] = []
    view_sections: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for gid, label in _GROUP_ORDER:
        labels[gid] = label
        items = groups[gid]
        if items:
            view_sections[gid] = items
            toc.append({"id": gid, "label": label, "count": len(items)})

    shown = sum(len(v) for v in view_sections.values())
    return {
        "toc": toc,
        "sections": view_sections,
        "section_labels": labels,
        "stats": {
            "raw_blocks": raw_blocks,
            "shown_blocks": shown,
            "filtered_blocks": filtered,
        },
    }
