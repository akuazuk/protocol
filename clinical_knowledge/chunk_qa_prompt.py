"""Промпты для LLM-QA rich-чанков (offline)."""
from __future__ import annotations

import json
from typing import Any

SYSTEM_CHUNK_QA = """Ты медицинский редактор клинических протоколов Минздрава РБ.
Проверь фрагменты (чанки) протокола: тип, заголовок, шум, сущности.

Правила:
- Не добавляй клинические факты, которых нет в исходном text.
- verdict=drop для юридической шапки, подписей, «утверждено», «согласовано» без клиники.
- verdict=merge_with_next если чанк - обрывок пункта (<80 символов) без самостоятельного смысла.
- clean_text - только удаление шума и склейка переносов; смысл сохранить.
- confidence 0.0-1.0.

Верни JSON-массив объектов (по одному на chunk_id):
[{"chunk_id":"...","verdict":"ok|fix|drop|merge_with_next","corrected_chunk_type":null,
  "corrected_section_title":null,"clean_text":null,"obligation":null,
  "entities":{"exam":[],"drug":[],"condition":[]},"noise_reasons":[],"confidence":0.0,"notes":""}]
Без markdown."""

SYSTEM_PROTOCOL_SECTIONS = """Ты медицинский редактор. По оглавлению/заголовкам протокола
сопоставь каждому разделу chunk_type из списка:
diagnostics, treatment, prevention, rehabilitation, dispensary, classification,
routing, pharmacotherapy, algorithm, criteria_block, drug_list, terms, appendix, body.

Верни JSON:
{"doc_id":"...","sections":[{"section_number":"1","section_title":"...","chunk_type":"diagnostics","page_from":1}],
 "confidence":0.0,"notes":""}
Без markdown."""


def build_chunk_qa_prompt(
    chunks: list[dict[str, Any]],
    *,
    protocol_title: str = "",
) -> str:
    rows = []
    for ch in chunks:
        rows.append({
            "chunk_id": ch.get("chunk_id"),
            "chunk_type": ch.get("chunk_type"),
            "section_title": ch.get("section_title"),
            "page_from": ch.get("page_from"),
            "text": (ch.get("text") or "")[:2000],
        })
    return (
        f"Протокол: {protocol_title}\n\n"
        f"Чанки для проверки:\n{json.dumps(rows, ensure_ascii=False, indent=2)}"
    )


def build_protocol_sections_prompt(
    *,
    doc_id: str,
    protocol_title: str,
    section_outline: list[dict[str, Any]],
) -> str:
    return (
        f"doc_id: {doc_id}\nПротокол: {protocol_title}\n\n"
        f"Разделы:\n{json.dumps(section_outline, ensure_ascii=False, indent=2)}"
    )
