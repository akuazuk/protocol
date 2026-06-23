"""Prompt templates for protocol summary LLM extraction."""
from __future__ import annotations

SYSTEM_SKELETON = """Ты медицинский редактор клинических протоколов Минздрава РБ.
Извлеки структуру протокола для навигации врача. Не выдумывай факты вне текста.

Верни ОДИН JSON без markdown:
{
  "title_ru": "нормализованное название без пост. МЗ",
  "population": ["adult"|"child"|"pregnant"|"adult_and_child"],
  "conditions": [
    {
      "condition_id": "латиница_snake",
      "name": "название нозологии",
      "icd10_codes": ["J02.9"],
      "section_pages": {"classification": 1, "diagnostics": 5, "treatment": 12}
    }
  ]
}

Правила:
- icd10_codes только из текста раздела классификации/диагноза или из подсказки каталога если есть в тексте
- не используй коды Y/T/X как основной диагноз
- если одна нозология - один condition
- condition_id: латиница, snake_case, до 40 символов"""


def prompt_skeleton(doc: dict, catalog_icd: list[str]) -> str:
    from .source_text import section_text_blob

    blob = section_text_blob(doc, ["classification", "criteria", "other"], max_chars=10000)
    icd_hint = ", ".join(catalog_icd[:12]) if catalog_icd else "нет"
    return (
        f"Протокол: {doc.get('title')}\n"
        f"Аудитория (каталог): {doc.get('audience')}\n"
        f"МКБ primary (каталог, проверь по тексту): {icd_hint}\n\n"
        f"Текст протокола:\n{blob}\n\n"
        "Верни JSON со списком conditions."
    )


SYSTEM_CONDITION_BLOCK = """Ты медицинский редактор. Извлеки блок протокола в едином стиле для врача.
Только факты из текста. Каждый пункт с цитатой quote (до 200 символов) и page_start.

Верни JSON:
{
  "required_exams": [{"name": "...", "level": "required|conditional", "quote": "...", "page_start": 1}],
  "diagnostic_criteria": ["краткая фраза"],
  "treatment_non_drug": ["..."],
  "drugs": [{"name": "...", "dose_text": "...", "quote": "...", "page_start": 1}],
  "red_flags": [{"text": "...", "severity": "high|medium", "quote": "...", "page_start": 1}],
  "follow_up": ["..."],
  "routing": ["госпитализация/амбулаторно"]
}"""


def prompt_condition_block(
    doc: dict,
    condition: dict,
    block: str,
) -> str:
    from .source_text import section_text_blob

    keys = {
        "diagnostics": ["diagnostics", "criteria"],
        "treatment": ["treatment", "other"],
        "classification": ["classification"],
        "routing": ["routing", "prevention"],
    }.get(block, ["other"])
    blob = section_text_blob(doc, keys, max_chars=11000)
    return (
        f"Протокол: {doc.get('title')}\n"
        f"Нозология: {condition.get('name')}\n"
        f"МКБ: {', '.join(condition.get('icd10_codes') or [])}\n"
        f"Блок: {block}\n\n"
        f"Текст:\n{blob}"
    )
