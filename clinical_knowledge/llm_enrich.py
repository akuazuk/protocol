"""Опциональное LLM-обогащение condition-блоков (кэш на диск, temperature=0)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

ENRICH_DIR = Path(__file__).resolve().parent.parent / "data" / "gastro_mvp" / "enrichment"

SYSTEM_CONDITION_ENRICH = """Ты медицинский редактор. По фрагменту клинического протокола Минздрава РБ извлеки структуру для проверки консультативных заключений.

Верни ОДИН JSON без markdown:
{"condition": "<название>", "icd10": ["..."], "diagnosis_required_components": ["..."], "diagnostic_criteria_summary": "<1-3 предложения>", "required_exams": ["..."], "red_flags": ["..."]}

Не выдумывай факты вне текста. Если данных нет — пустые массивы."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_json_loose(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _cache_path(condition_id: str, text_hash: str) -> Path:
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)
    return ENRICH_DIR / f"{condition_id}_{text_hash}.json"


def enrich_condition_text(
    condition_id: str,
    text_sample: str,
    *,
    model: Any = None,
    generate_fn: Callable[[Any, str], Any] | None = None,
    extract_text_fn: Callable[[Any], str] | None = None,
) -> dict[str, Any] | None:
    """LLM-обогащение с кэшем. Включение: CORPUS_LLM_ENRICH=1 + API key."""
    if not _env_bool("CORPUS_LLM_ENRICH", False):
        return None
    sample = (text_sample or "").strip()[:12_000]
    if len(sample) < 200:
        return None
    text_hash = hashlib.sha256(sample.encode()).hexdigest()[:16]
    cache = _cache_path(condition_id, text_hash)
    if cache.is_file():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    if model is None or generate_fn is None:
        return None
    prompt = (
        SYSTEM_CONDITION_ENRICH
        + f"\n\ncondition_id: {condition_id}\n\n--- ТЕКСТ ---\n\n"
        + sample
    )
    try:
        resp = generate_fn(model, prompt)
    except Exception:
        return None
    raw = extract_text_fn(resp) if extract_text_fn else str(resp)
    parsed = _parse_json_loose(raw)
    if not isinstance(parsed, dict):
        return None
    payload = {"condition_id": condition_id, "enrichment": parsed, "text_hash": text_hash}
    cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
