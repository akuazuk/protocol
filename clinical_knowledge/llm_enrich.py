"""Опциональное LLM-обогащение condition-блоков (кэш на диск, temperature=0)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ENRICH_DIR = Path(__file__).resolve().parent.parent / "data" / "gastro_mvp" / "enrichment"

SYSTEM_CONDITION_ENRICH = """Ты медицинский редактор. По фрагменту клинического протокола Минздрава РБ извлеки структуру для проверки консультативных заключений.

Верни ОДИН JSON без markdown:
{"condition": "<название>", "icd10": ["..."], "diagnosis_required_components": ["..."], "diagnostic_criteria_summary": "<1-3 предложения>", "required_exams": ["..."], "red_flags": ["..."]}

Не выдумывай факты вне текста. Если данных нет — пустые массивы."""


def _cache_path(condition_id: str, text_hash: str) -> Path:
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)
    return ENRICH_DIR / f"{condition_id}_{text_hash}.json"


def enrich_condition_text(
    condition_id: str,
    text_sample: str,
    *,
    model=None,
    generate_fn=None,
) -> dict[str, Any] | None:
    """LLM-обогащение с кэшем; model+generate_fn — как в rag_server.generate_gemini."""
    if not env_bool("CORPUS_LLM_ENRICH", False):
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
    from rag_server import _extract_gemini_text, _try_parse_json  # noqa: WPS433

    parsed = _try_parse_json(_extract_gemini_text(resp))
    if not isinstance(parsed, dict):
        return None
    payload = {"condition_id": condition_id, "enrichment": parsed, "text_hash": text_hash}
    cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")
