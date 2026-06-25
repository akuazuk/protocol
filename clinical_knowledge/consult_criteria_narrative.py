"""Опциональное LLM-перефразирование comment_ru в criteria (без изменения фактов)."""
from __future__ import annotations

import json
import os
import re
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _build_prompt(criteria: list[dict[str, Any]]) -> str:
    payload = []
    for i, c in enumerate(criteria[:7]):
        payload.append({
            "index": i,
            "block_id": c.get("block_id"),
            "name_ru": c.get("name_ru"),
            "comment_ru": c.get("comment_ru"),
            "findings_ru": (c.get("findings_ru") or [])[:6],
            "gaps_ru": (c.get("gaps_ru") or [])[:6],
            "score_pct": c.get("score_pct"),
        })
    return (
        "Перефразируй поле comment_ru для каждого критерия проверки клинического заключения.\n"
        "Правила:\n"
        "- Не добавляй новых фактов, диагнозов, препаратов или обследований.\n"
        "- Используй только findings_ru и gaps_ru.\n"
        "- 1–2 предложения, деловой стиль, русский язык.\n"
        "- Если данных мало — напиши «Недостаточно данных для вывода.»\n"
        "Ответ JSON: {\"comments\": [{\"index\": 0, \"comment_ru\": \"...\"}, ...]}\n\n"
        f"Критерии:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_response(text: str) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return []
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    rows = data.get("comments") or data.get("criteria") or []
    return rows if isinstance(rows, list) else []


def enrich_criteria_comments_llm(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Перефразировать comment_ru; при ошибке вернуть исходные criteria."""
    if not criteria:
        return criteria
    try:
        import rag_server as rs
        from rag_server import _extract_gemini_text, generate_gemini_consult_review_synthesize

        model = rs.get_gemini()
        if model is None:
            return criteria
        prompt = _build_prompt(criteria)
        timeout = int(os.environ.get("CONSULT_CRITERIA_NARRATIVE_TIMEOUT", "12"))
        resp = generate_gemini_consult_review_synthesize(
            model, prompt, max_out=1200,
        )
        txt = _extract_gemini_text(resp)
        updates = _parse_response(txt)
        by_idx = {int(u.get("index", -1)): str(u.get("comment_ru") or "").strip() for u in updates}
        out = []
        for i, c in enumerate(criteria):
            row = dict(c)
            new_comment = by_idx.get(i, "").strip()
            if new_comment and len(new_comment) >= 12:
                row["comment_ru"] = new_comment
                row["comment_narrative_llm"] = True
            out.append(row)
        return out
    except Exception:
        return criteria
