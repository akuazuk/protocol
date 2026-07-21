"""Кэш ИИ-обзоров разделов протокола (P5).

ИИ-обзор («суть раздела» одним абзацем) генерируется офлайн (Gemini) и кладётся в
`data/ml/section_overviews/{protocol_id}.json`. Навигатор читает кэш и показывает обзор
без обращения к LLM в рантайме. Генерация - скриптом `scripts/precompute_section_overviews.py`
(запуск из поддерживаемого региона; локально из РБ Gemini geo-blocked).

Формат файла:
{
  "protocol_id": "...",
  "sections": { "treatment": {"summary": "...", "points": [{"text": "...", "page": 16}]}, ... }
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "ml" / "section_overviews"


def _safe_id(protocol_id: str) -> str:
    return "".join(c for c in (protocol_id or "") if c.isalnum() or c in "._-") or "unknown"


def overview_file(protocol_id: str) -> Path:
    return CACHE_DIR / f"{_safe_id(protocol_id)}.json"


def load_section_overviews(protocol_id: str) -> dict[str, Any]:
    """Вернуть {section_id: {summary, points}} или {} если кэша нет."""
    p = overview_file(protocol_id)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    sections = data.get("sections")
    return sections if isinstance(sections, dict) else {}


def save_section_overviews(protocol_id: str, sections: dict[str, Any]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = overview_file(protocol_id)
    p.write_text(
        json.dumps({"protocol_id": protocol_id, "sections": sections}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p
