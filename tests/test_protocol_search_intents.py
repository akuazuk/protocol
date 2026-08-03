"""Intent-спеки навигации по протоколу и синхронизация с UI."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.protocol_search_intents import (
    INTENT_SPECS,
    detect_query_intents,
    is_drug_focus_query,
    is_table_noise_text,
)
from clinical_knowledge.protocol_source_view import format_rich_chunk_nav_item


ROOT = Path(__file__).resolve().parents[1]


def test_detect_tabletki_intent():
    intents = detect_query_intents("таблетки")
    assert "treatment" in intents
    assert is_drug_focus_query("таблетки", intents)


def test_table_noise_rejected():
    noisy = (
        "МКБ-10) Объемы оказания медицинской помощи Диагностика Лечение "
        "обязательная кратность 1 2 3 4 5 6 7"
    )
    assert is_table_noise_text(noisy)


def test_format_nav_item_query_aware_drugs():
    block = {
        "chunk_type": "treatment",
        "section_title": "Диагностика и лечение острого бронхита",
        "text": (
            "МКБ-10) Объемы оказания медицинской помощи Диагностика Лечение 1 2 3 4 5 6 7. "
            "Муколитические средства: ацетилцистеин внутрь 400-600 мг/сутки в 2 приема; "
            "амброксол 0,03 г 3 раза в сутки внутрь."
        ),
        "page_from": 5,
        "drugs": ["ацетилцистеин", "амброксол"],
    }
    item = format_rich_chunk_nav_item(block, query="таблетки", intents=["treatment"])
    assert item is not None
    assert "ацетилцистеин" in (item.get("lead") or "").lower()
    assert item.get("body_bullets") or item.get("body")
    chips = {c["label"]: c["items"] for c in item.get("entities") or []}
    assert "Препараты" in chips


def test_frontend_intent_terms_synced_with_python():
    """P3: ключевые treatment-термины есть в index.html."""
    html = (ROOT / "frontend" / "web" / "doctor" / "index.html").read_text(encoding="utf-8")
    terms = INTENT_SPECS["treatment"]["terms"]
    missing = [t for t in terms if t not in html]
    assert not missing, f"terms missing in index.html: {missing[:8]}"
