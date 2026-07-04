"""API resolve_protocol_source_text."""
from __future__ import annotations

from clinical_knowledge.protocol_summary.source_text import resolve_protocol_source_text


def test_resolve_protocol_source_text_venous() -> None:
    path = (
        "minzdrav_protocols/bolezni-sistemy-krovoobrashcheniya/"
        "КП_Диагностика_и_лечение_пациентов_с_хроническими_заболеваниями_вен_взр_население-пост_МЗ_2022_55.pdf"
    )
    out = resolve_protocol_source_text(path)
    assert out["available"] is True
    assert out["block_count"] > 0
    assert out["toc"]
    assert "diagnostics" in (out.get("sections") or {})


def test_resolve_protocol_source_text_missing() -> None:
    out = resolve_protocol_source_text("minzdrav_protocols/unknown/no_such.pdf")
    assert out["available"] is False
