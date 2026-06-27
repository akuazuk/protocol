"""Safe protocol quotes for B2C."""
from __future__ import annotations

from clinical_knowledge.patient_quote_quality import (
    filter_protocol_citations,
    is_unsafe_quote,
    sanitize_patient_text,
)


def test_unsafe_genetic_and_pediatric_quotes() -> None:
    assert is_unsafe_quote("Мутация гена PMP22 прилагается к протоколу")
    assert is_unsafe_quote("ИНСУЛЬТЕ У ДЕТЕЙ - экстренная помощь")
    assert is_unsafe_quote("прилагается")
    assert not is_unsafe_quote("При шейно-черепном синдроме рекомендуется МРТ шейного отдела позвоночника.")


def test_filter_citations() -> None:
    cites = [
        {"excerpt": "При шейно-черепном синдроме указывают локализацию боли.", "protocol_title": "КП"},
        {"excerpt": "SMN1/SMN2 при спinal muscular atrophy", "protocol_title": "КП дет"},
    ]
    out = filter_protocol_citations(cites)
    assert len(out) == 1


def test_sanitize_forbidden_phrasing() -> None:
    t = sanitize_patient_text("По протоколу положено сделать УЗИ")
    assert "положено" not in t.lower()
    assert "стандарту" in t.lower()
