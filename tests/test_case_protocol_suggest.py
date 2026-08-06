from __future__ import annotations

from clinical_knowledge.case_protocol_suggest import (
    build_case_fact_graph,
    suggest_protocols_for_case,
)


def test_build_case_fact_graph_extracts_icd_and_complaints() -> None:
    graph = build_case_fact_graph(
        clinical={
            "complaints": "боль в колене; хромота",
            "clinical_diagnosis": "M60 Миозит",
            "exam_recommendations": "УЗИ",
        },
        record={"visit_id": "v1", "specialty": "Ортопед"},
        findings=[{"code": "X1", "title_ru": "Нет обоснования диагноза"}],
    )
    assert graph["case_id"] == "v1"
    assert any(d.get("icd", "").startswith("M60") for d in graph["diagnoses"])
    assert "боль в колене" in graph["complaints"]
    assert graph["gaps"]


def test_suggest_protocols_returns_contract(monkeypatch) -> None:
    monkeypatch.setenv("CASE_PROTOCOL_SUGGEST", "1")

    def _fake_match(facts, specialty_slug=None, limit=8):
        return [
            {
                "protocol_id": "p1",
                "title": "Тестовый КП",
                "source_path": "minzdrav_protocols/test.pdf",
                "match_score": 77.0,
                "icd_fit": [{"code": "M60", "weight": 1.0}],
                "icd_fit_label": "M60 (1.00)",
            }
        ]

    monkeypatch.setattr(
        "clinical_knowledge.protocol_match.match_protocol_cards",
        _fake_match,
    )
    result = suggest_protocols_for_case(
        clinical={"clinical_diagnosis": "M60", "complaints": "боль"},
        record={"visit_id": "v2", "specialty": "Терапевт"},
        findings=[],
        limit=3,
    )
    assert result["ok"] is True
    assert result["available"] is True
    assert result["engine"] == "case_protocol_suggest_v1"
    assert result["items"][0]["protocol_id"] == "p1"
    assert result["items"][0]["match_kind_label"]
    assert result["items"][0]["reasons"]
    assert result["items"][0]["viewer_url"].startswith("/proto-viewer.html?path=")
    assert "minzdrav_protocols" in result["items"][0]["viewer_url"]
    assert "/proto?" not in result["items"][0]["viewer_url"]
