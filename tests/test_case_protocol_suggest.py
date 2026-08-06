from __future__ import annotations

from clinical_knowledge.case_protocol_suggest import (
    build_case_fact_graph,
    suggest_protocols_for_case,
)
from clinical_knowledge.protocol_links import protocol_display_name, title_looks_truncated


def test_build_case_fact_graph_extracts_icd_and_complaints() -> None:
    graph = build_case_fact_graph(
        clinical={
            "complaints": "боль в колене; хромота",
            "clinical_diagnosis": "M60 Миозит",
            "exam_recommendations": "УЗИ",
        },
        record={"visit_id": "v1", "specialty": "Ортопед"},
        findings=[
            {"code": "B_dx_no_support", "title_ru": "Нет обоснования диагноза"},
            {"code": "D_reg55_p0", "title_ru": "Критический дефект по №55"},
            {"code": "C_nsaid_dup", "title_ru": "Одновременно ≥2 НПВП"},
        ],
    )
    assert graph["case_id"] == "v1"
    assert any(d.get("icd", "").startswith("M60") for d in graph["diagnoses"])
    assert "боль в колене" in graph["complaints"]
    assert any(g.get("code") == "B_dx_no_support" for g in graph["gaps"])
    assert not any(str(g.get("code") or "").startswith("D_reg55") for g in graph["gaps"])
    assert not any(str(g.get("code") or "").startswith("C_nsaid") for g in graph["gaps"])
    assert graph["specialty"]["slug"] == "travmatologiya-ortopediya"


def test_truncated_registry_title_uses_filename() -> None:
    assert title_looks_truncated('клинический протокол «Диагностика и лечение')
    title = protocol_display_name(
        "minzdrav_protocols/urologiya/KP_cystitis_adult.pdf",
        registry_title='клинический протокол «Диагностика и лечение',
        prefer_filename_if_truncated=True,
    )
    assert "cystitis" in title.lower() or "KP" in title or "цист" in title.lower() or len(title) > 10
    assert "Диагностика и лечение" not in title or "»" in title


def test_suggest_protocols_returns_contract(monkeypatch) -> None:
    monkeypatch.setenv("CASE_PROTOCOL_SUGGEST", "1")

    def _fake_match(facts, specialty_slug=None, limit=8):
        return [
            {
                "protocol_id": "p1",
                "title": 'клинический протокол «Диагностика и лечение',
                "source_path": "minzdrav_protocols/travmatologiya-ortopediya/KP_myositis.pdf",
                "match_score": 77.0,
                "icd_fit": [{"code": "M60", "weight": 1.0}],
                "icd_fit_label": "M60 (1.00)",
                "specialty_slug": "travmatologiya-ortopediya",
            }
        ]

    monkeypatch.setattr(
        "clinical_knowledge.protocol_match.match_protocol_cards",
        _fake_match,
    )
    result = suggest_protocols_for_case(
        clinical={"clinical_diagnosis": "M60 миозит", "complaints": "боль"},
        record={"visit_id": "v2", "specialty": "Ортопед"},
        findings=[{"code": "D_reg55_p0", "title_ru": "Критический дефект по №55"}],
        limit=3,
    )
    assert result["ok"] is True
    assert result["available"] is True
    assert result["engine"] == "case_protocol_suggest_v2"
    item = result["items"][0]
    assert item["protocol_id"] == "p1"
    assert item["match_kind_label"]
    assert item["reasons"]
    assert not any("Критический дефект" in (r.get("text") or "") for r in item["reasons"])
    assert item["viewer_url"].startswith("/proto-viewer.html?path=")
    assert item["search_url"].startswith("/doctor/search?q=")
    assert "Диагностика и лечение" not in item["title"] or "»" in item["title"]
    assert "/proto?" not in item["viewer_url"]


def test_suggest_blocks_stomatology_for_urologist(monkeypatch) -> None:
    monkeypatch.setenv("CASE_PROTOCOL_SUGGEST", "1")

    def _fake_match(facts, specialty_slug=None, limit=8):
        return [
            {
                "protocol_id": "bad",
                "title": "Заболевания челюстно-лицевой области",
                "source_path": "minzdrav_protocols/stomatologiya/chelust.pdf",
                "match_score": 90.0,
                "icd_fit": [],
                "specialty_slug": "stomatologiya",
            },
            {
                "protocol_id": "good",
                "title": "Урология взрослых",
                "source_path": "minzdrav_protocols/urologiya/kp_urology.pdf",
                "match_score": 70.0,
                "icd_fit": [],
                "specialty_slug": "urologiya",
            },
        ]

    monkeypatch.setattr(
        "clinical_knowledge.protocol_match.match_protocol_cards",
        _fake_match,
    )
    result = suggest_protocols_for_case(
        clinical={
            "clinical_diagnosis": "Состояние после циркумцизио",
            "complaints": "боль в ране",
            "treatment_recommendations": 'Ибупрофен ("Кетопрофен" и т.п.)',
            "doctor_specialization": "Уролог",
        },
        record={"visit_id": "3650612"},
        findings=[{"code": "C_nsaid_dup", "title_ru": "Одновременно ≥2 НПВП"}],
        limit=3,
    )
    ids = [item.get("protocol_id") for item in result.get("items") or []]
    assert "bad" not in ids
    assert "good" in ids
