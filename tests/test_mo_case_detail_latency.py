"""Unit tests for MO case-detail latency flags and prewarm."""
from __future__ import annotations

from clinical_knowledge.mo_case_detail_latency import (
    findings_look_empty,
    prewarm_protocol_suggest_match,
    want_live_analyzers,
    want_prior_clinical,
    want_protocol_suggest_history,
)


def test_findings_look_empty() -> None:
    assert findings_look_empty(None) is True
    assert findings_look_empty([]) is True
    assert findings_look_empty([{"passed": True, "code": "X"}]) is True
    assert findings_look_empty([{"code": "C_ddi", "title_ru": "DDI"}]) is False


def test_want_live_analyzers_auto_skips_when_warehouse_has_findings(monkeypatch) -> None:
    monkeypatch.setenv("MO_CASE_DETAIL_LIVE_ANALYZERS", "auto")
    findings = [{"code": "C_ddi", "severity": "P1", "title_ru": "Major"}]
    assert want_live_analyzers(findings=findings) is False
    assert want_live_analyzers(findings=[]) is True


def test_want_live_analyzers_query_force(monkeypatch) -> None:
    monkeypatch.setenv("MO_CASE_DETAIL_LIVE_ANALYZERS", "0")
    assert want_live_analyzers(query_params={"live": "1"}, findings=[{"code": "X"}]) is True
    assert want_live_analyzers(query_params={"live": "0"}, findings=[]) is False


def test_want_prior_default_off(monkeypatch) -> None:
    monkeypatch.delenv("MO_CASE_DETAIL_PRIOR", raising=False)
    assert want_prior_clinical() is False
    assert want_prior_clinical(query_params={"prior": "1"}) is True


def test_want_protocol_suggest_history_default_off(monkeypatch) -> None:
    monkeypatch.delenv("MO_PROTOCOL_SUGGEST_ATTACH_HISTORY", raising=False)
    assert want_protocol_suggest_history() is False
    assert want_protocol_suggest_history(query_params={"attach_history": "1"}) is True


def test_prewarm_protocol_suggest_match_runs(monkeypatch) -> None:
    cards = [
        {
            "protocol_id": "kp-demo",
            "title": "Мигрень",
            "source_path": "demo.pdf",
            "specialty_slug": "nevrologiya",
            "icd10_codes": ["G43"],
        }
    ]
    monkeypatch.setattr(
        "clinical_knowledge.loader.load_protocol_cards_registry",
        lambda: cards,
    )
    monkeypatch.setattr(
        "clinical_knowledge.protocol_match.match_protocol_cards_by_diagnosis_text",
        lambda *a, **k: [{"protocol_id": "kp-demo", "match_score": 80}],
    )
    monkeypatch.setattr(
        "clinical_knowledge.protocol_match.match_protocol_cards",
        lambda *a, **k: [{"protocol_id": "kp-demo", "match_score": 90}],
    )
    out = prewarm_protocol_suggest_match()
    assert out.get("ok") is True
    assert out.get("cards") == 1
    assert out.get("text_hits") == 1
    assert out.get("icd_hits") == 1
