"""D3-D4: подоси drug/lab и флаг вклада в overall."""
from __future__ import annotations

from clinical_knowledge.kz_deep_eval import evaluate_kz_deep
from clinical_knowledge.mo_finding_families import (
    family_scores_from_findings,
    maybe_blend_family_into_axes,
)


def test_family_scores_penalties_and_caps() -> None:
    scores = family_scores_from_findings(
        [
            {"code": "C_ddi", "severity": "P2", "passed": False},
            {"code": "B_lab_unused_in_dx", "severity": "P2", "passed": False, "shadow": True},
        ]
    )
    assert scores["drug_score"] == 90.0
    assert scores["lab_score"] == 90.0
    assert scores["in_overall"] is False
    capped = family_scores_from_findings(
        [{"code": "C_ddi", "severity": "P0", "passed": False}]
    )
    assert capped["drug_score"] <= 40.0


def test_deep_eval_exposes_family_scores_without_changing_overall(monkeypatch) -> None:
    monkeypatch.delenv("MO_FAMILY_SCORES_IN_OVERALL", raising=False)
    case = {
        "complaints": "боль",
        "anamnesis_doctor": "длительно",
        "objective_status": "живот мягкий",
        "clinical_diagnosis": "Гастрит",
        "treatment_recommendations": "Омепразол 20 мг",
        "exam_recommendations": "",
    }
    first = evaluate_kz_deep(case, drug_ctx={})
    second = evaluate_kz_deep(case, drug_ctx={})
    assert "family_scores" in first
    assert first["family_scores"].get("drug_score") is not None
    assert first["overall_pct"] == second["overall_pct"]


def test_overall_blend_default_off(monkeypatch) -> None:
    monkeypatch.delenv("MO_FAMILY_SCORES_IN_OVERALL", raising=False)
    monkeypatch.setenv("MO_LAB_UNUSED_PRIMARY", "1")
    axes = {"clinical_concordance": 90.0, "safety": 90.0, "documentation": 90.0}
    scores = {"lab_score_primary": 40.0, "drug_score_primary": 40.0}
    out, meta = maybe_blend_family_into_axes(axes, scores)
    assert meta["applied"] is False
    assert out["clinical_concordance"] == 90.0


def test_overall_blend_min_when_flag_and_primary(monkeypatch) -> None:
    monkeypatch.setenv("MO_FAMILY_SCORES_IN_OVERALL", "1")
    monkeypatch.setenv("MO_LAB_UNUSED_PRIMARY", "1")
    monkeypatch.delenv("MO_CLASS_DUP_PRIMARY", raising=False)
    monkeypatch.delenv("MO_RCETH_LABEL_PRIMARY", raising=False)
    axes = {"clinical_concordance": 90.0, "safety": 90.0, "documentation": 90.0}
    scores = {"lab_score_primary": 40.0, "drug_score_primary": 40.0}
    out, meta = maybe_blend_family_into_axes(axes, scores)
    assert meta["applied"] is True
    assert out["clinical_concordance"] == 40.0
    assert out["safety"] == 90.0


def test_deep_eval_flag_without_primary_keeps_overall(monkeypatch) -> None:
    monkeypatch.setenv("MO_FAMILY_SCORES_IN_OVERALL", "1")
    monkeypatch.delenv("MO_LAB_UNUSED_PRIMARY", raising=False)
    monkeypatch.delenv("MO_LAB_ABNORMAL_PRIMARY", raising=False)
    monkeypatch.delenv("MO_CLASS_DUP_PRIMARY", raising=False)
    monkeypatch.delenv("MO_RCETH_LABEL_PRIMARY", raising=False)
    case = {
        "complaints": "боль",
        "anamnesis_doctor": "длительно",
        "objective_status": "живот мягкий",
        "clinical_diagnosis": "Гастрит",
        "treatment_recommendations": "Омепразол 20 мг",
        "exam_recommendations": "",
    }
    monkeypatch.delenv("MO_FAMILY_SCORES_IN_OVERALL", raising=False)
    baseline = evaluate_kz_deep(case, drug_ctx={})
    monkeypatch.setenv("MO_FAMILY_SCORES_IN_OVERALL", "1")
    flagged = evaluate_kz_deep(case, drug_ctx={})
    assert baseline["overall_pct"] == flagged["overall_pct"]
    assert flagged["family_scores"].get("overall_blend", {}).get("applied") is False
