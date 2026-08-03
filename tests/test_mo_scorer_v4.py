from __future__ import annotations

import sqlite3

import pytest

from clinical_knowledge.kz_evaluation_v4 import evaluate_kz_v4, load_v4_config
from clinical_knowledge.kz_evaluation_schema import EvaluationMode
from clinical_knowledge.mo_daily import build_daily_report, initialize_warehouse, upsert_warehouse
from clinical_knowledge.mo_llm_usage import calculate_cost_usd, record_llm_usage
from clinical_knowledge.mo_validation import build_gold_queue, evaluate_gold


CASE = {
    "mis_id": "m1",
    "visit_id": "v1",
    "date": "2026-07-29",
    "complaints": "боль в горле",
    "anamnesis_doctor": "болеет три дня",
    "objective_status": "зев гиперемирован",
    "clinical_diagnosis": "Острый фарингит",
    "mkb_code_main": "J02.9",
    "exam_recommendations": "Общий анализ крови",
    "treatment_recommendations": "парацетамол 500 мг",
}


def test_v4_weights_are_owner_approved_and_sum_to_one():
    weights = load_v4_config()["axis_weights"]
    assert weights == {
        "documentation": 0.25,
        "clinical_concordance": 0.35,
        "safety": 0.30,
        "regulatory": 0.10,
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_v4_is_primary_versioned_and_explainable():
    result = evaluate_kz_v4(CASE)
    assert result.schema_version == "4.0"
    assert result.scorer_version == "v4.0.0"
    assert result.mode.primary is False
    assert result.score_pct is not None
    assert sum(
        value for value in result.axis_contributions.values() if value is not None
    ) == pytest.approx(result.score_pct, abs=0.1)
    assert result.provenance.weights_version == "2026-07-30.1"


def test_untrusted_protocol_findings_are_advisory():
    protocol = {
        "review_status": "needs_review",
        "required_exams": ["КТ органов грудной клетки"],
        "name": "Неподтверждённый протокол",
    }
    result = evaluate_kz_v4(CASE, protocol_ctx=protocol)
    untrusted = [finding for finding in result.findings if finding.trust_level in {"C", "D"}]
    assert untrusted
    assert all(not finding.penalty_applied for finding in untrusted)


def test_llm_pricing_and_usage_are_persisted(tmp_path):
    assert calculate_cost_usd("gemini-3.6-flash", 1_000_000, 1_000_000) == 9.0
    warehouse = tmp_path / "mo.sqlite"
    initialize_warehouse(warehouse)
    usage = record_llm_usage(
        warehouse,
        run_id="run-1",
        tier="bulk",
        model="gemini-3.6-flash",
        case_id="v1",
        prompt_tokens=1000,
        completion_tokens=200,
        latency_ms=1200,
        status="ok",
    )
    assert usage["cost_usd"] > 0
    with sqlite3.connect(warehouse) as db:
        assert db.execute("SELECT COUNT(*) FROM fact_llm_usage").fetchone()[0] == 1


def test_warehouse_preserves_v3_and_visit_denominators(tmp_path):
    result = evaluate_kz_v4(
        CASE, mode=EvaluationMode(enabled=True, primary=True, gate=False)
    )
    case = {
        **CASE,
        "evaluation_v4": result.to_public_dict(),
        "overall_pct_v3": 87.0,
        "doctor_fio": "Иванов И.И.",
        "doctor_specialization": "Терапевт",
    }
    raw = {
        "id": "m1",
        "visit_id": "v1",
        "visit_date": "2026-07-29",
        "document_kind": "consultation",
        "doctor_fio": "Иванов И.И.",
        "doctor_specialization": "Терапевт",
        "patient_id": "p1",
    }
    report, _ = build_daily_report(
        [raw],
        [case],
        day=__import__("datetime").date(2026, 7, 29),
        run_id="v4-test",
        revision=1,
        quality={"passed": True},
    )
    warehouse = tmp_path / "mo.sqlite"
    upsert_warehouse(warehouse, [raw], [case], report)
    with sqlite3.connect(warehouse) as db:
        row = db.execute(
            "SELECT overall_pct_v3,scorer_version FROM fact_mo_case"
        ).fetchone()
        visit = db.execute(
            "SELECT records,scored_records,scorer_version FROM fact_mo_visit"
        ).fetchone()
    assert row == (87.0, "v4.0.0")
    assert visit == (1, 1, "v4.0.0")


def test_gold_queue_requires_real_double_label_acceptance():
    cases = [
        {
            "mis_id": str(index),
            "visit_id": str(index),
            "date": "2026-07-29",
            "doctor_specialization": "Терапевт",
            "evaluation_v4": {
                "score_pct": 80 + index % 10,
                "findings": [],
            },
        }
        for index in range(350)
    ]
    queue = build_gold_queue(cases, size=300)
    assert len(queue) == 300
    report = evaluate_gold(queue)
    assert report["n_completed"] == 0
    assert report["accepted"] is False
