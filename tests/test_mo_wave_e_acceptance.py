"""Permanent synthetic contract matrix E01-E23 (no PHI fixtures)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from clinical_knowledge.kz_deep_eval import evaluate_kz_deep
from clinical_knowledge.medication_normative_cards import (
    build_medication_normative_cards,
)
from clinical_knowledge.medication_parser import active_medication_assignments
from clinical_knowledge.mo_backend import _assessment_contract_from_row
from clinical_knowledge.mo_case_review_brief import synthesize_doctor_feedback
from clinical_knowledge.mo_daily import _assessment_input_integrity
from clinical_knowledge.mo_history_assessment import build_history_assessment_context
from clinical_knowledge.mo_lab_result_assessment import build_lab_result_assessment
from clinical_knowledge.mo_reg55_section import evaluate_reg55_section
from clinical_knowledge.mo_review_pack import save_review_pack


MATRIX = {
    "E01": "evaluator input loss",
    "E02": "stale revision",
    "E03": "next previous race",
    "E04": "list detail export parity",
    "E05": "protocol not evaluated",
    "E06": "history absent",
    "E07": "irrelevant prior",
    "E08": "relevant prior evidence",
    "E09": "culture and chemistry identity",
    "E10": "normal lab result",
    "E11": "abnormal without context",
    "E12": "post visit and unknown time",
    "E13": "empty and failed lab evaluator",
    "E14": "true zero score",
    "E15": "negation family hypothesis",
    "E16": "alternative and past medication",
    "E17": "consent unavailable source",
    "E18": "local N55 adaptation",
    "E19": "suspicion withheld from doctor message",
    "E20": "save replay concurrency and RBAC",
    "E21": "isolated widget failures",
    "E22": "keyboard narrow long text zoom",
    "E23": "lab assets and image evaluation",
}


def test_e_matrix_has_all_23_stable_ids() -> None:
    assert list(MATRIX) == [f"E{i:02d}" for i in range(1, 24)]


def test_e01_input_loss_and_conflicting_absence_are_not_confirmed() -> None:
    result = _assessment_input_integrity(
        raw={"complaints": "Синтетическая жалоба"},
        case={"input_presence": {"complaints": False}},
        evaluation={},
        findings=[{"code": "A_missing_complaints"}],
    )
    assert result["status"] == "error"
    assert result["dropped_domains"] == ["complaints"]
    assert result["conflicting_finding_codes"] == ["A_missing_complaints"]


def _assessment_row(*, content_hash: str = "same", score: float = 0.0) -> dict:
    snapshot = {
        "contract_version": 1,
        "evaluation_run_id": "run-synthetic",
        "document_revision": 7,
        "source_hash": "same",
        "status": "completed",
        "value": score,
        "scores": {"overall_pct": score},
        "protocol": {"applicability_status": "not_evaluated"},
    }
    return {
        "evaluation_run_id": "run-synthetic",
        "document_revision": 7,
        "source_hash": "same",
        "content_hash": content_hash,
        "overall_pct": score,
        "assessment_status": "completed",
        "evaluation_snapshot_json": json.dumps(snapshot),
    }


def test_e02_e04_e14_assessment_revision_parity_and_true_zero() -> None:
    stale = _assessment_contract_from_row(_assessment_row(content_hash="new"))
    assert stale["status"] == "stale"
    assert stale["confirmed_value"] is None
    row = _assessment_row(score=0.0)
    projections = [_assessment_contract_from_row(dict(row)) for _ in range(3)]
    assert projections[0] == projections[1] == projections[2]
    assert projections[0]["value"] == 0.0
    assert projections[0]["confirmed_value"] == 0.0


def test_e05_protocol_not_evaluated_has_no_protocol_verdict() -> None:
    protocol = {
        "required_exams": ["synthetic exam"],
        "diagnostic_criteria": ["synthetic criterion"],
        "treatment": ["warfarin"],
    }
    case = {
        "clinical_diagnosis": "synthetic diagnosis",
        "treatment_recommendations": "Метформин 500 мг",
        "protocol_check": "not_evaluated",
    }
    deep = evaluate_kz_deep(case, protocol_ctx=protocol, drug_ctx={})
    assert deep["protocol_used"] is False
    assert not {
        "B_exams_gap",
        "B_dx_criteria_gap",
        "B_tx_offprotocol",
    }.intersection(item["code"] for item in deep["findings"])
    cards = build_medication_normative_cards(
        case,
        assessment={"protocol": {"applicability_status": "not_evaluated"}},
        zones={"zone2b": {"kp_status": "unmatched"}},
        label_ctx={"by_inn": {}},
    )
    assert all(card["protocol_check"] == "not_evaluated" for card in cards["cards"])
    assert not any(
        source["source"] == "national_protocol"
        for card in cards["cards"]
        for source in card["instructions"]
    )


def test_e06_e07_e08_history_assessability_semantics() -> None:
    absent = build_history_assessment_context(history_bundle=None)
    assert absent["history_available"] is False
    assert absent["correction_assessable"] is False
    failed = build_history_assessment_context(
        history_bundle={"ok": False, "reason": "query_failed", "status": "error", "summary": {}}
    )
    assert failed["history_available"] is False
    assert failed["status"] == "error"
    irrelevant = build_history_assessment_context(
        history_bundle={
            "ok": True,
            "summary": {"n_visits": 1},
            "visits": [{"diagnosis_code": "A00", "date": "2026-01-01"}],
        },
        current_code="J00",
        current_text="synthetic respiratory episode",
    )
    assert irrelevant["any_prior_exists"] is True
    assert irrelevant["correction_assessable"] is False
    relevant = build_history_assessment_context(
        history_bundle={
            "ok": True,
            "summary": {"n_visits": 1},
            "visits": [{"diagnosis_code": "J00", "date": "2026-01-01"}],
        },
        current_code="J00",
        current_text="synthetic respiratory episode",
        episode_deep={
            "continuity": {
                "known_episode": True,
                "last_matched_date": "2026-01-01",
            },
            "prior_n_loaded": 1,
            "prior_clinical": {"treatment_recommendations": "synthetic plan"},
        },
    )
    assert relevant["relevant_episode_prior_exists"] is True
    assert relevant["correction_assessable"] is True
    assert relevant["sources"]["longitudinal"]["prior_n"] == 1


def _lab_bundle(*, value: str, available_at: str = "", reason: str = "") -> dict:
    return {
        "ok": True,
        "reason": reason,
        "lifecycle_schema_available": True,
        "days": [
            {
                "test_date": "2026-01-02",
                "types": [
                    {
                        "type_name": "Биохимия",
                        "test_id": "synthetic-1",
                        "indicators": [
                            {
                                "name": "Глюкоза",
                                "value": value,
                                "unit": "ммоль/л",
                                "available_at": available_at,
                                "result_status": "final",
                            }
                        ],
                    },
                    {
                        "type_name": "Бакпосев",
                        "test_id": "synthetic-2",
                        "indicators": [
                            {
                                "name": "Посев мочи",
                                "value": "роста нет",
                                "available_at": available_at,
                                "result_status": "final",
                            }
                        ],
                    },
                ],
            }
        ],
    }


def test_e09_to_e13_lab_identity_lifecycle_and_empty_failure() -> None:
    normal = build_lab_result_assessment(
        _lab_bundle(value="5.0", available_at="2026-01-02T08:00:00+03:00"),
        case={"patient_age_years": 40},
        cutoff_at="2026-01-02T12:00:00+03:00",
    )
    ids = {row["identity"]["panel_id"] for row in normal["results"]}
    assert {"glucose", "urine_culture"}.issubset(ids)
    glucose = next(
        row for row in normal["results"] if row["identity"]["panel_id"] == "glucose"
    )
    assert glucose["result_present"] is True
    assert glucose["actionable_for_review"] is False
    abnormal = build_lab_result_assessment(
        _lab_bundle(value="99", available_at="2026-01-02T08:00:00+03:00"),
        case={},
        cutoff_at="2026-01-02T12:00:00+03:00",
    )
    assert any(row["reference_available"] is False for row in abnormal["results"])
    post = build_lab_result_assessment(
        _lab_bundle(value="99", available_at="2026-01-03T08:00:00+03:00"),
        case={"patient_age_years": 40},
        cutoff_at="2026-01-02T12:00:00+03:00",
    )
    assert post["summary"]["post_cutoff_n"] == 2
    unknown = build_lab_result_assessment(
        _lab_bundle(value="99"),
        case={"patient_age_years": 40},
        cutoff_at="2026-01-02",
    )
    assert unknown["summary"]["unknown_availability_n"] == 2
    empty = build_lab_result_assessment(
        {"ok": True, "reason": "empty", "days": []},
        case={},
        cutoff_at="2026-01-02",
    )
    failed = build_lab_result_assessment(
        {"ok": False, "reason": "query_failed", "days": []},
        case={},
        cutoff_at="2026-01-02",
    )
    assert empty["status"] == "empty"
    assert failed["status"] == "unavailable"


def test_e15_e16_medication_assertion_alternative_and_past_guards() -> None:
    for text in (
        "Не принимает варфарин; Метформин 500 мг",
        "У отца варфарин; Метформин 500 мг",
        "Рассмотреть варфарин; Метформин 500 мг",
        "Ранее варфарин; Метформин 500 мг",
        "Метформин 500 мг (или варфарин 5 мг)",
    ):
        active = active_medication_assignments({"treatment_recommendations": text})
        assert all(item.get("inn") != "warfarin" for item in active)


def test_e17_e18_consent_and_local_methodology_scope() -> None:
    unavailable = evaluate_reg55_section(
        {"doctor_specialization": "терапевт", "raw_text": "Синтетическая запись"}
    )
    consent = [
        row for row in unavailable["criteria"] if "Согласие" in str(row.get("title"))
    ]
    assert consent and all(row["score"] is None for row in consent)
    payload = build_medication_normative_cards(
        {"treatment_recommendations": "Метформин 500 мг"},
        reg55={"pack_id": "synthetic-local", "pack_label_ru": "Локальный pack"},
        label_ctx={"by_inn": {}},
    )
    assert payload["methodology"]["n127_role"] == "evidence_helper"
    assert payload["methodology"]["local_pack_is_normative"] is False


def test_e19_candidate_suspicion_is_not_doctor_feedback() -> None:
    feedback = synthesize_doctor_feedback(
        zones={"criteria": []},
        findings=[
            {
                "code": "C_suspicion",
                "axis": "safety",
                "severity": "P1",
                "title_ru": "Подозрение",
                "assessment_status": "candidate",
            }
        ],
        protocol={"matched": True, "kp_status": "matched"},
        icd_status={},
        history_line="есть prior",
    )
    assert not any("Подозрение" in line for line in feedback)


def test_e20_review_pack_save_contract_exposes_all_guards() -> None:
    params = inspect.signature(save_review_pack).parameters
    assert {"role", "idempotency_key", "expected_document_revision"}.issubset(params)


def test_e23_lab_assets_and_image_verifier_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "data/lab_canons/lab_reference_ranges.json").is_file()
    assert (root / "data/lab_canons/lab_test_canons.json").is_file()
    verifier = (root / "deploy/gcp-app/verify_lab_assets.py").read_text(encoding="utf-8")
    assert "evaluate_lab_for_case" in verifier
    assert "lab_reference_ranges.json" in verifier

