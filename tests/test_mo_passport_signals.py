"""P2: теневые сигналы паспорта, без PHI и без сдвига итога."""
from __future__ import annotations

from clinical_knowledge.mo_overall_grade import attach_overall_grade
from clinical_knowledge.mo_passport_signals import (
    CODE_CROSS_SPEC,
    CODE_LAB_AFTER_PLAN,
    CODE_REPEAT_PLAN,
    evaluate_passport_signals,
    merge_passport_signals_into_findings,
    plan_fingerprint,
)
from clinical_knowledge.mo_patient_history_bundle import merge_patient_history_into_findings


def test_repeat_same_plan_is_shadow() -> None:
    plan = "контроль АД через 14 дней, эналаприл 10 мг утром, диета"
    findings = evaluate_passport_signals(
        prior_plans=[
            {
                "visit_date": "2026-08-02",
                "specialty": "Терапевт",
                "diagnosis_code": "I10",
                "treatment_recommendations": plan,
            }
        ],
        case={
            "visit_date": "2026-09-10",
            "specialty": "Терапевт",
            "diagnosis_code": "I10.0",
            "treatment_recommendations": plan,
        },
    )
    assert findings[0]["code"] == CODE_REPEAT_PLAN
    assert findings[0]["is_shadow"] is True
    assert findings[0]["shadow"] is True
    assert plan_fingerprint(plan)


def test_cross_spec_requires_missing_mention() -> None:
    cards = [
        {
            "visit_date": "2026-07-01",
            "specialty": "Кардиолог",
            "diagnosis_code": "I10",
        }
    ]
    silent = evaluate_passport_signals(
        prior_cards=cards,
        case={
            "visit_date": "2026-09-10",
            "specialty": "Терапевт",
            "diagnosis_code": "I10",
            "anamnesis": "наблюдается с гипертензией, жалоб мало",
        },
    )
    mentioned = evaluate_passport_signals(
        prior_cards=cards,
        case={
            "visit_date": "2026-09-10",
            "specialty": "Терапевт",
            "diagnosis_code": "I10",
            "anamnesis": "наблюдается у кардиолога с гипертензией",
        },
    )
    assert silent[0]["code"] == CODE_CROSS_SPEC
    assert mentioned == []


def test_lab_after_plan_when_exam_empty() -> None:
    findings = evaluate_passport_signals(
        prior_cards=[{"visit_date": "2026-08-01", "specialty": "Терапевт", "diagnosis_code": "E11"}],
        lab_dates=["2026-08-20"],
        case={
            "visit_date": "2026-09-10",
            "specialty": "Терапевт",
            "diagnosis_code": "E11.9",
            "exam_data": "",
        },
    )
    assert findings[0]["code"] == CODE_LAB_AFTER_PLAN


def test_shadow_does_not_change_overall_grade() -> None:
    payload = {
        "zone1_pct": 90.0,
        "zone1_band": "ok",
        "zone2a_pct": 100.0,
        "zone2a_band": "ok",
        "zone2b_pct": None,
        "zone2b_band": "na",
        "safety_band": "none",
        "attention_primary": "",
    }
    extra = evaluate_passport_signals(
        prior_plans=[
            {
                "visit_date": "2026-08-02",
                "specialty": "Терапевт",
                "diagnosis_code": "I10",
                "treatment_recommendations": "контроль АД через 14 дней, эналаприл 10 мг утром",
            }
        ],
        case={
            "visit_date": "2026-09-10",
            "specialty": "Терапевт",
            "diagnosis_code": "I10",
            "treatment_recommendations": "контроль АД через 14 дней, эналаприл 10 мг утром",
        },
    )
    before = attach_overall_grade(dict(payload))
    after = attach_overall_grade(dict(payload))
    assert extra[0]["is_shadow"] is True
    assert before["overall_grade"] == after["overall_grade"]


def test_merge_dedups_and_hides_patient_id() -> None:
    case = {
        "patient_key": "abc",
        "visit_date": "2026-09-10",
        "specialty": "Терапевт",
        "diagnosis_code": "I10",
        "treatment_recommendations": "контроль АД через 14 дней, эналаприл 10 мг утром",
        "_passport_prior_plans": [
            {
                "visit_date": "2026-08-02",
                "specialty": "Терапевт",
                "diagnosis_code": "I10",
                "treatment_recommendations": "контроль АД через 14 дней, эналаприл 10 мг утром",
            }
        ],
    }
    first = merge_passport_signals_into_findings([], case)
    second = merge_passport_signals_into_findings(first, case)
    assert [item["code"] for item in second].count(CODE_REPEAT_PLAN) == 1
    dumped = str(second)
    assert "patient_id" not in dumped


def test_history_merge_still_adds_passport(monkeypatch) -> None:
    monkeypatch.setenv("MO_PATIENT_HISTORY_BUNDLE", "0")
    case = {
        "visit_date": "2026-09-10",
        "specialty": "Терапевт",
        "diagnosis_code": "I10",
        "treatment_recommendations": "контроль АД через 14 дней, эналаприл 10 мг утром",
        "_passport_prior_plans": [
            {
                "visit_date": "2026-08-02",
                "specialty": "Терапевт",
                "diagnosis_code": "I10",
                "treatment_recommendations": "контроль АД через 14 дней, эналаприл 10 мг утром",
            }
        ],
    }
    out = merge_patient_history_into_findings([], case)
    assert any(item.get("code") == CODE_REPEAT_PLAN for item in out)
