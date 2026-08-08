"""Методика постановления №55: P0 только для реальных критических критериев."""
from __future__ import annotations

from clinical_knowledge.reg55_criteria import evaluate_reg55, format_failed_criteria_ru


def test_manual_review_status_is_not_p0() -> None:
    result = evaluate_reg55(
        {
            "status": "manual_review_required",
            "fields_present": {
                "complaints": True,
                "anamnesis": True,
                "objective_status": True,
                "diagnosis": True,
                "follow_up": True,
            },
            "diagnosis_short": "J06.9",
        }
    )
    assert result["has_p0_defect"] is False
    failed_ids = {item["id"] for item in result["failed"]}
    assert "no_unhandled_red_flag" in failed_ids
    assert all(item["severity"] != "P0" for item in result["failed"] if item["id"] == "no_unhandled_red_flag")


def test_evaluate_reg55_returns_criteria_detail_and_formula() -> None:
    result = evaluate_reg55(
        {
            "fields_present": {
                "complaints": True,
                "anamnesis": True,
                "objective_status": True,
                "diagnosis": True,
                "follow_up": False,
            },
            "clinical_diagnosis": "J06.9 ОРВИ",
            "status": "good",
        }
    )
    assert isinstance(result["regulatory_compliance_pct"], float)
    assert result["total"] >= 1
    assert result["criteria"]
    assert all("verdict" in row and "point" in row for row in result["criteria"])
    assert "100" in (result.get("formula_ru") or "")
    follow = next(row for row in result["criteria"] if row.get("id") == "follow_up_present")
    assert follow["verdict"] == "fail"
    assert follow["score"] == 0.0


def test_fields_present_from_clinical_text() -> None:
    from clinical_knowledge.reg55_criteria import fields_present_from_case

    fp = fields_present_from_case(
        {
            "clinical": {
                "complaints": "боль в горле",
                "clinical_diagnosis": "ОРВИ",
                "objective_status": "зев гиперемирован",
            }
        }
    )
    assert fp["complaints"] is True
    assert fp["diagnosis"] is True
    assert fp["objective_status"] is True


def test_format_failed_criteria_includes_how_checked() -> None:
    text = format_failed_criteria_ru(
        [
            {
                "title": "Наличие описания жалоб пациента",
                "point": "прил. 2, п. 4.2.5",
                "severity": "P2",
                "how_checked_ru": "Проверяется наличие заполненного поля «complaints».",
            }
        ]
    )
    assert "жалоб" in text.lower()
    assert "P2" in text
    assert "complaints" in text
