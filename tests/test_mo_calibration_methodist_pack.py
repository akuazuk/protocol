from __future__ import annotations

from scripts import build_mo_calibration_methodist_pack as packer


def _case(index: int) -> dict:
    return {
        "case_id": f"real-{index}",
        "clinical": {
            "age_years": 50,
            "sex": "female",
            "doctor_specialization": "Терапевт",
            "complaints": "жалобы",
            "anamnesis": "анамнез",
            "objective_status": "статус",
            "exam_data": "обследования",
            "clinical_diagnosis": "диагноз",
            "mkb_code_main": "I10",
            "recommendations_exam": "план обследований",
            "recommendations_treatment": "план лечения",
            "manipulations": "",
        },
    }


def _pilot() -> list[dict]:
    rows: list[dict] = []
    for index in range(1, 19):
        sample_id = f"S{index:03d}"
        for pass_no in (1, 2):
            rows.append(
                {
                    "kind": "pass",
                    "sample_id": sample_id,
                    "pass_no": pass_no,
                    "error": None,
                    "dx_evidence": {"verdict": "good", "dx_evidence_pct": 80},
                    "plan_concordance": {
                        "verdict": "partial",
                        "plan_general_llm_pct": 60,
                    },
                }
            )
        endpoints = ["dx", "plan"] if index <= 4 else ["dx" if index % 2 else "plan"]
        for endpoint in endpoints:
            rows.append(
                {
                    "kind": "adjudication",
                    "sample_id": sample_id,
                    "endpoint": endpoint,
                    "result": {"verdict": "partial"},
                }
            )
    return rows


def test_pack_covers_all_22_disagreements_without_real_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        packer,
        "protocol_context_for_case",
        lambda row, pack: (
            {"route": "llm_no_kp", "kp_status": "unmatched"},
            None,
        ),
    )
    cases, labels, sealed = packer.build_pack(
        [_case(index) for index in range(1, 31)],
        _pilot(),
    )
    assert len(cases) == 18
    assert len(labels) == 22
    assert len(sealed) == 22
    assert all(item["sample_id"].startswith("S") for item in cases)
    assert "real-" not in packer._canonical(cases)
    assert {(row["sample_id"], row["endpoint"]) for row in labels} == {
        (row["sample_id"], row["endpoint"]) for row in sealed
    }


def test_label_audit_requires_complete_human_fields() -> None:
    expected = {("S001", "dx"), ("S002", "plan")}
    template = [
        packer._label_template("S001", "dx"),
        packer._label_template("S002", "plan"),
    ]
    assert packer.audit_labels(template, expected=expected, minimum_cases=2)["passed"] is False
    completed = []
    for row in template:
        completed.append(
            {
                **row,
                "verdict": "partial",
                "score_pct": 65,
                "potential_harm": False,
                "icd_fit": "partial" if row["endpoint"] == "dx" else "na",
                "confidence": 0.8,
                "rationale": "Достаточное клиническое обоснование.",
                "reviewer_id": "methodist-1",
                "reviewed_at": "2026-08-09T14:00:00Z",
            }
        )
    audit = packer.audit_labels(completed, expected=expected, minimum_cases=2)
    assert audit["passed"] is True
    assert audit["complete_label_n"] == 2
