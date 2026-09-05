"""Endpoint C sees warehouse labs as panel labels and dates, never values."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from clinical_knowledge.mo_daily import patient_key_for
from clinical_knowledge.mo_dx_evidence_score import (
    dx_evidence_eligibility,
    nonsemantic_dx_result,
    validate_dx_evidence_result,
)
from clinical_knowledge.mo_lab_dx_evidence import (
    CODE_DX_LAB_CONTEXT,
    attach_lab_evidence_to_row,
    lab_dx_shadow_findings,
    lab_evidence_for_dx,
    lab_evidence_text_from_source,
)
from clinical_knowledge.mo_lab_shadow import apply_lab_to_result
from scripts.run_mo_calibration_blind_judge import (
    audit_prompt_input,
    blind_case_pack,
    build_dx_prompt,
)

DDL = """
CREATE TABLE fact_mo_lab (
  patient_key TEXT NOT NULL,
  test_date TEXT NOT NULL,
  test_id INTEGER NOT NULL,
  type_id INTEGER,
  type_name TEXT,
  indicator_id INTEGER,
  indicator_name TEXT,
  value TEXT,
  unit TEXT
);
"""


def _seed(path: Path) -> None:
    pk = patient_key_for("1001")
    db = sqlite3.connect(path)
    db.executescript(DDL)
    db.executemany(
        "INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (pk, "2026-08-10", 1, 10, "Общий анализ крови", 101, "Гемоглобин", "132", "г/л"),
            (pk, "2026-08-20", 2, 20, "Биохимический анализ крови", 201, "Глюкоза", "5.4", "ммоль/л"),
        ],
    )
    db.commit()
    db.close()


def test_lab_evidence_is_labels_and_dates_only(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    ev = lab_evidence_for_dx(
        {"patient_id": "1001", "visit_date": "2026-08-20"},
        lab_db=db,
    )
    assert ev["present"] is True
    assert ev["has_values"] is False
    labels = {item["label"] for item in ev["panels"]}
    assert "ОАК" in labels
    assert "БАК" in labels
    assert ev["same_day_n"] >= 1
    blob = json.dumps(ev, ensure_ascii=False)
    assert "132" not in blob
    assert "5.4" not in blob
    assert "г/л" not in blob
    assert "1001" not in blob
    assert "patient_key" not in blob


def test_diagnosis_plus_lab_is_eligible_without_doctor_text() -> None:
    eligibility = dx_evidence_eligibility(
        {
            "slots": {
                "clinical_diagnosis": "Артериальная гипертензия",
                "lab": "ОАК (2026-08-20, день визита)",
            }
        }
    )
    assert eligibility["status"] == "eligible"
    assert "lab" in eligibility["evidence_slots_present"]
    assert nonsemantic_dx_result(
        {
            "slots": {
                "clinical_diagnosis": "Артериальная гипертензия",
                "lab": "ОАК (2026-08-20, день визита)",
            }
        }
    ) is None


def test_lab_payload_object_does_not_count_until_sanitized() -> None:
    blocked = dx_evidence_eligibility(
        {
            "slots": {"clinical_diagnosis": "Диагноз"},
            "lab": {"days": [{"types": [{"indicators": [{"value": "132"}]}]}]},
        }
    )
    assert blocked["status"] == "blocked"
    assert "lab" not in blocked["evidence_slots_present"]


def test_contract_accepts_lab_slot() -> None:
    result = validate_dx_evidence_result(
        {
            "dx_evidence_pct": 70,
            "verdict": "partial",
            "supported_by": [{"slot": "lab", "evidence": "ОАК (2026-08-20, день визита)"}],
            "icd_fit": "unknown",
            "provenance": "llm_blind",
        }
    )
    assert result["supported_by"][0]["slot"] == "lab"


def test_blind_prompt_sees_panels_not_values(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    row = {
        "visit_id": "real-visit-id-must-not-leak",
        "patient_id": "1001",
        "visit_date": "2026-08-20",
        "clinical_diagnosis": "анемия",
        "exam_data": "",
        "complaints": "",
        "lab_evidence": lab_evidence_for_dx(
            {"patient_id": "1001", "visit_date": "2026-08-20"},
            lab_db=db,
        ),
    }
    pack = blind_case_pack(row, sample_id="S-LAB")
    prompt, prompt_input = build_dx_prompt(pack)
    assert pack["evidence"]["lab"]
    assert "ОАК" in pack["evidence"]["lab"]
    assert "ОАК" in prompt
    assert "132" not in prompt
    assert "5.4" not in json.dumps(prompt_input, ensure_ascii=False)
    assert "real-visit-id-must-not-leak" not in prompt
    assert "слот lab" in prompt.lower() or "Слот lab" in prompt
    assert audit_prompt_input(prompt_input, source_row=row)["passed"] is True


def test_text_from_source_strips_ui_values() -> None:
    text = lab_evidence_text_from_source(
        {
            "visit_date": "2026-08-20",
            "lab": {
                "window": {"visit_date": "2026-08-20"},
                "days": [
                    {
                        "test_date": "2026-08-20",
                        "same_day": True,
                        "types": [
                            {
                                "type_name": "Общий анализ крови",
                                "indicators": [{"name": "Гемоглобин", "value": "132", "unit": "г/л"}],
                            }
                        ],
                    }
                ],
            },
        }
    )
    assert "ОАК" in text
    assert "132" not in text
    assert "г/л" not in text


def test_shadow_finding_only_when_diagnosis_has_no_clinical_text(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    ev = lab_evidence_for_dx(
        {"patient_id": "1001", "visit_date": "2026-08-20"},
        lab_db=db,
    )
    empty_text = lab_dx_shadow_findings(
        ev,
        {"clinical_diagnosis": "анемия", "exam_data": "", "complaints": ""},
    )
    assert [item["code"] for item in empty_text] == [CODE_DX_LAB_CONTEXT]
    assert all(item.get("is_shadow") for item in empty_text)
    assert "132" not in str(empty_text)
    with_exam = lab_dx_shadow_findings(
        ev,
        {"clinical_diagnosis": "анемия", "exam_data": "ОАК без патологии"},
    )
    assert with_exam == []


def test_apply_attaches_dx_evidence_and_keeps_primary_off(
    tmp_path: Path, monkeypatch
) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    monkeypatch.delenv("MO_LAB_IN_PRIMARY", raising=False)
    result: dict = {"findings": []}
    apply_lab_to_result(
        result,
        {
            "patient_id": "1001",
            "visit_date": "2026-08-20",
            "clinical_diagnosis": "анемия",
            "exam_data": "",
            "complaints": "",
        },
        lab_db=db,
    )
    assert result["lab"]["dx_evidence"]["present"] is True
    assert "ОАК" in result["lab"]["dx_evidence"]["text"]
    assert "132" not in json.dumps(result["lab"]["dx_evidence"], ensure_ascii=False)
    codes = {item["code"] for item in result["findings"]}
    assert CODE_DX_LAB_CONTEXT in codes
    assert all(
        item.get("is_shadow")
        for item in result["findings"]
        if item.get("code") == CODE_DX_LAB_CONTEXT
    )


def test_attach_does_not_query_without_identity() -> None:
    row = attach_lab_evidence_to_row({"clinical_diagnosis": "диагноз"})
    assert row["lab_evidence"]["present"] is False
