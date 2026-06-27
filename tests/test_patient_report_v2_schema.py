"""Patient report v2 schema contract."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.patient_report import build_patient_report
from clinical_knowledge.patient_report_v2 import enrich_patient_report_v2

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "patient_report_v2.schema.json"


def _sample_v2_report() -> dict:
    l1 = {
        "confidence_score": 80,
        "matched_protocols_count": 1,
        "structured_analysis": {"matches": [{"title": "КП", "source_path": "x/kp.pdf", "match_score": 60}]},
        "alignment": {
            "alignment_mean_score": 65,
            "alignment_cards": [
                {"block_id": "diagnosis", "name_ru": "Диагноз", "score_pct": 80, "gaps_ru": []},
                {"block_id": "treatment", "name_ru": "Лечение", "score_pct": 55, "gaps_ru": ["нет срока"]},
            ],
        },
    }
    kz = Path(__file__).parent / "fixtures" / "neurology_kz_adult.txt"
    text = kz.read_text(encoding="utf-8")
    base = build_patient_report(l1)
    return enrich_patient_report_v2(base, l1_result=l1, kz_text=text, patient_context={"age_group": "adult"})


def test_v2_required_fields_present() -> None:
    rep = _sample_v2_report()
    assert rep["report_schema_version"] == 2
    assert rep.get("top_summary")
    assert rep.get("scores")
    assert rep.get("understood_from_document") is not None
    assert rep.get("clarification_points") is not None
    assert rep.get("message_to_doctor")
    assert rep.get("visit_sheet")
    assert rep.get("protocol_confidence_bucket") in ("low", "medium", "high")


def test_v2_schema_file_validates_core_shape() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rep = _sample_v2_report()
    required = set(schema.get("required") or [])
    for key in required:
        assert key in rep, f"missing required field {key}"
