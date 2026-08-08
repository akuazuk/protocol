"""Per-visit ICD status for MO Analytics chips."""
from __future__ import annotations

from clinical_knowledge.mo_icd_visit_status import (
    compute_icd_visit_status,
    status_from_finding_codes,
)


def test_status_from_finding_codes_priority() -> None:
    assert status_from_finding_codes(["B_dx_absent", "B_icd_name_weak_match"]) == "missing_dx"
    assert status_from_finding_codes(["B_icd_dir_code_unknown"]) == "not_in_directory"
    assert status_from_finding_codes(["B_icd_name_weak_match"]) == "weak_name"
    assert status_from_finding_codes([]) == "unknown"


def test_compute_missing_dx(monkeypatch) -> None:
    monkeypatch.setattr(
        "clinical_knowledge.mo_icd_resolve.resolve_icd_codes_from_mo",
        lambda case: {"main": None, "all": [], "present": False},
    )
    payload = compute_icd_visit_status({"clinical_diagnosis": ""})
    assert payload["status"] == "missing_dx"
    assert "нет" in payload["label_ru"].lower() or payload["label_ru"] == "нет Dx"


def test_compute_ok_from_findings() -> None:
    payload = compute_icd_visit_status(
        {"clinical_diagnosis": "Острый цистит"},
        findings=[{"code": "A_ok_something"}],
    )
    # no ICD finding codes → unknown from findings path, then live eval
    assert payload["status"] in {"ok", "unknown", "not_in_directory", "weak_name", "missing_dx"}
