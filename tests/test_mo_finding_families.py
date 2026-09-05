"""D0: реестр семейств findings - покрытие, без сирот, фильтр, KPI."""
from __future__ import annotations

from clinical_knowledge.mo_backend import _filter_records
from clinical_knowledge.mo_finding_families import (
    codes_for_family,
    duplicate_family_codes,
    family_dashboard_from_rows,
    family_for_code,
    orphan_drug_lab_codes,
    required_drug_lab_codes,
    tile_codes,
)


def test_no_orphan_drug_lab_codes() -> None:
    assert orphan_drug_lab_codes() == []
    assert duplicate_family_codes() == []
    required = required_drug_lab_codes()
    assigned = codes_for_family("drug") | codes_for_family("lab")
    assert required <= assigned
    assert codes_for_family("drug") & codes_for_family("lab") == set()


def test_known_codes_map_to_families() -> None:
    assert family_for_code("C_ddi") == "drug"
    assert family_for_code("B_tx_offprotocol") == "drug"
    assert family_for_code("B_lab_unused_in_dx") == "lab"
    assert family_for_code("B_exams_gap") == "lab"
    assert family_for_code("B_dx_absent") == "other"


def test_filter_records_finding_family() -> None:
    records = [
        {"finding_codes": ["C_ddi"], "date": "2026-09-01", "document_kind": "clinical_visit"},
        {"finding_codes": ["B_lab_unused_in_dx"], "date": "2026-09-01", "document_kind": "clinical_visit"},
        {"finding_codes": ["B_dx_absent"], "date": "2026-09-01", "document_kind": "clinical_visit"},
    ]
    drug = _filter_records(records, {"finding_family": "drug"})
    lab = _filter_records(records, {"finding_family": "lab"})
    both = _filter_records(records, {"finding_family": "drug", "finding_codes": "C_ddi"})
    miss = _filter_records(records, {"finding_family": "lab", "finding_codes": "C_ddi"})
    assert [r["finding_codes"][0] for r in drug] == ["C_ddi"]
    assert [r["finding_codes"][0] for r in lab] == ["B_lab_unused_in_dx"]
    assert len(both) == 1
    assert miss == []


def test_family_dashboard_unused_uses_lab_denominator() -> None:
    rows = [
        {"mis_id": 1, "finding_code": "B_lab_unused_in_dx", "specialty": "терапия", "doctor": "A", "visit_date": "2026-09-01"},
        {"mis_id": 2, "finding_code": "C_ddi", "specialty": "терапия", "doctor": "B", "visit_date": "2026-09-01"},
        {"mis_id": 2, "finding_code": "C_ppi_dup", "specialty": "терапия", "doctor": "B", "visit_date": "2026-09-01"},
    ]
    dash = family_dashboard_from_rows(
        rows,
        total_cases=10,
        cases_with_lab=4,
        lab_coverage_available=True,
    )
    unused_tile = next(t for t in dash["families"]["lab"]["tiles"] if t["id"] == "unused")
    assert unused_tile["cases"] == 1
    assert unused_tile["pct"] == 25.0
    assert unused_tile["denominator"] == "cases_with_lab"
    assert dash["families"]["drug"]["cases"] == 1
    assert dash["families"]["drug"]["pct"] == 10.0
    ddi = next(t for t in dash["families"]["drug"]["tiles"] if t["id"] == "interactions")
    assert ddi["cases"] == 1
    assert "C_ddi" in tile_codes("drug", "interactions")
    assert dash["strips"]["lab"]["pct"] == 25.0
