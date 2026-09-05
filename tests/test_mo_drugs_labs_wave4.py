"""Wave 4: dual scores, anomalies, risk-adjust, agreement."""
from __future__ import annotations

from clinical_knowledge.mo_anomaly_kpis import (
    classify_case_anomalies,
    kpi_from_finding_rows,
    load_anomaly_catalog,
)
from clinical_knowledge.mo_dual_score import (
    dual_admission_scores,
    form_content_matrix,
)
from clinical_knowledge.mo_risk_adjust import agreement_report, risk_adjust_doctor_rows


def test_dual_admission_takes_minimum() -> None:
    dual = dual_admission_scores(clinical_pct=64, document_ready_pct=93)
    assert dual["admission_pct"] == 64.0
    assert dual["rule"].startswith("min")


def test_form_content_matrix_form_over_content() -> None:
    cell = form_content_matrix(clinical_pct=55, document_ready_pct=90)
    assert cell["cell"] == "form_over_content"


def test_anomaly_catalog_has_ten() -> None:
    cat = load_anomaly_catalog()
    assert len(cat) == 10
    hits = classify_case_anomalies(
        [
            {"code": "B_lab_unused_in_dx"},
            {"code": "C_ddi"},
            {"code": "E_template_copy"},
        ]
    )
    nums = {h["n"] for h in hits}
    assert 1 in nums and 2 in nums and 6 in nums


def test_kpi_unused_and_drug_columns() -> None:
    rows = [
        {"mis_id": 1, "finding_code": "B_lab_unused_in_dx"},
        {"mis_id": 1, "finding_code": "C_ddi"},
        {"mis_id": 2, "finding_code": "C_ppi_dup"},
        {"mis_id": 3, "finding_code": "C_rceth_off_label"},
    ]
    kpi = kpi_from_finding_rows(rows, total_cases=10)
    assert kpi["unused_lab_cases"] == 1
    assert kpi["unused_lab_pct"] == 10.0
    assert kpi["drug_columns"]["interactions_cases"] == 1
    assert kpi["drug_columns"]["duplicates_cases"] == 1
    assert kpi["drug_columns"]["dose_label_cases"] == 1


def test_risk_adjust_requires_20_cases() -> None:
    rows = [
        {"doctor": "A", "specialty": "терапия", "cases": 30, "avg_score": 70},
        {"doctor": "B", "specialty": "терапия", "cases": 25, "avg_score": 80},
        {"doctor": "C", "specialty": "терапия", "cases": 5, "avg_score": 50},
    ]
    out = risk_adjust_doctor_rows(rows)
    by_doc = {r["doctor"]: r for r in out}
    assert by_doc["A"]["eligible"] is True
    assert by_doc["C"]["eligible"] is False
    assert by_doc["A"]["delta_vs_expected"] is not None


def test_agreement_report_kappa() -> None:
    pairs = [
        {"axis": "lab_unused", "system": True, "expert": True},
        {"axis": "lab_unused", "system": True, "expert": False},
        {"axis": "lab_unused", "system": False, "expert": False},
        {"axis": "lab_unused", "system": False, "expert": False},
        {"axis": "drug_label", "system": True, "expert": True},
        {"axis": "drug_label", "system": False, "expert": False},
    ]
    rep = agreement_report(pairs)
    assert rep["overall"]["n"] == 6
    assert rep["overall"]["agreement_pct"] is not None
    assert "lab_unused" in rep["by_axis"]
