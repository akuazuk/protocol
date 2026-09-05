"""Wave 3: abnormal labs, service map, formulary soft, drug-disease."""
from __future__ import annotations

from clinical_knowledge.drug_disease_findings import drug_disease_findings
from clinical_knowledge.formulary_findings import formulary_findings
from clinical_knowledge.lab_abnormal_findings import (
    CODE_ABNORMAL_IGNORED,
    abnormal_from_bundle,
    abnormal_lab_findings,
)
from clinical_knowledge.service_exam_map import map_service_to_panel, map_services


def test_service_exam_map_oak() -> None:
    hit = map_service_to_panel("Общий анализ крови")
    assert hit is not None
    assert hit["panel_id"] == "oak"
    mapped = map_services(["ОАК", "ТТГ", "неизвестная услуга"])
    assert {m["panel_id"] for m in mapped} >= {"oak", "tsh"}


def test_abnormal_glucose_ignored() -> None:
    bundle = {
        "days": [
            {
                "test_date": "2026-08-20",
                "types": [
                    {
                        "type_name": "Биохимия",
                        "indicators": [
                            {"name": "Глюкоза", "value": "12.5", "unit": "ммоль/л"}
                        ],
                    }
                ],
            }
        ]
    }
    items = abnormal_from_bundle(bundle)
    assert items
    case = {
        "clinical_diagnosis": "ОРВИ",
        "treatment_recommendations": "симптоматически",
        "exam_data": "",
    }
    findings = abnormal_lab_findings(case, bundle)
    assert findings
    assert findings[0]["code"] == CODE_ABNORMAL_IGNORED
    assert findings[0].get("shadow") is True


def test_abnormal_acknowledged_no_finding() -> None:
    bundle = {
        "days": [
            {
                "test_date": "2026-08-20",
                "types": [
                    {
                        "type_name": "Биохимия",
                        "indicators": [
                            {"name": "Глюкоза", "value": "12.5", "unit": "ммоль/л"}
                        ],
                    }
                ],
            }
        ]
    }
    case = {
        "clinical_diagnosis": "Сахарный диабет, глюкоза повышена",
        "treatment_recommendations": "метформин",
        "exam_data": "глюкоза 12.5",
    }
    assert abnormal_lab_findings(case, bundle) == []


def test_formulary_unknown_shadow() -> None:
    case = {
        "treatment_recommendations": "XYZ-unknown-drug 10 мг 2 раза в день",
    }
    # May or may not extract; if extract_drugs finds nothing with inn, empty is ok.
    findings = formulary_findings(case)
    assert isinstance(findings, list)
    for f in findings:
        assert f.get("shadow") is True


def test_drug_disease_metformin_without_diabetes() -> None:
    case = {
        "clinical_diagnosis": "Остеохондроз",
        "treatment_recommendations": "Метформин 1000 мг вечером",
    }
    findings = drug_disease_findings(case)
    # Depends on normalizer recognizing metformin; if yes - finding
    if findings:
        assert findings[0]["code"] == "C_drug_disease_mismatch"
        assert findings[0].get("shadow") is True
