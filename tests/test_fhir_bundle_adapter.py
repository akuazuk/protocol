"""Тесты адаптера FHIR BY Bundle → КЗ."""
from __future__ import annotations

from clinical_knowledge.fhir_bundle_adapter import (
    bundle_to_consultation_document,
    bundle_to_consultation_text,
)

SAMPLE_BUNDLE = {
    "resourceType": "Bundle",
    "type": "document",
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "birthDate": "1989-01-31",
                "gender": "male",
                "name": [{"family": "Иванов", "given": ["Василий", "Васильевич"]}],
            }
        },
        {
            "resource": {
                "resourceType": "Encounter",
                "actualPeriod": {"start": "2024-04-07T11:12:21Z"},
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "code": {
                    "coding": [
                        {
                            "system": "https://fhir.by/ValueSet/InternClassificDiseases10",
                            "code": "E11",
                        }
                    ],
                    "text": "E11 - Сахарный диабет 2 типа",
                },
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "code": {
                    "coding": [
                        {"system": "https://fhir.by/ValueSet/VitalSignsCodes", "code": "heart-rate"}
                    ]
                },
                "valueQuantity": {"value": 80, "unit": "/min"},
            }
        },
    ],
}


def test_bundle_to_text_contains_diagnosis_and_vitals():
    text = bundle_to_consultation_text(SAMPLE_BUNDLE)
    assert "E11" in text
    assert "диабет" in text.lower()
    assert "heart-rate" in text or "80" in text


def test_bundle_to_document_patient_fields():
    doc = bundle_to_consultation_document(SAMPLE_BUNDLE, consultation_id="b1")
    assert doc.consultation_id == "b1"
    assert doc.patient.sex == "male"
    assert doc.patient.birth_date is not None
    assert doc.source_file_type == "fhir_bundle"


def test_bundle_with_subjective_and_medication_in_text():
    bundle = {
        **SAMPLE_BUNDLE,
        "entry": list(SAMPLE_BUNDLE["entry"])
        + [
            {
                "resource": {
                    "resourceType": "Observation",
                    "meta": {
                        "profile": ["https://fhir.by/StructureDefinition/ObservationSubjective"]
                    },
                    "valueString": "Жалобы на жажду",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "encounter": {"reference": "Encounter/1"},
                    "medication": {"concept": {"text": "Метформин 500 мг"}},
                    "dosageInstruction": [{"text": "1 таблетка утром"}],
                }
            },
        ],
    }
    text = bundle_to_consultation_text(bundle)
    assert "жалоб" in text.lower()
    assert "метформин" in text.lower()
