"""B2C protocol crosscheck (phase 2c)."""
from __future__ import annotations

from clinical_knowledge.patient_protocol_crosscheck import crosscheck_protocol_requirements


def test_crosscheck_finds_missing_required_exam() -> None:
    l1 = {
        "structured_analysis": {
            "matches": [{"title": "КП флеботромбоз"}],
            "compliance": {
                "exam_assessments": [
                    {
                        "exam_name": "УЗИ вен нижних конечностей",
                        "status": "missing_required",
                        "protocol_evidence": ["УЗИ вен обязательно при подозрении на ТГВ."],
                    },
                    {
                        "exam_name": "ОАК",
                        "status": "present",
                    },
                ],
            },
        },
    }
    kz = "Диагноз: флеботромбоз. Лечение: антикоагулянты."
    out = crosscheck_protocol_requirements(l1_result=l1, kz_text=kz, lab_text="")
    missing = out["missing_recommended_exams"]
    assert len(missing) == 1
    assert missing[0]["exam_name"] == "УЗИ вен нижних конечностей"
    assert missing[0]["severity"] == "high"
    assert out["protocol_title"] == "КП флеботромбоз"
    assert out["notes_ru"]


def test_crosscheck_skips_exam_mentioned_in_kz() -> None:
    l1 = {
        "structured_analysis": {
            "compliance": {
                "exam_assessments": [
                    {
                        "exam_name": "ОАК",
                        "status": "missing_required",
                        "protocol_evidence": [],
                    },
                ],
            },
        },
    }
    kz = "Назначено: ОАК, биохимия."
    out = crosscheck_protocol_requirements(l1_result=l1, kz_text=kz)
    assert out["missing_recommended_exams"] == []
