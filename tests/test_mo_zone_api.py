"""Контракт zones в overview / payload."""
from __future__ import annotations

from clinical_knowledge.mo_zone_scores import compute_mo_zone_scores, zones_api_payload


def test_zones_api_payload_shape() -> None:
    zones = compute_mo_zone_scores(
        {
            "clinical": {
                "complaints": "Боль в горле локально 3 дня, интенсивность умеренная",
                "anamnesis_doctor": "Болеет третий день. Курение отрицает. Аллергии нет.",
                "objective_status": "Состояние удовлетворительное. Зев гиперемирован. ЧСС 72.",
                "clinical_diagnosis": "Острый тонзиллит J03.9",
                "exam_recommendations": "ОАК",
                "treatment_recommendations": "Полоскание, явка через 5 дней",
            },
            "meta": {
                "visit_date": "2026-08-02",
                "visit_time": "10:00",
                "diagnosis_code": "J03.9",
            },
            "document_kind": "clinical_visit",
        }
    )
    payload = zones_api_payload(zones)
    assert payload["ok"] is True
    assert payload["engine"] == "mo_zones_v1"
    assert payload["zone1"]["label_ru"] == "Оформление"
    assert payload["zone2a"]["label_ru"] == "Диагноз"
    assert payload["zone2b"]["label_ru"] == "План по протоколу"
    assert "attention_primary" in payload
    assert isinstance(payload["criteria"], list)
    assert len(payload["criteria"]) == 13
