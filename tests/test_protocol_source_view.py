"""Unit tests for clinical protocol source view."""
from __future__ import annotations

from clinical_knowledge.protocol_source_view import prepare_protocol_source_view


def test_prepare_view_filters_duplicates_and_noise() -> None:
    doc = {
        "sections": {
            "classification": [
                {
                    "section_title": "1. Настоящий клинический протокол устанавливает общие требования",
                    "page_from": 2,
                    "text": (
                        "оказания медицинской помощи пациентам с хроническими заболеваниями вен. "
                        "Оказание медицинской помощи осуществляется в соответствии с клиническим протоколом."
                    ),
                },
                {
                    "section_title": "4. Для установления диагноза применяется классификация СЕАР",
                    "page_from": 4,
                    "text": (
                        "4. Для установления диагноза применяется базовый вариант классификации СЕАР, "
                        "которая учитывает клинические проявления, этиологию, анатомическую локализацию "
                        "и патогенез заболевания."
                    ),
                },
                {
                    "section_title": "4. Для установления диагноза применяется классификация СЕАР",
                    "page_from": 4,
                    "text": (
                        "4. Для установления диагноза применяется базовый вариант классификации СЕАР, "
                        "которая учитывает клинические проявления, этиологию, анатомическую локализацию "
                        "и патогенез заболевания."
                    ),
                },
            ],
            "other": [
                {
                    "section_title": "Документ",
                    "page_from": 1,
                    "text": "Национальный правовой Интернет-портал Республики Беларусь, 23.07.2022, 8/38363",
                }
            ],
        }
    }
    view = prepare_protocol_source_view(doc)
    assert view["stats"]["shown_blocks"] >= 1
    assert view["stats"]["filtered_blocks"] >= 2
    diagnosis = view["sections"].get("diagnosis") or []
    assert diagnosis
    assert "СЕАР" in diagnosis[0]["lead"]
