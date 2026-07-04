"""Unit tests for clinical protocol source view."""
from __future__ import annotations

from clinical_knowledge.protocol_source_view import prepare_protocol_source_view


def _doc():
    return {
        "sections": {
            "any": [
                {
                    "chunk_type": "classification",
                    "section_title": "4. Для установления диагноза применяется классификация СЕАР",
                    "page_from": 4,
                    "text": (
                        "которая учитывает клинические проявления, этиологию, "
                        "анатомическую локализацию и патогенез заболевания."
                    ),
                },
                {
                    "chunk_type": "classification",
                    "section_title": "4. Для установления диагноза применяется классификация СЕАР",
                    "page_from": 4,
                    "text": (
                        "которая учитывает клинические проявления, этиологию, "
                        "анатомическую локализацию и патогенез заболевания."
                    ),
                },
                {
                    "chunk_type": "body",
                    "section_title": "Документ",
                    "page_from": 1,
                    "text": "Национальный правовой Интернет-портал Республики Беларусь, 23.07.2022",
                },
                {
                    "chunk_type": "diagnostics",
                    "section_title": "Обязательными диагностическими мероприятиями являются",
                    "page_from": 5,
                    "text": (
                        "общий анализ крови; биохимический анализ крови; УЗДС вен "
                        "нижних конечностей обеих ног."
                    ),
                    "lab_tests": ["ОАК", "БАК"],
                    "imaging": ["УЗДС"],
                },
            ]
        }
    }


def test_prepare_view_groups_by_chunk_type_and_dedupes() -> None:
    view = prepare_protocol_source_view(_doc())
    assert view["stats"]["shown_blocks"] >= 2
    # дубль + административный body отфильтрованы
    assert view["stats"]["filtered_blocks"] >= 2
    diagnosis = view["sections"].get("diagnosis") or []
    assert diagnosis and "СЕАР" in diagnosis[0]["lead"]
    diagnostics = view["sections"].get("diagnostics") or []
    assert diagnostics
    chips = {c["label"]: c["items"] for c in diagnostics[0]["entities"]}
    assert "Анализы" in chips
    assert "УЗДС" in chips.get("Визуализация", [])


def test_prepare_view_drops_administrative_only() -> None:
    doc = {
        "sections": {
            "any": [
                {
                    "chunk_type": "body",
                    "section_title": "ПОСТАНОВЛЯЕТ:",
                    "page_from": 1,
                    "text": "1. Утвердить клинический протокол (прилагается).",
                }
            ]
        }
    }
    view = prepare_protocol_source_view(doc)
    assert view["stats"]["shown_blocks"] == 0
