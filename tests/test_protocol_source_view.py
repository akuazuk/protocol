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
    assert "diagnosis" in (diagnosis[0].get("intent_tags") or [])
    assert "классификация" in (diagnosis[0].get("search_blob") or "")
    diagnostics = view["sections"].get("diagnostics") or []
    assert diagnostics
    assert "diagnostics" in (diagnostics[0].get("intent_tags") or [])
    chips = {c["label"]: c["items"] for c in diagnostics[0]["entities"]}
    assert "Анализы" in chips
    assert "УЗДС" in chips.get("Визуализация", [])


def test_treatment_items_searchable_by_drug_intent() -> None:
    doc = {
        "sections": {
            "any": [
                {
                    "chunk_type": "treatment",
                    "section_title": "При хроническом венозном отеке ФЛП назначаются курсами",
                    "page_from": 8,
                    "text": "на 3-6 месяцев и более не реже 2 раз в год.",
                    "drugs": ["диосмин", "гесперидин"],
                }
            ]
        }
    }
    view = prepare_protocol_source_view(doc)
    items = (view["sections"] or {}).get("treatment") or []
    assert items
    blob = items[0].get("search_blob") or ""
    assert "лекарства" in blob
    assert "диосмин" in blob
    assert "treatment" in (items[0].get("intent_tags") or [])


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
