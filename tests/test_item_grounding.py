"""Трек 2: grounding пунктов (препараты/обследования) с вероятностью и цитатой."""
from __future__ import annotations

from clinical_knowledge.item_grounding import (
    build_extraction_grounding,
    ground_items,
    score_item,
)


CHUNKS = [
    {
        "text": (
            "Диагностика включает общий анализ крови и рентгенографию органов "
            "грудной клетки. При подтверждении диагноза назначают амоксициллин "
            "500 мг три раза в сутки в течение 7 дней."
        ),
        "page_from": 4,
    },
    {
        "text": "Наблюдение за пациентом проводится врачом по месту жительства.",
        "page_from": 5,
    },
]


def test_present_item_has_high_support_and_quote() -> None:
    rows = ground_items(["Амоксициллин 500 мг 3 раза в сутки"], CHUNKS)
    r = rows[0]
    assert r["support"] >= 0.34
    assert r["verified"] is True
    assert r["page"] == 4
    assert "амоксициллин" in r["quote"].lower()


def test_absent_item_low_support_not_verified() -> None:
    rows = ground_items(["Цефтриаксон внутривенно 2 г"], CHUNKS)
    r = rows[0]
    assert r["verified"] is False
    assert r["support"] < 0.34


def test_obligation_from_profile_floors_support() -> None:
    profile = [{"text": "рентгенография органов грудной клетки", "obligation": "required"}]
    rows = ground_items(
        ["Рентгенография ОГК"], CHUNKS, profile_items=profile
    )
    r = rows[0]
    assert r["obligation"] == "required"
    assert r["support"] >= 0.6
    assert r["source"] == "icd_profile"


def test_build_extraction_grounding_summary() -> None:
    ext = {
        "medications": ["Амоксициллин 500 мг"],
        "investigations": ["Общий анализ крови", "МРТ головного мозга"],
        "treatment_methods": [],
    }
    g = build_extraction_grounding(ext, CHUNKS)
    assert g["summary"]["items"] == 3
    assert g["summary"]["verified"] >= 2
    assert set(g.keys()) >= {"medications", "investigations", "treatment_methods", "summary"}
    inv = {r["text"]: r for r in g["investigations"]}
    assert inv["Общий анализ крови"]["verified"] is True
    assert inv["МРТ головного мозга"]["verified"] is False


def test_score_item_empty_tokens() -> None:
    r = score_item("и в на", [("текст протокола о лечении", 1)])
    assert r["support"] == 0.0


def test_abbreviation_matches_full_form() -> None:
    chunks = [
        {"text": "Проводится общий анализ крови и общий анализ мочи.", "page_from": 3},
    ]
    rows = ground_items(["ОАК", "ОАМ"], chunks)
    assert rows[0]["verified"] is True
    assert rows[1]["verified"] is True


def test_full_form_matches_abbreviation_in_text() -> None:
    chunks = [{"text": "Назначают ОАК и УЗИ органов брюшной полости.", "page_from": 2}]
    rows = ground_items(["Общий анализ крови", "УЗИ ОБП"], chunks)
    assert rows[0]["verified"] is True
    assert rows[1]["verified"] is True
