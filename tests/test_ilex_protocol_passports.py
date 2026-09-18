from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.ilex_protocol_passports import (
    extract_icd_chapter1,
    infer_audience,
    match_passports_for_diagnosis,
    overlay_ilex_passports,
    parse_approval_from_name,
    parse_ilex_document,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "ilex_sample" / "pages" / "0.html"


def test_parse_approval_and_audience() -> None:
    date, number = parse_approval_from_name(
        'Постановление ... от 25.04.2026 N 38 "Об утверждении клинического протокола"'
    )
    assert date == "2026-04-25"
    assert number == "38"
    assert infer_audience("Диагностика ... (взрослое население) с артериальной гипертензией") == "adult"
    assert infer_audience("... (детское население) с астмой") == "child"


def test_chapter1_icd_and_inner_title_from_synthetic_html() -> None:
    html = FIX.read_text(encoding="utf-8")
    assert extract_icd_chapter1(html)[:4] == ["I10", "I11", "I12", "I13"]
    rows = parse_ilex_document(
        meta={
            "ilex_id": "BELAW/1",
            "bank": "BELAW",
            "doc_id": 1,
            "name": (
                "Постановление Министерства здравоохранения Республики Беларусь "
                'от 25.04.2026 N 38 "Об утверждении клинического протокола" '
                '(вместе с "Клиническим протоколом "Диагностика и лечение пациентов '
                '(взрослое население) с артериальной гипертензией")'
            ),
            "status": "Действующий",
            "inner_protocols": [],
        },
        html_text=html,
    )
    assert len(rows) == 1
    row = rows[0]
    assert "артериальной гипертензией" in row["protocol_title"]
    assert row["icd10_primary"][:4] == ["I10", "I11", "I12", "I13"]
    assert row["audience"] == "adult"
    assert row["status"] == "active"
    assert row["approval_number"] == "38"
    assert any(ch.startswith("Глава 1") for ch in row["chapters"])


def test_overlay_replaces_truncated_title_and_fills_icd(tmp_path, monkeypatch) -> None:
    from clinical_knowledge import ilex_protocol_passports as mod

    path = tmp_path / "passports.jsonl"
    path.write_text(
        json.dumps(
            {
                "ilex_id": "BELAW/1",
                "protocol_title": "Диагностика и лечение пациентов (взрослое население) с артериальной гипертензией",
                "audience": "adult",
                "status": "active",
                "approval_date": "2026-04-25",
                "approval_number": "38",
                "approval_year": "2026",
                "icd10_primary": ["I10", "I11", "I12", "I13"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ILEX_PASSPORTS", "1")
    monkeypatch.setenv("ILEX_PASSPORTS_PATH", str(path))
    mod.clear_ilex_passport_cache()
    cards = [
        {
            "title": "КЛИНИЧЕСКИЙ ПРОТОКОЛ",
            "source_path": "minzdrav_protocols/kardiologiya/КП_артериальная_гипертензия_пост_2026_38.pdf",
            "approval": {"date": "2026-01-01", "number": "38"},
            "icd10_primary": [],
            "icd10_all": [],
            "status": "active",
        }
    ]
    overlay_ilex_passports(cards)
    assert "гипертензией" in cards[0]["title"]
    assert cards[0]["icd10_primary"][:4] == ["I10", "I11", "I12", "I13"]
    assert cards[0]["ilex_title_overlay"] is True
    other = {
        "title": "КЛИНИЧЕСКИЙ ПРОТОКОЛ",
        "source_path": "minzdrav_protocols/akusherstvo/омнибус_2026_38.pdf",
        "approval": {"date": "2026-01-01", "number": "38"},
        "icd10_primary": [],
        "status": "active",
    }
    overlay_ilex_passports([other])
    assert other["title"] == "КЛИНИЧЕСКИЙ ПРОТОКОЛ"
    assert not other.get("ilex_title_overlay")
    mod.clear_ilex_passport_cache()


def test_split_icd_per_inner_protocol() -> None:
    html = """
    <p><ogl-tag tag-level="1" tag-value="Клинический протокол &quot;с острым бронхитом&quot;"></ogl-tag></p>
    <p>шифр по Международной статистической классификации болезней и проблем, связанных со здоровьем, десятого пересмотра: J20 Острый бронхит. 2. Далее.</p>
    <p><ogl-tag tag-level="1" tag-value="Клинический протокол &quot;с хроническим бронхитом&quot;"></ogl-tag></p>
    <p>шифр по Международной статистической классификации болезней и проблем, связанных со здоровьем, десятого пересмотра: J41 Хронический бронхит. 2. Далее.</p>
    """
    rows = parse_ilex_document(
        meta={
            "ilex_id": "BELAW/2",
            "name": 'Постановление от 01.04.2026 N 64 "Об утверждении"',
            "status": "Действующий",
            "inner_protocols": ["с острым бронхитом", "с хроническим бронхитом"],
        },
        html_text=html,
    )
    by_title = {row["protocol_title"]: row["icd10_primary"] for row in rows}
    assert by_title["с острым бронхитом"][:1] == ["J20"]
    assert by_title["с хроническим бронхитом"][:1] == ["J41"]


def test_overlay_skips_unrelated_year(tmp_path, monkeypatch) -> None:
    from clinical_knowledge import ilex_protocol_passports as mod

    path = tmp_path / "passports.jsonl"
    path.write_text(
        json.dumps(
            {
                "protocol_title": "с артериальной гипертензией",
                "status": "active",
                "approval_number": "38",
                "approval_year": "2026",
                "icd10_primary": ["I10"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ILEX_PASSPORTS_PATH", str(path))
    monkeypatch.setenv("ILEX_PASSPORTS", "1")
    mod.clear_ilex_passport_cache()
    cards = [
        {
            "title": "КЛИНИЧЕСКИЙ ПРОТОКОЛ",
            "approval": {"date": "2018-02-19", "number": "38"},
            "icd10_primary": ["X00"],
        }
    ]
    overlay_ilex_passports(cards)
    assert cards[0]["title"] == "КЛИНИЧЕСКИЙ ПРОТОКОЛ"
    assert cards[0]["icd10_primary"] == ["X00"]
    mod.clear_ilex_passport_cache()


def test_match_passports_picks_hypertension_not_gsk(tmp_path, monkeypatch) -> None:
    from clinical_knowledge import ilex_protocol_passports as mod

    path = tmp_path / "passports.jsonl"
    rows = [
        {
            "protocol_title": "Диагностика и лечение пациентов (взрослое население) с артериальной гипертензией",
            "status": "active",
            "audience": "adult",
            "approval_year": "2026",
            "icd10_primary": ["I10", "I11"],
        },
        {
            "protocol_title": "Трансплантация гемопоэтических стволовых клеток пациентам (детское население) с первичными иммунодефицитами",
            "status": "active",
            "audience": "child",
            "approval_year": "2025",
            "icd10_primary": ["D82"],
        },
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("ILEX_PASSPORTS_PATH", str(path))
    monkeypatch.setenv("ILEX_PASSPORTS", "1")
    mod.clear_ilex_passport_cache()
    hits = match_passports_for_diagnosis(
        "Гипертоническая болезнь",
        icd_codes=["I11.9"],
        audience="adult",
        limit=3,
    )
    assert hits
    assert "гипертензией" in hits[0]["protocol_title"]
    assert "стволов" not in hits[0]["protocol_title"]
    mod.clear_ilex_passport_cache()
