"""Тесты рубрико-специфичных извлечений (ТЗ раздел 22)."""
from __future__ import annotations

from clinical_knowledge.rubric_extractors import (
    RUBRIC_TERMS,
    extract_measurements,
    extract_rubric_specifics,
    normalize_rubric_slug,
    rubric_slugs_from_matches,
)


def test_all_24_rubrics_present():
    assert len(RUBRIC_TERMS) == 24


def test_oncology_tnm_and_stage():
    text = "Диагноз: рак желудка, стадия III, cT3N1M0, гистология аденокарцинома. Химиотерапия по схеме."
    res = extract_rubric_specifics(text, ["novoobrazovaniya"])
    meas = res["measurements"]
    assert "tnm" in meas
    assert meas["tnm"]["value"].lower().startswith("ct3n1m0")
    assert "stage" in meas
    info = res["by_rubric"]["novoobrazovaniya"]
    assert "стади" in info["matched_terms"]
    assert "химиотерап" in info["matched_terms"]


def test_endocrinology_hba1c():
    text = "Сахарный диабет 2 типа. HbA1c 8.5%. Назначена сахароснижающая терапия."
    res = extract_rubric_specifics(text, ["endokrinologiya-narusheniya-obmena-veshchestv"])
    assert res["measurements"]["hba1c"]["value"] == "8.5"
    assert res["measurements"]["hba1c"]["unit"] == "%"


def test_pulmonology_spiro_spo2():
    text = "Жалобы на кашель и одышку. Спирометрия: ОФВ1 65%. Сатурация 94%."
    res = extract_rubric_specifics(text, ["pulmonologiya-ftiziatriya"])
    meas = res["measurements"]
    assert meas["fev1"]["value"] == "65"
    assert meas["spo2"]["value"] == "94"


def test_obstetrics_gestational_weeks():
    text = "Беременность 32 недели. Угроза преждевременных родов."
    res = extract_rubric_specifics(text, ["akusherstvo-ginekologiya"])
    assert res["measurements"]["gestational_weeks"]["value"] == "32"


def test_rheumatology_das28_crp_esr():
    text = "Ревматоидный артрит. DAS28 5.2. СРБ 24 мг/л. СОЭ 38 мм/ч."
    meas = extract_measurements(text)
    assert meas["das28"]["value"] == "5.2"
    assert meas["crp"]["value"] == "24"
    assert meas["esr"]["value"] == "38"


def test_normalize_and_from_matches():
    assert normalize_rubric_slug("minzdrav_protocols/khirurgiya/file.pdf") == "khirurgiya"
    assert normalize_rubric_slug(None) is None
    matches = [{"source_path": "minzdrav_protocols/nefrologiya/a.pdf"}, {"source_path": "x/khirurgiya/b.pdf"}]
    slugs = rubric_slugs_from_matches(matches)
    assert "nefrologiya" in slugs and "khirurgiya" in slugs


def test_auto_rubric_when_no_slug():
    text = "Острота зрения снижена, внутриглазное давление повышено, глаукома."
    res = extract_rubric_specifics(text)
    assert "oftalmologiya" in res["rubrics"]


def test_resilient_empty():
    res = extract_rubric_specifics("")
    assert res["measurements"] == {}
    assert res["rubrics"] == []
