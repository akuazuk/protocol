"""Rceth identity: бренд → INN / форма, без ложного override."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.drug_normalizer import clear_cache, extract_drugs, normalize_drug
from clinical_knowledge.rceth_sync.identity import (
    build_identity_index,
    canon_inn,
    form_keywords,
    is_combo_inn,
    load_identity_index,
    merge_brand_overrides,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "rceth"

# 20 белорусских брендов из seed / пилота (Фенибут, Ибупрофен Дансон, Нимесулид Фармлэнд).
_PILOT_BRANDS = {
    "фенибут": "phenibut",
    "ибупрофен дансон": "ibuprofen",
    "нимесулид фармлэнд": "nimesulide",
    "нимесулид белмед": "nimesulide",
    "кетонал": "ketoprofen",
    "мидокалм": "tolperisone",
    "милдронат": "meldonium",
    "эутирокс": "levothyroxine",
    "детралекс": "diosmin",
    "арбидол": "umifenovir",
    "мезим": "pancreatin",
    "диклофенак фармлэнд": "diclofenac",
    "мелоксикам фармлэнд": "meloxicam",
    "амелотекс": "meloxicam",
    "валсакор": "valsartan",
    "тромбонет": "clopidogrel",
    "предуктал": "trimetazidine",
    "флебодиа": "diosmin",
    "стопдиар": "nifuroxazide",
    "эспумизан": "simeticone",
}


def test_form_keywords_gel_and_suspension():
    assert "gel" in form_keywords("гель 2.5%")
    assert "suspension" in form_keywords("суспензия 100мг/5мл")
    assert "tablet" in form_keywords("таблетки 100мг")
    assert form_keywords("без формы") == []


def test_combo_inn_skipped():
    assert is_combo_inn("Ibuprofen + Paracetamol")
    assert is_combo_inn("ибупрофен и парацетамол")
    assert canon_inn("Ibuprofen + Paracetamol") is None
    assert canon_inn("Ibuprofen") == "ibuprofen"
    assert canon_inn("Нимесулид") == "nimesulide"


def test_ambiguous_brand_dropped():
    idx = build_identity_index(
        [
            {"trade_name_ru": "ОДИНБРЕНД", "inn": "Ibuprofen", "status": "active"},
            {"trade_name_ru": "ОДИНБРЕНД", "inn": "Diclofenac", "status": "active"},
        ]
    )
    assert "одинбренд" not in idx


def test_merge_keeps_curated_meloxicam():
    merged = merge_brand_overrides(
        {"мовалис": "meloxicam", "мелоксикам": "meloxicam"},
        {"мовалис": "diclofenac", "амелотекс": "meloxicam"},
    )
    assert merged["мовалис"] == "meloxicam"
    assert merged["мелоксикам"] == "meloxicam"
    assert merged["амелотекс"] == "meloxicam"


def test_search_fixture_identity():
    from clinical_knowledge.rceth_sync.parse import parse_search_results

    html = (FIX / "search_results_sample.html").read_text(encoding="utf-8")
    idx = build_identity_index(parse_search_results(html))
    assert idx["фенибут"]["inn"] == "phenibut"
    assert "powder" in idx["фенибут"]["forms"]
    assert idx["ибупрофен дансон"]["inn"] == "ibuprofen"
    assert "suspension" in idx["ибупрофен дансон"]["forms"]
    assert "старое" not in idx


def test_seed_twenty_belarus_brands():
    clear_cache()
    idx = load_identity_index()
    missing = [name for name in _PILOT_BRANDS if name not in idx]
    assert not missing, missing
    for brand, inn in _PILOT_BRANDS.items():
        assert idx[brand]["inn"] == inn, (brand, idx[brand])
        rec = normalize_drug(brand)
        assert rec["inn"] == inn, (brand, rec)


def test_rceth_brands_in_extract_and_forms():
    clear_cache()
    txt = "Кетонал гель местно; Ибупрофен Дансон 5 мл; Амелотекс при боли."
    inns = {d["inn"] for d in extract_drugs(txt)}
    assert "ketoprofen" in inns, inns
    assert "ibuprofen" in inns, inns
    assert "meloxicam" in inns, inns
    ket = normalize_drug("Кетонал")
    assert ket["inn"] == "ketoprofen"
    assert "gel" in (ket.get("forms") or [])


def test_curated_meloxicam_not_diclofenac_with_rceth_loaded():
    clear_cache()
    assert normalize_drug("мелоксикам")["inn"] == "meloxicam"
    assert normalize_drug("мовалис")["inn"] == "meloxicam"
    assert normalize_drug("диклофенак")["inn"] == "diclofenac"
    assert normalize_drug("мелоксикам фармлэнд")["inn"] == "meloxicam"
    inns = {
        d["inn"]
        for d in extract_drugs("Мовалис 7.5 мг; Мелоксикам Фармлэнд 15 мг; Вольтарен гель.")
    }
    assert "meloxicam" in inns
    assert "diclofenac" in inns
