"""Тесты нормализации ЛС рус/бренд → INN."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.drug_normalizer import (
    clear_cache,
    extract_drugs,
    normalize_drug,
    transliterate,
)


def test_override_brand():
    assert normalize_drug("Ксарелто")["inn"] == "rivaroxaban"
    assert normalize_drug("но-шпа")["inn"] == "drotaverine"
    assert normalize_drug("конкор")["inn"] == "bisoprolol"


def test_transliterate_fuzzy():
    r = normalize_drug("амоксициллин")
    assert r["inn"] == "amoxicillin", r
    r2 = normalize_drug("варфарин")
    assert r2["inn"] == "warfarin", r2


def test_unresolved_low_conf():
    r = normalize_drug("нечтонепонятное")
    assert r["inn"] is None
    assert r["confidence"] == 0.0


def test_translit_basic():
    assert transliterate("аспирин") == "aspirin"


def test_extract_from_treatment_text():
    txt = "Рекомендовано: Амоксициллин 500 мг 3 раза в день внутрь; Омез 20 мг утром; Ксарелто 20 мг."
    drugs = extract_drugs(txt)
    inns = {d["inn"] for d in drugs}
    assert "amoxicillin" in inns, inns
    assert "omeprazole" in inns, inns
    assert "rivaroxaban" in inns, inns


def test_multiword_override():
    r = normalize_drug("калия хлорид")
    assert r["inn"] == "potassium chloride", r


def test_meloxicam_not_diclofenac_override():
    """Регрессия 3600047: мелоксикам не должен мапиться в diclofenac через STOPP-seed."""
    clear_cache()
    assert normalize_drug("мелоксикам")["inn"] == "meloxicam"
    assert normalize_drug("мовалис")["inn"] == "meloxicam"
    assert normalize_drug("диклофенак")["inn"] == "diclofenac"
    assert normalize_drug("эсциталопрам")["inn"] == "escitalopram"
    assert normalize_drug("суматриптан")["inn"] == "sumatriptan"
    assert normalize_drug("ципралекс")["inn"] == "escitalopram"


def test_extract_visit_3600047_plan_drugs():
    """План мигрени: SSRI + триптан + мелоксикам, без ложного diclofenac."""
    clear_cache()
    txt = (
        "Эсциталопрам 10 мг утром постоянно; Суматриптан 50 мг при приступе мигрени; "
        "Мелоксикам 7,5 мг при головной боли."
    )
    inns = {d["inn"] for d in extract_drugs(txt)}
    assert "escitalopram" in inns, inns
    assert "sumatriptan" in inns, inns
    assert "meloxicam" in inns, inns
    assert "diclofenac" not in inns, inns


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
