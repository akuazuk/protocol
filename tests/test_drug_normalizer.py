"""Тесты нормализации ЛС рус/бренд → INN."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.drug_normalizer import extract_drugs, normalize_drug, transliterate


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


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
