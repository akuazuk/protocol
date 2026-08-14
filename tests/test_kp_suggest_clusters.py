from __future__ import annotations

import os
import time

from clinical_knowledge.case_protocol_suggest import suggest_protocols_for_case
from clinical_knowledge.kp_card_enrich import icd_codes_from_text
from clinical_knowledge.protocol_candidate_index import select_candidate_cards


def _suggest(code: str, *, age: int, dx: str = "") -> dict:
    os.environ["CASE_PROTOCOL_SUGGEST"] = "1"
    return suggest_protocols_for_case(
        clinical={
            "clinical_diagnosis": dx,
            "mis_diagnos": code,
            "patient_age_years": age,
            "visit_date": "2026-07-15",
        },
        record={"visit_id": "cluster", "date": "2026-07-15"},
        limit=3,
    )


def test_icd_codes_from_content_text() -> None:
    codes = icd_codes_from_text("Гастроэзофагеальная рефлюксная болезнь K21 K21.9 Функциональная диспепсия K30")
    assert "K21" in codes
    assert "K21.9" in codes
    assert "K30" in codes


def test_suggest_k21_hits_esophagus_stomach_kp() -> None:
    result = _suggest("K21.9", age=41)
    assert result["available"] is True
    blob = " ".join(str(item.get("source_path") or "") for item in result["items"]).lower()
    assert "пищевод" in blob or "желудк" in blob
    assert "паллиатив" not in blob
    assert "дет" not in blob or "вз" in blob


def test_suggest_e03_hits_thyroid_kp() -> None:
    result = _suggest("E03.9", age=45)
    assert result["available"] is True
    blob = " ".join(str(item.get("source_path") or "") for item in result["items"]).lower()
    assert "щитовид" in blob
    assert "гипотиреоз" in blob or "щитовид" in blob


def test_suggest_k30_hits_stomach_kp() -> None:
    result = _suggest("K30", age=40)
    assert result["available"] is True
    blob = " ".join(str(item.get("source_path") or "") for item in result["items"]).lower()
    assert "пищевод" in blob or "желудк" in blob or "диспепс" in blob


def test_suggest_a63_and_b07_stay_empty() -> None:
    for code in ("A63.0", "B07"):
        result = _suggest(code, age=30)
        blob = " ".join(str(item.get("source_path") or "") for item in result.get("items") or []).lower()
        assert "урологическими заболеваниями" not in blob
        assert "болезнями кожи" not in blob
        assert result["available"] is False or not result.get("items")


def test_candidate_prefilter_is_much_smaller_than_registry() -> None:
    from clinical_knowledge.loader import load_protocol_cards_registry

    n_all = len(load_protocol_cards_registry())
    cands = select_candidate_cards(diag_text="Гастроэзофагеальный рефлюкс", icd_list=["K21.9"])
    assert cands
    assert len(cands) < n_all * 0.25
    assert any("пищевод" in str(c.get("source_path") or "").lower() for c in cands)


def test_suggest_k21_is_fast_enough() -> None:
    started = time.perf_counter()
    _suggest("K21.9", age=41)
    elapsed = time.perf_counter() - started
    assert elapsed < 8.0
