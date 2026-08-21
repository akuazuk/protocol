from __future__ import annotations

from clinical_knowledge.kp_icd_passport import apply_icd_passport, suggest_icd_codes


def test_apply_icd_passport_moves_external_and_body_codes() -> None:
    card = {
        "icd10_primary": ["Y40", "Y45", "K29.5"],
        "icd10_all": ["Y40", "Y45", "K29.5", "K21.9"],
    }
    catalog = {"icd10_primary": ["K21.9", "K29.5", "Y40"], "icd10_all": ["K21.9", "K29.5", "Y40", "Y45"]}
    apply_icd_passport(card, catalog)
    assert "K21.9" in card["icd10_primary"]
    assert "K29.5" in card["icd10_primary"]
    assert "Y40" not in card["icd10_primary"]
    assert "Y45" in card["icd10_mentions"]
    assert "Y40" in card["icd10_mentions"]
    assert "Y45" not in suggest_icd_codes(card)
    assert "K21.9" in suggest_icd_codes(card)


def test_attach_content_does_not_fill_primary() -> None:
    card = {"icd10_primary": [], "icd10_all": [], "title": "x"}
    # content lookup may be empty; inject as if found
    card["icd10_mentions"] = []
    from clinical_knowledge import kp_card_enrich as enrich

    found = enrich.icd_codes_from_text("в тексте протокола Y45.1 и J06.9")
    assert "Y45.1" in found
    card["icd10_all"] = []
    extra = found
    card["icd10_mentions"] = extra
    apply_icd_passport(card)
    assert "Y45.1" not in card["icd10_primary"]
    assert "Y45.1" in card["icd10_mentions"]


def test_y45_alone_does_not_select_gi_card() -> None:
    from clinical_knowledge.loader import load_protocol_cards_registry
    from clinical_knowledge.protocol_candidate_index import _built_index, select_candidate_cards

    load_protocol_cards_registry.cache_clear()
    _built_index.cache_clear()
    cands = select_candidate_cards(diag_text="", icd_list=["Y45.1"])
    paths = " ".join(str(c.get("source_path") or "") for c in cands).lower()
    assert "пищевода_желудка" not in paths
