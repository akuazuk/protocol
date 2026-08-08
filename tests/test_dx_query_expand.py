from __future__ import annotations

from clinical_knowledge.dx_query_expand import (
    bridge_icd_candidates,
    expand_diagnosis_query,
    strip_icd_tokens,
)


def test_strip_icd_tokens_removes_codes() -> None:
    assert "M21" not in strip_icd_tokens("M21.0 Вальгусная деформация")
    assert "вальгусная" in strip_icd_tokens("M21.0 Вальгусная деформация").lower()


def test_expand_flatfoot_aliases() -> None:
    q = expand_diagnosis_query("Плосковальгусная установка стоп")
    low = q.lower()
    assert "плоская стопа" in low or "pes planus" in low
    assert "вальгус" in low


def test_bridge_flatfoot_prefers_m21_or_q66() -> None:
    cands = bridge_icd_candidates(
        "Плосковальгусная установка стоп с вальгированием пяточных костей"
    )
    codes = {c["code"] for c in cands}
    assert codes & {"M21.0", "M21.4", "Q66.5", "Q66.9"}
    assert not any(c["code"].startswith(("C", "T", "W")) for c in cands)


def test_bridge_orvi_stays_on_j06_family() -> None:
    cands = bridge_icd_candidates("ОРВИ")
    assert cands
    assert all(str(c["code"]).startswith("J") for c in cands)
    assert any(c["code"].startswith("J06") or c["code"].startswith("J39") for c in cands)
