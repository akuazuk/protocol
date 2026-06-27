"""Protocol age/specialty filter for B2C."""
from __future__ import annotations

from clinical_knowledge.patient_context import extract_patient_context
from clinical_knowledge.patient_protocol_filter import filter_l1_protocols, should_reject_protocol


def test_reject_pediatric_for_adult() -> None:
    ctx = {"age_group": "adult", "specialty": "neurology", "icd10_codes": ["M53.0"]}
    reject, reason = should_reject_protocol(
        path="minzdrav/kp_insult_u_detey.pdf",
        title="КП диагностика инсульта у детей (дет нас)",
        patient_context=ctx,
    )
    assert reject is True
    assert reason


def test_filter_l1_removes_pediatric_match() -> None:
    l1 = {
        "structured_analysis": {
            "matches": [
                {"title": "КП инсульт у детей", "source_path": "x/kp_detsk_insult.pdf", "match_score": 90},
                {"title": "КП шейно-черепной синдром взр нас", "source_path": "x/kp_spine_adult.pdf", "match_score": 70},
            ],
        },
        "alignment": {"alignment_cards": []},
    }
    ctx = {"age_group": "adult", "specialty": "neurology", "icd10_codes": ["M53.0"]}
    out = filter_l1_protocols(l1, ctx)
    matches = out["structured_analysis"]["matches"]
    assert len(matches) == 1
    assert "взр" in matches[0]["title"].lower() or "spine" in matches[0]["source_path"]


def test_extract_adult_context_from_kz() -> None:
    kz = "Дата рождения: 12.05.1965\nДиагноз: M53.0\nВрач-невролог"
    l1 = {"structured_analysis": {"document": {"patient": {}}}}
    ctx = extract_patient_context(l1, kz_text=kz, demographics_meta={"age_years": 61})
    assert ctx["age_group"] == "adult"
    assert ctx["specialty"] == "neurology"
    assert "M53.0" in ctx["icd10_codes"]
