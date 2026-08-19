"""Shadow label-check Rceth: не штрафует overall и не идёт в очередь."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.rceth_label_findings import (
    evaluate_rceth_label_findings,
    merge_rceth_label_into_findings,
)
from clinical_knowledge.rceth_sync.label_ctx import build_label_ctx, lookup_label
from clinical_knowledge.rceth_sync.label_parse import build_label_record

FIX = Path(__file__).resolve().parent / "fixtures" / "rceth"


def _ibu_label() -> dict:
    text = (FIX / "oxlp_ibuprofen_sample.txt").read_text(encoding="utf-8")
    return build_label_record(
        reg_id="11349_24",
        text=text,
        meta={
            "trade_name_ru": "ИБУПРОФЕН ДАНСОН",
            "inn": "Ibuprofen",
            "form_text": "суспензия 100мг/5мл",
            "status": "active",
            "term_from": "2024-10-23",
        },
    )


def _ctx() -> dict:
    return build_label_ctx([_ibu_label()])


def _case(**over: object) -> dict:
    row = {
        "treatment_recommendations": "Ибупрофен Дансон 5 мл 3 раза в день",
        "clinical_diagnosis": "Острая респираторная инфекция, лихорадка",
        "patient_age_years": 6,
        "mkb_code_main": "J06.9",
    }
    row.update(over)
    return row


def test_lookup_by_inn_and_form():
    ctx = _ctx()
    rec = lookup_label(ctx, "ibuprofen", "suspension")
    assert rec and rec["reg_id"] == "11349_24"
    assert lookup_label(ctx, "meloxicam") is None


def test_age_below_contra_minimum():
    findings = evaluate_rceth_label_findings(_case(patient_age_years=0), _ctx())
    codes = {f["code"] for f in findings}
    assert "C_rceth_age_outside_label" in codes
    assert all(f.get("shadow") is True for f in findings)
    assert all(f.get("severity") == "P2" for f in findings)


def test_adult_outside_pediatric_posology():
    findings = evaluate_rceth_label_findings(_case(patient_age_years=41), _ctx())
    assert any(f["code"] == "C_rceth_age_outside_label" for f in findings)


def test_matching_indication_no_off_label():
    findings = evaluate_rceth_label_findings(_case(), _ctx())
    assert not any(f["code"] == "C_rceth_off_label" for f in findings)


def test_off_label_when_dx_unrelated():
    findings = evaluate_rceth_label_findings(
        _case(clinical_diagnosis="Закрытый перелом бедра", mkb_code_main="S72.0"),
        _ctx(),
    )
    hit = [f for f in findings if f["code"] == "C_rceth_off_label"]
    assert hit
    assert "ibuprofen" in hit[0]["detail_ru"].lower()
    assert "инструкция rceth" in hit[0]["detail_ru"]


def test_contraindication_ulcer_in_dx():
    findings = evaluate_rceth_label_findings(
        _case(clinical_diagnosis="Язвенная болезнь желудка, обострение", mkb_code_main="K25"),
        _ctx(),
    )
    assert any(f["code"] == "C_rceth_contraindication" for f in findings)


def test_needs_human_without_sections_skipped():
    bare = {
        "reg_id": "x",
        "status": "active",
        "inn": "Ibuprofen",
        "sections": {},
        "parse": {"ok": False, "needs_human": True},
    }
    findings = evaluate_rceth_label_findings(_case(), build_label_ctx([bare]))
    assert findings == []


def test_merge_sets_is_shadow():
    merged = merge_rceth_label_into_findings(
        [],
        _case(patient_age_years=0),
        label_ctx=_ctx(),
    )
    assert merged
    assert all(item.get("is_shadow") or item.get("shadow") for item in merged)


def test_shadow_not_in_action_queue():
    from clinical_knowledge.mo_action_queue_select import signal_band_for_finding

    assert (
        signal_band_for_finding(
            {"code": "C_rceth_off_label", "severity": "P2", "is_shadow": True}
        )
        is None
    )


def test_evaluate_kz_deep_keeps_primary_score(monkeypatch):
    monkeypatch.setenv("MO_CONCORDANCE_FINDINGS", "0")
    monkeypatch.setenv("MO_ICD_NAME_MATCH", "0")
    monkeypatch.setenv("MO_ICD_DIRECTORY_EVAL", "0")
    monkeypatch.setenv("MO_PATIENT_HISTORY", "0")
    from clinical_knowledge.kz_deep_eval import evaluate_kz_deep

    case = _case(
        clinical_diagnosis="Закрытый перелом бедра",
        mkb_code_main="S72.0",
        patient_age_years=41,
        complaints="боль в бедре",
        objective_status="без особенностей",
        exam_recommendations="",
    )
    empty = evaluate_kz_deep(case, label_ctx={"by_inn": {}})
    filled = evaluate_kz_deep(case, label_ctx=_ctx())
    assert empty["overall_pct"] == filled["overall_pct"]
    primary_codes = {f["code"] for f in filled["findings"]}
    assert not any(c.startswith("C_rceth_") for c in primary_codes)
    shadow_codes = {f["code"] for f in filled["shadow_findings"]}
    assert any(c.startswith("C_rceth_") for c in shadow_codes)
