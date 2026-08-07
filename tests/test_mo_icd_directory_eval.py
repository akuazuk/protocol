"""P1/P2: оценка диагноза ↔ справочник МКБ (shadow)."""
from __future__ import annotations

from clinical_knowledge.mo_icd_directory_eval import (
    TEXT_FIT_OK,
    TEXT_FIT_REVIEW,
    evaluate_diagnosis_against_icd_directory,
    merge_icd_directory_into_findings,
    title_match_score,
)


def test_title_match_thresholds_aligned_with_consult() -> None:
    assert TEXT_FIT_OK == 0.35
    assert TEXT_FIT_REVIEW == 0.25
    assert title_match_score("острый цистит", "Острый цистит") >= TEXT_FIT_OK


def test_directory_eval_ok_when_text_matches_known_code(monkeypatch) -> None:
    monkeypatch.setattr(
        "icd_mkb.is_code_in_ru_reference",
        lambda code: str(code).upper().startswith("N30"),
    )
    monkeypatch.setattr("icd_mkb.ru_title", lambda code: "Острый цистит")
    monkeypatch.setattr("icd_mkb.suggest_icd_from_russian", lambda text, max_results=8: [])
    result = evaluate_diagnosis_against_icd_directory("Острый цистит", ["N30.0"])
    assert result["code_in_directory"] is True
    assert result["text_rubric_fit"] >= TEXT_FIT_OK
    assert result["verdict"] == "ok"
    assert result["findings"] == []


def test_directory_eval_code_unknown(monkeypatch) -> None:
    monkeypatch.setattr("icd_mkb.is_code_in_ru_reference", lambda code: False)
    monkeypatch.setattr("icd_mkb.ru_title", lambda code: None)
    monkeypatch.setattr(
        "icd_mkb.suggest_icd_from_russian",
        lambda text, max_results=8: [
            {"code": "M08.0", "title_ru": "Ювенильный артрит", "score": 0.5, "match_method": "lexicon_ru"}
        ],
    )
    result = evaluate_diagnosis_against_icd_directory("Ювенильный артрит", ["ZZ99.9"])
    codes = {f["code"] for f in result["findings"]}
    assert "B_icd_dir_code_unknown" in codes
    assert result["directory_hit"] is True


def test_directory_eval_no_match_on_garbage_text(monkeypatch) -> None:
    monkeypatch.setattr("icd_mkb.suggest_icd_from_russian", lambda text, max_results=8: [])
    monkeypatch.setattr("icd_mkb.is_code_in_ru_reference", lambda code: False)
    result = evaluate_diagnosis_against_icd_directory("ааа ббб ввв ггг без нозологии", [])
    assert result["directory_hit"] is False
    assert any(f["code"] == "B_icd_dir_no_match" for f in result["findings"])
    assert result["verdict"] == "fail"


def test_directory_eval_text_mismatch(monkeypatch) -> None:
    monkeypatch.setattr("icd_mkb.is_code_in_ru_reference", lambda code: True)
    monkeypatch.setattr("icd_mkb.ru_title", lambda code: "Инфаркт миокарда")
    monkeypatch.setattr("icd_mkb.suggest_icd_from_russian", lambda text, max_results=8: [])
    result = evaluate_diagnosis_against_icd_directory("Хронический тонзиллит", ["I21.0"])
    assert result["text_rubric_fit"] < TEXT_FIT_REVIEW
    assert any(f["code"] == "B_icd_dir_text_mismatch" for f in result["findings"])


def test_merge_icd_directory_shadow_default(monkeypatch) -> None:
    monkeypatch.setenv("MO_ICD_DIRECTORY_EVAL", "1")
    monkeypatch.setenv("MO_ICD_DIR_IN_PRIMARY", "0")
    monkeypatch.setattr(
        "clinical_knowledge.mo_icd_directory_eval.evaluate_mo_icd_directory",
        lambda case: [
            {
                "code": "B_icd_dir_no_match",
                "shadow": True,
                "title_ru": "нет в справочнике",
                "severity": "P2",
            }
        ],
    )
    merged = merge_icd_directory_into_findings([], {"clinical_diagnosis": "xyz"})
    assert len(merged) == 1
    assert merged[0]["shadow"] is True
    again = merge_icd_directory_into_findings(merged, {"clinical_diagnosis": "xyz"})
    assert len(again) == 1


def test_p2_calibration_three_etalons_directory(monkeypatch) -> None:
    """P2: три эталона - цистит ok, мусор fail, код без текста review/mismatch."""
    def _ref(code: str) -> bool:
        return code.upper() in {"N30.0", "I21.0"}

    def _title(code: str) -> str | None:
        return {"N30.0": "Острый цистит", "I21.0": "Острый инфаркт миокарда"}.get(code.upper())

    monkeypatch.setattr("icd_mkb.is_code_in_ru_reference", _ref)
    monkeypatch.setattr("icd_mkb.ru_title", _title)

    def _suggest(text: str, max_results: int = 8):
        low = (text or "").lower()
        if "цистит" in low:
            return [{"code": "N30.0", "title_ru": "Острый цистит", "score": 0.6, "match_method": "lexicon_ru"}]
        return []

    monkeypatch.setattr("icd_mkb.suggest_icd_from_russian", _suggest)

    cystitis = evaluate_diagnosis_against_icd_directory("Острый цистит", ["N30.0"])
    assert cystitis["verdict"] == "ok"

    garbage = evaluate_diagnosis_against_icd_directory("тест тест тест плейсхолдер", [])
    assert garbage["verdict"] == "fail"

    mismatch = evaluate_diagnosis_against_icd_directory("Хронический тонзиллит", ["I21.0"])
    assert any(f["code"] == "B_icd_dir_text_mismatch" for f in mismatch["findings"])
