"""P1/P2: name_only сверка диагноза со справочником МКБ (без кодов)."""
from __future__ import annotations

from clinical_knowledge.mo_icd_name_match import (
    NAME_OK,
    NAME_REVIEW,
    evaluate_diagnosis_name_only,
    merge_icd_name_match_into_findings,
)


def _suggest_factory(mapping: dict[str, list[dict]]):
    def _suggest(text: str, max_results: int = 8):
        low = (text or "").lower()
        for key, rows in mapping.items():
            if key in low:
                return rows[:max_results]
        return []

    return _suggest


def test_name_only_ok_ignores_wrong_code_in_text(monkeypatch) -> None:
    monkeypatch.setattr(
        "icd_mkb.suggest_icd_from_russian",
        _suggest_factory(
            {
                "цистит": [
                    {
                        "code": "N30.0",
                        "title_ru": "N30.0 - Острый цистит",
                        "score": 0.55,
                        "match_method": "lexicon_ru",
                    }
                ]
            }
        ),
    )
    # В тексте чужой код - name_only всё равно ok по названию
    result = evaluate_diagnosis_name_only("Острый цистит I21.0")
    assert result["verdict"] == "ok"
    assert result["name_fit"] >= NAME_OK
    assert result["findings"] == []
    assert result["best_code"] == "N30.0"


def test_name_only_fail_on_garbage(monkeypatch) -> None:
    monkeypatch.setattr("icd_mkb.suggest_icd_from_russian", lambda text, max_results=8: [])
    result = evaluate_diagnosis_name_only("ааа ббб ввв без нозологии")
    assert result["verdict"] == "fail"
    assert any(f["code"] == "B_icd_name_no_match" for f in result["findings"])


def test_name_only_weak_match_typo(monkeypatch) -> None:
    monkeypatch.setattr(
        "icd_mkb.suggest_icd_from_russian",
        _suggest_factory(
            {
                "цисти": [
                    {
                        "code": "N30.0",
                        "title_ru": "Острый цистит",
                        "score": 0.2,
                        "match_method": "lexicon_ru",
                    }
                ]
            }
        ),
    )
    result = evaluate_diagnosis_name_only("острый циститт")
    assert result["name_fit"] >= NAME_REVIEW
    # точная опечатка может дать ok или review - не fail
    assert result["verdict"] in {"ok", "review"}
    if result["verdict"] == "review":
        assert any(f["code"] == "B_icd_name_weak_match" for f in result["findings"])


def test_merge_name_match_shadow_default(monkeypatch) -> None:
    monkeypatch.setenv("MO_ICD_NAME_MATCH", "1")
    monkeypatch.setenv("MO_ICD_NAME_IN_PRIMARY", "0")
    monkeypatch.setattr(
        "clinical_knowledge.mo_icd_name_match.evaluate_mo_icd_name_match",
        lambda case: [
            {
                "code": "B_icd_name_no_match",
                "shadow": True,
                "title_ru": "нет",
                "severity": "P2",
            }
        ],
    )
    merged = merge_icd_name_match_into_findings([], {"clinical_diagnosis": "xyz"})
    assert len(merged) == 1
    assert merged[0]["shadow"] is True
    again = merge_icd_name_match_into_findings(merged, {"clinical_diagnosis": "xyz"})
    assert len(again) == 1
