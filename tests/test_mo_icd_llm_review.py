"""Фаза 4: LLM grey-zone ICD review (без живого Gemini)."""
from __future__ import annotations

import json

import pytest

from clinical_knowledge.mo_icd_llm_review import (
    build_llm_review_pack,
    findings_from_review,
    icd_llm_review_enabled,
    review_one,
    validate_llm_review,
)


def test_flag_default_off(monkeypatch) -> None:
    monkeypatch.delenv("MO_ICD_LLM_REVIEW", raising=False)
    assert icd_llm_review_enabled() is False


def test_validate_llm_review_contract() -> None:
    v = validate_llm_review(
        {
            "agree": "partial",
            "reason_ru": "код подходит, формулировка шире рубрики",
            "suggested_code": "K29.3",
        }
    )
    assert v["agree"] == "partial"
    assert v["suggested_code"] == "K29.3"
    assert len(v["reason_ru"]) <= 160


def test_validate_rejects_bad_agree() -> None:
    with pytest.raises(ValueError):
        validate_llm_review({"agree": "maybe", "reason_ru": "x"})


def test_pack_only_when_needs_llm() -> None:
    assert build_llm_review_pack({"needs_llm_review": False, "diag_text": "ОРВИ"}) is None
    pack = build_llm_review_pack(
        {
            "needs_llm_review": True,
            "diag_text": "хронический гастрит",
            "codes": ["K29.3"],
            "pipeline_verdict": "review",
            "chip": {"status": "weak_name"},
            "name_only": {
                "candidates": [
                    {"code": "K29.3", "title_ru": "Хронический гастрит", "score": 0.5}
                ]
            },
            "findings": [{"code": "B_icd_dir_text_mismatch"}],
        }
    )
    assert pack is not None
    assert pack["code"] == "K29.3"
    assert pack["has_text_mismatch"] is True
    assert len(pack["candidates"]) == 1


def test_review_one_mocked_agree_yes(monkeypatch) -> None:
    monkeypatch.setenv("MO_ICD_LLM_REVIEW", "1")

    def _gen(prompt: str) -> str:
        assert "diag_text" in prompt
        return json.dumps(
            {
                "agree": "yes",
                "reason_ru": "формулировка соответствует рубрике",
                "suggested_code": None,
            },
            ensure_ascii=False,
        )

    pipe = {
        "needs_llm_review": True,
        "diag_text": "Острый цистит",
        "codes": ["N30.0"],
        "pipeline_verdict": "review",
        "chip": {"status": "weak_name"},
        "name_only": {"candidates": []},
        "findings": [{"code": "B_icd_name_weak_match"}],
    }
    out = review_one(pipe, generate_fn=_gen)
    assert out["skipped"] is False
    assert out["review"]["agree"] == "yes"
    codes = {f["code"] for f in out["findings"]}
    assert "B_icd_llm_review_yes" in codes
    assert all(f.get("shadow") is True for f in out["findings"])


def test_review_one_flag_off_skips(monkeypatch) -> None:
    monkeypatch.setenv("MO_ICD_LLM_REVIEW", "0")
    out = review_one(
        {
            "needs_llm_review": True,
            "diag_text": "x",
            "codes": ["N30.0"],
            "findings": [],
            "name_only": {},
            "chip": {},
        },
        generate_fn=lambda p: "{}",
    )
    assert out["skipped"] is True
    assert out["reason"] == "flag_off"


def test_findings_disagree() -> None:
    findings = findings_from_review(
        {"agree": "no", "reason_ru": "код про мышцы, текст про желудок", "suggested_code": "K29.3"},
        pack={"diag_text": "гастрит", "code": "M60"},
    )
    assert findings[0]["code"] == "B_icd_llm_review_no"
    assert findings[0]["passed"] is False
    assert findings[0]["shadow"] is True
