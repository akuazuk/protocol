"""Тесты чистых функций LLM-грейдера (без сети)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.grade_kz_llm import (
    _checklist_from_protocol,
    _should_escalate,
    build_grader_prompt,
    parse_grader_response,
)


def test_checklist_fallback_to_rubric():
    items = _checklist_from_protocol(None)
    ids = {i[0] for i in items}
    assert "B_dx_from_data" in ids
    assert len(items) >= 10


def test_checklist_from_protocol_merges():
    proto = {"kz_checklist": ["Измерено АД", "Оценён неврологический статус"], "name": "X"}
    items = _checklist_from_protocol(proto)
    texts = " ".join(t for _, t in items)
    assert "Измерено АД" in texts
    assert "B_icd_coded" in {i[0] for i in items}  # рубрика тоже добавлена


def test_build_prompt_contains_sections():
    case = {"complaints": "боль в горле", "clinical_diagnosis": "J02.9", "treatment_recommendations": "парацетамол"}
    p = build_grader_prompt(case, _checklist_from_protocol(None), protocol_name="Острый фарингит")
    assert "боль в горле" in p
    assert "цепочк" in p.lower() or "цепоч" in p.lower()
    assert "JSON" in p
    assert "Острый фарингит" in p


def test_parse_plain_json():
    r = parse_grader_response('{"overall_pct": 80, "verdict": "good", "potential_harm": false}')
    assert r["overall_pct"] == 80
    assert r["verdict"] == "good"


def test_parse_markdown_fenced():
    r = parse_grader_response('```json\n{"overall_pct": 55, "needs_human": true}\n```')
    assert r["overall_pct"] == 55
    assert r["needs_human"] is True


def test_parse_embedded_json():
    r = parse_grader_response('Вот оценка: {"overall_pct": 40, "potential_harm": true} - конец.')
    assert r["overall_pct"] == 40


def test_parse_garbage():
    r = parse_grader_response("не удалось")
    assert r.get("_parse_error")


def test_escalate_low_confidence():
    do, reason = _should_escalate({"confidence": 0.4}, None)
    assert do and reason == "low_confidence"


def test_escalate_harm_disagreement():
    do, reason = _should_escalate({"confidence": 0.9, "potential_harm": False}, {"has_potential_harm": True})
    assert do and reason == "harm_disagreement"


def test_no_escalate_high_conf_agree():
    do, _ = _should_escalate(
        {"confidence": 0.9, "potential_harm": True, "needs_human": False},
        {"has_potential_harm": True},
    )
    assert do is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
