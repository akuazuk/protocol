"""Регрессия воспроизводимости: одинаковый вход -> одинаковый результат.

Причина бага: вызовы модели шли с temperature>0 без seed, поэтому повторная загрузка одного
и того же PDF давала разный «Ориентировочное соответствие». Тесты фиксируют детерминированные
настройки генерации и стабильный порядок отбора фрагментов.
"""
from __future__ import annotations

import pytest


def test_generation_config_is_deterministic() -> None:
    import rag_server as rs

    genai = pytest.importorskip("google.generativeai")
    cfg = rs._make_generation_config(genai, max_output_tokens=1024, json_mode=True)
    assert float(getattr(cfg, "temperature")) == 0.0
    assert int(getattr(cfg, "candidate_count")) == 1
    assert getattr(cfg, "response_mime_type") == "application/json"


def test_generation_temperature_override(monkeypatch) -> None:
    import rag_server as rs

    genai = pytest.importorskip("google.generativeai")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.7")
    cfg = rs._make_generation_config(genai, max_output_tokens=64, json_mode=False)
    assert abs(float(getattr(cfg, "temperature")) - 0.7) < 1e-9


def test_retrieve_order_is_stable() -> None:
    import rag_server as rs

    runs = [
        [r.get("path") for r in rs.retrieve("кашель бронхит J20", max_chunks=5, max_per_path=2)]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_overall_compliance_is_mean_of_criteria() -> None:
    import rag_server as rs

    parsed = {
        "overall_compliance_pct": 91,  # «свободное» число модели — должно быть пересчитано
        "criteria": [
            {"name_ru": "A", "score_pct": 80},
            {"name_ru": "B", "score_pct": 60},
            {"name_ru": "C", "score_pct": 70},
        ],
    }
    rs._stabilize_overall_compliance(parsed)
    assert parsed["overall_compliance_pct"] == 70  # round((80+60+70)/3)
    assert parsed["overall_compliance_method"] == "mean_of_criteria"


def test_overall_compliance_no_criteria_keeps_value() -> None:
    import rag_server as rs

    parsed = {"overall_compliance_pct": 88, "criteria": []}
    rs._stabilize_overall_compliance(parsed)
    assert parsed["overall_compliance_pct"] == 88
