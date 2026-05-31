"""Tests for clinical rules benchmark and LLM context."""
from __future__ import annotations

from clinical_knowledge.benchmark import run_gastro_gold_benchmark
from clinical_knowledge.llm_context import format_clinical_rules_for_llm


def test_format_clinical_rules_for_llm_includes_missing():
    block = format_clinical_rules_for_llm(
        {
            "matched_protocols": [{"title": "КП ГЭРБ"}],
            "rules_check": {
                "rules_compliance_pct": 42.0,
                "missing_required_items": ["Нет фазы заболевания"],
                "findings": [
                    {
                        "passed": False,
                        "severity": "critical",
                        "message_ru": "Протокол для adult, аудитория child",
                    }
                ],
            },
            "consult_facts": {"consultation": {"icd10": ["K21.9"]}},
        }
    )
    assert "ДЕТЕРМИНИРОВАННАЯ ПРОВЕРКА" in block
    assert "Нет фазы" in block
    assert "K21.9" in block


def test_gastro_gold_benchmark_pass_rate():
    rep = run_gastro_gold_benchmark()
    assert rep["cases_total"] >= 9
    assert rep["cases_passed"] >= 8
    assert rep["pass_rate_pct"] >= 85
