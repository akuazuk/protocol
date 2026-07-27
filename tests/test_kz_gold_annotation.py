"""Тесты инфраструктуры gold-разметки (Workstream K ТЗ overnight-v1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.kz_gold_annotation import (
    GoldAnnotation,
    build_sample_slots,
    deterministic_visit_ref,
    evaluate_scorer_vs_gold,
    quadratic_weighted_kappa,
    synthetic_example,
    validate_annotation,
)


def test_visit_ref_deterministic_and_anonymized():
    r1 = deterministic_visit_ref("spec:band:0")
    r2 = deterministic_visit_ref("spec:band:0")
    assert r1 == r2
    assert r1.startswith("kzref_")
    # не содержит исходного seed
    assert "spec" not in r1


def test_sample_slots_deterministic():
    s1 = build_sample_slots({"therapist:70-79": 2, "lor:60-69": 1}, seed=1)
    s2 = build_sample_slots({"therapist:70-79": 2, "lor:60-69": 1}, seed=1)
    assert s1 == s2
    assert len(s1) == 3


def test_validate_annotation_harm_consistency():
    a = GoldAnnotation(visit_ref="r", annotator_id="a1", potential_harm=True, harm_class="none")
    issues = validate_annotation(a)
    assert any("harm_class" in i for i in issues)


def test_qwk_perfect_and_worst():
    assert quadratic_weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == 1.0
    v = quadratic_weighted_kappa([0, 0, 4, 4], [4, 4, 0, 0])
    assert v is not None and v < 0


def test_evaluator_harm_recall():
    gold = [
        GoldAnnotation(visit_ref="r1", annotator_id="a", potential_harm=True, harm_class="big_three", verdict="non_compliant"),
        GoldAnnotation(visit_ref="r2", annotator_id="a", potential_harm=False, verdict="mostly_compliant"),
    ]
    m = evaluate_scorer_vs_gold(gold, scorer_status=["critical", "good"], scorer_harm=[True, False])
    assert m["harm_recall"] == 1.0
    assert m["false_critical_rate"] == 0.0
    assert m["n"] == 2


def test_synthetic_example_no_consensus_when_disagree():
    ex = synthetic_example()
    # A=mostly, B=partially -> расхождение 1 балл -> консенсус = A
    assert ex.consensus() is not None
    assert ex.annotation_a.is_synthetic is True
