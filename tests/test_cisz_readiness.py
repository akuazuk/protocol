"""Тесты готовности к ЦИСЗ (программа испытаний МИС v.1.3-4)."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.cisz_readiness import (
    attach_cisz_readiness,
    evaluate_cisz_readiness,
    merge_send_gate_with_cisz,
)
from clinical_knowledge.fhir_bundle_inspect import detect_bundle_scenario, inspect_bundle_checks

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fhir_by"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_inspect_bundle_primary_ambulatory_fixture():
    bundle = _load_fixture("primary_ambulatory_min.json")
    checks = inspect_bundle_checks(bundle)
    assert checks["patient"]
    assert checks["encounter"]
    assert checks["diagnosis_icd10"]
    assert checks["complaints"]
    assert checks["vitals"]
    assert detect_bundle_scenario(bundle) == "primary_ambulatory"


def test_evaluate_cisz_readiness_bundle_high_score():
    bundle = _load_fixture("primary_ambulatory_min.json")
    out = evaluate_cisz_readiness(bundle=bundle)
    assert out["ok"] is True
    assert out["source"] == "fhir_bundle"
    assert out["overall_score"] is not None
    assert out["overall_score"] >= 70
    assert len(out["checks"]) > 0


def test_evaluate_cisz_readiness_text_heuristics():
    text = """\
Врач: терапевт
Дата консультации: 2024-04-07
Пациент: Иванов И.И., дата рождения 1989-01-31, пол: мужской
Жалобы: слабость
Объективный статус: АД 120/80, пульс 80
Диагноз: E11 Сахарный диабет 2 типа
Рекомендации по лечению: метформин 500 мг
"""
    out = evaluate_cisz_readiness(text=text)
    assert out["source"] == "text"
    assert out["overall_score"] is not None
    assert out["overall_score"] >= 50


def test_merge_send_gate_with_cisz_lowers_gate_score():
    sg = {"gate_score": 90.0, "gate_allowed": True, "gate_mode": "hard_gate"}
    cisz = {"overall_score": 55.0, "critical_failures": 1}
    merged = merge_send_gate_with_cisz(sg, cisz)
    assert merged["gate_score"] == 55.0
    assert merged["cisz_score"] == 55.0
    assert merged["clinical_gate_score"] == 90.0
    assert not merged["gate_allowed"]


def test_attach_cisz_readiness_adds_block_to_payload():
    payload = {"send_gate": {"gate_score": 80.0, "gate_allowed": True}}
    out = attach_cisz_readiness(
        payload,
        text="Дата консультации: 2024-01-01\nДиагноз: I10\nЖалобы: головная боль",
    )
    assert "cisz_readiness" in out
    assert out["cisz_readiness"]["overall_score"] is not None
