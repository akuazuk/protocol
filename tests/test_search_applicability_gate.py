"""Regression-тесты applicability-gate поиска (ТЗ №2, §A1-A4, §15.1).

Ключевой инвариант: детский протокол не может стать Top-1/recommended для взрослого
или аудиторно-неопределённого запроса без подтверждения детской аудитории.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_knowledge.search_applicability_gate import (
    STATUS_CLARIFY,
    STATUS_EXACT,
    STATUS_NOT_FOR_AUDIENCE,
    STATUS_OUTDATED,
    apply_applicability_gate,
    classify_result,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "search_applicability_golden.jsonl"


def _load_rows() -> list[dict]:
    rows = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _run(row: dict) -> list[dict]:
    patient = row.get("patient") or {}
    audience = (patient.get("adult_or_child") or "").lower()
    pediatric_signal = audience in ("child", "newborn")
    return apply_applicability_gate(
        row["candidates"], patient, row.get("icd_query") or [],
        pediatric_signal=pediatric_signal, keep_not_applicable=True,
    )


@pytest.mark.parametrize("row", _load_rows(), ids=lambda r: r["query"])
def test_no_invalid_top1(row: dict) -> None:
    ranked = _run(row)
    assert ranked, f"пустая выдача для {row['query']}"
    top = ranked[0]
    gate = top["_gate"]

    if "forbid_top1_population" in row:
        assert gate["population"] not in _pop_group(row["forbid_top1_population"]), (
            f"{row['query']}: недопустимый Top-1 population={gate['population']}"
        )
    if "expect_top1_population" in row:
        assert gate["population"] in _pop_group(row["expect_top1_population"]), (
            f"{row['query']}: ожидали Top-1 population={row['expect_top1_population']}, "
            f"получили {gate['population']}"
        )
    if "expect_top1_not_status" in row:
        assert row["expect_top1_not_status"] != "obsolete" or gate["status"] != STATUS_OUTDATED
        # устаревший не должен быть Top-1
        assert gate["status"] != STATUS_OUTDATED, f"{row['query']}: устаревший стал Top-1"

    if "forbid_recommended_population" in row:
        bad = _pop_group(row["forbid_recommended_population"])
        for item in ranked:
            g = item["_gate"]
            if g["population"] in bad:
                assert not g["recommended"], (
                    f"{row['query']}: {g['population']}-протокол помечен recommended"
                )


def _pop_group(pop: str) -> set[str]:
    if pop == "child":
        return {"child", "children", "pediatric"}
    return {pop}


def test_child_not_top1_unknown_audience() -> None:
    """I10 без аудитории: детский протокол (score 95) не Top-1 и не recommended."""
    cards = [
        {"title": "Гипертензия у детского населения", "population": "child",
         "icd10_primary": "I10", "match_score": 95.0},
        {"title": "Гипертензия (взрослое население)", "population": "adult",
         "icd10_primary": "I10", "match_score": 80.0},
    ]
    ranked = apply_applicability_gate(cards, {}, ["I10"], pediatric_signal=False)
    assert ranked[0]["_gate"]["population"] == "adult"
    assert all(not r["_gate"]["recommended"] or r["_gate"]["population"] != "child" for r in ranked)
    # оба population-specific при неизвестной аудитории -> needs_clarification
    assert ranked[0]["_gate"]["status"] == STATUS_CLARIFY


def test_child_confirmed_audience_ranks_child_first() -> None:
    cards = [
        {"title": "Гипертензия у детского населения", "population": "child",
         "icd10_primary": "I10", "match_score": 95.0},
        {"title": "Гипертензия (взрослое население)", "population": "adult",
         "icd10_primary": "I10", "match_score": 80.0},
    ]
    ranked = apply_applicability_gate(
        cards, {"adult_or_child": "child", "age_years": 8}, ["I10"], pediatric_signal=True
    )
    assert ranked[0]["_gate"]["population"] == "child"
    # взрослый протокол для ребёнка отфильтрован как not_for_audience (или помечен)


def test_adult_query_blocks_child_protocol() -> None:
    card = {"title": "Гипертензия у детского населения", "population": "child",
            "icd10_primary": "I10", "match_score": 95.0}
    g = classify_result(card, {"adult_or_child": "adult", "age_years": 50}, ["I10"])
    assert g["status"] == STATUS_NOT_FOR_AUDIENCE
    assert not g["recommended"]


def test_exact_match_neutral_recommended() -> None:
    card = {"title": "Гипертензия (амбулаторно)", "population": "any",
            "icd10_primary": "I10", "match_score": 85.0}
    g = classify_result(card, {"adult_or_child": "adult"}, ["I10"])
    assert g["status"] == STATUS_EXACT
    assert g["recommended"] is True


def test_pregnancy_unconfirmed_not_recommended() -> None:
    card = {"title": "Артериальная гипертензия при беременности", "population": "adult",
            "icd10_primary": "O16", "match_score": 90.0}
    g = classify_result(card, {"sex": "female"}, ["O16"])
    # беременность не подтверждена -> possibly_applicable, не recommended
    assert not g["recommended"]


def test_low_confidence_not_recommended() -> None:
    card = {"title": "Гипертензия (амбулаторно)", "population": "any",
            "icd10_primary": "I10", "match_score": 30.0}
    g = classify_result(card, {"adult_or_child": "adult"}, ["I10"])
    assert not g["recommended"], "низкий score не должен давать recommended"


def test_gate_never_labels_recommended_below_threshold_across_dataset() -> None:
    for row in _load_rows():
        for item in _run(row):
            g = item["_gate"]
            if g["recommended"]:
                assert float(item.get("match_score") or 0) >= 60.0
                assert g["applicability"] == "applicable"
                assert g["status"] in (STATUS_EXACT, "icd_match")
