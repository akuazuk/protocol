"""Контроль качества клинических выдержек (отсев обрывков и мусора)."""
from __future__ import annotations

from clinical_knowledge.extract_quality import (
    best_meaningful_excerpt,
    clean_clinical_text,
    is_meaningful_clinical_text,
    meaningful_clinical_excerpt,
)


def test_single_word_is_not_meaningful() -> None:
    assert is_meaningful_clinical_text("УЗИ") is False
    assert is_meaningful_clinical_text("контроль") is False
    assert is_meaningful_clinical_text("норадреналином") is False


def test_full_sentence_is_meaningful() -> None:
    t = "Рентгенография органов грудной клетки выполняется всем пациентам с подозрением на пневмонию."
    assert is_meaningful_clinical_text(t) is True


def test_enumeration_prefix_stripped() -> None:
    assert clean_clinical_text("4.1. малые критерии тяжелого течения") == "малые критерии тяжелого течения"


def test_dangling_tail_and_unbalanced_paren_trimmed() -> None:
    t = "4.1. малые критерии тяжёлого течения пневмонии (далее - малый"
    out = meaningful_clinical_excerpt(t, limit=240)
    assert "(" not in out
    assert not out.endswith("далее")
    assert out.startswith("малые критерии")


def test_garbled_short_returns_empty() -> None:
    assert meaningful_clinical_excerpt("УЗИ", limit=240) == ""
    assert meaningful_clinical_excerpt("контроль", limit=240) == ""


def test_best_excerpt_prefers_first_meaningful() -> None:
    out = best_meaningful_excerpt(
        ["Амоксициллин", "Амоксициллин назначают внутрь по 500 мг три раза в сутки семь дней."],
        limit=240,
    )
    assert out.startswith("Амоксициллин назначают")


def test_best_excerpt_all_bad_returns_empty() -> None:
    assert best_meaningful_excerpt(["УЗИ", "ЭКГ", None], limit=240) == ""


def test_excerpt_ends_on_sentence_boundary() -> None:
    t = (
        "Диагноз устанавливают на основании клинической картины. "
        "Далее следует множество дополнительных деталей, которые не должны попасть в выдержку "
        "полностью, поскольку превышают лимит символов и продолжаются очень долго без остановки."
    )
    out = meaningful_clinical_excerpt(t, limit=60)
    assert out
    assert len(out) <= 62
