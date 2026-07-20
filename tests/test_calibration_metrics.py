"""Шаг 3: метрики калибровки (Brier, ECE, reliability)."""
from __future__ import annotations

from clinical_knowledge.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_table,
    summarize_calibration,
)


def test_brier_perfect() -> None:
    assert brier_score([(1.0, 1), (0.0, 0), (1.0, 1)]) == 0.0


def test_brier_worst() -> None:
    assert brier_score([(1.0, 0), (0.0, 1)]) == 1.0


def test_ece_well_calibrated() -> None:
    # уверенность 0.9 -> 9 из 10 верно; ECE близок к 0
    pairs = [(0.9, 1)] * 9 + [(0.9, 0)]
    assert expected_calibration_error(pairs, n_bins=10) < 0.05


def test_ece_miscalibrated() -> None:
    # заявлено 0.95, но все неверны -> большой ECE
    pairs = [(0.95, 0)] * 10
    assert expected_calibration_error(pairs, n_bins=10) > 0.5


def test_reliability_table_shape() -> None:
    table = reliability_table([(0.1, 0), (0.9, 1), (0.85, 1)], n_bins=5)
    assert len(table) == 5
    assert all(set(r.keys()) >= {"lo", "hi", "count", "avg_conf", "accuracy", "gap"} for r in table)
    assert sum(r["count"] for r in table) == 3


def test_summarize_shape() -> None:
    s = summarize_calibration([(0.8, 1), (0.2, 0)], n_bins=5)
    assert s["n"] == 2
    assert 0.0 <= s["accuracy"] <= 1.0
    assert "brier_score" in s and "ece" in s and "reliability" in s


def test_empty_safe() -> None:
    assert brier_score([]) == 0.0
    assert expected_calibration_error([]) == 0.0
    assert summarize_calibration([])["n"] == 0
