"""Трек 4: калибровка уверенности выдачи."""
from __future__ import annotations

from clinical_knowledge.confidence_calibration import (
    calibrate_confidence,
    confidence_band,
)


def test_bounds() -> None:
    assert 0.0 <= calibrate_confidence(rag_support=0.0, icd_relevance=0, llm_confidence=0) <= 1.0
    assert 0.0 <= calibrate_confidence(rag_support=1.0, icd_relevance=1, llm_confidence=1) <= 1.0


def test_high_signals_high_band() -> None:
    c = calibrate_confidence(rag_support=0.95, icd_relevance=95, llm_confidence=0.9)
    assert c >= 0.75
    assert confidence_band(c) == "высокая"


def test_low_signals_low_band() -> None:
    c = calibrate_confidence(rag_support=0.05, icd_relevance=5, llm_confidence=0.1)
    assert c < 0.5
    assert confidence_band(c) == "низкая"


def test_monotonic_in_rag_support() -> None:
    lo = calibrate_confidence(rag_support=0.2, icd_relevance=50, llm_confidence=0.5)
    hi = calibrate_confidence(rag_support=0.8, icd_relevance=50, llm_confidence=0.5)
    assert hi > lo


def test_percent_normalization() -> None:
    # icd_relevance как процент (92) и как доля (0.92) дают одно и то же
    a = calibrate_confidence(rag_support=0.5, icd_relevance=92)
    b = calibrate_confidence(rag_support=0.5, icd_relevance=0.92)
    assert abs(a - b) < 1e-9


def test_partial_signals() -> None:
    assert calibrate_confidence(rag_support=0.7) > calibrate_confidence(rag_support=0.3)


def test_no_signals_zero() -> None:
    assert calibrate_confidence() == 0.0


def test_band_thresholds() -> None:
    assert confidence_band(0.9) == "высокая"
    assert confidence_band(0.6) == "средняя"
    assert confidence_band(0.2) == "низкая"
    assert confidence_band(None) == "низкая"
