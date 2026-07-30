from datetime import datetime, timezone

import pytest

from clinical_knowledge.mo_metrics import (
    mean_confidence_interval,
    resolve_periods,
    suppress_values,
)


def test_yesterday_uses_minsk_date_near_utc_midnight() -> None:
    # 21:30 UTC уже является следующим календарным днём в Минске.
    resolved = resolve_periods(
        period="yesterday",
        now=datetime(2026, 7, 29, 21, 30, tzinfo=timezone.utc),
    )
    assert resolved.current.to_dict() == {
        "date_from": "2026-07-29",
        "date_to": "2026-07-29",
    }
    assert resolved.timezone == "Europe/Minsk"


def test_month_is_mtd_through_minsk_yesterday() -> None:
    resolved = resolve_periods(
        period="month",
        now=datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc),
    )
    assert resolved.current.to_dict() == {
        "date_from": "2026-07-01",
        "date_to": "2026-07-29",
    }


def test_previous_comparison_has_equal_length() -> None:
    resolved = resolve_periods(
        period="7d",
        compare="previous",
        now=datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc),
    )
    assert resolved.current.to_dict() == {
        "date_from": "2026-07-23",
        "date_to": "2026-07-29",
    }
    assert resolved.comparison is not None
    assert resolved.comparison.to_dict() == {
        "date_from": "2026-07-16",
        "date_to": "2026-07-22",
    }


def test_weekday_comparison_shifts_seven_days() -> None:
    resolved = resolve_periods(
        period="custom",
        date_from="2026-07-27",
        date_to="2026-07-29",
        compare="weekday",
        now=datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc),
    )
    assert resolved.comparison is not None
    assert resolved.comparison.to_dict() == {
        "date_from": "2026-07-20",
        "date_to": "2026-07-22",
    }


def test_invalid_period_and_custom_dates_are_clear() -> None:
    with pytest.raises(ValueError, match="Неизвестный period"):
        resolve_periods(period="quarter")
    with pytest.raises(ValueError, match="date_from"):
        resolve_periods(period="custom", date_from="", date_to="2026-07-20")


def test_suppression_and_confidence_interval() -> None:
    hidden = suppress_values({"doctor": "A", "avg_score": 80.0}, n=3, threshold=5, protected={"doctor"})
    assert hidden == {
        "doctor": "A",
        "n": None,
        "n_bucket": "<5",
        "suppressed": True,
    }
    interval = mean_confidence_interval([70, 80, 90])
    assert interval["mean"] == 80
    assert interval["low"] < 80 < interval["high"]
