"""Семантический слой аналитики медицинских осмотров."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

MINSK = ZoneInfo("Europe/Minsk")
SCHEMA_VERSION = 1
VALID_PERIODS = frozenset({"yesterday", "7d", "month", "custom"})
VALID_COMPARE = frozenset({"previous", "weekday", "none"})

METRICS: dict[str, dict[str, Any]] = {
    "overall": {
        "label": "Оценка МО",
        "description": "Средняя оценка среди оценённых записей",
        "unit": "percent",
    },
    "documentation": {
        "label": "Оформление",
        "description": "Полнота и структура медицинской записи",
        "unit": "percent",
    },
    "clinical_concordance": {
        "label": "Клиническая согласованность",
        "description": "Согласованность диагноза, обследований и лечения",
        "unit": "percent",
    },
    "safety": {
        "label": "Безопасность",
        "description": "Отсутствие клинически значимых рисков",
        "unit": "percent",
    },
    "regulatory": {
        "label": "Регуляторика",
        "description": "Выполнение обязательных требований к записи",
        "unit": "percent",
    },
    "volume": {
        "label": "Количество записей",
        "description": "Число записей из МИС",
        "unit": "count",
    },
    "coverage": {
        "label": "Покрытие оценки",
        "description": "Доля допущенных записей, для которых рассчитана оценка",
        "unit": "percent",
    },
    "critical": {
        "label": "Критические случаи",
        "description": "Число записей с критическими замечаниями",
        "unit": "count",
    },
}


@dataclass(frozen=True)
class DateRange:
    date_from: date
    date_to: date

    @property
    def days(self) -> int:
        return (self.date_to - self.date_from).days + 1

    def to_dict(self) -> dict[str, str]:
        return {
            "date_from": self.date_from.isoformat(),
            "date_to": self.date_to.isoformat(),
        }


@dataclass(frozen=True)
class ResolvedPeriods:
    period: str
    current: DateRange
    compare: str
    comparison: DateRange | None
    timezone: str = "Europe/Minsk"

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "current": self.current.to_dict(),
            "compare": self.compare,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "timezone": self.timezone,
        }


def minsk_today(now: datetime | None = None) -> date:
    current = now or datetime.now(MINSK)
    if current.tzinfo is None:
        current = current.replace(tzinfo=MINSK)
    return current.astimezone(MINSK).date()


def _parse_date(value: str | None, name: str) -> date:
    try:
        return date.fromisoformat(str(value or ""))
    except ValueError as exc:
        raise ValueError(f"{name} должна быть датой YYYY-MM-DD") from exc


def _month_range(month: str, *, last_available: date) -> DateRange:
    try:
        start = date.fromisoformat(f"{month}-01")
    except ValueError as exc:
        raise ValueError("month должен иметь формат YYYY-MM") from exc
    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    end = next_month - timedelta(days=1)
    if start > last_available:
        raise ValueError("Выбранный месяц ещё не содержит завершённых дней Europe/Minsk")
    return DateRange(start, min(end, last_available))


def resolve_periods(
    *,
    period: str = "month",
    month: str | None = None,
    compare: str = "none",
    date_from: str | None = None,
    date_to: str | None = None,
    now: datetime | None = None,
) -> ResolvedPeriods:
    """Разрешить пользовательский период по календарю Europe/Minsk."""
    period = (period or "month").strip().lower()
    compare = (compare or "none").strip().lower()
    if period not in VALID_PERIODS:
        raise ValueError(f"Неизвестный period={period!r}; допустимо: yesterday, 7d, month, custom")
    if compare not in VALID_COMPARE:
        raise ValueError(f"Неизвестный compare={compare!r}; допустимо: previous, weekday, none")

    yesterday = minsk_today(now) - timedelta(days=1)
    if period == "yesterday":
        current = DateRange(yesterday, yesterday)
    elif period == "7d":
        current = DateRange(yesterday - timedelta(days=6), yesterday)
    elif period == "month":
        selected_month = month or yesterday.strftime("%Y-%m")
        current = _month_range(selected_month, last_available=yesterday)
    else:
        start = _parse_date(date_from, "date_from")
        end = _parse_date(date_to, "date_to")
        if start > end:
            raise ValueError("date_from не может быть позже date_to")
        if end > yesterday:
            raise ValueError("date_to должна быть не позже вчера Europe/Minsk")
        current = DateRange(start, end)

    comparison: DateRange | None = None
    if compare == "previous":
        comparison = DateRange(
            current.date_from - timedelta(days=current.days),
            current.date_from - timedelta(days=1),
        )
    elif compare == "weekday":
        comparison = DateRange(
            current.date_from - timedelta(days=7),
            current.date_to - timedelta(days=7),
        )
    return ResolvedPeriods(period, current, compare, comparison)


def suppress_values(
    values: dict[str, Any],
    *,
    n: int,
    threshold: int,
    protected: Iterable[str] = (),
) -> dict[str, Any]:
    """Скрыть метрики малой группы, сохранив безопасные поля."""
    if n >= threshold:
        return {**values, "n": n, "suppressed": False}
    protected_keys = set(protected)
    return {
        **{key: value for key, value in values.items() if key in protected_keys},
        "n": None,
        "n_bucket": f"<{threshold}",
        "suppressed": True,
    }


def mean_confidence_interval(values: Iterable[float], confidence: float = 0.95) -> dict[str, float | int | None]:
    """Нормальный 95% ДИ среднего; для n<2 границы не публикуются."""
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    if not numbers:
        return {"n": 0, "mean": None, "low": None, "high": None}
    mean = statistics.fmean(numbers)
    if len(numbers) < 2:
        return {"n": 1, "mean": round(mean, 2), "low": None, "high": None}
    # Для аналитической витрины используем устойчивое нормальное приближение.
    z = 1.959963984540054 if confidence == 0.95 else 1.959963984540054
    margin = z * statistics.stdev(numbers) / math.sqrt(len(numbers))
    return {
        "n": len(numbers),
        "mean": round(mean, 2),
        "low": round(mean - margin, 2),
        "high": round(mean + margin, 2),
    }


def metric_catalog() -> list[dict[str, Any]]:
    return [{"key": key, **definition} for key, definition in METRICS.items()]
