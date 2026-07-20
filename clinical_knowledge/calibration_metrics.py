"""Метрики калибровки уверенности: Brier score, ECE и таблица надёжности (reliability).

Позволяют проверить, что заявленный процент уверенности выдачи соответствует фактической
доле правильных ответов - основа для измеримого «идеального» поиска.
"""
from __future__ import annotations

from typing import Any

Pair = tuple[float, int]


def _clean_pairs(pairs: list[Pair]) -> list[Pair]:
    out: list[Pair] = []
    for p, o in pairs:
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        pf = 0.0 if pf < 0 else 1.0 if pf > 1 else pf
        out.append((pf, 1 if int(o) else 0))
    return out


def brier_score(pairs: list[Pair]) -> float:
    """Средний квадрат ошибки прогноза (0 - идеально, 1 - худший)."""
    data = _clean_pairs(pairs)
    if not data:
        return 0.0
    return round(sum((p - o) ** 2 for p, o in data) / len(data), 4)


def reliability_table(pairs: list[Pair], *, n_bins: int = 5) -> list[dict[str, Any]]:
    """Разбивка по корзинам уверенности: [{lo, hi, count, avg_conf, accuracy}]."""
    data = _clean_pairs(pairs)
    n_bins = max(1, int(n_bins))
    bins: list[list[Pair]] = [[] for _ in range(n_bins)]
    for p, o in data:
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, o))
    table: list[dict[str, Any]] = []
    for i, b in enumerate(bins):
        lo = round(i / n_bins, 3)
        hi = round((i + 1) / n_bins, 3)
        if b:
            avg_conf = sum(p for p, _ in b) / len(b)
            acc = sum(o for _, o in b) / len(b)
        else:
            avg_conf = 0.0
            acc = 0.0
        table.append(
            {
                "lo": lo,
                "hi": hi,
                "count": len(b),
                "avg_conf": round(avg_conf, 4),
                "accuracy": round(acc, 4),
                "gap": round(abs(avg_conf - acc), 4),
            }
        )
    return table


def expected_calibration_error(pairs: list[Pair], *, n_bins: int = 10) -> float:
    """ECE: средневзвешенный разрыв между уверенностью и точностью по корзинам (0 - идеально)."""
    data = _clean_pairs(pairs)
    if not data:
        return 0.0
    table = reliability_table(data, n_bins=n_bins)
    n = len(data)
    ece = sum(row["gap"] * row["count"] / n for row in table)
    return round(ece, 4)


def summarize_calibration(pairs: list[Pair], *, n_bins: int = 10) -> dict[str, Any]:
    data = _clean_pairs(pairs)
    return {
        "n": len(data),
        "accuracy": round(sum(o for _, o in data) / len(data), 4) if data else 0.0,
        "avg_confidence": round(sum(p for p, _ in data) / len(data), 4) if data else 0.0,
        "brier_score": brier_score(data),
        "ece": expected_calibration_error(data, n_bins=n_bins),
        "reliability": reliability_table(data, n_bins=min(n_bins, 5)),
    }
