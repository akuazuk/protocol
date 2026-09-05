"""Грубая поправка на поток: сравнение врачей от ≥20 случаев (wave 4)."""
from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Mapping


DEFAULT_MIN_CASES = 20


def risk_adjust_doctor_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_cases: int = DEFAULT_MIN_CASES,
    score_key: str = "avg_score",
    cases_key: str = "cases",
    specialty_key: str = "specialty",
) -> list[dict[str, Any]]:
    """Ожидаемая оценка = среднее по специальности (врачи с ≥min_cases).

    Отклонение = факт − ожидание. Без specialty - глобальное среднее.
    """
    material = [dict(r) for r in rows if isinstance(r, Mapping)]
    eligible = [
        r for r in material if int(r.get(cases_key) or 0) >= int(min_cases)
    ]
    by_spec: dict[str, list[float]] = {}
    global_scores: list[float] = []
    for row in eligible:
        try:
            score = float(row.get(score_key))
        except (TypeError, ValueError):
            continue
        spec = str(row.get(specialty_key) or "").strip() or "_all"
        by_spec.setdefault(spec, []).append(score)
        global_scores.append(score)
    global_exp = mean(global_scores) if global_scores else None
    out: list[dict[str, Any]] = []
    for row in material:
        cases = int(row.get(cases_key) or 0)
        try:
            score = float(row.get(score_key))
        except (TypeError, ValueError):
            score = None
        spec = str(row.get(specialty_key) or "").strip() or "_all"
        expected = None
        if cases >= min_cases and score is not None:
            bucket = by_spec.get(spec) or global_scores
            expected = round(mean(bucket), 1) if bucket else global_exp
        delta = None
        if score is not None and expected is not None:
            delta = round(score - expected, 1)
        out.append(
            {
                **row,
                "eligible": cases >= min_cases and score is not None,
                "expected_score": expected,
                "delta_vs_expected": delta,
                "min_cases": min_cases,
                "note_ru": (
                    None
                    if cases >= min_cases
                    else f"Сравнение с {min_cases}+ случаев; сейчас {cases}."
                ),
            }
        )
    return out


def agreement_report(
    pairs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """pairs: [{system: bool, expert: bool, axis: 'lab_unused'|'drug_label'}].

    Простой % agreement и kappa-lite (для 2×2).
    """
    by_axis: dict[str, list[tuple[bool, bool]]] = {}
    for row in pairs:
        if not isinstance(row, Mapping):
            continue
        axis = str(row.get("axis") or "all")
        by_axis.setdefault(axis, []).append(
            (bool(row.get("system")), bool(row.get("expert")))
        )

    def _stats(items: list[tuple[bool, bool]]) -> dict[str, Any]:
        n = len(items)
        if n == 0:
            return {"n": 0, "agreement_pct": None, "kappa": None}
        agree = sum(1 for s, e in items if s == e)
        # Cohen's kappa for binary
        tp = sum(1 for s, e in items if s and e)
        tn = sum(1 for s, e in items if (not s) and (not e))
        fp = sum(1 for s, e in items if s and (not e))
        fn = sum(1 for s, e in items if (not s) and e)
        po = agree / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else 0
        kappa = None if pe >= 1 else round((po - pe) / (1 - pe), 3)
        return {
            "n": n,
            "agreement_pct": round(100.0 * po, 1),
            "kappa": kappa,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "false_positive_pct": round(100.0 * fp / n, 1) if n else None,
        }

    axes = {axis: _stats(items) for axis, items in by_axis.items()}
    flat = [pair for items in by_axis.values() for pair in items]
    return {
        "overall": _stats(flat),
        "by_axis": axes,
        "gate_ru": (
            "Primary только при agreement ≥ 0.7 и FP ≤ 15% "
            "на контрольной выборке (план wave 1-2)."
        ),
    }
