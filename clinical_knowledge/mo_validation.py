"""Human validation workflow for scorer v4 and protocol trust."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


def build_gold_queue(
    cases: Iterable[dict[str, Any]], *, size: int = 300, seed: int = 20260730
) -> list[dict[str, Any]]:
    candidates = []
    for case in cases:
        evaluation = case.get("evaluation_v4") or {}
        if evaluation.get("score_pct") is None:
            continue
        findings = evaluation.get("findings") or []
        severity = next(
            (
                level
                for level in ("P0", "P1", "P2", "P3")
                if any(
                    not finding.get("passed") and finding.get("severity") == level
                    for finding in findings
                )
            ),
            "ok",
        )
        candidates.append(
            {
                "case_id": str(case.get("mis_id") or case.get("visit_id") or ""),
                "visit_id": str(case.get("visit_id") or ""),
                "date": str(case.get("date") or "")[:10],
                "specialty": str(case.get("doctor_specialization") or "Не указано"),
                "v4_score": evaluation["score_pct"],
                "v4_severity": severity,
                "findings": findings,
                "reviewer_a": None,
                "reviewer_b": None,
                "adjudication": None,
            }
        )
    rng = random.Random(seed)
    by_severity: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_severity.setdefault(row["v4_severity"], []).append(row)
    selected: list[dict[str, Any]] = []
    # Oversample rare clinical risks, then fill deterministically from all strata.
    targets = {"P0": 40, "P1": 80, "P2": 80, "P3": 30, "ok": 70}
    for severity, target in targets.items():
        rows = by_severity.get(severity, [])
        rng.shuffle(rows)
        selected.extend(rows[:target])
    selected_ids = {row["case_id"] for row in selected}
    remainder = [row for row in candidates if row["case_id"] not in selected_ids]
    rng.shuffle(remainder)
    selected.extend(remainder[: max(0, size - len(selected))])
    return selected[:size]


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = (
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    ) ** 0.5
    return numerator / denominator if denominator else None


def evaluate_gold(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    completed = [row for row in rows if isinstance(row.get("adjudication"), dict)]
    predicted = [float(row["v4_score"]) for row in completed]
    expert = [float(row["adjudication"]["score"]) for row in completed]
    spearman = _correlation(_ranks(predicted), _ranks(expert))
    actual_p0 = [row for row in completed if row["adjudication"].get("severity") == "P0"]
    recalled = sum(row.get("v4_severity") == "P0" for row in actual_p0)
    return {
        "n_total": len(rows),
        "n_completed": len(completed),
        "double_labeled": sum(
            isinstance(row.get("reviewer_a"), dict)
            and isinstance(row.get("reviewer_b"), dict)
            for row in completed
        ),
        "spearman": round(spearman, 4) if spearman is not None else None,
        "p0_recall": round(recalled / len(actual_p0), 4) if actual_p0 else None,
        "accepted": bool(
            len(completed) >= 300
            and spearman is not None
            and spearman >= 0.70
            and actual_p0
            and recalled / len(actual_p0) >= 0.90
        ),
    }


def protocol_trust_status(summary_dir: Path) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for path in summary_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status = str(payload.get("review_status") or "missing")
        statuses[status] = statuses.get(status, 0) + 1
    ready = sum(statuses.get(status, 0) for status in ("approved", "reviewed"))
    return {"total": sum(statuses.values()), "penalty_ready": ready, "statuses": statuses}
