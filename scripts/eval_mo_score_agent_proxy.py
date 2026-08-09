#!/usr/bin/env python3
"""Compare score families with an independent AI proxy without claiming human gold."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

BAD_VERDICTS = frozenset({"poor", "critical"})
EXCLUDED_VERDICTS = frozenset({"blocked", "na"})


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _plan_score(value: Mapping[str, Any]) -> float | None:
    for key in ("plan_protocol_pct", "plan_general_llm_pct"):
        score = _number(value.get(key))
        if score is not None:
            return score
    return None


def _endpoint_score(value: Mapping[str, Any], endpoint: str) -> float | None:
    return (
        _number(value.get("dx_evidence_pct"))
        if endpoint == "dx"
        else _plan_score(value)
    )


def _average(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            out[index] = rank
        cursor = end
    return out


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - mean_left) ** 2 for x in left)
        * sum((y - mean_right) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _roc_auc(labels: list[bool], quality_scores: list[float]) -> float | None:
    bad_scores = [100 - score for label, score in zip(labels, quality_scores) if label]
    good_scores = [100 - score for label, score in zip(labels, quality_scores) if not label]
    if not bad_scores or not good_scores:
        return None
    wins = sum(
        1.0 if bad > good else 0.5 if bad == good else 0.0
        for bad in bad_scores
        for good in good_scores
    )
    return wins / (len(bad_scores) * len(good_scores))


def _average_precision(labels: list[bool], quality_scores: list[float]) -> float | None:
    positive_n = sum(labels)
    if positive_n == 0:
        return None
    ranked = sorted(
        zip(labels, quality_scores),
        key=lambda item: (100 - item[1]),
        reverse=True,
    )
    hits = 0
    precision_sum = 0.0
    for rank, (label, _) in enumerate(ranked, 1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positive_n


def _classification(labels: list[bool], quality_scores: list[float]) -> dict[str, Any]:
    predicted = [score < 55 for score in quality_scores]
    tp = sum(actual and guess for actual, guess in zip(labels, predicted))
    tn = sum(not actual and not guess for actual, guess in zip(labels, predicted))
    fp = sum(not actual and guess for actual, guess in zip(labels, predicted))
    fn = sum(actual and not guess for actual, guess in zip(labels, predicted))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    return {
        "threshold": 55,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None
            else None
        ),
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_ci(
    labels: list[bool],
    candidate: list[float],
    proxy: list[float],
    metric: Callable[[list[bool], list[float], list[float]], float | None],
    *,
    iterations: int,
    seed: int,
) -> list[float] | None:
    if len(candidate) < 5 or iterations <= 0:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(len(candidate)) for _ in candidate]
        value = metric(
            [labels[index] for index in indices],
            [candidate[index] for index in indices],
            [proxy[index] for index in indices],
        )
        if value is not None and math.isfinite(value):
            values.append(value)
    if len(values) < max(20, iterations // 5):
        return None
    low = _percentile(values, 0.025)
    high = _percentile(values, 0.975)
    return [round(float(low), 4), round(float(high), 4)] if low is not None and high is not None else None


def _metric_row(
    references: Mapping[str, dict[str, Any]],
    candidate: Mapping[str, float],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    sample_ids = sorted(set(references) & set(candidate))
    labels = [bool(references[sample_id]["bad"]) for sample_id in sample_ids]
    proxy = [float(references[sample_id]["score"]) for sample_id in sample_ids]
    scores = [float(candidate[sample_id]) for sample_id in sample_ids]
    mae = statistics.fmean(abs(left - right) for left, right in zip(scores, proxy))
    spearman = _spearman(scores, proxy)
    roc_auc = _roc_auc(labels, scores)
    average_precision = _average_precision(labels, scores)
    classification = _classification(labels, scores)

    def mae_metric(_: list[bool], values: list[float], gold: list[float]) -> float:
        return statistics.fmean(abs(left - right) for left, right in zip(values, gold))

    def rho_metric(_: list[bool], values: list[float], gold: list[float]) -> float | None:
        return _spearman(values, gold)

    def auc_metric(actual: list[bool], values: list[float], _: list[float]) -> float | None:
        return _roc_auc(actual, values)

    def ap_metric(actual: list[bool], values: list[float], _: list[float]) -> float | None:
        return _average_precision(actual, values)

    return {
        "n": len(sample_ids),
        "proxy_bad_n": sum(labels),
        "proxy_good_n": len(labels) - sum(labels),
        "mae": round(mae, 4),
        "mae_ci95": _bootstrap_ci(
            labels, scores, proxy, mae_metric, iterations=bootstrap_iterations, seed=seed
        ),
        "spearman": round(spearman, 4) if spearman is not None else None,
        "spearman_ci95": _bootstrap_ci(
            labels, scores, proxy, rho_metric, iterations=bootstrap_iterations, seed=seed + 1
        ),
        "roc_auc_bad": round(roc_auc, 4) if roc_auc is not None else None,
        "roc_auc_bad_ci95": _bootstrap_ci(
            labels, scores, proxy, auc_metric, iterations=bootstrap_iterations, seed=seed + 2
        ),
        "pr_auc_bad": round(average_precision, 4) if average_precision is not None else None,
        "pr_auc_bad_ci95": _bootstrap_ci(
            labels, scores, proxy, ap_metric, iterations=bootstrap_iterations, seed=seed + 3
        ),
        "classification_at_55": classification,
    }


def _snapshot_candidates(
    snapshots: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    candidates: dict[str, dict[str, float]] = {}
    mis_to_sample: dict[str, str] = {}
    for index, row in enumerate(snapshots, 1):
        raw_sample_id = str(row.get("sample_id") or "")
        sample_id = (
            raw_sample_id
            if re.fullmatch(r"S\d{3}", raw_sample_id)
            else f"S{index:03d}"
        )
        source_ids = row.get("source_ids") if isinstance(row.get("source_ids"), dict) else {}
        mis_to_sample[str(source_ids.get("mis_id") or "")] = sample_id
        scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        values: dict[str, Any] = {
            "snapshot.overall": scores.get("overall_pct"),
            "snapshot.overall_v3": scores.get("overall_pct_v3"),
            "snapshot.rubric": scores.get("rubric_pct"),
            "snapshot.reg55": (scores.get("reg55") or {}).get("score_pct"),
        }
        for axis, value in (scores.get("axes") or {}).items():
            values[f"snapshot.axis.{axis}"] = value
        for zone, value in (scores.get("zones") or {}).items():
            values[f"snapshot.{zone}"] = (
                value.get("score_pct") if isinstance(value, dict) else value
            )
        for name, value in values.items():
            numeric = _number(value)
            if numeric is not None:
                candidates.setdefault(name, {})[sample_id] = numeric
    return candidates, mis_to_sample


def _replay_candidates(
    replays: list[dict[str, Any]],
    mis_to_sample: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    candidates: dict[str, dict[str, float]] = {}
    for row in replays:
        sample_id = mis_to_sample.get(str(row.get("case_key") or ""))
        if not sample_id:
            continue
        for key, comparison in (row.get("comparisons") or {}).items():
            if not isinstance(comparison, dict):
                continue
            numeric = _number(comparison.get("replayed"))
            if numeric is not None:
                name = f"arm_d.{str(key).replace(':', '.')}"
                candidates.setdefault(name, {})[sample_id] = numeric
    return candidates


def _blind_candidates(rows: list[dict[str, Any]], endpoint: str) -> dict[str, dict[str, float]]:
    values: dict[str, dict[int, float]] = {}
    adjudicated: dict[str, float] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if row.get("kind", "pass") == "pass" and not row.get("error"):
            payload = row.get("dx_evidence" if endpoint == "dx" else "plan_concordance")
            score = _endpoint_score(payload, endpoint) if isinstance(payload, dict) else None
            if score is not None:
                values.setdefault(sample_id, {})[int(row.get("pass_no") or 0)] = score
        elif row.get("kind") == "adjudication" and row.get("endpoint") == endpoint:
            payload = row.get("result")
            score = _endpoint_score(payload, endpoint) if isinstance(payload, dict) else None
            if score is not None:
                adjudicated[sample_id] = score
    out: dict[str, dict[str, float]] = {
        "blind.pass_1": {},
        "blind.pass_2": {},
        "blind.mean_2": {},
        "blind.adjudicated_or_mean": {},
    }
    for sample_id, passes in values.items():
        if 1 in passes:
            out["blind.pass_1"][sample_id] = passes[1]
        if 2 in passes:
            out["blind.pass_2"][sample_id] = passes[2]
        mean = _average(passes.values())
        if mean is not None:
            out["blind.mean_2"][sample_id] = mean
            out["blind.adjudicated_or_mean"][sample_id] = adjudicated.get(sample_id, mean)
    return out


def _proxy_reference(rows: list[dict[str, Any]], endpoint: str) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("kind", "pass") != "pass":
            continue
        payload = row.get("dx_evidence" if endpoint == "dx" else "plan_concordance")
        if not isinstance(payload, dict):
            continue
        verdict = str(payload.get("verdict") or "")
        score = _endpoint_score(payload, endpoint)
        if verdict in EXCLUDED_VERDICTS or score is None:
            continue
        sample_id = str(row.get("sample_id") or "")
        references[sample_id] = {
            "score": score,
            "bad": verdict in BAD_VERDICTS or bool(payload.get("potential_harm")),
        }
    return references


def _ensembles(
    candidates: dict[str, dict[str, float]],
    endpoint: str,
) -> dict[str, dict[str, float]]:
    relevant = "snapshot.zone2a" if endpoint == "dx" else "snapshot.zone2b"
    pairs = {
        "ensemble.zone_blind_mean": (relevant, "blind.mean_2"),
        "ensemble.zone_blind_adjudicated": (relevant, "blind.adjudicated_or_mean"),
        "ensemble.arm_d_blind_mean": ("arm_d.axis.clinical_concordance", "blind.mean_2"),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (left_name, right_name) in pairs.items():
        left = candidates.get(left_name, {})
        right = candidates.get(right_name, {})
        for sample_id in set(left) & set(right):
            out.setdefault(name, {})[sample_id] = (left[sample_id] + right[sample_id]) / 2
    return out


def _primary_candidate(name: str, endpoint: str) -> bool:
    relevant_zone = "snapshot.zone2a" if endpoint == "dx" else "snapshot.zone2b"
    return (
        name == relevant_zone
        or name in {
            "snapshot.axis.clinical_concordance",
            "arm_d.axis.clinical_concordance",
        }
        or name.startswith("blind.")
        or name.startswith("ensemble.")
    )


def evaluate_agent_proxy(
    snapshots: list[dict[str, Any]],
    replays: list[dict[str, Any]],
    blind: list[dict[str, Any]],
    proxy: list[dict[str, Any]],
    *,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    snapshot_candidates, mis_to_sample = _snapshot_candidates(snapshots)
    base_candidates = {
        **snapshot_candidates,
        **_replay_candidates(replays, mis_to_sample),
    }
    endpoint_reports: dict[str, Any] = {}
    for endpoint in ("dx", "plan"):
        references = _proxy_reference(proxy, endpoint)
        payload_key = "dx_evidence" if endpoint == "dx" else "plan_concordance"
        payload_n = sum(
            row.get("kind", "pass") == "pass"
            and isinstance(row.get(payload_key), dict)
            for row in proxy
        )
        candidates = {**base_candidates, **_blind_candidates(blind, endpoint)}
        candidates.update(_ensembles(candidates, endpoint))
        all_metrics = {
            name: _metric_row(
                references,
                values,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed + index * 10,
            )
            for index, (name, values) in enumerate(sorted(candidates.items()))
            if len(set(references) & set(values)) >= 5
        }
        metrics = {
            name: value
            for name, value in all_metrics.items()
            if _primary_candidate(name, endpoint)
        }
        controls = {
            name: value
            for name, value in all_metrics.items()
            if name not in metrics
        }
        ranking = sorted(
            metrics,
            key=lambda name: (
                metrics[name].get("pr_auc_bad") is not None,
                metrics[name].get("pr_auc_bad") or -1,
                metrics[name].get("roc_auc_bad") or -1,
                -(metrics[name].get("mae") or 10_000),
            ),
            reverse=True,
        )
        endpoint_reports[endpoint] = {
            "proxy_payload_n": payload_n,
            "proxy_labeled_n": len(references),
            "proxy_abstention_n": payload_n - len(references),
            "proxy_bad_n": sum(bool(value["bad"]) for value in references.values()),
            "candidate_n": len(metrics),
            "control_candidate_n": len(controls),
            "ranking_by_proxy_pr_auc": ranking,
            "metrics": metrics,
            "control_metrics": controls,
        }
    proxy_models = sorted(
        {
            str(row.get("model"))
            for row in proxy
            if row.get("model") and row.get("kind", "pass") == "pass"
        }
    )
    return {
        "schema_version": 1,
        "analysis": "agent_proxy_exploratory_c7a",
        "proxy_models": proxy_models,
        "proxy_run_quality": {
            "row_n": sum(row.get("kind", "pass") == "pass" for row in proxy),
            "error_row_n": sum(
                row.get("kind", "pass") == "pass" and bool(row.get("error"))
                for row in proxy
            ),
            "error_class_counts": {
                error_class: sum(
                    row.get("kind", "pass") == "pass"
                    and bool(row.get("error"))
                    and str(row.get("error")).split(":", 1)[0] == error_class
                    for row in proxy
                )
                for error_class in sorted(
                    {
                        str(row.get("error")).split(":", 1)[0]
                        for row in proxy
                        if row.get("kind", "pass") == "pass" and row.get("error")
                    }
                )
            },
            "leakage_failure_n": sum(
                row.get("kind", "pass") == "pass"
                and not all(
                    bool((audit or {}).get("passed"))
                    for audit in (row.get("leakage_audit") or {}).values()
                )
                for row in proxy
            ),
        },
        "endpoints": endpoint_reports,
        "limitations": {
            "proxy_is_human_gold": False,
            "formal_c6_passed": False,
            "formal_c7_c9_completed": False,
            "production_decision_allowed": False,
            "required_next_gate": "independent_methodist_labels",
        },
        "phi_check": {
            "contains_source_identifiers": False,
            "contains_sample_identifiers": False,
            "contains_clinical_text": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = evaluate_agent_proxy(
        _rows(args.snapshot),
        _rows(args.replay),
        _rows(args.blind),
        _rows(args.proxy),
        bootstrap_iterations=max(0, args.bootstrap_iterations),
        seed=args.seed,
    )
    report["provenance"] = {
        "evaluator_sha256": _sha256(Path(__file__)),
        "input_sha256": {
            "snapshot": _sha256(args.snapshot),
            "replay": _sha256(args.replay),
            "blind": _sha256(args.blind),
            "proxy": _sha256(args.proxy),
        },
        "bootstrap_iterations": max(0, args.bootstrap_iterations),
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "analysis": report["analysis"],
                "proxy_models": report["proxy_models"],
                "dx_labeled_n": report["endpoints"]["dx"]["proxy_labeled_n"],
                "plan_labeled_n": report["endpoints"]["plan"]["proxy_labeled_n"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
