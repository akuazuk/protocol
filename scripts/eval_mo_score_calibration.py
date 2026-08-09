#!/usr/bin/env python3
"""Create PHI-safe aggregate diagnostics for MO score calibration artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def _equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 0.01
    except (TypeError, ValueError):
        return False


def replay_drift(
    cases_path: Path,
    snapshot_path: Path,
    replay_path: Path,
) -> dict[str, Any]:
    cases = _rows(cases_path)
    snapshots = _rows(snapshot_path)
    replays = _rows(replay_path)
    if not (len(cases) == len(snapshots) == len(replays)):
        raise ValueError("cases, snapshots, and replay must have identical row counts")
    source_versions: Counter[str] = Counter()
    overall: Counter[str] = Counter()
    axis_source_snapshot: dict[str, Counter[str]] = {}
    replay_mismatches: Counter[str] = Counter()
    replay_delta: dict[str, list[float]] = {}
    source_hash_comparable = source_hash_equal = 0
    arm_d_fingerprints: Counter[str] = Counter()
    for case, snapshot, replay in zip(cases, snapshots, replays):
        source_eval = case.get("evaluation_v4") if isinstance(case.get("evaluation_v4"), dict) else {}
        scores = snapshot.get("scores") if isinstance(snapshot.get("scores"), dict) else {}
        source_versions[str(source_eval.get("scorer_version") or "missing")] += 1
        for label, value in (
            ("evaluation_v4", source_eval.get("score_pct")),
            ("row_overall", case.get("overall_pct")),
            ("deep_overall", (case.get("deep") or {}).get("overall_pct")),
        ):
            if value is not None and scores.get("overall_pct") is not None:
                overall[f"{label}_n"] += 1
                overall[f"{label}_equal"] += _equal(value, scores["overall_pct"])
        for axis, value in (source_eval.get("axes") or {}).items():
            stored = (scores.get("axes") or {}).get(axis)
            if value is None or stored is None:
                continue
            counts = axis_source_snapshot.setdefault(str(axis), Counter())
            counts["n"] += 1
            counts["equal"] += _equal(value, stored)
        for field, comparison in (replay.get("comparisons") or {}).items():
            if not isinstance(comparison, dict) or not comparison.get("comparable"):
                continue
            if not comparison.get("match"):
                replay_mismatches[str(field)] += 1
            delta = comparison.get("delta")
            if delta is not None:
                replay_delta.setdefault(str(field), []).append(float(delta))
        source_hash = str(case.get("_content_hash") or "")
        snapshot_hash = str((snapshot.get("versions") or {}).get("content_hash") or "")
        if source_hash and snapshot_hash:
            source_hash_comparable += 1
            source_hash_equal += source_hash == snapshot_hash
        arm_d_fingerprints[str(replay.get("arm_d_fingerprint") or "missing")] += 1
    return {
        "schema_version": 1,
        "case_n": len(cases),
        "source_scorer_versions": dict(source_versions),
        "overall_source_vs_snapshot": dict(overall),
        "axis_source_vs_snapshot": {
            axis: dict(counts) for axis, counts in axis_source_snapshot.items()
        },
        "source_content_hash": {
            "comparable_n": source_hash_comparable,
            "equal_n": source_hash_equal,
        },
        "replay_mismatch_counts": dict(replay_mismatches),
        "replay_delta_ranges": {
            field: {"min": min(values), "max": max(values)}
            for field, values in replay_delta.items()
            if values
        },
        "arm_d_fingerprints": dict(arm_d_fingerprints),
        "conclusion": (
            "stored_snapshot_not_replay_baseline"
            if replay_mismatches
            else "stored_snapshot_reproducible"
        ),
        "phi_check": {
            "contains_row_identifiers": False,
            "contains_clinical_text": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = replay_drift(args.cases, args.snapshot, args.replay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_n": report["case_n"], "conclusion": report["conclusion"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
