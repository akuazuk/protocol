#!/usr/bin/env python3
"""Build a blinded, PHI-contained methodist review pack for C6."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.run_mo_calibration_blind_judge import (
    blind_case_pack,
    protocol_context_for_case,
)

VERDICTS = frozenset({"good", "partial", "poor", "critical", "blocked", "na"})
ICD_FITS = frozenset({"fit", "partial", "mismatch", "unknown", "na"})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: row must be an object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(_canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _methodist_case(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    endpoints: list[str],
) -> dict[str, Any]:
    pack = blind_case_pack(row, sample_id=sample_id)
    route, protocol_context = protocol_context_for_case(row, pack)
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "required_endpoints": endpoints,
        "clinical_case": pack,
        "plan_route": route,
        "protocol_context": protocol_context,
        "instructions": {
            "blind": (
                "Оценивать исходный клинический текст независимо. "
                "LLM и engine scores скрыты до фиксации labels."
            ),
            "dx": (
                "Проверить, подтверждается ли диагноз жалобами, анамнезом, "
                "объективным статусом и обследованиями."
            ),
            "plan": (
                "Принять диагноз как premise и проверить обследования/лечение; "
                "при KP-grounded использовать только показанный контекст КП."
            ),
        },
    }


def _label_template(sample_id: str, endpoint: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "endpoint": endpoint,
        "verdict": None,
        "score_pct": None,
        "potential_harm": None,
        "icd_fit": None if endpoint == "dx" else "na",
        "confidence": None,
        "rationale": "",
        "reviewer_id": "",
        "reviewed_at": "",
    }


def build_pack(
    cases: list[dict[str, Any]],
    pilot: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cases_by_sample = {f"S{index:03d}": row for index, row in enumerate(cases, 1)}
    adjudications = [
        row for row in pilot if row.get("kind") == "adjudication"
    ]
    endpoint_map: dict[str, set[str]] = {}
    for row in adjudications:
        sample_id = str(row.get("sample_id") or "")
        endpoint = str(row.get("endpoint") or "")
        if sample_id not in cases_by_sample or endpoint not in {"dx", "plan"}:
            raise ValueError("adjudication refers to an unknown sample or endpoint")
        endpoint_map.setdefault(sample_id, set()).add(endpoint)
    if not endpoint_map:
        raise ValueError("pilot has no disagreement adjudications")

    methodist_cases: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []
    primary_by_key = {
        (str(row.get("sample_id")), int(row.get("pass_no") or 0)): row
        for row in pilot
        if row.get("kind", "pass") == "pass" and not row.get("error")
    }
    adjudication_by_key = {
        (str(row.get("sample_id")), str(row.get("endpoint"))): row
        for row in adjudications
    }
    for sample_id in sorted(endpoint_map):
        endpoints = sorted(endpoint_map[sample_id])
        methodist_cases.append(
            _methodist_case(
                cases_by_sample[sample_id],
                sample_id=sample_id,
                endpoints=endpoints,
            )
        )
        for endpoint in endpoints:
            labels.append(_label_template(sample_id, endpoint))
            first = primary_by_key.get((sample_id, 1))
            second = primary_by_key.get((sample_id, 2))
            if first is None or second is None:
                raise ValueError("each disagreement requires two successful blind passes")
            result_key = "dx_evidence" if endpoint == "dx" else "plan_concordance"
            comparison.append(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "endpoint": endpoint,
                    "pass_1": first[result_key],
                    "pass_2": second[result_key],
                    "llm_adjudication": adjudication_by_key[(sample_id, endpoint)]["result"],
                }
            )
    return methodist_cases, labels, comparison


def audit_labels(
    labels: list[dict[str, Any]],
    *,
    expected: set[tuple[str, str]],
    minimum_cases: int = 15,
) -> dict[str, Any]:
    seen: set[tuple[str, str]] = set()
    invalid: Counter[str] = Counter()
    complete = 0
    for row in labels:
        row_invalid: set[str] = set()
        key = (str(row.get("sample_id") or ""), str(row.get("endpoint") or ""))
        if key in seen:
            invalid["duplicate"] += 1
        seen.add(key)
        endpoint = key[1]
        verdict = str(row.get("verdict") or "")
        try:
            score = float(row.get("score_pct"))
            confidence = float(row.get("confidence"))
        except (TypeError, ValueError):
            score = confidence = -1
        if endpoint not in {"dx", "plan"}:
            invalid["endpoint"] += 1
            row_invalid.add("endpoint")
        if verdict not in VERDICTS:
            invalid["verdict"] += 1
            row_invalid.add("verdict")
        if not 0 <= score <= 100:
            invalid["score_pct"] += 1
            row_invalid.add("score_pct")
        if not isinstance(row.get("potential_harm"), bool):
            invalid["potential_harm"] += 1
            row_invalid.add("potential_harm")
        if endpoint == "dx" and str(row.get("icd_fit") or "") not in ICD_FITS:
            invalid["icd_fit"] += 1
            row_invalid.add("icd_fit")
        if not 0 <= confidence <= 1:
            invalid["confidence"] += 1
            row_invalid.add("confidence")
        if len(str(row.get("rationale") or "").strip()) < 10:
            invalid["rationale"] += 1
            row_invalid.add("rationale")
        if not str(row.get("reviewer_id") or "").strip():
            invalid["reviewer_id"] += 1
            row_invalid.add("reviewer_id")
        if not str(row.get("reviewed_at") or "").strip():
            invalid["reviewed_at"] += 1
            row_invalid.add("reviewed_at")
        if not row_invalid:
            complete += 1
    missing = expected - seen
    extra = seen - expected
    case_n = len({sample_id for sample_id, _ in seen})
    passed = (
        not invalid
        and not missing
        and not extra
        and case_n >= minimum_cases
        and complete == len(expected)
    )
    return {
        "schema_version": 1,
        "expected_label_n": len(expected),
        "seen_label_n": len(seen),
        "case_n": case_n,
        "complete_label_n": complete,
        "invalid_counts": dict(invalid),
        "missing_n": len(missing),
        "extra_n": len(extra),
        "minimum_cases": minimum_cases,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--secret-out-dir", type=Path, required=True)
    parser.add_argument("--public-status", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Validate completed labels instead of recreating template")
    args = parser.parse_args()

    cases = _load_rows(args.cases)
    pilot = _load_rows(args.pilot)
    methodist_cases, template, comparison = build_pack(cases, pilot)
    expected = {(row["sample_id"], row["endpoint"]) for row in template}
    args.secret_out_dir.mkdir(parents=True, exist_ok=True)
    try:
        args.secret_out_dir.chmod(0o700)
    except OSError:
        pass
    cases_out = args.secret_out_dir / "methodist_cases.jsonl"
    template_out = args.secret_out_dir / "methodist_labels.jsonl"
    comparison_out = args.secret_out_dir.parent / "methodist_llm_comparison_unsealed.jsonl"
    legacy_comparison_out = args.secret_out_dir / "llm_comparison_sealed.jsonl"
    if legacy_comparison_out.exists():
        legacy_comparison_out.unlink()
    _write_jsonl(cases_out, methodist_cases)
    if args.labels is None:
        if template_out.is_file():
            labels = _load_rows(template_out)
        else:
            _write_jsonl(template_out, template)
            labels = template
        labels_out = template_out
    else:
        labels = _load_rows(args.labels)
        labels_out = args.labels
    audit = audit_labels(labels, expected=expected)
    comparison_bytes = "".join(_canonical(row) + "\n" for row in comparison).encode("utf-8")
    if audit["passed"]:
        _write_jsonl(comparison_out, comparison)
    elif comparison_out.exists():
        comparison_out.unlink()
    public = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "review_case_n": len(methodist_cases),
        "required_endpoint_n": len(template),
        "endpoint_counts": dict(Counter(row["endpoint"] for row in template)),
        "artifact_hashes": {
            "methodist_cases.jsonl": _hash_file(cases_out),
            "methodist_labels.jsonl": _hash_file(labels_out),
            "llm_comparison_frozen": hashlib.sha256(comparison_bytes).hexdigest(),
        },
        "comparison_unsealed": audit["passed"],
        "label_audit": audit,
        "phi_boundary": {
            "secret_directory_mode": oct(args.secret_out_dir.stat().st_mode & 0o777),
            "public_contains_case_ids": False,
            "public_contains_clinical_text": False,
        },
    }
    args.public_status.parent.mkdir(parents=True, exist_ok=True)
    args.public_status.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(public, ensure_ascii=False))
    return 0 if (args.labels is None or audit["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
