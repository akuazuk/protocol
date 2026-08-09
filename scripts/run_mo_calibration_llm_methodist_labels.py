#!/usr/bin/env python3
"""Fill C6 methodist labels with an independent blind LLM proxy.

This is an explicit owner-approved substitute for human gold. Labels are marked
``llm_proxy_c6b_not_human_gold`` and must not be treated as methodist review.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_calibration_methodist_ui import (  # noqa: E402
    load_review_pack,
    save_label,
)
from clinical_knowledge.mo_dx_evidence_score import validate_dx_evidence_result  # noqa: E402
from clinical_knowledge.mo_llm_action_judge import extract_json_object  # noqa: E402
from clinical_knowledge.mo_plan_protocol_score import (  # noqa: E402
    validate_plan_concordance_result,
)
from scripts.run_mo_calibration_blind_judge import (  # noqa: E402
    assert_gce_live_contour,
    build_dx_prompt,
    build_plan_prompt,
    pin_dx_semantics,
    pin_plan_route,
)

DEFAULT_MODEL = "gemini-3.1-pro-preview"
REVIEWER_ID = "llm_proxy_c6b_not_human_gold"


def _score(endpoint: str, payload: Mapping[str, Any]) -> float:
    if endpoint == "dx":
        value = payload.get("dx_evidence_pct")
    else:
        value = payload.get("plan_protocol_pct")
        if value is None:
            value = payload.get("plan_general_llm_pct")
    if value is None:
        return 0.0
    return float(value)


def _confidence(payload: Mapping[str, Any]) -> float:
    raw = payload.get("confidence")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.7
    return min(1.0, max(0.0, value))


def _rationale(payload: Mapping[str, Any]) -> str:
    text = str(payload.get("summary_ru") or payload.get("rationale") or "").strip()
    if len(text) < 10:
        text = (
            "LLM-proxy label without human methodist review; "
            f"verdict={payload.get('verdict')}"
        )
    return text[:2000]


def label_from_result(endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    verdict = str(payload.get("verdict") or "")
    icd_fit = "na" if endpoint == "plan" else str(payload.get("icd_fit") or "unknown")
    return {
        "verdict": verdict,
        "score_pct": round(_score(endpoint, payload), 2),
        "potential_harm": bool(payload.get("potential_harm")),
        "icd_fit": icd_fit,
        "confidence": round(_confidence(payload), 3),
        "rationale": _rationale(payload),
    }


def _generate(prompt: str, *, model: str) -> tuple[str, int]:
    from scripts.run_mo_action_queue_llm_judge import _generate_gemini

    return _generate_gemini(prompt, model_name=model)


def judge_endpoint(
    case: Mapping[str, Any],
    *,
    endpoint: str,
    model: str,
    dry_run: bool,
) -> dict[str, Any]:
    pack = case["clinical_case"]
    route = str((case.get("plan_route") or {}).get("route") or "llm_no_kp")
    protocol_context = case.get("protocol_context")
    if endpoint == "dx":
        prompt, _ = build_dx_prompt(pack)
    else:
        prompt, _ = build_plan_prompt(
            pack,
            route=route,
            protocol_context=protocol_context if isinstance(protocol_context, dict) else None,
        )
    if dry_run:
        return {
            "verdict": "partial",
            "score_pct": 60.0,
            "potential_harm": False,
            "icd_fit": "unknown" if endpoint == "dx" else "na",
            "confidence": 0.5,
            "rationale": "dry-run placeholder label for contract checks",
        }
    assert_gce_live_contour()
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw, _latency = _generate(prompt, model=model)
            parsed = extract_json_object(raw)
            if endpoint == "dx":
                result = validate_dx_evidence_result(
                    pin_dx_semantics(parsed),
                    case_id=str(case["sample_id"]),
                )
            else:
                result = validate_plan_concordance_result(
                    pin_plan_route(
                        parsed,
                        route=route,
                        protocol_context=protocol_context
                        if isinstance(protocol_context, dict)
                        else None,
                    ),
                    case_id=str(case["sample_id"]),
                )
            return label_from_result(endpoint, result)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 1:
                raise
            time.sleep(1)
    raise RuntimeError(str(last_error))


def run_labels(
    *,
    model: str,
    dry_run: bool,
    limit: int = 0,
) -> dict[str, Any]:
    pack = load_review_pack(actor=REVIEWER_ID, role="methodist")
    pending = []
    for item in pack["items"]:
        for endpoint in item.get("required_endpoints") or []:
            label = (item.get("labels") or {}).get(endpoint) or {}
            if label.get("verdict"):
                continue
            pending.append((item, str(endpoint)))
    if limit > 0:
        pending = pending[:limit]
    written = 0
    errors: list[dict[str, str]] = []
    for case, endpoint in pending:
        try:
            values = judge_endpoint(case, endpoint=endpoint, model=model, dry_run=dry_run)
            save_label(
                sample_id=str(case["sample_id"]),
                endpoint=endpoint,
                reviewer_id=REVIEWER_ID,
                reviewer_role="methodist",
                expected_reviewed_at=str(
                    ((case.get("labels") or {}).get(endpoint) or {}).get("reviewed_at")
                    or ""
                ),
                **values,
            )
            written += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "sample_id": str(case.get("sample_id")),
                    "endpoint": endpoint,
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
            )
    final = load_review_pack(actor=REVIEWER_ID, role="methodist")
    status = final["status"]
    return {
        "schema_version": 1,
        "model": model,
        "reviewer_id": REVIEWER_ID,
        "proxy_not_human_gold": True,
        "attempted_n": len(pending),
        "written_n": written,
        "error_n": len(errors),
        "errors": errors[:20],
        "label_audit": status,
        "comparison_unsealed": bool(final.get("comparison_unsealed")),
        "passed": bool(status.get("passed")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MO_CALIBRATION_PROXY_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()
    if not args.dry_run:
        assert_gce_live_contour()
    summary = run_labels(model=args.model, dry_run=args.dry_run, limit=max(0, args.limit))
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            args.summary_out.chmod(0o600)
        except OSError:
            pass
    print(
        json.dumps(
            {
                "written_n": summary["written_n"],
                "error_n": summary["error_n"],
                "passed": summary["passed"],
                "comparison_unsealed": summary["comparison_unsealed"],
                "complete_label_n": (summary.get("label_audit") or {}).get("complete_label_n"),
            }
        )
    )
    return 0 if summary["passed"] and summary["error_n"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
