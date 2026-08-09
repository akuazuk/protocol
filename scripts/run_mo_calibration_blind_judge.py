#!/usr/bin/env python3
"""Run blind Endpoint C/D calibration passes.

Live model calls are hard-blocked unless the process explicitly runs in the GCE
LLM contour.  Inputs and detailed outputs are PHI-bearing and must remain under
``/var/data/medical_exams`` (or another explicitly secured data root).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_dx_evidence_score import validate_dx_evidence_result  # noqa: E402
from clinical_knowledge.mo_plan_protocol_score import (  # noqa: E402
    resolve_plan_route,
    validate_plan_concordance_result,
)
from clinical_knowledge.mo_llm_action_judge import extract_json_object  # noqa: E402

ENGINE = "mo_score_calibration_blind_v1"
SCHEMA_VERSION = 1
DEFAULT_MODEL = "gemini-3.6-flash"
FORBIDDEN_PROMPT_KEYS = frozenset(
    {
        "overall_pct",
        "overall_pct_v3",
        "deep",
        "evaluation_v4",
        "axes",
        "findings",
        "queue_reason",
        "queue_severity",
        "attention_primary",
        "attention_reason_ru",
        "rubric_pct",
        "reg55_section_pct",
        "regulatory_pct",
        "zone1_pct",
        "zone2a_pct",
        "zone2b_pct",
        "protocol_suggest",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clip(value: Any, limit: int = 4000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _source(row: Mapping[str, Any]) -> dict[str, Any]:
    for name in ("clinical", "slots", "detail"):
        value = row.get(name)
        if isinstance(value, dict) and value:
            return {**value, **row}
    return dict(row)


def _value(source: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = source.get(name)
        if isinstance(value, str) and value.strip():
            return _clip(value)
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def blind_case_pack(row: Mapping[str, Any], *, sample_id: str) -> dict[str, Any]:
    """Allowlist the only fields that may reach a blind prompt."""
    source = _source(row)
    return {
        "sample_id": sample_id,
        "meta": {
            "age_group": _value(source, "age_group", "patient_age_group"),
            "age_years": _value(source, "age_years", "patient_age"),
            "sex": _value(source, "sex", "patient_sex", "gender"),
            "specialty": _value(source, "doctor_specialization", "specialty", "specialization"),
            "visit_type": _value(source, "document_kind", "visit_type"),
        },
        "evidence": {
            "complaints": _value(source, "complaints", "complaint"),
            "anamnesis": _value(source, "anamnesis", "anamnesis_doctor", "anamnesis_auto", "history"),
            "objective_status": _value(source, "objective_status", "objective", "status_localis"),
            "exam_data": _value(source, "exam_data", "exam_results", "investigations"),
        },
        "diagnosis": {
            "text": _value(source, "clinical_diagnosis", "diagnosis", "diagnosis_main_text"),
            "icd": _value(source, "mkb_code_main", "diagnosis_code", "icd_main"),
        },
        "plan": {
            "exam_recommendations": _value(
                source, "exam_recommendations", "exam_recs", "recommendations_exam"
            ),
            "treatment_recommendations": _value(
                source, "treatment_recommendations", "treatment_recs", "recommendations_treatment"
            ),
            "follow_up": _value(source, "follow_up", "dispensary_info", "return_date"),
        },
    }


def _walk_forbidden(value: Any, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_PROMPT_KEYS:
                found.append(f"{path}.{key_text}")
            found.extend(_walk_forbidden(child, path=f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_forbidden(child, path=f"{path}[{index}]"))
    return found


def audit_prompt_input(
    prompt_input: Mapping[str, Any],
    *,
    source_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    forbidden_paths = _walk_forbidden(prompt_input)
    rendered = _canonical(prompt_input)
    leaked_canaries: list[str] = []
    if source_row:
        for key in FORBIDDEN_PROMPT_KEYS:
            value = source_row.get(key)
            if isinstance(value, (str, int, float)) and len(str(value)) >= 4 and str(value) in rendered:
                leaked_canaries.append(key)
    return {
        "passed": not forbidden_paths and not leaked_canaries,
        "forbidden_paths": forbidden_paths,
        "leaked_canaries": leaked_canaries,
        "input_hash": _hash(prompt_input),
    }


def build_dx_prompt(case_pack: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = {
        "sample_id": case_pack["sample_id"],
        "meta": case_pack["meta"],
        "clinical_evidence": case_pack["evidence"],
        "diagnosis": case_pack["diagnosis"],
    }
    schema = {
        "dx_evidence_pct": 0,
        "verdict": "good|partial|poor|critical|blocked|na",
        "supported_by": [{"slot": "complaints|anamnesis|objective_status|exam_data|diagnosis|icd|meta", "evidence": ""}],
        "not_supported_by": [],
        "contradictions": [],
        "icd_fit": "fit|partial|mismatch|unknown|na",
        "potential_harm": False,
        "summary_ru": "",
        "provenance": "llm_blind",
    }
    prompt = "\n".join(
        [
            "Оцени только Endpoint C: следует ли поставленный диагноз из представленных клинических данных.",
            "Не оценивай назначения, лечение, полноту плана или соответствие протоколу.",
            "Валидный код МКБ сам по себе не доказывает диагноз. Отсутствие МКБ при текстовом диагнозе не штрафуй.",
            "Если клинических данных недостаточно, verdict=blocked и dx_evidence_pct=null.",
            "Если нет ни диагноза, ни МКБ, verdict=na и dx_evidence_pct=null.",
            "Каждый вывод привяжи к дословному evidence и его slot. Не додумывай отсутствующие факты.",
            "Ответь одним JSON без markdown по схеме:",
            _canonical(schema),
            "Слепой вход:",
            _canonical(payload),
        ]
    )
    return prompt, payload


def build_plan_prompt(
    case_pack: Mapping[str, Any],
    *,
    route: str,
    protocol_context: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if route not in {"kp_grounded", "llm_no_kp"}:
        raise ValueError("unknown plan route")
    payload: dict[str, Any] = {
        "sample_id": case_pack["sample_id"],
        "meta": case_pack["meta"],
        "accepted_diagnosis": case_pack["diagnosis"],
        "plan": case_pack["plan"],
    }
    if route == "kp_grounded":
        if not protocol_context:
            raise ValueError("kp_grounded prompt requires protocol context")
        payload["protocol_requirements"] = dict(protocol_context)
        schema = {
            "exam_protocol_pct": 0,
            "treatment_protocol_pct": 0,
            "followup_protocol_pct": 0,
            "plan_protocol_pct": 0,
            "verdict": "good|partial|poor|critical|blocked|na",
            "kp_status": "matched",
            "kp_path": protocol_context.get("kp_path"),
            "kp_trust": protocol_context.get("kp_trust"),
            "missing_required": [],
            "off_protocol": [],
            "source_refs": [],
            "potential_harm": False,
            "summary_ru": "",
            "provenance": "kp_grounded",
        }
        instruction = (
            "Оцени Endpoint D по переданным требованиям КП. Диагноз является принятой premise: "
            "не переоценивай и не оспаривай его. Используй только переданные требования и source_refs."
        )
    else:
        schema = {
            "plan_general_llm_pct": 0,
            "verdict": "good|partial|poor|critical|blocked|na",
            "kp_status": "unmatched",
            "missing_required": [],
            "off_protocol": [],
            "source_refs": [],
            "potential_harm": False,
            "summary_ru": "",
            "provenance": "llm_no_kp",
        }
        instruction = (
            "Оцени общую клиническую достаточность плана без утверждений о соответствии КП. "
            "Диагноз является принятой premise: не переоценивай и не оспаривай его."
        )
    prompt = "\n".join(
        [
            instruction,
            "Раздельно учти обследования, лечение и follow-up. Не додумывай отсутствующие назначения.",
            "Если данных недостаточно, verdict=blocked и итоговый процент=null.",
            "Ответь одним JSON без markdown по схеме:",
            _canonical(schema),
            "Слепой вход:",
            _canonical(payload),
        ]
    )
    return prompt, payload


def _ref_text(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, Mapping):
        return _clip(
            value.get("source_ref")
            or value.get("local_path")
            or value.get("path")
            or value.get("section")
            or _canonical(value),
            300,
        )
    return _clip(value, 300)


def _item_text(value: Any, *names: str) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, Mapping):
        parts = [_clip(value.get(name), 160) for name in names if value.get(name)]
        return "; ".join(parts)
    return _clip(value, 200)


def protocol_context_for_case(
    row: Mapping[str, Any],
    case_pack: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    suggest = row.get("protocol_suggest")
    if not isinstance(suggest, Mapping):
        from clinical_knowledge.case_protocol_suggest import suggest_protocols_for_mo_case

        suggest = suggest_protocols_for_mo_case(
            clinical=_source(row),
            record=dict(row),
            findings=[],
            llm_judge={},
            limit=3,
            attach_history=False,
        )
    route = resolve_plan_route(suggest)
    if route["route"] != "kp_grounded":
        return route, None
    hit = route["hit"] or {}
    source_path = str(hit.get("source_path") or "").strip()
    context: dict[str, Any] = {
        "kp_path": source_path,
        "kp_trust": route["kp_trust"],
        "applicability": {},
        "required_exams": [],
        "treatment": [],
        "follow_up": [],
        "safety": [],
        "source_refs": [source_path] if source_path else [],
    }
    try:
        from clinical_knowledge.protocol_summary.loader import load_summary_by_path

        summary = load_summary_by_path(source_path, usable_only=True) if source_path else None
        if summary:
            context["applicability"] = summary.applicability.model_dump()
            diagnosis = str((case_pack.get("diagnosis") or {}).get("text") or "").lower()
            condition = next(
                (
                    candidate
                    for candidate in summary.conditions
                    if candidate.name.lower() in diagnosis
                    or any(code in str((case_pack.get("diagnosis") or {}).get("icd") or "") for code in candidate.icd10_codes)
                ),
                summary.conditions[0] if summary.conditions else None,
            )
            if condition:
                context["required_exams"] = [
                    _item_text(item, "name", "timing", "comment") for item in condition.required_exams[:16]
                ]
                treatment = condition.treatment
                if treatment:
                    context["treatment"] = [
                        *[_item_text(item, "text") for item in treatment.non_drug[:8]],
                        *[_item_text(item, "drug_group", "indication") for item in treatment.drug_groups[:8]],
                        *[
                            _item_text(
                                item,
                                "drug_name",
                                "active_substance",
                                "drug_group",
                                "dose_text",
                                "frequency_text",
                                "duration_text",
                                "indication",
                            )
                            for item in treatment.drugs[:12]
                        ],
                        *[_item_text(item, "name", "indication") for item in treatment.procedures[:8]],
                    ]
                context["follow_up"] = [
                    _item_text(item, "text", "timing") for item in condition.follow_up[:12]
                ]
                context["safety"] = [
                    *[_item_text(item, "text") for item in condition.contraindications[:8]],
                    *[_item_text(item, "text", "severity") for item in condition.red_flags[:8]],
                ]
                refs = [
                    *condition.source_refs,
                    *[item.source_ref for item in condition.required_exams[:16]],
                    *[item.source_ref for item in condition.follow_up[:12]],
                ]
                context["source_refs"] = sorted(
                    {text for text in (_ref_text(ref) for ref in refs) if text}
                )[:24] or context["source_refs"]
    except Exception as exc:  # noqa: BLE001
        route = {
            "route": "llm_no_kp",
            "kp_status": "unmatched",
            "fallback_reason": f"summary_unavailable:{type(exc).__name__}",
            "hit": hit,
        }
        return route, None
    if not context["source_refs"]:
        route = {
            "route": "llm_no_kp",
            "kp_status": "unmatched",
            "fallback_reason": "summary_has_no_source_refs",
            "hit": hit,
        }
        return route, None
    return route, context


def assert_gce_live_contour() -> None:
    explicit = (os.environ.get("MO_LLM_EXECUTION_HOST") or "").strip().lower()
    run_host = (os.environ.get("RUN_HOST") or "").strip().lower()
    if explicit != "gce" or run_host not in {"gcp", "gce"}:
        raise RuntimeError(
            "live calibration judge is allowed only on GCE "
            "(MO_LLM_EXECUTION_HOST=gce and RUN_HOST=gcp)"
        )


def pin_plan_route(
    raw: Mapping[str, Any],
    *,
    route: str,
    protocol_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Pin controller-owned route fields; the model grades content, not routing."""
    normalized = dict(raw)
    if route == "kp_grounded":
        if not protocol_context:
            raise ValueError("grounded route requires protocol context")
        normalized.update(
            {
                "provenance": "kp_grounded",
                "kp_status": "matched",
                "kp_path": protocol_context.get("kp_path"),
                "kp_trust": protocol_context.get("kp_trust"),
            }
        )
        if not normalized.get("source_refs"):
            normalized["source_refs"] = list(protocol_context.get("source_refs") or [])
    elif route == "llm_no_kp":
        normalized.update(
            {
                "provenance": "llm_no_kp",
                "kp_status": "unmatched",
                "kp_path": "",
                "kp_trust": None,
                "plan_protocol_pct": None,
                "source_refs": [],
            }
        )
    else:
        raise ValueError("unknown plan route")
    return normalized


def pin_dx_semantics(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Remove unsupported categorical claims and pin blind provenance."""
    normalized = dict(raw)
    normalized["provenance"] = "llm_blind"
    if (
        str(normalized.get("icd_fit") or "").lower() == "mismatch"
        and not normalized.get("not_supported_by")
        and not normalized.get("contradictions")
    ):
        normalized["icd_fit"] = "unknown"
    return normalized


def _generate(prompt: str, *, model: str) -> tuple[str, int]:
    from scripts.run_mo_action_queue_llm_judge import _generate_gemini

    return _generate_gemini(prompt, model_name=model)


def judge_case(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    pass_no: int,
    model: str,
    dry_run: bool,
) -> dict[str, Any]:
    pack = blind_case_pack(row, sample_id=sample_id)
    route, protocol_context = protocol_context_for_case(row, pack)
    dx_prompt, dx_input = build_dx_prompt(pack)
    plan_prompt, plan_input = build_plan_prompt(
        pack,
        route=route["route"],
        protocol_context=protocol_context,
    )
    dx_audit = audit_prompt_input(dx_input, source_row=row)
    plan_audit = audit_prompt_input(plan_input, source_row=row)
    if not dx_audit["passed"] or not plan_audit["passed"]:
        raise ValueError("blind prompt leakage audit failed")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "sample_id": sample_id,
        "pass_no": pass_no,
        "model": model,
        "route": route["route"],
        "kp_status": route["kp_status"],
        "prompt_hashes": {"dx": _hash(dx_prompt), "plan": _hash(plan_prompt)},
        "input_hashes": {"dx": dx_audit["input_hash"], "plan": plan_audit["input_hash"]},
        "leakage_audit": {"dx": dx_audit, "plan": plan_audit},
        "dx_evidence": None,
        "plan_concordance": None,
        "latency_ms": {"dx": None, "plan": None},
        "retry_count": {"dx": 0, "plan": 0},
        "error": None,
    }
    if dry_run:
        result["dry_run"] = True
        return result
    assert_gce_live_contour()
    try:
        dx_elapsed = 0
        for attempt in range(2):
            dx_raw, dx_ms = _generate(dx_prompt, model=model)
            dx_elapsed += dx_ms
            try:
                result["dx_evidence"] = validate_dx_evidence_result(
                    pin_dx_semantics(extract_json_object(dx_raw)),
                    case_id=sample_id,
                )
                result["retry_count"]["dx"] = attempt
                break
            except (ValueError, json.JSONDecodeError):
                if attempt == 1:
                    raise
        result["latency_ms"]["dx"] = dx_elapsed

        plan_elapsed = 0
        for attempt in range(2):
            plan_raw, plan_ms = _generate(plan_prompt, model=model)
            plan_elapsed += plan_ms
            try:
                result["plan_concordance"] = validate_plan_concordance_result(
                    pin_plan_route(
                        extract_json_object(plan_raw),
                        route=route["route"],
                        protocol_context=protocol_context,
                    ),
                    case_id=sample_id,
                )
                result["retry_count"]["plan"] = attempt
                break
            except (ValueError, json.JSONDecodeError):
                if attempt == 1:
                    raise
        result["latency_ms"]["plan"] = plan_elapsed
    except Exception as exc:  # noqa: BLE001 - keep the smoke batch complete
        result["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    return result


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: row must be object")
            rows.append(value)
    return rows


def _select_smoke_rows(
    rows: list[dict[str, Any]],
    *,
    manifest_path: Path | None,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or manifest_path is None:
        return rows if limit <= 0 else rows[:limit]
    manifest = _load_rows(manifest_path)
    if len(manifest) != len(rows):
        raise ValueError("secret manifest and cases must have identical row counts")
    paired = list(zip(rows, manifest))
    matched = [
        row
        for row, item in paired
        if bool((item.get("signals") or {}).get("kp_matched"))
    ]
    unmatched = [
        row
        for row, item in paired
        if not bool((item.get("signals") or {}).get("kp_matched"))
    ]
    selected = [*matched[:2], *unmatched[: max(0, limit - min(2, len(matched)))]]
    if len(selected) < limit:
        selected_ids = {id(row) for row in selected}
        selected.extend(row for row in rows if id(row) not in selected_ids)
    return selected[:limit]


def _summary(
    results: list[dict[str, Any]],
    *,
    model: str,
    passes: int,
    require_route_coverage: bool,
) -> dict[str, Any]:
    routes = Counter(str(row.get("route") or "unknown") for row in results)
    errors = [str(row.get("error")) for row in results if row.get("error")]
    leakage_failures = sum(
        1
        for row in results
        if not all(
            bool((row.get("leakage_audit") or {}).get(stage, {}).get("passed"))
            for stage in ("dx", "plan")
        )
    )
    parse_success = sum(
        1
        for row in results
        if row.get("dx_evidence") is not None and row.get("plan_concordance") is not None
    )
    route_coverage = bool(routes.get("kp_grounded") and routes.get("llm_no_kp"))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "generated_at": _utc_now(),
        "model": model,
        "passes": passes,
        "result_n": len(results),
        "unique_sample_n": len({row.get("sample_id") for row in results}),
        "route_counts": dict(routes),
        "parse_success_n": parse_success,
        "error_n": len(errors),
        "leakage_failure_n": leakage_failures,
        "geo_error_n": sum(
            1 for error in errors if "location is not supported" in error.lower() or "geo" in error.lower()
        ),
        "route_coverage_passed": route_coverage,
        "passed": bool(results)
        and parse_success == len(results)
        and not errors
        and leakage_failures == 0
        and (route_coverage or not require_route_coverage),
        "config_hash": _hash(
            {
                "engine": ENGINE,
                "schema_version": SCHEMA_VERSION,
                "model": model,
                "forbidden_prompt_keys": sorted(FORBIDDEN_PROMPT_KEYS),
            }
        ),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--model", default=os.environ.get("MO_CALIBRATION_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-route-coverage", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        assert_gce_live_contour()
    rows = _load_rows(args.cases)
    rows = _select_smoke_rows(rows, manifest_path=args.manifest, limit=args.limit)
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        sample_id = f"S{index:03d}"
        for pass_no in range(1, max(1, args.passes) + 1):
            started = time.perf_counter()
            result = judge_case(
                row,
                sample_id=sample_id,
                pass_no=pass_no,
                model=args.model,
                dry_run=args.dry_run,
            )
            result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            results.append(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(_canonical(result) + "\n")
    try:
        args.out.chmod(0o600)
    except OSError:
        pass
    summary = _summary(
        results,
        model=args.model,
        passes=max(1, args.passes),
        require_route_coverage=args.require_route_coverage,
    )
    if args.dry_run:
        summary["dry_run"] = True
        summary["passed"] = (
            summary["leakage_failure_n"] == 0
            and summary["error_n"] == 0
            and (summary["route_coverage_passed"] or not args.require_route_coverage)
        )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
