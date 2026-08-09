#!/usr/bin/env python3
"""Build the frozen MO score-calibration sample and deterministic snapshot.

Clinical payloads and identifiers are written only to ``--secret-dir``.  The
public manifest contains aggregate coverage and hashes, but no row identifiers,
doctor labels, diagnoses, or clinical text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_FROM = "2026-08-01"
DEFAULT_TO = "2026-08-08"
DEFAULT_SEED = 42
DEFAULT_TARGET = 30
DEFAULT_SENTINEL = "3643940"
BANDS = ("0-49", "50-59", "60-69", "70-79", "80+")
AXES = ("documentation", "clinical_concordance", "safety", "regulatory")
SECRET_FILE_NAMES = frozenset({"secret_manifest.jsonl", "secret_cases.jsonl", "engine_snapshot.jsonl"})
ARM_D_FILES = (
    "clinical_knowledge/kz_evaluation_v4.py",
    "clinical_knowledge/kz_evaluation_engine.py",
    "clinical_knowledge/kz_evaluation_schema.py",
    "clinical_knowledge/kz_deep_eval.py",
    "clinical_knowledge/reg55_criteria.py",
    "config/mo_scorer_v4.yaml",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def arm_d_fingerprint() -> dict[str, Any]:
    """Freeze code, config, summary corpus, and relevant scorer flags."""
    component_hashes: dict[str, str] = {}
    for relative in ARM_D_FILES:
        path = ROOT / relative
        component_hashes[relative] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        )
    summary_root = ROOT / "data" / "protocol_summaries"
    summary_hash = hashlib.sha256()
    summary_files = 0
    if summary_root.is_dir():
        for path in sorted(item for item in summary_root.rglob("*") if item.is_file()):
            summary_hash.update(str(path.relative_to(summary_root)).encode("utf-8"))
            summary_hash.update(b"\0")
            summary_hash.update(hashlib.sha256(path.read_bytes()).digest())
            summary_files += 1
    flags = {
        name: os.environ.get(name)
        for name in (
            "KZ_EVALUATION_V4_ENABLED",
            "KZ_EVALUATION_V4_PRIMARY",
            "KZ_EVALUATION_V4_GATE",
            "CASE_PROTOCOL_SUGGEST",
            "MO_PATIENT_HISTORY_ENABLED",
        )
    }
    payload = {
        "engine": "kz_evaluation_v4",
        "component_hashes": component_hashes,
        "protocol_summary_tree_hash": summary_hash.hexdigest(),
        "protocol_summary_file_n": summary_files,
        "environment_flags": flags,
    }
    return {**payload, "fingerprint": _sha256(payload)}


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if 0 <= result <= 100 else None


def _as_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return default
    return parsed if isinstance(parsed, type(default)) else default


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def score_band(value: Any) -> str:
    score = _as_float(value)
    if score is None:
        return "na"
    if score < 50:
        return "0-49"
    if score < 60:
        return "50-59"
    if score < 70:
        return "60-69"
    if score < 80:
        return "70-79"
    return "80+"


def _nested(row: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = row.get(name)
    return value if isinstance(value, dict) else {}


def _case_key(row: Mapping[str, Any]) -> str:
    return str(_first(row, "mis_id", "id", "case_id", "visit_id") or "").strip()


def _case_aliases(row: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get(name) or "").strip()
        for name in ("mis_id", "id", "case_id", "visit_id")
        if str(row.get(name) or "").strip()
    }


def _clinical(row: Mapping[str, Any]) -> dict[str, Any]:
    for name in ("clinical", "slots", "detail"):
        value = row.get(name)
        if isinstance(value, dict) and value:
            return value
    return dict(row)


def _text_present(row: Mapping[str, Any], *names: str) -> bool:
    source = _clinical(row)
    for name in names:
        value = source.get(name)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, dict)) and value:
            return True
    return False


def _all_findings(row: Mapping[str, Any], warehouse: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for value in (
        row.get("findings"),
        _nested(row, "deep").get("findings"),
        _nested(row, "evaluation_v4").get("findings"),
        warehouse.get("findings"),
    ):
        if isinstance(value, list):
            out.extend(item for item in value if isinstance(item, dict))
    return out


def _axis_map(row: Mapping[str, Any], warehouse: Mapping[str, Any]) -> dict[str, float | None]:
    sources = (
        warehouse.get("axes"),
        row.get("axes"),
        _nested(row, "deep").get("axes"),
        _nested(row, "evaluation_v4").get("axes"),
    )
    out: dict[str, float | None] = {axis: None for axis in AXES}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for axis in AXES:
            value = source.get(axis)
            if isinstance(value, dict):
                value = _first(value, "score", "score_pct", "pct")
            parsed = _as_float(value)
            if parsed is not None and out[axis] is None:
                out[axis] = parsed
    return out


def _protocol_status(
    row: Mapping[str, Any],
    warehouse: Mapping[str, Any] | None = None,
) -> tuple[bool, str, bool]:
    warehouse = warehouse or {}
    suggest = row.get("protocol_suggest")
    if not isinstance(suggest, dict):
        suggest = _nested(row, "system").get("protocol_suggest")
    if isinstance(suggest, dict):
        for item in suggest.get("items") or []:
            if not isinstance(item, dict) or str(item.get("match_kind") or "") != "clinical":
                continue
            score = _as_float(item.get("score"))
            if score is not None and score >= 50:
                return True, str(item.get("trust") or item.get("trust_level") or "B").upper(), True
    zone_status = str(
        _first(warehouse, "zone2b_kp_status")
        or _first(row, "zone2b_kp_status", "kp_status")
        or _nested(row, "zone2b").get("kp_status")
        or ""
    ).lower()
    return zone_status in {"matched", "found", "available"}, "", bool(zone_status or isinstance(suggest, dict))


def normalize_candidate(
    row: Mapping[str, Any],
    *,
    warehouse: Mapping[str, Any] | None = None,
    training_use: bool = False,
    compute_kp: bool = False,
) -> dict[str, Any] | None:
    row = dict(row)
    warehouse = warehouse or {}
    key = _case_key(row)
    if not key:
        return None
    deep = _nested(row, "deep")
    evaluation = _nested(row, "evaluation_v4")
    overall = _as_float(
        _first(warehouse, "overall_pct")
        or _first(row, "overall_pct")
        or _first(evaluation, "score_pct", "overall_pct")
        or _first(deep, "overall_pct")
    )
    axes = _axis_map(row, warehouse)
    findings = _all_findings(row, warehouse)
    severities = {
        str(item.get("severity") or "").upper()
        for item in findings
        if item.get("passed") is not True
    }
    counts = deep.get("n_by_severity") if isinstance(deep.get("n_by_severity"), dict) else {}
    has_p0p1 = bool({"P0", "P1"} & severities) or any(int(counts.get(x, 0) or 0) > 0 for x in ("P0", "P1"))
    attention = str(_first(warehouse, "attention_primary") or _first(row, "attention_primary") or "")
    status = str(_first(warehouse, "status") or _first(row, "status") or "").lower()
    action = bool(attention.strip()) or has_p0p1 or status in {
        "manual_review_required",
        "review",
        "poor",
        "critical",
    }
    reg55 = _as_float(
        _first(warehouse, "reg55_section_pct")
        or _first(row, "reg55_section_pct")
        or _nested(row, "reg55_section").get("reg55_section_pct")
    )
    weak_points = (
        warehouse.get("reg55_weak_points")
        or _as_json(warehouse.get("reg55_weak_points_json"), [])
        or row.get("reg55_weak_points")
        or _nested(row, "reg55_section").get("weak_points")
        or []
    )
    if not isinstance(weak_points, list):
        weak_points = []
    kp_matched, kp_trust, kp_checked = _protocol_status(row, warehouse)
    if not kp_matched and compute_kp:
        try:
            from clinical_knowledge.case_protocol_suggest import (
                clinical_kp_hit,
                suggest_protocols_for_mo_case,
            )

            suggest = suggest_protocols_for_mo_case(
                clinical=_clinical(row),
                record=row,
                findings=findings,
                llm_judge={},
                limit=3,
                attach_history=False,
            )
            row["protocol_suggest"] = suggest
            hit = clinical_kp_hit(suggest)
            if hit:
                kp_matched = True
                kp_trust = str(hit.get("trust") or hit.get("trust_level") or "B").upper()
            kp_checked = True
        except Exception:  # noqa: BLE001 - unmatched is an explicit calibration stratum
            kp_checked = True
    icd_status = " ".join(
        str(value or "").lower()
        for value in (
            row.get("icd_fit"),
            row.get("icd_status"),
            _nested(row, "icd_review").get("verdict"),
            _nested(row, "llm_icd_review").get("verdict"),
            row.get("mkb_code_agreement"),
        )
    )
    finding_codes = {
        str(item.get("finding_code") or item.get("code") or "").strip()
        for item in findings
    }
    dispute_codes = {
        "B_icd_mismatch_mis",
        "B_icd_dir_no_match",
        "B_icd_dir_text_mismatch",
        "B_icd_name_no_match",
        "B_icd_name_weak_match",
        "B_icd_llm_review_no",
        "B_icd_llm_review_partial",
    }
    dispute = bool(finding_codes & dispute_codes) or any(
        token in icd_status for token in ("mismatch", "dispute", "contradiction", "weak")
    )
    visit_date = str(_first(warehouse, "visit_date") or _first(row, "visit_date", "date") or "")[:10]
    specialty = str(
        _first(warehouse, "specialty")
        or _first(row, "specialty", "doctor_specialization", "specialization")
        or "Не указано"
    ).strip()
    doctor = str(
        _first(warehouse, "doctor_key", "doctor_id")
        or _first(row, "doctor_key", "doctor_id", "specialist_id_from_visit", "doctor_fio")
        or f"unknown:{key}"
    ).strip()
    return {
        "case_key": key,
        "aliases": sorted(_case_aliases(row)),
        "visit_date": visit_date,
        "overall_pct": overall,
        "band": score_band(overall),
        "specialty": specialty,
        "doctor_key": doctor,
        "training_use": bool(training_use or row.get("training_use")),
        "high_action": bool(overall is not None and overall >= 80 and action),
        "has_p0p1": has_p0p1,
        "action": action,
        "reg55_pct": reg55,
        "regulatory_pct": axes["regulatory"],
        "reg55_gap": bool(
            reg55 is not None
            and axes["regulatory"] is not None
            and abs(reg55 - float(axes["regulatory"])) >= 5
        ),
        "reg55_high_weak": bool(reg55 is not None and reg55 >= 80 and weak_points),
        "icd_dx_dispute": dispute,
        "kp_matched": kp_matched,
        "kp_trust": kp_trust,
        "kp_checked": kp_checked,
        "has_exam_results": _text_present(row, "exam_data", "exam_results", "investigations"),
        "has_treatment": _text_present(
            row, "treatment_recommendations", "treatment_recs", "recommendations_treatment", "treatment"
        ),
        "axes": axes,
        "row": dict(row),
        "warehouse": dict(warehouse),
    }


def _load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                if isinstance(row, dict) and not row.get("error"):
                    rows.append(row)
    return rows


def _load_clinical_csv(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item = dict(row)
            for key in ("visit_id", "id", "mis_id"):
                value = str(item.get(key) or "").strip()
                if value:
                    index[value] = item
    return index


def _load_warehouse(path: Path | None) -> tuple[dict[str, dict[str, Any]], set[str]]:
    if path is None:
        return {}, set()
    if not path.is_file():
        raise FileNotFoundError(path)
    cases: dict[str, dict[str, Any]] = {}
    training: set[str] = set()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        columns = {row[1] for row in db.execute("PRAGMA table_info(fact_mo_case)")}
        wanted = [
            name
            for name in (
                "mis_id", "visit_id", "visit_date", "overall_pct", "overall_pct_v3", "status",
                "scorer_version", "score_schema_version", "doctor_key", "doctor_id", "specialty",
                "diagnosis_code", "diagnosis_text", "zone1_pct", "zone2a_pct", "zone2b_pct",
                "zone1_band", "zone2a_band", "zone2b_band", "zone2b_kp_status",
                "attention_primary", "attention_reason_ru", "rubric_json", "rubric_pct",
                "reg55_section_pct", "reg55_band", "reg55_pack", "reg55_applicable_n",
                "reg55_weak_points_json", "content_hash", "updated_at",
            )
            if name in columns
        ]
        if wanted:
            for row in db.execute(f"SELECT {', '.join(wanted)} FROM fact_mo_case"):
                item = dict(row)
                item["axes"] = {}
                item["findings"] = []
                for key in (str(item.get("mis_id") or ""), str(item.get("visit_id") or "")):
                    if key:
                        cases[key] = item
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "fact_mo_score_axis" in tables:
            for row in db.execute("SELECT mis_id, axis, score FROM fact_mo_score_axis"):
                item = cases.get(str(row[0]))
                if item is not None:
                    item["axes"][str(row[1])] = row[2]
        if "fact_mo_finding" in tables:
            fcols = {row[1] for row in db.execute("PRAGMA table_info(fact_mo_finding)")}
            selected = [name for name in ("mis_id", "finding_code", "severity", "passed", "axis", "is_shadow") if name in fcols]
            for row in db.execute(f"SELECT {', '.join(selected)} FROM fact_mo_finding"):
                item = cases.get(str(row[0]))
                if item is not None:
                    item["findings"].append(dict(zip(selected, row)))
        if "crm_review_pack" in tables:
            for row in db.execute("SELECT case_id, visit_id, mis_id FROM crm_review_pack WHERE training_use=1"):
                training.update(str(value or "").strip() for value in row if str(value or "").strip())
    return cases, training


def _requirements(target: int) -> dict[str, int]:
    if target < 30:
        raise ValueError("pilot target must be at least 30")
    req = {f"band:{band}": 4 for band in BANDS}
    req.update(
        {
            "high_action": 3,
            "reg55_gap": 3,
            "reg55_high_weak": 2,
            "icd_dx_dispute": 4,
            "kp_matched": 8,
            "kp_unmatched": 6,
            "has_exam_results": 4,
            "has_treatment": 4,
            "specialties": 4,
        }
    )
    return req


def prepare_kp_checked_pool(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the canonical KP matcher on a bounded, balanced candidate pool."""
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    chosen: dict[str, dict[str, Any]] = {}

    def add(items: Iterable[dict[str, Any]], limit: int) -> None:
        added = 0
        for item in items:
            if item["case_key"] in chosen:
                continue
            chosen[item["case_key"]] = item
            added += 1
            if added >= limit:
                break

    add((item for item in shuffled if item["training_use"]), 30)
    add(
        (
            item
            for item in shuffled
            if DEFAULT_SENTINEL in set(item.get("aliases") or [item["case_key"]])
        ),
        1,
    )
    for band in BANDS:
        per_doctor: Counter[str] = Counter()
        diverse: list[dict[str, Any]] = []
        for item in shuffled:
            if item["band"] != band or per_doctor[item["doctor_key"]] >= 2:
                continue
            diverse.append(item)
            per_doctor[item["doctor_key"]] += 1
            if len(diverse) >= 24:
                break
        add(diverse, 24)
    for signal in (
        "high_action",
        "reg55_gap",
        "reg55_high_weak",
        "icd_dx_dispute",
        "has_exam_results",
        "has_treatment",
    ):
        add((item for item in shuffled if item[signal]), 24)
    specialties: Counter[str] = Counter()
    for item in shuffled:
        if specialties[item["specialty"]] >= 3:
            continue
        chosen.setdefault(item["case_key"], item)
        specialties[item["specialty"]] += 1
        if len(specialties) >= 12 and all(value >= 3 for value in specialties.values()):
            break

    enriched: dict[str, dict[str, Any]] = {}
    for item in chosen.values():
        if item.get("kp_checked"):
            enriched[item["case_key"]] = item
            continue
        normalized = normalize_candidate(
            item["row"],
            warehouse=item.get("warehouse") or {},
            training_use=bool(item.get("training_use")),
            compute_kp=True,
        )
        if normalized is not None:
            enriched[item["case_key"]] = normalized
    return [enriched.get(item["case_key"], item) for item in candidates]


def _coverage(selected: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in selected:
        counts[f"band:{item['band']}"] += 1
        for name in (
            "high_action", "reg55_gap", "reg55_high_weak", "icd_dx_dispute",
            "kp_matched", "has_exam_results", "has_treatment",
        ):
            if item[name]:
                counts[name] += 1
        if item.get("kp_checked", False) and not item["kp_matched"]:
            counts["kp_unmatched"] += 1
    counts["specialties"] = len({item["specialty"] for item in selected})
    counts["training_use"] = sum(1 for item in selected if item["training_use"])
    return dict(counts)


def _eligible(item: dict[str, Any], selected: list[dict[str, Any]], target: int) -> bool:
    if len(selected) >= target or any(current["case_key"] == item["case_key"] for current in selected):
        return False
    return sum(1 for current in selected if current["doctor_key"] == item["doctor_key"]) < 3


def _satisfies_requirement(
    item: dict[str, Any],
    requirement: str,
    selected: list[dict[str, Any]],
) -> bool:
    if requirement.startswith("band:"):
        return item["band"] == requirement.split(":", 1)[1]
    if requirement == "kp_unmatched":
        return bool(item.get("kp_checked", False) and not item["kp_matched"])
    if requirement == "specialties":
        return item["specialty"] not in {current["specialty"] for current in selected}
    return bool(item.get(requirement))


def _constraint_search(
    candidates: list[dict[str, Any]],
    mandatory: list[dict[str, Any]],
    *,
    target: int,
    requirements: dict[str, int],
    seed: int,
    attempts: int = 400,
) -> list[dict[str, Any]] | None:
    """Randomized rarest-constraint-first search for an exact feasible pilot."""
    rng = random.Random(seed + 991)
    for _ in range(attempts):
        selected = list(mandatory)
        while len(selected) < target:
            coverage = _coverage(selected)
            unmet = [
                name
                for name, minimum in requirements.items()
                if coverage.get(name, 0) < minimum
            ]
            if not unmet:
                break
            eligible_by_req: dict[str, list[dict[str, Any]]] = {}
            for name in unmet:
                eligible_by_req[name] = [
                    item
                    for item in candidates
                    if _eligible(item, selected, target)
                    and _satisfies_requirement(item, name, selected)
                ]
            if any(not values for values in eligible_by_req.values()):
                break
            rarest_n = min(len(values) for values in eligible_by_req.values())
            rarest = [name for name, values in eligible_by_req.items() if len(values) == rarest_n]
            requirement = rng.choice(rarest)
            options = eligible_by_req[requirement]

            def overlap(item: dict[str, Any]) -> float:
                return sum(
                    int(_satisfies_requirement(item, name, selected))
                    for name in unmet
                ) + rng.random()

            selected.append(max(options, key=overlap))
        if len(selected) < target:
            remaining = [item for item in candidates if _eligible(item, selected, target)]
            rng.shuffle(remaining)
            selected.extend(remaining[: target - len(selected)])
        coverage = _coverage(selected)
        if (
            len(selected) == target
            and all(coverage.get(name, 0) >= minimum for name, minimum in requirements.items())
        ):
            return selected
    return None


def select_sample(
    candidates: list[dict[str, Any]],
    *,
    target: int = DEFAULT_TARGET,
    seed: int = DEFAULT_SEED,
    sentinel: str = DEFAULT_SENTINEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requirements = _requirements(target)
    sentinel_item = next(
        (item for item in candidates if sentinel in set(item.get("aliases") or [item["case_key"]])),
        None,
    )
    if sentinel_item is None:
        raise ValueError(f"sentinel {sentinel} not found")
    mandatory = [
        sentinel_item,
        *[
            item
            for item in candidates
            if item["training_use"] and item["case_key"] != sentinel_item["case_key"]
        ],
    ]
    if len(mandatory) > target:
        raise ValueError(f"training-use + sentinel rows exceed target: {len(mandatory)} > {target}")
    selected: list[dict[str, Any]] = []
    for item in mandatory:
        if not _eligible(item, selected, target):
            raise ValueError("mandatory rows violate max 3 cases per doctor")
        selected.append(item)

    rng = random.Random(seed)
    tie_break = {item["case_key"]: rng.random() for item in candidates}

    # Freeze the pre-registered minimum of four cases in each score band first.
    # Other rare strata are then optimized inside the remaining ten slots.
    for band in BANDS:
        while sum(1 for item in selected if item["band"] == band) < requirements[f"band:{band}"]:
            available = [
                item
                for item in candidates
                if item["band"] == band and _eligible(item, selected, target)
            ]
            if not available:
                break
            selected.append(
                max(
                    available,
                    key=lambda item: (
                        sum(
                            int(bool(item[name]))
                            for name in (
                                "high_action", "reg55_gap", "reg55_high_weak",
                                "icd_dx_dispute", "kp_matched",
                                "has_exam_results", "has_treatment",
                            )
                        ),
                        tie_break[item["case_key"]],
                    ),
                )
            )

    while len(selected) < target:
        coverage = _coverage(selected)
        deficits = {name: max(0, minimum - coverage.get(name, 0)) for name, minimum in requirements.items()}
        if not any(deficits.values()):
            break

        def gain(item: dict[str, Any]) -> tuple[float, float]:
            value = 0.0
            if deficits.get(f"band:{item['band']}", 0):
                value += 20
            for name in (
                "high_action", "reg55_gap", "reg55_high_weak", "icd_dx_dispute",
                "kp_matched", "has_exam_results", "has_treatment",
            ):
                if deficits.get(name, 0) and item[name]:
                    value += 10
            if deficits.get("kp_unmatched", 0) and not item["kp_matched"]:
                value += 10
            if deficits.get("specialties", 0) and item["specialty"] not in {
                current["specialty"] for current in selected
            }:
                value += 12
            return value, tie_break[item["case_key"]]

        available = [item for item in candidates if _eligible(item, selected, target)]
        if not available:
            break
        choice = max(available, key=gain)
        if gain(choice)[0] <= 0:
            break
        selected.append(choice)

    for item in sorted(candidates, key=lambda value: tie_break[value["case_key"]]):
        if _eligible(item, selected, target):
            selected.append(item)
        if len(selected) >= target:
            break

    # Repair a full greedy sample; neutral swaps escape one-swap local minima.
    mandatory_keys = {item["case_key"] for item in mandatory}
    seen_samples = {frozenset(item["case_key"] for item in selected)}
    for _ in range(120):
        current_coverage = _coverage(selected)
        current_cost = sum(
            max(0, minimum - current_coverage.get(name, 0))
            for name, minimum in requirements.items()
        )
        if current_cost == 0:
            break
        best: tuple[int, dict[str, Any], dict[str, Any]] | None = None
        neutral: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        selected_keys = {item["case_key"] for item in selected}
        doctor_counts = Counter(item["doctor_key"] for item in selected)
        for incoming in candidates:
            if incoming["case_key"] in selected_keys:
                continue
            for outgoing in selected:
                if outgoing["case_key"] in mandatory_keys:
                    continue
                if (
                    incoming["doctor_key"] != outgoing["doctor_key"]
                    and doctor_counts[incoming["doctor_key"]] >= 3
                ):
                    continue
                trial = [item for item in selected if item["case_key"] != outgoing["case_key"]]
                trial.append(incoming)
                coverage = _coverage(trial)
                if any(
                    current_coverage.get(name, 0) >= minimum
                    and coverage.get(name, 0) < minimum
                    for name, minimum in requirements.items()
                ):
                    continue
                cost = sum(
                    max(0, minimum - coverage.get(name, 0))
                    for name, minimum in requirements.items()
                )
                signature = frozenset(
                    item["case_key"] for item in trial
                )
                if signature in seen_samples:
                    continue
                if cost < current_cost and (best is None or cost < best[0]):
                    best = (cost, outgoing, incoming)
                elif cost == current_cost:
                    neutral.append((cost, outgoing, incoming))
        if best is None:
            if not neutral:
                break
            best = rng.choice(neutral)
        _, outgoing, incoming = best
        selected = [item for item in selected if item["case_key"] != outgoing["case_key"]]
        selected.append(incoming)
        seen_samples.add(frozenset(item["case_key"] for item in selected))
    if any(
        _coverage(selected).get(name, 0) < minimum
        for name, minimum in requirements.items()
    ):
        feasible = _constraint_search(
            candidates,
            mandatory,
            target=target,
            requirements=requirements,
            seed=seed,
        )
        if feasible is not None:
            selected = feasible
    coverage = _coverage(selected)
    deficits = {name: minimum - coverage.get(name, 0) for name, minimum in requirements.items() if coverage.get(name, 0) < minimum}
    doctor_max = max(Counter(item["doctor_key"] for item in selected).values(), default=0)
    audit = {
        "target": target,
        "selected": len(selected),
        "sentinel_present": any(
            sentinel in set(item.get("aliases") or [item["case_key"]]) for item in selected
        ),
        "all_training_use_present": all(
            any(chosen["case_key"] == item["case_key"] for chosen in selected)
            for item in candidates
            if item["training_use"]
        ),
        "max_cases_per_doctor": doctor_max,
        "coverage": coverage,
        "requirements": requirements,
        "deficits": deficits,
        "passed": (
            len(selected) == target
            and not deficits
            and doctor_max <= 3
            and any(
                sentinel in set(item.get("aliases") or [item["case_key"]])
                for item in selected
            )
        ),
    }
    return selected, audit


def extract_engine_snapshot(item: Mapping[str, Any]) -> dict[str, Any]:
    row = item["row"]
    warehouse = item.get("warehouse") if isinstance(item.get("warehouse"), dict) else {}
    deep = _nested(row, "deep")
    evaluation = _nested(row, "evaluation_v4")
    axes = _axis_map(row, warehouse)
    findings = _all_findings(row, warehouse)
    severity = Counter(
        str(finding.get("severity") or "none").upper()
        for finding in findings
        if finding.get("passed") is not True
    )
    zones = {}
    for name in ("zone1", "zone2a", "zone2b"):
        value = _as_float(
            _first(warehouse, f"{name}_pct")
            or _first(row, f"{name}_pct")
            or _nested(row, name).get("score_pct")
        )
        zones[name] = {
            "score_pct": value,
            "band": _first(warehouse, f"{name}_band") or _first(row, f"{name}_band"),
        }
    reg55_weak = (
        _as_json(warehouse.get("reg55_weak_points_json"), [])
        or row.get("reg55_weak_points")
        or _nested(row, "reg55_section").get("weak_points")
        or []
    )
    snapshot = {
        "sample_id": _sha256({"case_key": item["case_key"]})[:16],
        "source_ids": {
            "case_id": str(_first(row, "case_id") or ""),
            "visit_id": str(_first(row, "visit_id") or ""),
            "mis_id": str(_first(row, "mis_id", "id") or ""),
        },
        "visit_date": item["visit_date"],
        "scores": {
            "overall_pct": item["overall_pct"],
            "overall_pct_v3": _as_float(
                _first(warehouse, "overall_pct_v3")
                or _first(row, "overall_pct_v3")
                or _first(deep, "overall_pct")
            ),
            "axes": axes,
            "zones": zones,
            "rubric_pct": _as_float(
                _first(warehouse, "rubric_pct")
                or _first(row, "rubric_pct")
                or _nested(row, "rubric_mz").get("rubric_pct")
            ),
            "reg55": {
                "score_pct": item["reg55_pct"],
                "band": _first(warehouse, "reg55_band") or _first(row, "reg55_band"),
                "applicable_n": _first(warehouse, "reg55_applicable_n") or _first(row, "reg55_applicable_n"),
                "weak_points": reg55_weak if isinstance(reg55_weak, list) else [],
            },
        },
        "findings": {
            "n_by_severity": dict(severity),
            "codes": sorted(
                {
                    str(finding.get("finding_code") or finding.get("code") or "")
                    for finding in findings
                    if finding.get("finding_code") or finding.get("code")
                }
            ),
        },
        "action": {
            "member": item["action"],
            "attention_primary": _first(warehouse, "attention_primary") or _first(row, "attention_primary"),
            "reason": _first(warehouse, "attention_reason_ru") or _first(row, "attention_reason_ru"),
        },
        "icd_pipeline": {
            "code": _first(warehouse, "diagnosis_code") or _first(row, "mkb_code_main", "diagnosis_code"),
            "directory": row.get("icd_directory"),
            "name_review": row.get("icd_name_review"),
            "llm_review": row.get("llm_icd_review") or row.get("icd_review"),
        },
        "existing_llm": row.get("llm_grade") or row.get("llm_action_judge"),
        "versions": {
            "scorer": _first(warehouse, "scorer_version") or _first(row, "scorer_version") or evaluation.get("scorer_version"),
            "schema": _first(warehouse, "score_schema_version") or _first(row, "score_schema_version"),
            "content_hash": _first(warehouse, "content_hash") or _sha256(row),
        },
    }
    snapshot["snapshot_hash"] = _sha256(snapshot)
    return snapshot


def replay_v4(
    item: Mapping[str, Any],
    *,
    drug_ctx: dict[str, Any] | None = None,
    fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay the primary deterministic scorer from the selected full payload."""
    from clinical_knowledge.kz_deep_eval import load_drug_ctx, resolve_protocol_ctx
    from clinical_knowledge.kz_evaluation_v4 import evaluate_kz_v4

    row = dict(item["row"])
    deep = _nested(row, "deep")
    result = evaluate_kz_v4(
        row,
        protocol_ctx=resolve_protocol_ctx(row),
        drug_ctx=drug_ctx if drug_ctx is not None else load_drug_ctx(),
        legacy={
            "deep_overall_pct": deep.get("overall_pct"),
            "deep_status": deep.get("status"),
            "l1_overall_pct": row.get("overall_pct"),
            "v3_score_pct": _nested(row, "evaluation_v3").get("score_pct"),
        },
    )
    axes = result.axes.model_dump()
    stored_axes = item.get("axes") if isinstance(item.get("axes"), dict) else {}
    comparisons: dict[str, dict[str, Any]] = {}

    def compare(name: str, stored: Any, replayed: Any) -> None:
        left = _as_float(stored)
        right = _as_float(replayed)
        comparable = left is not None and right is not None
        comparisons[name] = {
            "stored": left,
            "replayed": right,
            "delta": round(right - left, 6) if comparable else None,
            "comparable": comparable,
            "match": bool(comparable and abs(right - left) <= 0.01),
        }

    compare("overall_pct", item.get("overall_pct"), result.score_pct)
    for axis in AXES:
        compare(f"axis:{axis}", stored_axes.get(axis), axes.get(axis))
    comparable = [value for value in comparisons.values() if value["comparable"]]
    return {
        "case_key": item["case_key"],
        "replay_hash": _sha256(
            {
                "overall_pct": result.score_pct,
                "axes": axes,
                "scorer_version": result.scorer_version,
                "schema_version": result.schema_version,
            }
        ),
        "scorer_version": result.scorer_version,
        "schema_version": result.schema_version,
        "arm_d_fingerprint": str((fingerprint or {}).get("fingerprint") or ""),
        "comparisons": comparisons,
        "comparable_n": len(comparable),
        "passed": bool(comparable) and all(value["match"] for value in comparable),
    }


def replay_selected(
    selected: list[dict[str, Any]],
    *,
    fingerprint: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay once with shared immutable contexts and return a PHI-safe aggregate."""
    from clinical_knowledge.kz_deep_eval import load_drug_ctx

    drug_ctx = load_drug_ctx()
    rows: list[dict[str, Any]] = []
    errors = 0
    for item in selected:
        try:
            rows.append(replay_v4(item, drug_ctx=drug_ctx, fingerprint=fingerprint))
        except Exception as exc:  # noqa: BLE001 - preserve every pilot row for audit
            errors += 1
            rows.append(
                {
                    "case_key": item["case_key"],
                    "passed": False,
                    "comparable_n": 0,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            )
    failed = sum(1 for row in rows if not row.get("passed"))
    audit = {
        "attempted_n": len(rows),
        "passed_n": len(rows) - failed,
        "failed_n": failed,
        "error_n": errors,
        "all_cases_reproducible": bool(rows) and failed == 0,
        "audit_complete": bool(rows)
        and errors == 0
        and all(int(row.get("comparable_n") or 0) == 1 + len(AXES) for row in rows),
    }
    return rows, audit


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def write_outputs(
    selected: list[dict[str, Any]],
    audit: dict[str, Any],
    *,
    secret_dir: Path,
    public_manifest: Path,
    source_paths: list[Path],
    seed: int,
    replay_rows: list[dict[str, Any]] | None = None,
    replay_audit: dict[str, Any] | None = None,
    fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if public_manifest.name in SECRET_FILE_NAMES or secret_dir in public_manifest.parents:
        raise ValueError("public manifest must be outside secret-dir")
    secret_dir.mkdir(parents=True, exist_ok=True)
    try:
        secret_dir.chmod(0o700)
    except OSError:
        pass
    manifest_rows = [
        {
            "sample_order": index,
            "case_key": item["case_key"],
            "visit_date": item["visit_date"],
            "band": item["band"],
            "specialty": item["specialty"],
            "doctor_key": item["doctor_key"],
            "signals": {
                name: item[name]
                for name in (
                    "training_use", "high_action", "reg55_gap", "reg55_high_weak",
                    "icd_dx_dispute", "kp_matched", "has_exam_results", "has_treatment",
                )
            },
        }
        for index, item in enumerate(selected, 1)
    ]
    case_rows = [item["row"] for item in selected]
    snapshots = [extract_engine_snapshot(item) for item in selected]
    manifest_path = secret_dir / "secret_manifest.jsonl"
    cases_path = secret_dir / "secret_cases.jsonl"
    snapshot_path = secret_dir / "engine_snapshot.jsonl"
    replay_path = secret_dir / "engine_replay.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(cases_path, case_rows)
    _write_jsonl(snapshot_path, snapshots)
    secret_paths = [manifest_path, cases_path, snapshot_path]
    if replay_rows is not None:
        _write_jsonl(replay_path, replay_rows)
        secret_paths.append(replay_path)
    for path in secret_paths:
        try:
            path.chmod(0o600)
        except OSError:
            pass
    public = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "period": {"from": DEFAULT_FROM, "to": DEFAULT_TO},
        "seed": seed,
        "source_file_count": len(source_paths),
        "selected_n": len(selected),
        "audit": audit,
        "specialty_count": len({item["specialty"] for item in selected}),
        "band_counts": dict(Counter(item["band"] for item in selected)),
        "signal_counts": {
            name: sum(1 for item in selected if item[name])
            for name in (
                "training_use", "high_action", "reg55_gap", "reg55_high_weak",
                "icd_dx_dispute", "kp_matched", "has_exam_results", "has_treatment",
            )
        },
        "replay_audit": replay_audit,
        "arm_d_fingerprint": dict(fingerprint or {}),
        "secret_artifact_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in secret_paths
        },
        "phi_check": {
            "contains_row_identifiers": False,
            "contains_doctor_labels": False,
            "contains_clinical_text": False,
        },
    }
    public_manifest.parent.mkdir(parents=True, exist_ok=True)
    public_manifest.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--warehouse", type=Path)
    parser.add_argument("--clinical-csv", type=Path)
    parser.add_argument("--secret-dir", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--date-from", default=DEFAULT_FROM)
    parser.add_argument("--date-to", default=DEFAULT_TO)
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sentinel", default=DEFAULT_SENTINEL)
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="build C0 only; C1 replay is required for the frozen pilot",
    )
    args = parser.parse_args()
    source_rows = _load_jsonl(args.cases)
    warehouse, training_ids = _load_warehouse(args.warehouse)
    clinical = _load_clinical_csv(args.clinical_csv)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source_rows:
        source_aliases = _case_aliases(row)
        clinical_row = next(
            (clinical[alias] for alias in source_aliases if alias in clinical),
            {},
        )
        if clinical_row:
            row = {**clinical_row, **row}
        key = _case_key(row)
        aliases = _case_aliases(row)
        wh = next((warehouse[alias] for alias in aliases if alias in warehouse), {})
        visit_date = str(_first(wh, "visit_date") or _first(row, "visit_date", "date") or "")[:10]
        if not args.date_from <= visit_date <= args.date_to or key in seen:
            continue
        normalized = normalize_candidate(
            row,
            warehouse=wh,
            training_use=bool(aliases & training_ids),
        )
        if normalized is not None:
            candidates.append(normalized)
            seen.add(key)
    candidates = prepare_kp_checked_pool(candidates, seed=args.seed)
    selected, audit = select_sample(
        candidates,
        target=args.target_n,
        seed=args.seed,
        sentinel=str(args.sentinel),
    )
    fingerprint = arm_d_fingerprint()
    replay_rows = replay_audit = None
    if not args.skip_replay:
        replay_rows, replay_audit = replay_selected(selected, fingerprint=fingerprint)
    public = write_outputs(
        selected,
        audit,
        secret_dir=args.secret_dir,
        public_manifest=args.public_manifest,
        source_paths=args.cases,
        seed=args.seed,
        replay_rows=replay_rows,
        replay_audit=replay_audit,
        fingerprint=fingerprint,
    )
    passed = bool(audit["passed"]) and (
        args.skip_replay or bool((replay_audit or {}).get("audit_complete"))
    )
    print(
        json.dumps(
            {
                "selected_n": public["selected_n"],
                "sample_audit_passed": audit["passed"],
                "replay_audit": replay_audit,
            }
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
