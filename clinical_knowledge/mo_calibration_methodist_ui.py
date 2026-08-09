"""Secure read/write service for the blinded C6 methodist calibration pack."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.build_mo_calibration_methodist_pack import ICD_FITS, VERDICTS, audit_labels

DEFAULT_RUN = "mo-score-v3-2026-08-01-2026-08-08"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def calibration_root() -> Path:
    configured = (os.environ.get("MO_CALIBRATION_C6_ROOT") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    data_root = Path(
        (os.environ.get("MO_DATA_ROOT") or "/var/data/medical_exams").strip()
    ).expanduser()
    return (data_root / "calibration" / DEFAULT_RUN).resolve()


def _paths() -> dict[str, Path]:
    root = calibration_root()
    review = root / "secret" / "methodist"
    return {
        "root": root,
        "cases": review / "methodist_cases.jsonl",
        "labels": review / "methodist_labels.jsonl",
        "lock": review / ".methodist_labels.lock",
        "audit": review / "methodist_access_audit.jsonl",
        "audit_lock": review / ".methodist_access_audit.lock",
        "pilot": root / "secret" / "blind_pilot.jsonl",
        "comparison": root / "secret" / "methodist_llm_comparison_unsealed.jsonl",
        "status": root / "methodist_status.json",
    }


def _append_audit(
    paths: Mapping[str, Path],
    *,
    actor: str,
    role: str,
    action: str,
    sample_id: str = "",
    endpoint: str = "",
) -> None:
    paths["audit_lock"].parent.mkdir(parents=True, exist_ok=True)
    with paths["audit_lock"].open("a+", encoding="utf-8") as lock:
        paths["audit_lock"].chmod(0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with paths["audit"].open("a", encoding="utf-8") as handle:
            try:
                paths["audit"].chmod(0o600)
            except OSError:
                pass
            handle.write(
                json.dumps(
                    {
                        "created_at": _utc_now(),
                        "actor": actor[:120],
                        "role": role[:30],
                        "action": action[:80],
                        "sample_id": sample_id[:12],
                        "endpoint": endpoint[:12],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path.name)
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_no}: row must be object")
        rows.append(value)
    return rows


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _expected(cases: list[dict[str, Any]]) -> set[tuple[str, str]]:
    expected: set[tuple[str, str]] = set()
    for case in cases:
        sample_id = str(case.get("sample_id") or "")
        for endpoint in case.get("required_endpoints") or []:
            if sample_id and endpoint in {"dx", "plan"}:
                expected.add((sample_id, str(endpoint)))
    return expected


def _public_label(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "sample_id",
            "endpoint",
            "verdict",
            "score_pct",
            "potential_harm",
            "icd_fit",
            "confidence",
            "rationale",
            "reviewer_id",
            "reviewed_at",
        )
    }


def _validate_label(
    *,
    endpoint: str,
    verdict: str,
    score_pct: float,
    potential_harm: bool,
    icd_fit: str,
    confidence: float,
    rationale: str,
) -> None:
    if endpoint not in {"dx", "plan"}:
        raise ValueError("invalid_endpoint")
    if verdict not in VERDICTS:
        raise ValueError("invalid_verdict")
    if not 0 <= score_pct <= 100:
        raise ValueError("invalid_score_pct")
    if not isinstance(potential_harm, bool):
        raise ValueError("invalid_potential_harm")
    if endpoint == "dx" and icd_fit not in ICD_FITS:
        raise ValueError("invalid_icd_fit")
    if endpoint == "plan" and icd_fit != "na":
        raise ValueError("plan_icd_fit_must_be_na")
    if not 0 <= confidence <= 1:
        raise ValueError("invalid_confidence")
    if len(rationale.strip()) < 10 or len(rationale) > 2000:
        raise ValueError("invalid_rationale")


def _comparison_rows(pilot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = {
        (str(row.get("sample_id")), int(row.get("pass_no") or 0)): row
        for row in pilot
        if row.get("kind", "pass") == "pass" and not row.get("error")
    }
    output: list[dict[str, Any]] = []
    for adjudication in pilot:
        if adjudication.get("kind") != "adjudication":
            continue
        sample_id = str(adjudication.get("sample_id") or "")
        endpoint = str(adjudication.get("endpoint") or "")
        first = primary.get((sample_id, 1))
        second = primary.get((sample_id, 2))
        if first is None or second is None or endpoint not in {"dx", "plan"}:
            raise ValueError("incomplete_pilot_comparison")
        result_key = "dx_evidence" if endpoint == "dx" else "plan_concordance"
        output.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "endpoint": endpoint,
                "pass_1": first[result_key],
                "pass_2": second[result_key],
                "llm_adjudication": adjudication.get("result"),
            }
        )
    return sorted(output, key=lambda row: (row["sample_id"], row["endpoint"]))


def _update_status(
    paths: Mapping[str, Path],
    *,
    labels: list[dict[str, Any]],
    expected: set[tuple[str, str]],
) -> dict[str, Any]:
    audit = audit_labels(labels, expected=expected)
    status: dict[str, Any] = {}
    if paths["status"].is_file():
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
    status["generated_at"] = _utc_now()
    status["label_audit"] = audit
    status["comparison_unsealed"] = bool(audit["passed"])
    if audit["passed"]:
        comparison = _comparison_rows(_load_rows(paths["pilot"]))
        _atomic_jsonl(paths["comparison"], comparison)
        status.setdefault("artifact_hashes", {})["llm_comparison_unsealed.jsonl"] = (
            hashlib.sha256(paths["comparison"].read_bytes()).hexdigest()
        )
    paths["status"].write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return status


def load_review_pack(*, actor: str = "system", role: str = "system") -> dict[str, Any]:
    paths = _paths()
    cases = _load_rows(paths["cases"])
    labels = _load_rows(paths["labels"])
    expected = _expected(cases)
    labels_by_key = {
        (str(row.get("sample_id")), str(row.get("endpoint"))): _public_label(row)
        for row in labels
    }
    if set(labels_by_key) != expected:
        raise ValueError("methodist_pack_label_keys_mismatch")
    items: list[dict[str, Any]] = []
    for case in cases:
        sample_id = str(case["sample_id"])
        item = dict(case)
        item["labels"] = {
            endpoint: labels_by_key[(sample_id, endpoint)]
            for endpoint in case.get("required_endpoints") or []
        }
        items.append(item)
    audit = audit_labels(labels, expected=expected)
    if not audit["passed"] and paths["comparison"].exists():
        paths["comparison"].unlink()
    _append_audit(paths, actor=actor, role=role, action="calibration_pack_open")
    return {
        "schema_version": 1,
        "items": items,
        "status": audit,
        "comparison_unsealed": bool(audit["passed"] and paths["comparison"].is_file()),
    }


def save_label(
    *,
    sample_id: str,
    endpoint: str,
    verdict: str,
    score_pct: float,
    potential_harm: bool,
    icd_fit: str,
    confidence: float,
    rationale: str,
    reviewer_id: str,
    reviewer_role: str = "methodist",
    expected_reviewed_at: str = "",
) -> dict[str, Any]:
    _validate_label(
        endpoint=endpoint,
        verdict=verdict,
        score_pct=score_pct,
        potential_harm=potential_harm,
        icd_fit=icd_fit,
        confidence=confidence,
        rationale=rationale,
    )
    paths = _paths()
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    with paths["lock"].open("a+", encoding="utf-8") as lock:
        try:
            paths["lock"].chmod(0o600)
        except OSError:
            pass
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cases = _load_rows(paths["cases"])
        labels = _load_rows(paths["labels"])
        expected = _expected(cases)
        key = (sample_id, endpoint)
        if key not in expected:
            raise KeyError("label_not_found")
        updated = False
        for row in labels:
            if (str(row.get("sample_id")), str(row.get("endpoint"))) != key:
                continue
            current_reviewed_at = str(row.get("reviewed_at") or "")
            if current_reviewed_at != str(expected_reviewed_at or ""):
                raise ValueError("label_changed_by_another_reviewer")
            row.update(
                {
                    "verdict": verdict,
                    "score_pct": round(float(score_pct), 2),
                    "potential_harm": potential_harm,
                    "icd_fit": icd_fit,
                    "confidence": round(float(confidence), 3),
                    "rationale": rationale.strip(),
                    "reviewer_id": reviewer_id[:120],
                    "reviewed_at": _utc_now(),
                }
            )
            updated = True
            break
        if not updated:
            raise KeyError("label_not_found")
        _atomic_jsonl(paths["labels"], labels)
        status = _update_status(paths, labels=labels, expected=expected)
        saved = next(
            row
            for row in labels
            if (str(row.get("sample_id")), str(row.get("endpoint"))) == key
        )
        _append_audit(
            paths,
            actor=reviewer_id,
            role=reviewer_role,
            action="calibration_label_save",
            sample_id=sample_id,
            endpoint=endpoint,
        )
        return {
            "ok": True,
            "label": _public_label(saved),
            "status": status["label_audit"],
            "comparison_unsealed": bool(status["comparison_unsealed"]),
        }
