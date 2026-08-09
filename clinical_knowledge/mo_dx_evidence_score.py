"""Explicit contract for Endpoint C: diagnosis-to-evidence concordance.

This module validates shadow calibration output only.  It does not change the
production MO score or infer semantic correctness deterministically.
"""
from __future__ import annotations

from typing import Any, Mapping

ENGINE = "mo_dx_evidence_v1"
SCHEMA_VERSION = 1
VERDICTS = frozenset({"good", "partial", "poor", "critical", "blocked", "na"})
ICD_FITS = frozenset({"fit", "partial", "mismatch", "unknown", "na"})
PROVENANCE = frozenset({"deterministic", "llm_blind", "methodist"})
EVIDENCE_SLOTS = frozenset(
    {
        "complaints",
        "anamnesis",
        "objective_status",
        "exam_data",
        "diagnosis",
        "icd",
        "meta",
    }
)


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _pct(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 100 else None


def _evidence_list(value: Any, *, field: str) -> list[dict[str, str]]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    out: list[dict[str, str]] = []
    for index, item in enumerate(value[:12]):
        if isinstance(item, str):
            slot = "meta"
            evidence = _clip(item, 240)
        elif isinstance(item, Mapping):
            slot = str(item.get("slot") or "").strip()
            evidence = _clip(item.get("evidence") or item.get("quote"), 240)
        else:
            raise ValueError(f"{field}[{index}] must be an object")
        if slot not in EVIDENCE_SLOTS:
            raise ValueError(f"{field}[{index}].slot is invalid")
        if not evidence:
            raise ValueError(f"{field}[{index}].evidence is required")
        out.append({"slot": slot, "evidence": evidence})
    return out


def validate_dx_evidence_result(
    raw: Mapping[str, Any],
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    verdict = str(raw.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"invalid dx evidence verdict: {verdict}")
    score = _pct(raw.get("dx_evidence_pct"))
    if verdict in {"blocked", "na"}:
        if score is not None:
            raise ValueError(f"{verdict} result must not have dx_evidence_pct")
    elif score is None:
        raise ValueError("dx_evidence_pct 0-100 is required")
    fit = str(raw.get("icd_fit") or "unknown").strip().lower()
    if fit not in ICD_FITS:
        raise ValueError(f"invalid icd_fit: {fit}")
    provenance = str(raw.get("provenance") or "").strip().lower()
    if provenance not in PROVENANCE:
        raise ValueError(f"invalid provenance: {provenance}")
    supported = _evidence_list(raw.get("supported_by"), field="supported_by")
    unsupported = _evidence_list(raw.get("not_supported_by"), field="not_supported_by")
    contradictions = _evidence_list(raw.get("contradictions"), field="contradictions")
    if verdict in {"poor", "critical"} and not (unsupported or contradictions):
        raise ValueError("poor/critical verdict requires unsupported or contradictory evidence")
    if fit == "mismatch" and not (unsupported or contradictions):
        raise ValueError("ICD mismatch requires evidence")
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "case_id": str(raw.get("case_id") or case_id or "").strip(),
        "dx_evidence_pct": score,
        "verdict": verdict,
        "supported_by": supported,
        "not_supported_by": unsupported,
        "contradictions": contradictions,
        "icd_fit": fit,
        "potential_harm": bool(raw.get("potential_harm")),
        "summary_ru": _clip(raw.get("summary_ru"), 600),
        "provenance": provenance,
    }


def dx_evidence_eligibility(case_pack: Mapping[str, Any]) -> dict[str, Any]:
    """Determine only whether semantic grading is possible, without grading it."""
    slots = case_pack.get("slots") if isinstance(case_pack.get("slots"), Mapping) else case_pack
    diagnosis = str(
        slots.get("clinical_diagnosis")
        or slots.get("diagnosis")
        or slots.get("diagnosis_text")
        or ""
    ).strip()
    icd = str(slots.get("mkb_code_main") or slots.get("diagnosis_code") or slots.get("icd") or "").strip()
    evidence_present = [
        name
        for name, aliases in {
            "complaints": ("complaints", "complaint"),
            "anamnesis": ("anamnesis", "anamnesis_doctor", "history"),
            "objective_status": ("objective_status", "objective", "status_localis"),
            "exam_data": ("exam_data", "exam_results", "investigations"),
        }.items()
        if any(str(slots.get(alias) or "").strip() for alias in aliases)
    ]
    if not diagnosis and not icd:
        status = "na"
        reason = "missing_diagnosis_and_icd"
    elif not evidence_present:
        status = "blocked"
        reason = "insufficient_clinical_evidence"
    else:
        status = "eligible"
        reason = ""
    return {
        "status": status,
        "reason": reason,
        "diagnosis_present": bool(diagnosis),
        "icd_present": bool(icd),
        "evidence_slots_present": evidence_present,
        "icd_absence_penalty": False,
    }


def nonsemantic_dx_result(
    case_pack: Mapping[str, Any],
    *,
    case_id: str | None = None,
) -> dict[str, Any] | None:
    """Return NA/blocked result when semantic grading must not be attempted."""
    eligibility = dx_evidence_eligibility(case_pack)
    if eligibility["status"] == "eligible":
        return None
    return validate_dx_evidence_result(
        {
            "case_id": case_id,
            "dx_evidence_pct": None,
            "verdict": eligibility["status"],
            "supported_by": [],
            "not_supported_by": [],
            "contradictions": [],
            "icd_fit": "na" if not eligibility["icd_present"] else "unknown",
            "potential_harm": False,
            "summary_ru": eligibility["reason"],
            "provenance": "deterministic",
        }
    )
