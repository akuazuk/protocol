"""Explicit contract for Endpoint D: diagnosis-conditioned plan concordance."""
from __future__ import annotations

from typing import Any, Mapping

ENGINE = "mo_plan_protocol_v1"
SCHEMA_VERSION = 1
VERDICTS = frozenset({"good", "partial", "poor", "critical", "blocked", "na"})
KP_TRUST = frozenset({"A", "B", "C", "D"})
GROUNDED_TRUST = frozenset({"A", "B"})
PROVENANCE = frozenset({"kp_grounded", "llm_no_kp"})


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


def _strings(value: Any, *, limit: int = 12, width: int = 240) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("expected a list")
    return [_clip(item, width) for item in value[:limit] if _clip(item, width)]


def resolve_plan_route(protocol_suggest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Use the same clinical hit threshold as MO UI and route low trust to fallback."""
    from clinical_knowledge.case_protocol_suggest import clinical_kp_hit

    hit = clinical_kp_hit(dict(protocol_suggest) if isinstance(protocol_suggest, Mapping) else None)
    if not hit:
        return {"route": "llm_no_kp", "kp_status": "unmatched", "hit": None}
    trust = str(hit.get("trust") or hit.get("trust_level") or "B").strip().upper()
    if trust not in KP_TRUST:
        trust = "D"
    if trust not in GROUNDED_TRUST:
        return {
            "route": "llm_no_kp",
            "kp_status": "unmatched",
            "fallback_reason": "kp_trust_below_threshold",
            "hit": hit,
        }
    return {"route": "kp_grounded", "kp_status": "matched", "kp_trust": trust, "hit": hit}


def validate_plan_concordance_result(
    raw: Mapping[str, Any],
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    verdict = str(raw.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"invalid plan verdict: {verdict}")
    provenance = str(raw.get("provenance") or "").strip().lower()
    if provenance not in PROVENANCE:
        raise ValueError(f"invalid plan provenance: {provenance}")
    kp_status = str(raw.get("kp_status") or ("matched" if provenance == "kp_grounded" else "unmatched")).lower()
    blocked = verdict in {"blocked", "na"}
    base = {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "case_id": str(raw.get("case_id") or case_id or "").strip(),
        "verdict": verdict,
        "missing_required": _strings(raw.get("missing_required")),
        "off_protocol": _strings(raw.get("off_protocol")),
        "source_refs": _strings(raw.get("source_refs")),
        "potential_harm": bool(raw.get("potential_harm")),
        "summary_ru": _clip(raw.get("summary_ru"), 600),
        "provenance": provenance,
        "kp_status": kp_status,
    }
    if provenance == "kp_grounded":
        if kp_status != "matched":
            raise ValueError("kp_grounded requires kp_status=matched")
        trust = str(raw.get("kp_trust") or "").strip().upper()
        if trust not in GROUNDED_TRUST:
            raise ValueError("kp_grounded requires trust A or B")
        kp_path = _clip(raw.get("kp_path"), 500)
        if not kp_path:
            raise ValueError("kp_grounded requires kp_path")
        if not base["source_refs"]:
            raise ValueError("kp_grounded requires source_refs")
        scores = {
            name: _pct(raw.get(name))
            for name in (
                "exam_protocol_pct",
                "treatment_protocol_pct",
                "followup_protocol_pct",
                "plan_protocol_pct",
            )
        }
        if blocked:
            if any(score is not None for score in scores.values()):
                raise ValueError("blocked/na grounded result must not have scores")
        elif any(score is None for score in scores.values()):
            raise ValueError("all grounded plan scores 0-100 are required")
        if raw.get("plan_general_llm_pct") is not None:
            raise ValueError("grounded result cannot contain plan_general_llm_pct")
        return {
            **base,
            **scores,
            "plan_general_llm_pct": None,
            "kp_path": kp_path,
            "kp_trust": trust,
        }

    if kp_status != "unmatched":
        raise ValueError("llm_no_kp requires kp_status=unmatched")
    if raw.get("plan_protocol_pct") is not None or raw.get("kp_path") or raw.get("source_refs"):
        raise ValueError("no-KP fallback cannot claim protocol compliance or references")
    general = _pct(raw.get("plan_general_llm_pct"))
    if blocked:
        if general is not None:
            raise ValueError("blocked/na fallback must not have a score")
    elif general is None:
        raise ValueError("plan_general_llm_pct 0-100 is required")
    return {
        **base,
        "exam_protocol_pct": None,
        "treatment_protocol_pct": None,
        "followup_protocol_pct": None,
        "plan_protocol_pct": None,
        "plan_general_llm_pct": general,
        "kp_path": "",
        "kp_trust": None,
        "source_refs": [],
    }


def selected_plan_score(result: Mapping[str, Any]) -> int | None:
    """Return exactly one score for Endpoint D without mixing KP and fallback."""
    provenance = str(result.get("provenance") or "")
    if provenance == "kp_grounded":
        return _pct(result.get("plan_protocol_pct"))
    if provenance == "llm_no_kp":
        return _pct(result.get("plan_general_llm_pct"))
    raise ValueError("unknown plan provenance")
