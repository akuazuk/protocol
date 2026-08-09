"""Primary MO scorer v4 built on the trust-aware v3 rule engine.

V4 keeps the proven axis detectors and trust gate from v3, but makes the
production contract explicit: configurable weights, one primary score,
severity-first attention, and complete scorer provenance.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from .kz_evaluation_engine import evaluate_kz_v3
from .kz_evaluation_schema import EvaluationMode, KzEvaluationResultV3

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "mo_scorer_v4.yaml"
AXES = ("documentation", "clinical_concordance", "safety", "regulatory")


class KzEvaluationResultV4(KzEvaluationResultV3):
    schema_version: str = "4.0"
    scorer_version: str = "v4.0.0"
    axis_contributions: dict[str, float | None] = Field(default_factory=dict)
    attention_required: bool = False
    attention_reasons: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def load_v4_config() -> dict[str, Any]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    weights = raw.get("axis_weights") or {}
    missing = set(AXES) - set(weights)
    if missing:
        raise ValueError(f"mo_scorer_v4_missing_weights:{','.join(sorted(missing))}")
    normalized = {axis: float(weights[axis]) for axis in AXES}
    if any(weight < 0 for weight in normalized.values()):
        raise ValueError("mo_scorer_v4_negative_weight")
    if abs(sum(normalized.values()) - 1.0) > 1e-9:
        raise ValueError("mo_scorer_v4_weights_must_sum_to_one")
    raw["axis_weights"] = normalized
    try:
        from clinical_knowledge.mo_scoring_profile import apply_profile_to_v4_config

        raw = apply_profile_to_v4_config(raw)
    except Exception:  # noqa: BLE001
        pass
    return raw


def _enabled(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def resolve_v4_mode() -> EvaluationMode:
    return EvaluationMode(
        enabled=_enabled("KZ_EVALUATION_V4_ENABLED", "1"),
        primary=_enabled("KZ_EVALUATION_V4_PRIMARY", "0"),
        gate=_enabled("KZ_EVALUATION_V4_GATE", "0"),
    )


def evaluate_kz_v4(
    case: dict[str, Any],
    *,
    protocol_ctx: Any = None,
    drug_ctx: dict | None = None,
    icd_client=None,
    legacy: dict | None = None,
    mode: EvaluationMode | None = None,
) -> KzEvaluationResultV4:
    config = load_v4_config()
    mode = mode or resolve_v4_mode()
    base = evaluate_kz_v3(
        case,
        protocol_ctx=protocol_ctx,
        drug_ctx=drug_ctx,
        icd_client=icd_client,
        legacy=legacy,
        mode=mode,
    )
    axes = base.axes.model_dump()
    available = [(axis, axes.get(axis)) for axis in AXES if axes.get(axis) is not None]
    denominator = sum(config["axis_weights"][axis] for axis, _ in available)
    score = (
        round(
            sum(float(value) * config["axis_weights"][axis] for axis, value in available)
            / denominator,
            1,
        )
        if denominator
        else None
    )
    contributions = {
        axis: (
            round(float(value) * config["axis_weights"][axis] / denominator, 2)
            if value is not None and denominator
            else None
        )
        for axis, value in axes.items()
    }

    worst = base.risk.worst_severity
    cap = (config.get("risk_caps") or {}).get(worst)
    if score is not None and cap is not None:
        score = min(score, float(cap))

    attention = config.get("attention") or {}
    attention_reasons: list[str] = []
    if worst in set(attention.get("severities") or ("P0", "P1")):
        attention_reasons.append(f"severity={worst}")
    if attention.get("include_low_evidence", True):
        coverage = base.coverage.overall
        confidence = base.confidence.overall
        if coverage is not None and coverage < float(attention.get("minimum_coverage", 0.5)):
            attention_reasons.append("low_coverage")
        if confidence is not None and confidence < float(attention.get("minimum_confidence", 0.5)):
            attention_reasons.append("low_confidence")

    payload = base.model_dump()
    payload.update(
        {
            "schema_version": str(config["schema_version"]),
            "scorer_version": str(config["scorer_version"]),
            "score_pct": score,
            "mode": mode.model_dump(),
            "axis_contributions": contributions,
            "attention_required": bool(attention_reasons),
            "attention_reasons": attention_reasons,
        }
    )
    provenance = dict(payload.get("provenance") or {})
    provenance.update(
        {
            "schema_version": str(config["schema_version"]),
            "scorer_version": str(config["scorer_version"]),
            "weights_version": str(config["weights_version"]),
        }
    )
    payload["provenance"] = provenance
    return KzEvaluationResultV4.model_validate(payload)
