"""Подсчёт итоговой оценки соответствия КЗ (ТЗ §19 / improve_kz §4, §8).

Веса из config/compliance_weights.yaml. Блоки None не штрафуют - веса перенормируются.
"""
from __future__ import annotations

from .consult_config import load_compliance_weights
from .consult_schema import ScoreBreakdown

# v2 primary keys → yaml keys (fallback to legacy names)
_BLOCK_TO_WEIGHT: dict[str, str] = {
    "documentation_score": "documentation_score",
    "patient_data_score": "patient_data_score",
    "protocol_applicability_score": "protocol_applicability_score",
    "diagnosis_score": "diagnosis_score",
    "diagnostic_criteria_score": "diagnostic_criteria_score",
    "required_exams_score": "required_exams_score",
    "treatment_score": "treatment_score",
    "safety_score": "safety_score",
    "follow_up_score": "follow_up_score",
    # legacy aliases
    "structural_score": "structural_score",
    "protocol_match_score": "protocol_match_score",
    "documentation_quality_score": "documentation_quality_score",
}

_LEGACY_BLOCKS = (
    "protocol_match_score",
    "protocol_applicability_score",
    "diagnosis_score",
    "required_exams_score",
    "treatment_score",
    "safety_score",
    "documentation_quality_score",
    "documentation_score",
    "structural_score",
)


def sync_score_aliases(bd: ScoreBreakdown) -> ScoreBreakdown:
    """Синхронизирует v2 и legacy поля для обратной совместимости."""
    if bd.documentation_score is None and bd.structural_score is not None:
        bd.documentation_score = bd.structural_score
    if bd.structural_score is None and bd.documentation_score is not None:
        bd.structural_score = bd.documentation_score
    if bd.protocol_applicability_score is None and bd.protocol_match_score is not None:
        bd.protocol_applicability_score = bd.protocol_match_score
    if bd.protocol_match_score is None and bd.protocol_applicability_score is not None:
        bd.protocol_match_score = bd.protocol_applicability_score
    if bd.documentation_quality_score is None and bd.documentation_score is not None:
        bd.documentation_quality_score = bd.documentation_score
    return bd


def compute_overall(
    breakdown: ScoreBreakdown,
    *,
    force_manual_review: bool = False,
    has_protocol_data: bool = True,
    min_blocks: int = 2,
    status_thresholds: dict | None = None,
) -> tuple[float | None, str]:
    """Возвращает (overall_score, overall_status).

    status_thresholds: переопределяет пороги статусов (Э4). Нужен для axes-режима,
    где overall считается по более строгим alignment-блокам и требует своих
    (калиброванных на эталоне) порогов - иначе прежние 90/75/50 заливают non_compliant.
    """
    breakdown = sync_score_aliases(breakdown)
    cfg = load_compliance_weights()
    weights = cfg.get("weights") or {}
    legacy = cfg.get("legacy_weights") or {}
    thr = status_thresholds if status_thresholds else (cfg.get("status_thresholds") or {})

    present: list[tuple[float, float]] = []
    seen_vals: set[str] = set()
    for block, wkey in _BLOCK_TO_WEIGHT.items():
        val = getattr(breakdown, block, None)
        if val is None:
            continue
        # dedupe documentation_score vs structural_score
        dedupe_key = f"{wkey}:{val}"
        if wkey in ("structural_score", "documentation_score", "documentation_quality_score"):
            if "doc_block" in seen_vals:
                continue
            seen_vals.add("doc_block")
        elif wkey in ("protocol_match_score", "protocol_applicability_score"):
            if "proto_block" in seen_vals:
                continue
            seen_vals.add("proto_block")
        w = float(weights.get(wkey, 0.0))
        if w <= 0:
            w = float(weights.get(
                "structural_score" if wkey == "documentation_score" else wkey, 0.0,
            ))
        if w <= 0 and block in _LEGACY_BLOCKS:
            w = float(legacy.get(wkey, legacy.get("protocol_match_score", 0.0)))
        if w <= 0:
            continue
        present.append((float(val), w))

    if len(present) < min_blocks or sum(w for _, w in present) <= 0:
        return None, "insufficient_data"

    total_w = sum(w for _, w in present)
    overall = round(sum(s * w for s, w in present) / total_w, 1)

    if force_manual_review:
        return overall, "manual_review_required"

    if not has_protocol_data:
        return overall, "insufficient_protocol_data"

    if overall >= float(thr.get("compliant", 90)):
        status = "compliant"
    elif overall >= float(thr.get("mostly_compliant", 75)):
        status = "mostly_compliant"
    elif overall >= float(thr.get("partially_compliant", 50)):
        status = "partially_compliant"
    elif overall >= float(thr.get("non_compliant", 1)):
        status = "non_compliant"
    else:
        status = "insufficient_data"
    return overall, status
