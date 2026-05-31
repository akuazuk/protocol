"""Подсчёт итоговой оценки соответствия КЗ (ТЗ раздел 19 / kz_compliance_checker §14).

Веса берутся из config/compliance_weights.yaml. Блоки без данных (None) не штрафуют
балл — веса перенормируются по доступным блокам.
"""
from __future__ import annotations

from .consult_config import load_compliance_weights
from .consult_schema import ScoreBreakdown

_BLOCK_TO_WEIGHT = {
    "structural_score": "structural_score",
    "patient_data_score": "patient_data_score",
    "protocol_match_score": "protocol_match_score",
    "diagnosis_score": "diagnosis_score",
    "required_exams_score": "required_exams_score",
    "treatment_score": "treatment_score",
    "safety_score": "safety_score",
    "follow_up_score": "follow_up_score",
    "documentation_quality_score": "documentation_quality_score",
}

_LEGACY_BLOCKS = (
    "protocol_match_score",
    "diagnosis_score",
    "required_exams_score",
    "treatment_score",
    "safety_score",
    "documentation_quality_score",
)


def compute_overall(
    breakdown: ScoreBreakdown,
    *,
    force_manual_review: bool = False,
    min_blocks: int = 2,
) -> tuple[float | None, str]:
    """Возвращает (overall_score, overall_status)."""
    cfg = load_compliance_weights()
    weights = cfg.get("weights") or {}
    legacy = cfg.get("legacy_weights") or {}
    thr = cfg.get("status_thresholds") or {}

    present: list[tuple[float, float]] = []
    for block, wkey in _BLOCK_TO_WEIGHT.items():
        val = getattr(breakdown, block, None)
        if val is None:
            continue
        w = float(weights.get(wkey, 0.0))
        if w <= 0 and block in _LEGACY_BLOCKS:
            w = float(legacy.get(wkey, 0.0))
        if w <= 0:
            continue
        present.append((float(val), w))

    if len(present) < min_blocks or sum(w for _, w in present) <= 0:
        return None, "insufficient_data"

    total_w = sum(w for _, w in present)
    overall = round(sum(s * w for s, w in present) / total_w, 1)

    if force_manual_review:
        return overall, "manual_review_required"

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
