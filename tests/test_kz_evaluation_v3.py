"""Тесты контракта и движка KzEvaluationResultV3 (Workstream A ТЗ overnight-v1)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.kz_evaluation_engine import (
    _attach_evidence_spans,
    evaluate_kz_v3,
    resolve_mode,
)
from clinical_knowledge.kz_evaluation_schema import (
    SCHEMA_VERSION,
    AxisScores,
    ConfidenceInfo,
    CoverageInfo,
    EvaluationFinding,
    KzEvaluationResultV3,
)

_CASE = {
    "complaints": "боль в горле",
    "anamnesis_doctor": "3 дня",
    "objective_status": "зев гиперемирован",
    "clinical_diagnosis": "Острый фарингит",
    "mkb_code_main": "J02.9",
    "exam_recommendations": "ОАК",
    "treatment_recommendations": "парацетамол 500 мг 3 раза",
}


def test_schema_defaults_safe():
    r = KzEvaluationResultV3()
    assert r.schema_version == SCHEMA_VERSION
    assert r.score_pct is None
    assert r.status == "insufficient_data"
    assert r.axes.documentation is None
    d = r.to_public_dict()
    assert "evaluation" not in d  # это сам объект
    assert set(["axes", "coverage", "confidence", "risk", "provenance", "legacy"]).issubset(d)


def test_score_bounds_clamped():
    a = AxisScores(documentation=150, clinical_concordance=-20)
    assert a.documentation == 100.0
    assert a.clinical_concordance == 0.0
    c = CoverageInfo(overall=2.5, documentation=-1)
    assert c.overall == 1.0
    assert c.documentation == 0.0
    conf = ConfidenceInfo(overall=float("nan"), protocol_match=float("inf"))
    assert conf.overall is None
    assert conf.protocol_match is None


def test_no_nan_inf_in_output():
    r = evaluate_kz_v3(_CASE)
    d = r.to_public_dict()

    def _walk(o):
        if isinstance(o, float):
            assert not math.isnan(o) and not math.isinf(o)
        elif isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    _walk(d)


def test_legacy_preserved_additively():
    legacy = {"structural_score": 71.0, "overall_compliance_pct": 68.0}
    r = evaluate_kz_v3(_CASE, legacy=legacy)
    assert r.legacy == legacy


def test_provenance_and_versions():
    r = evaluate_kz_v3(_CASE)
    assert r.provenance.schema_version == SCHEMA_VERSION
    assert r.scorer_version
    # provenance не роняет NaN и присутствует
    assert r.provenance.scorer_version == r.scorer_version


def test_finding_evidence_gets_exact_document_span():
    finding = EvaluationFinding(
        code="B_diagnosis",
        axis="clinical_concordance",
        evidence="Острый фарингит",
    )
    _attach_evidence_spans(_CASE, [finding])
    assert finding.evidence_span is not None
    assert finding.evidence_span.field == "clinical_diagnosis"
    assert _CASE["clinical_diagnosis"][
        finding.evidence_span.start : finding.evidence_span.end
    ] == "Острый фарингит"


def test_mode_defaults_shadow():
    m = resolve_mode()
    assert m.enabled is True
    assert m.primary is False
    assert m.gate is False


def test_axes_present_for_documented_case():
    r = evaluate_kz_v3(_CASE)
    assert r.axes.documentation is not None
    assert r.axes.clinical_concordance is not None
    assert r.axes.safety is not None
    assert r.status in (
        "good", "acceptable", "review", "limited_evidence", "insufficient_evidence",
    )
