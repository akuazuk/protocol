"""Единый версионированный контракт результата оценки КЗ - ``KzEvaluationResultV3``.

Workstream A ТЗ ``docs/plans/2026-07-27-kz-evaluation-quality-overnight-v1.md`` (§5).

Принципы контракта (§5.2):
- у всех полей безопасные defaults;
- никаких ``NaN``/``Infinity``;
- score: 0-100, coverage/confidence: 0-1;
- отсутствие данных выражается ``None``, а не нулём;
- ``legacy`` хранит старые баллы только для shadow-сравнения;
- сериализуется Pydantic и стабилен в API.

Контракт аддитивен: он не заменяет ``structured_analysis``/``review``/``send_gate``,
а добавляется рядом (см. §5.3).
"""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "3.0"
SCORER_VERSION = "2026-07-27.1"

AXES = ("documentation", "clinical_concordance", "safety", "regulatory")

Severity = Literal["P0", "P1", "P2", "P3", "ok"]
TrustLevelLit = Literal["A", "B", "C", "D"]
FindingKind = Literal[
    "documentation_gap",
    "protocol_mismatch",
    "safety_warning",
    "insufficient_context",
    "needs_human",
    "regulatory_defect",
]
EvalStatus = Literal[
    "good",
    "acceptable",
    "review",
    "limited_evidence",
    "insufficient_evidence",
    "insufficient_data",
    "critical",
    "poor",
]


def _finite(v: Any) -> float | None:
    """Отбросить None/NaN/Inf, привести к float. Иначе None (§5.2 - без NaN/Inf)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def clamp_score(v: Any) -> float | None:
    f = _finite(v)
    if f is None:
        return None
    return round(max(0.0, min(100.0, f)), 1)


def clamp_unit(v: Any) -> float | None:
    f = _finite(v)
    if f is None:
        return None
    return round(max(0.0, min(1.0, f)), 3)


class _V3Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class AxisScores(_V3Base):
    documentation: float | None = None
    clinical_concordance: float | None = None
    safety: float | None = None
    regulatory: float | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _clamp(cls, v):
        return clamp_score(v)


class CoverageInfo(_V3Base):
    overall: float | None = None
    documentation: float | None = None
    clinical_concordance: float | None = None
    safety: float | None = None
    regulatory: float | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _clamp(cls, v):
        return clamp_unit(v)


class ConfidenceInfo(_V3Base):
    overall: float | None = None
    document_parse: float | None = None
    protocol_match: float | None = None
    evidence_match: float | None = None
    protocol_knowledge: float | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _clamp(cls, v):
        return clamp_unit(v)


class RiskInfo(_V3Base):
    worst_severity: str | None = None
    cap_applied: bool = False
    cap_value: float | None = None
    reasons: list[str] = Field(default_factory=list)

    @field_validator("cap_value", mode="before")
    @classmethod
    def _clamp(cls, v):
        return clamp_score(v)


class ProtocolMatchInfo(_V3Base):
    condition_id: str | None = None
    name: str | None = None
    applicability_confidence: float | None = None
    retrieval_confidence: float | None = None
    population_match: bool | None = None
    care_setting_match: bool | None = None
    specialty_match: bool | None = None
    version_current: bool | None = None
    penalty_eligible: bool = False
    trust_level: TrustLevelLit = "D"
    reasons: list[str] = Field(default_factory=list)

    @field_validator("applicability_confidence", "retrieval_confidence", mode="before")
    @classmethod
    def _clamp(cls, v):
        return clamp_unit(v)


class EvidenceSpan(_V3Base):
    field: str
    start: int
    end: int
    text: str = ""


class EvaluationFinding(_V3Base):
    code: str
    axis: str
    severity: Severity = "P3"
    kind: FindingKind = "needs_human"
    passed: bool = False
    title_ru: str = ""
    detail_ru: str = ""
    evidence: str = ""
    evidence_span: EvidenceSpan | None = None
    source_ref: str = ""
    trust_level: TrustLevelLit = "D"
    penalty_applied: bool = False
    needs_human: bool = False

    @field_validator("evidence", mode="before")
    @classmethod
    def _trim(cls, v):
        return str(v or "")[:400]


class RuleTrustDiagnostics(_V3Base):
    rules_total: int = 0
    rules_penalty_eligible: int = 0
    rules_advisory: int = 0
    rules_heuristic: int = 0


class Provenance(_V3Base):
    corpus_version: str | None = None
    rules_version: str | None = None
    weights_version: str | None = None
    build_version: str | None = None
    scorer_version: str = SCORER_VERSION
    schema_version: str = SCHEMA_VERSION


class EvaluationMode(_V3Base):
    enabled: bool = True
    primary: bool = False
    gate: bool = False


class KzEvaluationResultV3(_V3Base):
    schema_version: str = SCHEMA_VERSION
    scorer_version: str = SCORER_VERSION
    score_pct: float | None = None
    status: EvalStatus = "insufficient_data"
    axes: AxisScores = Field(default_factory=AxisScores)
    coverage: CoverageInfo = Field(default_factory=CoverageInfo)
    confidence: ConfidenceInfo = Field(default_factory=ConfidenceInfo)
    risk: RiskInfo = Field(default_factory=RiskInfo)
    protocols: list[ProtocolMatchInfo] = Field(default_factory=list)
    findings: list[EvaluationFinding] = Field(default_factory=list)
    diagnostics: RuleTrustDiagnostics = Field(default_factory=RuleTrustDiagnostics)
    mode: EvaluationMode = Field(default_factory=EvaluationMode)
    provenance: Provenance = Field(default_factory=Provenance)
    legacy: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score_pct", mode="before")
    @classmethod
    def _clamp(cls, v):
        return clamp_score(v)

    def to_public_dict(self) -> dict[str, Any]:
        """Стабильный сериализуемый словарь для API/JSON/FHIR."""
        return self.model_dump(mode="json")
