"""Pydantic-модели для структурного анализа консультативных заключений (КЗ).

Реализуют схему из docs/cursor_task_protocols_and_consultations.md (разделы 7, 9, 11, 13-19).

Принципы (см. docs/implementation_plan.md, этап 1):
- Все поля имеют дефолты — частично распознанное КЗ не должно ронять разбор.
- ``extra="ignore"`` — лишние ключи из эвристик/LLM не вызывают ошибок.
- Модели чистые: без побочных эффектов и тяжёлых импортов.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Sex = Literal["male", "female", "any", "unknown"]
AgeGroup = Literal["newborn", "infant", "child", "adult", "elderly", "unknown"]
AdultOrChild = Literal["adult", "child", "newborn", "unknown"]
Severity = Literal["low", "medium", "high", "critical"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


# --------------------------------------------------------------------------- #
# Общие сущности
# --------------------------------------------------------------------------- #
class SourceRef(_Base):
    document_url: str | None = None
    normalized_document_url: str | None = None
    local_path: str | None = None
    protocol_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    section_type: str | None = None
    quote: str | None = None


class ComplianceIssue(_Base):
    issue_type: str
    severity: Literal["info", "low", "medium", "high", "critical", "warning"] = "info"
    category: str | None = None
    message_ru: str
    field_target: str | None = None
    expected: str | None = None
    actual: str | None = None
    consultation_evidence: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)


class ApplicabilityFilter(_Base):
    age_min_years: int | None = None
    age_max_years: int | None = None
    age_groups: list[str] = Field(default_factory=list)
    sex: Sex = "unknown"
    pregnancy: Literal["required", "excluded", "any", "unknown"] = "unknown"
    care_setting: list[str] = Field(default_factory=list)
    specialty: list[str] = Field(default_factory=list)
    condition_status: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Консультативное заключение
# --------------------------------------------------------------------------- #
class PatientContext(_Base):
    full_name: str | None = None
    birth_date: _dt.date | None = None
    age_years: int | None = None
    age_months: int | None = None
    sex: Literal["male", "female", "unknown"] = "unknown"
    age_group: AgeGroup = "unknown"
    adult_or_child: AdultOrChild = "unknown"
    pregnancy: bool | None = None

    height_cm: float | None = None
    weight_kg: float | None = None
    bmi: float | None = None

    allergies: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    surgeries: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    social_history: list[str] = Field(default_factory=list)

    vitals: dict[str, Any] = Field(default_factory=dict)


class ConsultationSections(_Base):
    header: str | None = None
    consent_text: str | None = None
    consultation_purpose: str | None = None
    complaints: str | None = None
    anamnesis: str | None = None
    life_history: str | None = None
    allergy_history: str | None = None
    medication_history: str | None = None
    surgical_history: str | None = None
    objective_status: str | None = None
    local_status: str | None = None
    exam_results: str | None = None
    diagnosis_text: str | None = None
    recommendations_exams: str | None = None
    recommendations_treatment: str | None = None
    non_drug_recommendations: str | None = None
    general_recommendations: str | None = None
    routing: str | None = None
    follow_up_text: str | None = None
    doctor_signature: str | None = None


class ConsultationDiagnosis(_Base):
    diagnosis_id: str
    raw_text: str
    icd10_code: str | None = None
    diagnosis_name: str | None = None
    diagnosis_role: Literal[
        "primary", "secondary", "comorbidity", "symptom",
        "suspected", "finding", "red_flag_finding", "unknown",
    ] = "unknown"
    certainty: Literal["confirmed", "suspected", "excluded", "unclear"] = "unclear"
    is_protocol_relevant: bool = True
    safety_flags: list[str] = Field(default_factory=list)
    source_section: str | None = None
    source_text: str | None = None


class ExamItem(_Base):
    exam_id: str
    exam_name: str
    exam_type: Literal[
        "laboratory", "instrumental", "imaging",
        "functional", "consultation", "pathology", "unknown",
    ] = "unknown"
    status: Literal[
        "performed", "recommended", "planned", "control", "unknown",
    ] = "unknown"
    date: _dt.date | None = None
    result_text: str | None = None
    result_value: str | None = None
    abnormal_flag: bool | None = None
    source_section: str | None = None


class MedicationScheduleStep(_Base):
    start_date: _dt.date | None = None
    end_date: _dt.date | None = None
    dose_text: str
    frequency_text: str | None = None
    daily_dose_text: str | None = None


class MedicationItem(_Base):
    medication_id: str
    raw_text: str
    drug_name: str | None = None
    active_substance: str | None = None
    dose_value: float | None = None
    dose_unit: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    start_date: _dt.date | None = None
    end_date: _dt.date | None = None
    schedule: list[MedicationScheduleStep] = Field(default_factory=list)
    indication_text: str | None = None
    source_section: str | None = None


class FollowUpItem(_Base):
    follow_up_id: str | None = None
    raw_text: str | None = None
    date: _dt.date | None = None
    interval_text: str | None = None
    source_section: str | None = None


class TemplateBlock(_Base):
    block_diagnosis_text: str
    icd10_code: str | None = None
    block_type: Literal[
        "required_exams", "additional_exams",
        "follow_up", "treatment", "care", "unknown",
    ] = "unknown"
    items: list[str] = Field(default_factory=list)
    source_text: str = ""


class ExtractionQuality(_Base):
    raw_text_length: int = 0
    parsed_sections_count: int = 0
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    has_undefined: bool = False
    has_question_mark_diagnosis: bool = False
    has_unparsed_medication_schedule: bool = False
    has_missing_birth_date: bool = False
    has_missing_consultation_date: bool = False
    has_missing_doctor_specialty: bool = False


class ConsultationDocument(_Base):
    consultation_id: str
    source_file: str = ""
    source_file_type: str = ""
    raw_text: str = ""
    pages: list[dict[str, Any]] = Field(default_factory=list)

    clinic_name: str | None = None
    doctor_specialty: str | None = None
    doctor_name: str | None = None
    doctor_category: str | None = None

    consultation_date: _dt.date | None = None
    consultation_datetime: _dt.datetime | None = None

    patient: PatientContext = Field(default_factory=PatientContext)
    sections: ConsultationSections = Field(default_factory=ConsultationSections)
    diagnoses: list[ConsultationDiagnosis] = Field(default_factory=list)
    medications: list[MedicationItem] = Field(default_factory=list)
    planned_exams: list[ExamItem] = Field(default_factory=list)
    performed_exams: list[ExamItem] = Field(default_factory=list)
    follow_up: list[FollowUpItem] = Field(default_factory=list)
    template_blocks: list[TemplateBlock] = Field(default_factory=list)

    extraction_quality: ExtractionQuality = Field(default_factory=ExtractionQuality)


# --------------------------------------------------------------------------- #
# Подбор протоколов
# --------------------------------------------------------------------------- #
class ProtocolMatchResult(_Base):
    protocol_id: str
    diagnosis_id: str | None = None
    rubric_name: str | None = None
    document_title: str | None = None
    source_path: str | None = None
    matched_condition: str | None = None
    match_score: float = 0.0
    match_reasons: list[str] = Field(default_factory=list)
    mismatch_reasons: list[str] = Field(default_factory=list)
    applicability: Literal[
        "applicable", "possibly_applicable", "not_applicable", "unknown",
    ] = "unknown"
    age_applicability: str | None = None
    sex_applicability: str | None = None
    pregnancy_applicability: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Оценки соответствия
# --------------------------------------------------------------------------- #
class DiagnosisAssessment(_Base):
    diagnosis_id: str
    diagnosis_text: str
    icd10_code: str | None = None
    matched_condition: str | None = None
    matched_protocol_id: str | None = None
    status: Literal[
        "supported", "partially_supported", "not_supported",
        "suspected_needs_confirmation", "insufficient_data", "not_assessed",
    ] = "not_assessed"
    issues: list[ComplianceIssue] = Field(default_factory=list)
    evidence_found: list[str] = Field(default_factory=list)
    evidence_missing: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)


class ExamAssessment(_Base):
    protocol_rule_id: str | None = None
    exam_name: str
    exam_type: str = "unknown"
    status: Literal[
        "present_performed", "present_recommended", "missing_required",
        "missing_conditional", "not_applicable", "extra_not_assessed", "unknown",
    ] = "not_applicable"
    reason: str = ""
    consultation_evidence: list[str] = Field(default_factory=list)
    protocol_evidence: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)


class TreatmentAssessment(_Base):
    medication_id: str | None = None
    treatment_text: str
    matched_protocol_rule_id: str | None = None
    status: Literal[
        "matches_protocol", "partially_matches_protocol", "not_in_protocol",
        "dose_mismatch", "duration_mismatch", "frequency_mismatch",
        "age_contraindication", "contraindication_warning",
        "insufficient_data", "not_assessed",
    ] = "not_assessed"
    issues: list[ComplianceIssue] = Field(default_factory=list)
    protocol_evidence: list[str] = Field(default_factory=list)
    consultation_evidence: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)


class SafetyAssessment(_Base):
    issue_type: Literal[
        "red_flag", "urgent_referral", "possible_malignancy", "thrombosis",
        "severe_infection", "drug_safety", "missing_control",
        "manual_review_required",
    ]
    severity: Severity = "medium"
    finding_text: str = ""
    expected_action: str | None = None
    actual_action: str | None = None
    status: Literal[
        "handled", "partially_handled", "not_handled", "not_assessed",
    ] = "not_assessed"
    source_refs: list[SourceRef] = Field(default_factory=list)


class SectionQualityAssessment(_Base):
    has_complaints: bool = False
    has_anamnesis: bool = False
    has_objective_status: bool = False
    has_exam_results: bool = False
    has_diagnosis: bool = False
    has_recommendations: bool = False
    has_treatment: bool = False
    has_follow_up: bool = False
    missing_sections: list[str] = Field(default_factory=list)
    duplicate_sections: list[str] = Field(default_factory=list)
    suspicious_placeholders: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)


class StructuralAssessment(_Base):
    """Проверка обязательных/условных рубрик КЗ (ТЗ §9)."""
    filled_sections: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    missing_conditional: list[str] = Field(default_factory=list)
    missing_recommended: list[str] = Field(default_factory=list)
    structural_score: float | None = None
    patient_data_score: float | None = None


class ProtocolAssessment(_Base):
    """Сводка применимости подобранных протоколов (ТЗ §11–12)."""
    matched_count: int = 0
    applicable_count: int = 0
    top_protocol_id: str | None = None
    top_protocol_title: str | None = None
    rules_compliance_pct: float | None = None
    summary_ru: str = ""


class EvidenceMapItem(_Base):
    rule_id: str
    rule_type: str
    required_item: str | None = None
    found_in_consultation: bool = False
    found_status: Literal[
        "performed", "recommended", "mentioned",
        "not_found", "not_applicable", "unknown",
    ] = "unknown"
    consultation_evidence: list[str] = Field(default_factory=list)
    protocol_evidence: list[str] = Field(default_factory=list)
    decision: Literal[
        "satisfied", "satisfied_by_recommendation", "missing",
        "not_applicable", "manual_review", "unknown",
    ] = "unknown"
    explanation: str = ""
    source_refs: list[SourceRef] = Field(default_factory=list)


class SafetyCapInfo(_Base):
    applied: bool = False
    reason: str | None = None
    cap_value: float | None = None


class ScoreBreakdown(_Base):
    # v2 names (ТЗ improve_kz §4)
    documentation_score: float | None = None
    patient_data_score: float | None = None
    protocol_applicability_score: float | None = None
    diagnosis_score: float | None = None
    diagnostic_criteria_score: float | None = None
    required_exams_score: float | None = None
    treatment_score: float | None = None
    safety_score: float | None = None
    follow_up_score: float | None = None
    confidence_score: float | None = None
    # legacy aliases (backward compat)
    structural_score: float | None = None
    protocol_match_score: float | None = None
    documentation_quality_score: float | None = None
    overall_score: float | None = None


OverallStatus = Literal[
    "compliant", "mostly_compliant", "partially_compliant",
    "non_compliant", "insufficient_data", "insufficient_protocol_data",
    "low_confidence", "manual_review_required",
]


class ComplianceReport(_Base):
    consultation_id: str
    source_file: str = ""
    overall_score: float | None = None
    confidence_score: float | None = None
    overall_status: OverallStatus = "insufficient_data"
    score_source: str = "deterministic"
    llm_score_ignored: bool = True
    llm_used_for: list[str] = Field(default_factory=lambda: [
        "query_focus", "evidence_summarization", "expert_explanation",
    ])

    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    protocol_matches: list[ProtocolMatchResult] = Field(default_factory=list)
    not_applicable_protocols: list[ProtocolMatchResult] = Field(default_factory=list)
    diagnosis_assessments: list[DiagnosisAssessment] = Field(default_factory=list)
    section_quality: SectionQualityAssessment = Field(default_factory=SectionQualityAssessment)
    structural_assessment: StructuralAssessment = Field(default_factory=StructuralAssessment)
    protocol_assessment: ProtocolAssessment = Field(default_factory=ProtocolAssessment)
    exam_assessments: list[ExamAssessment] = Field(default_factory=list)
    treatment_assessments: list[TreatmentAssessment] = Field(default_factory=list)
    safety_assessments: list[SafetyAssessment] = Field(default_factory=list)
    evidence_map: list[EvidenceMapItem] = Field(default_factory=list)

    missing_required_items: list[ComplianceIssue] = Field(default_factory=list)
    major_issues: list[ComplianceIssue] = Field(default_factory=list)
    warnings: list[ComplianceIssue] = Field(default_factory=list)
    critical_issues: list[ComplianceIssue] = Field(default_factory=list)

    safety_cap: SafetyCapInfo = Field(default_factory=SafetyCapInfo)
    limitations: list[str] = Field(default_factory=list)
    explanation: str = ""
    source_refs: list[SourceRef] = Field(default_factory=list)


# Alias для совместимости с ТЗ §7.9
KzComplianceReport = ComplianceReport
