"""Pydantic-схема Protocol Summary Cards."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..consult_schema import SourceRef


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


Population = Literal["adult", "child", "newborn", "pregnant", "adult_and_child", "unknown"]
CareSetting = Literal[
    "outpatient", "inpatient", "emergency", "intensive_care",
    "rehabilitation", "palliative", "unknown",
]
ExtractionStatus = Literal[
    "draft", "auto_extracted", "llm_extracted", "needs_human_review", "reviewed", "deprecated",
]
ReviewStatus = Literal["not_reviewed", "needs_review", "reviewed", "approved", "rejected"]
ValidationStatus = Literal["valid", "valid_with_warnings", "invalid", "needs_human_review"]

CriterionOperator = Literal[
    "present", "absent", "contains", "any_of", "all_of",
    "numeric_gte", "numeric_lte", "duration_gte", "frequency_gte", "unknown",
]
EvidenceTarget = Literal[
    "complaints", "anamnesis", "objective_status", "local_status",
    "performed_exams", "recommended_exams", "diagnosis", "treatment", "follow_up",
]
ExamType = Literal[
    "laboratory", "instrumental", "imaging", "functional",
    "consultation", "pathology", "unknown",
]
RequirementLevel = Literal["required", "conditional", "recommended", "optional"]
ExamAcceptedStatus = Literal["performed", "recommended", "planned", "control"]
RedFlagType = Literal[
    "possible_malignancy", "thrombosis", "severe_infection", "systemic_autoimmune",
    "drug_safety", "urgent_referral", "other",
]


class SummarySourceRef(_Base):
    """Расширенная ссылка на фрагмент протокола (совместима с consult_schema.SourceRef)."""

    protocol_id: str | None = None
    document_url: str | None = None
    local_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    section_type: str | None = None
    table_index: int | None = None
    row_index: int | None = None
    quote: str | None = None

    def to_source_ref(self) -> SourceRef:
        return SourceRef(
            protocol_id=self.protocol_id,
            document_url=self.document_url,
            local_path=self.local_path,
            page_start=self.page_start,
            page_end=self.page_end,
            section_title=self.section_title,
            section_type=self.section_type,
            quote=(self.quote or "")[:400] or None,
        )


class ExtractionMetadata(_Base):
    extracted_at: str | None = None
    extractor: str = "heuristic"
    extractor_version: str = "1.0"
    source_document_sha256: str | None = None
    notes: list[str] = Field(default_factory=list)


class ValidationIssue(_Base):
    level: Literal["error", "warning", "info"] = "warning"
    code: str
    message: str
    path: str | None = None


class SummaryValidationResult(_Base):
    status: ValidationStatus = "needs_human_review"
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    validated_at: str | None = None


class ProtocolSource(_Base):
    title: str = ""
    url: str | None = None
    local_path: str | None = None
    document_sha256: str | None = None
    approval_date: str | None = None
    approval_number: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    document_year: int | None = None
    pages_total: int | None = None


class ProtocolRubric(_Base):
    name: str = ""
    slug: str | None = None
    specialty_slugs: list[str] = Field(default_factory=list)


class ProtocolApplicability(_Base):
    population: list[Population] = Field(default_factory=list)
    age_min_years: int | None = None
    age_max_years: int | None = None
    sex: Literal["male", "female", "any", "unknown"] = "unknown"
    pregnancy: Literal["required", "excluded", "any", "unknown"] = "unknown"
    care_setting: list[CareSetting] = Field(default_factory=list)


class SummaryNote(_Base):
    text: str
    source_ref: SummarySourceRef | None = None


class DiagnosisComponent(_Base):
    name: str
    required: bool = True
    description: str | None = None
    source_ref: SummarySourceRef


class DiagnosisExample(_Base):
    text: str
    source_ref: SummarySourceRef


class DiagnosisStructure(_Base):
    required_components: list[DiagnosisComponent] = Field(default_factory=list)
    optional_components: list[DiagnosisComponent] = Field(default_factory=list)
    examples: list[DiagnosisExample] = Field(default_factory=list)
    source_refs: list[SummarySourceRef] = Field(default_factory=list)


class CriterionItem(_Base):
    text: str
    logic_group: str | None = None
    operator: CriterionOperator = "unknown"
    values: list[str] = Field(default_factory=list)
    numeric_value: float | None = None
    unit: str | None = None
    evidence_targets: list[EvidenceTarget] = Field(default_factory=list)
    source_ref: SummarySourceRef


class CriteriaBlock(_Base):
    required: list[CriterionItem] = Field(default_factory=list)
    optional: list[CriterionItem] = Field(default_factory=list)
    exclusion: list[CriterionItem] = Field(default_factory=list)


class ExamRequirement(_Base):
    name: str
    aliases: list[str] = Field(default_factory=list)
    exam_type: ExamType = "unknown"
    requirement_level: RequirementLevel
    accepted_statuses: list[ExamAcceptedStatus] = Field(
        default_factory=lambda: ["performed", "recommended"],
    )
    required_if: list[str] = Field(default_factory=list)
    timing: str | None = None
    comment: str | None = None
    source_ref: SummarySourceRef


class NonDrugTreatmentItem(_Base):
    text: str
    source_ref: SummarySourceRef


class DrugGroupItem(_Base):
    drug_group: str
    indication: str | None = None
    source_ref: SummarySourceRef


class DrugTreatmentItem(_Base):
    drug_name: str | None = None
    active_substance: str | None = None
    drug_group: str | None = None
    dose_text: str | None = None
    frequency_text: str | None = None
    duration_text: str | None = None
    route: str | None = None
    indication: str | None = None
    contraindications: list[str] = Field(default_factory=list)
    monitoring: list[str] = Field(default_factory=list)
    applicability: ProtocolApplicability | None = None
    source_ref: SummarySourceRef


class ProcedureTreatmentItem(_Base):
    name: str
    indication: str | None = None
    source_ref: SummarySourceRef


class SurgeryTreatmentItem(_Base):
    name: str
    indication: str | None = None
    source_ref: SummarySourceRef


class TreatmentBlock(_Base):
    non_drug: list[NonDrugTreatmentItem] = Field(default_factory=list)
    drug_groups: list[DrugGroupItem] = Field(default_factory=list)
    drugs: list[DrugTreatmentItem] = Field(default_factory=list)
    procedures: list[ProcedureTreatmentItem] = Field(default_factory=list)
    surgery: list[SurgeryTreatmentItem] = Field(default_factory=list)
    source_refs: list[SummarySourceRef] = Field(default_factory=list)


class FollowUpRequirement(_Base):
    text: str
    timing: str | None = None
    required_if: list[str] = Field(default_factory=list)
    expected_actions: list[str] = Field(default_factory=list)
    source_ref: SummarySourceRef


class RoutingRequirement(_Base):
    text: str
    timing: str | None = None
    required_if: list[str] = Field(default_factory=list)
    source_ref: SummarySourceRef


class RedFlagItem(_Base):
    text: str
    aliases: list[str] = Field(default_factory=list)
    red_flag_type: RedFlagType = "other"
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    expected_actions: list[str] = Field(default_factory=list)
    cap_if_unhandled: int | None = None
    source_ref: SummarySourceRef


class ContraindicationItem(_Base):
    text: str
    source_ref: SummarySourceRef


class ComplicationItem(_Base):
    text: str
    source_ref: SummarySourceRef


class KzChecklist(_Base):
    must_have: list[str] = Field(default_factory=list)
    should_have: list[str] = Field(default_factory=list)
    conditional: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ConditionSummary(_Base):
    condition_id: str
    name: str
    synonyms: list[str] = Field(default_factory=list)
    abbreviations: list[str] = Field(default_factory=list)
    icd10_codes: list[str] = Field(default_factory=list)
    condition_applicability: ProtocolApplicability | None = None
    diagnosis_structure: DiagnosisStructure | None = None
    clinical_criteria: CriteriaBlock | None = None
    diagnostic_criteria: CriteriaBlock | None = None
    required_exams: list[ExamRequirement] = Field(default_factory=list)
    conditional_exams: list[ExamRequirement] = Field(default_factory=list)
    treatment: TreatmentBlock | None = None
    follow_up: list[FollowUpRequirement] = Field(default_factory=list)
    hospitalization: list[RoutingRequirement] = Field(default_factory=list)
    routing: list[RoutingRequirement] = Field(default_factory=list)
    red_flags: list[RedFlagItem] = Field(default_factory=list)
    contraindications: list[ContraindicationItem] = Field(default_factory=list)
    complications: list[ComplicationItem] = Field(default_factory=list)
    kz_checklist: KzChecklist | None = None
    source_refs: list[SummarySourceRef] = Field(default_factory=list)


class ProtocolSummary(_Base):
    protocol_id: str
    summary_version: str = "1.0"
    extraction_status: ExtractionStatus = "draft"
    review_status: ReviewStatus = "not_reviewed"
    source: ProtocolSource = Field(default_factory=ProtocolSource)
    rubric: ProtocolRubric = Field(default_factory=ProtocolRubric)
    applicability: ProtocolApplicability = Field(default_factory=ProtocolApplicability)
    conditions: list[ConditionSummary] = Field(default_factory=list)
    global_red_flags: list[RedFlagItem] = Field(default_factory=list)
    global_contraindications: list[ContraindicationItem] = Field(default_factory=list)
    global_notes: list[SummaryNote] = Field(default_factory=list)
    extraction_metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    validation: SummaryValidationResult | None = None
