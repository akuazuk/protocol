"""Валидация Protocol Summary Cards."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config as _cfg_mod
from .schema import (
    ConditionSummary,
    CriteriaBlock,
    DrugTreatmentItem,
    ExamRequirement,
    ProtocolSummary,
    RedFlagItem,
    SummarySourceRef,
    SummaryValidationResult,
    ValidationIssue,
    ValidationStatus,
)

_MAX_QUOTE_LEN = 1200
_REVIEW_RANK = {"not_reviewed": 0, "needs_review": 1, "reviewed": 2, "approved": 3, "rejected": -1}
_MIN_REVIEW_RANK = {"draft": 0, "reviewed": 2, "approved": 3}


def _has_source_anchor(ref: SummarySourceRef | None) -> bool:
    if ref is None:
        return False
    return ref.page_start is not None or bool(ref.section_title) or bool(ref.quote)


def _check_source_ref(
    ref: SummarySourceRef | None,
    path: str,
    *,
    strict: bool,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    if ref is None:
        errors.append(ValidationIssue(code="missing_source_ref", message="Нет source_ref", path=path))
        return
    if not _has_source_anchor(ref):
        errors.append(
            ValidationIssue(
                code="incomplete_source_ref",
                message="source_ref без page_start и section_title",
                path=path,
            ),
        )
    if strict and not (ref.quote or "").strip():
        errors.append(
            ValidationIssue(code="missing_quote", message="STRICT: нет цитаты в source_ref", path=path),
        )
    if ref.quote and len(ref.quote) > _MAX_QUOTE_LEN:
        warnings.append(
            ValidationIssue(code="long_quote", message="Слишком длинная цитата", path=path),
        )


def _iter_criterion_items(block: CriteriaBlock | None, prefix: str) -> Iterable[tuple[str, Any]]:
    if not block:
        return
    for i, item in enumerate(block.required):
        yield f"{prefix}.required[{i}]", item
    for i, item in enumerate(block.optional):
        yield f"{prefix}.optional[{i}]", item
    for i, item in enumerate(block.exclusion):
        yield f"{prefix}.exclusion[{i}]", item


def _validate_applicability(summary: ProtocolSummary, warnings: list[ValidationIssue]) -> None:
    appl = summary.applicability
    pops = set(appl.population or [])
    if "adult" in pops and appl.age_max_years is not None and appl.age_max_years < 18:
        warnings.append(
            ValidationIssue(
                code="age_population_conflict",
                message="population=adult, но age_max_years < 18",
                path="applicability",
            ),
        )
    if "child" in pops and appl.age_min_years is not None and appl.age_min_years >= 18:
        warnings.append(
            ValidationIssue(
                code="age_population_conflict",
                message="population=child, но age_min_years >= 18",
                path="applicability",
            ),
        )


def _validate_condition(
    cond: ConditionSummary,
    *,
    strict: bool,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
    seen_ids: set[str],
    seen_exams: set[str],
    seen_drugs: set[str],
) -> None:
    base = f"conditions[{cond.condition_id}]"
    if cond.condition_id in seen_ids:
        errors.append(ValidationIssue(code="duplicate_condition", message="Дублирующийся condition_id", path=base))
    seen_ids.add(cond.condition_id)
    if not cond.name.strip():
        errors.append(ValidationIssue(code="missing_condition_name", message="Нет name", path=base))
    if not cond.icd10_codes:
        warnings.append(
            ValidationIssue(code="missing_icd10", message="Нет icd10_codes у condition", path=base),
        )

    if cond.diagnosis_structure:
        for i, comp in enumerate(cond.diagnosis_structure.required_components):
            _check_source_ref(comp.source_ref, f"{base}.diagnosis_structure.required[{i}]", strict=strict, errors=errors, warnings=warnings)
        for i, ex in enumerate(cond.diagnosis_structure.examples):
            _check_source_ref(ex.source_ref, f"{base}.diagnosis_structure.examples[{i}]", strict=strict, errors=errors, warnings=warnings)

    for path, item in _iter_criterion_items(cond.clinical_criteria, f"{base}.clinical_criteria"):
        _check_source_ref(item.source_ref, path, strict=strict, errors=errors, warnings=warnings)
    for path, item in _iter_criterion_items(cond.diagnostic_criteria, f"{base}.diagnostic_criteria"):
        _check_source_ref(item.source_ref, path, strict=strict, errors=errors, warnings=warnings)

    for i, exam in enumerate(cond.required_exams):
        if exam.requirement_level != "required":
            warnings.append(
                ValidationIssue(
                    code="required_exam_level",
                    message="required_exams содержит элемент не с level=required",
                    path=f"{base}.required_exams[{i}]",
                ),
            )
        key = _norm_exam_key(exam)
        if key in seen_exams:
            warnings.append(ValidationIssue(code="duplicate_exam", message="Дублирующееся обследование", path=f"{base}.required_exams[{i}]"))
        seen_exams.add(key)
        _check_source_ref(exam.source_ref, f"{base}.required_exams[{i}]", strict=strict, errors=errors, warnings=warnings)

    for i, exam in enumerate(cond.conditional_exams):
        if exam.requirement_level not in ("conditional", "recommended"):
            warnings.append(
                ValidationIssue(
                    code="conditional_exam_level",
                    message="conditional_exams: ожидается conditional/recommended",
                    path=f"{base}.conditional_exams[{i}]",
                ),
            )
        _check_source_ref(exam.source_ref, f"{base}.conditional_exams[{i}]", strict=strict, errors=errors, warnings=warnings)

    if cond.treatment:
        for i, item in enumerate(cond.treatment.drugs):
            key = _norm_drug_key(item)
            if key in seen_drugs:
                warnings.append(ValidationIssue(code="duplicate_drug", message="Дублирующийся препарат", path=f"{base}.treatment.drugs[{i}]"))
            seen_drugs.add(key)
            _check_source_ref(item.source_ref, f"{base}.treatment.drugs[{i}]", strict=strict, errors=errors, warnings=warnings)
        for i, item in enumerate(cond.treatment.drug_groups):
            _check_source_ref(item.source_ref, f"{base}.treatment.drug_groups[{i}]", strict=strict, errors=errors, warnings=warnings)
        for i, item in enumerate(cond.treatment.non_drug):
            _check_source_ref(item.source_ref, f"{base}.treatment.non_drug[{i}]", strict=strict, errors=errors, warnings=warnings)

    for i, fu in enumerate(cond.follow_up):
        _check_source_ref(fu.source_ref, f"{base}.follow_up[{i}]", strict=strict, errors=errors, warnings=warnings)

    for i, rf in enumerate(cond.red_flags):
        if rf.severity == "critical" and not rf.expected_actions:
            errors.append(
                ValidationIssue(
                    code="critical_red_flag_actions",
                    message="critical red flag без expected_actions",
                    path=f"{base}.red_flags[{i}]",
                ),
            )
        _check_source_ref(rf.source_ref, f"{base}.red_flags[{i}]", strict=strict, errors=errors, warnings=warnings)


def _norm_exam_key(exam: ExamRequirement) -> str:
    return re.sub(r"\s+", " ", exam.name.lower().strip())


def _norm_drug_key(drug: DrugTreatmentItem) -> str:
    return (drug.drug_name or drug.active_substance or drug.drug_group or "").lower().strip()


def validate_protocol_summary(
    summary: ProtocolSummary,
    *,
    strict: bool | None = None,
    source_blob: str | None = None,
) -> SummaryValidationResult:
    """Проверяет карточку; не изменяет summary."""
    strict = _cfg_mod.protocol_summary_config.strict_validation if strict is None else strict
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if source_blob:
        try:
            from .quote_validator import quote_found_in_source

            for cond in summary.conditions:
                base = f"conditions[{cond.condition_id}]"
                for i, ex in enumerate(cond.required_exams):
                    q = (ex.source_ref.quote or "") if ex.source_ref else ""
                    if q and not quote_found_in_source(q, source_blob):
                        warnings.append(
                            ValidationIssue(
                                code="quote_not_in_source",
                                message="Цитата не найдена в исходном тексте",
                                path=f"{base}.required_exams[{i}]",
                            ),
                        )
                if cond.treatment:
                    for i, d in enumerate(cond.treatment.drugs):
                        q = (d.source_ref.quote or "") if d.source_ref else ""
                        if q and not quote_found_in_source(q, source_blob):
                            warnings.append(
                                ValidationIssue(
                                    code="quote_not_in_source",
                                    message="Цитата препарата не найдена в исходном тексте",
                                    path=f"{base}.treatment.drugs[{i}]",
                                ),
                            )
        except Exception:
            pass

    if not summary.protocol_id.strip():
        errors.append(ValidationIssue(code="missing_protocol_id", message="protocol_id пуст"))
    if not summary.source.title.strip():
        errors.append(ValidationIssue(code="missing_source_title", message="source.title пуст", path="source"))
    if not summary.source.url and not summary.source.local_path:
        errors.append(ValidationIssue(code="missing_source_location", message="Нет url/local_path", path="source"))
    if not summary.rubric.name.strip():
        errors.append(ValidationIssue(code="missing_rubric", message="rubric.name пуст", path="rubric"))
    if not summary.applicability.population:
        warnings.append(
            ValidationIssue(code="unknown_population", message="applicability.population пуст", path="applicability"),
        )
    if not summary.conditions:
        errors.append(ValidationIssue(code="empty_conditions", message="conditions пуст"))

    _validate_applicability(summary, warnings)

    seen_ids: set[str] = set()
    seen_exams: set[str] = set()
    seen_drugs: set[str] = set()
    for cond in summary.conditions:
        _validate_condition(
            cond, strict=strict, errors=errors, warnings=warnings,
            seen_ids=seen_ids, seen_exams=seen_exams, seen_drugs=seen_drugs,
        )

    for i, rf in enumerate(summary.global_red_flags):
        _check_source_ref(rf.source_ref, f"global_red_flags[{i}]", strict=strict, errors=errors, warnings=warnings)

    status: ValidationStatus
    if errors:
        status = "invalid"
    elif warnings:
        status = "valid_with_warnings"
    else:
        status = "valid"

    if summary.review_status in ("needs_review", "not_reviewed") and status == "valid_with_warnings":
        status = "needs_human_review"

    return SummaryValidationResult(
        status=status,
        errors=errors,
        warnings=warnings,
        validated_at=datetime.now(timezone.utc).isoformat(),
    )


def review_status_acceptable(summary: ProtocolSummary, min_status: str | None = None) -> bool:
    """Карточка достаточного review_status для использования в summary/hybrid."""
    min_status = min_status or _cfg_mod.protocol_summary_config.min_review_status
    min_rank = _MIN_REVIEW_RANK.get(min_status, 0)
    ext_rank = {"draft": 0, "auto_extracted": 0, "llm_extracted": 1, "needs_human_review": 0, "reviewed": 2, "deprecated": -1}
    if summary.extraction_status == "deprecated":
        return False
    if summary.review_status == "rejected":
        return False
    rev_ok = _REVIEW_RANK.get(summary.review_status, 0) >= _MIN_REVIEW_RANK.get(
        "reviewed" if min_status == "reviewed" else min_status, min_rank,
    )
    if min_status == "draft":
        return ext_rank.get(summary.extraction_status, 0) >= 0 and summary.review_status != "rejected"
    if min_status == "reviewed":
        return rev_ok or summary.review_status in ("reviewed", "approved")
    if min_status == "approved":
        return summary.review_status == "approved"
    return rev_ok


def summary_is_usable(summary: ProtocolSummary, *, strict: bool | None = None) -> bool:
    """Карточка валидна и проходит порог review для режима summary/hybrid."""
    result = validate_protocol_summary(summary, strict=strict)
    if result.status == "invalid":
        return False
    return review_status_acceptable(summary)


def write_validation_report(
    summary: ProtocolSummary,
    result: SummaryValidationResult,
    out_dir: Path | None = None,
) -> Path:
    """Сохраняет markdown-отчёт в validation_reports/."""
    root = Path(_cfg_mod.protocol_summary_config.data_root)
    out_dir = out_dir or (root / "validation_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{summary.protocol_id}.md"
    lines = [
        f"# Validation: {summary.protocol_id}",
        "",
        f"- **status:** {result.status}",
        f"- **validated_at:** {result.validated_at}",
        f"- **review_status:** {summary.review_status}",
        f"- **extraction_status:** {summary.extraction_status}",
        "",
    ]
    if result.errors:
        lines.append("## Errors")
        for e in result.errors:
            lines.append(f"- `{e.code}` {e.path or ''}: {e.message}")
        lines.append("")
    if result.warnings:
        lines.append("## Warnings")
        for w in result.warnings:
            lines.append(f"- `{w.code}` {w.path or ''}: {w.message}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
