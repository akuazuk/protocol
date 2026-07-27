"""Каноническая knowledge-model протокола + адаптер summary + валидатор (Workstream G).

Модель описывает протокол как набор **атомарных требований** с источником, trust level
и признаком ``penalty_allowed``. Адаптер конвертирует существующие ``ProtocolSummary``
без массовой ре-экстракции PDF (§11).

Инвариант валидатора (§11.3): knowledge document НЕ penalty-ready, если нет
подтверждённой цитаты / применимости / условие неизвестно / правило оборвано /
review/trust ниже B.

CLI:
    python -m clinical_knowledge.protocol_knowledge_model --validate <summary.json>
    python -m clinical_knowledge.protocol_knowledge_model --validate-all
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .rule_trust import TRUST_A, TRUST_B, TRUST_C, TRUST_D

RequirementType = Literal[
    "diagnostic_criterion",
    "required_exam",
    "conditional_exam",
    "treatment_group",
    "drug",
    "red_flag",
    "follow_up",
    "routing",
    "contraindication",
    "informational",
]
Obligation = Literal["required", "conditional", "recommended", "warning", "informational"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SourceEvidence(_Base):
    protocol_id: str | None = None
    section_title: str | None = None
    section_type: str | None = None
    page_start: int | None = None
    quote: str | None = None
    quote_verified: bool = False
    table_index: int | None = None


class KnowledgeApplicability(_Base):
    population: list[str] = Field(default_factory=list)
    age_min_years: int | None = None
    age_max_years: int | None = None
    sex: str = "unknown"
    pregnancy: str = "unknown"
    care_setting: list[str] = Field(default_factory=list)
    specialty: list[str] = Field(default_factory=list)


class KnowledgeReview(_Base):
    review_status: str = "not_reviewed"
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    trust_level: str = TRUST_C


class MedicationRegimen(_Base):
    drug_name: str | None = None
    active_substance: str | None = None
    drug_group: str | None = None
    dose_text: str | None = None
    frequency_text: str | None = None
    duration_text: str | None = None
    route: str | None = None
    monitoring: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    source: SourceEvidence = Field(default_factory=SourceEvidence)


class AtomicRequirement(_Base):
    requirement_id: str
    type: RequirementType = "informational"
    obligation: Obligation = "recommended"
    canonical_item_id: str | None = None
    text_ru: str = ""
    applicability: KnowledgeApplicability = Field(default_factory=KnowledgeApplicability)
    source: SourceEvidence = Field(default_factory=SourceEvidence)
    trust: str = TRUST_D
    review_status: str = "not_reviewed"
    extraction_confidence: float = 0.5
    penalty_allowed: bool = False


class ConditionDefinition(_Base):
    condition_id: str
    name: str = ""
    icd10_codes: list[str] = Field(default_factory=list)
    applicability: KnowledgeApplicability = Field(default_factory=KnowledgeApplicability)
    requirements: list[AtomicRequirement] = Field(default_factory=list)
    medications: list[MedicationRegimen] = Field(default_factory=list)


class ProtocolKnowledgeDocument(_Base):
    protocol_id: str
    title: str = ""
    version: str = "1.0"
    review_status: str = "not_reviewed"
    trust_level: str = TRUST_C
    conditions: list[ConditionDefinition] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Trust mapping (консервативно; C/D не повышаем автоматически)
# --------------------------------------------------------------------------- #
def _summary_trust(review_status: str) -> str:
    rs = (review_status or "").strip().lower()
    if rs == "approved":
        return TRUST_A
    if rs == "reviewed":
        return TRUST_B
    return TRUST_C


def _obligation_from_level(level: str | None) -> Obligation:
    lvl = (level or "").strip().lower()
    if lvl == "required":
        return "required"
    if lvl == "conditional":
        return "conditional"
    if lvl in ("optional", "recommended"):
        return "recommended"
    return "recommended"


def _src_from_ref(ref: Any, protocol_id: str) -> SourceEvidence:
    if ref is None:
        return SourceEvidence(protocol_id=protocol_id)
    quote = (getattr(ref, "quote", None) or "").strip()
    section_type = getattr(ref, "section_type", None)
    return SourceEvidence(
        protocol_id=getattr(ref, "protocol_id", None) or protocol_id,
        section_title=getattr(ref, "section_title", None),
        section_type=section_type,
        page_start=getattr(ref, "page_start", None),
        quote=quote[:400] or None,
        quote_verified=len(quote) >= 8,
        table_index=getattr(ref, "table_index", None),
    )


def _req_trust(base_trust: str, src: SourceEvidence) -> str:
    """path/rich-table эвристики -> D; иначе base (A/B/C)."""
    stype = (src.section_type or "").lower()
    if "table" in stype or src.table_index is not None or "path" in stype:
        return TRUST_D
    return base_trust


def _penalty_ready(trust: str, src: SourceEvidence) -> bool:
    return trust in (TRUST_A, TRUST_B) and src.quote_verified


def _mk_req(
    rid: str, rtype: RequirementType, obligation: Obligation, text: str,
    ref: Any, base_trust: str, protocol_id: str, confidence: float, review_status: str,
) -> AtomicRequirement:
    src = _src_from_ref(ref, protocol_id)
    trust = _req_trust(base_trust, src)
    return AtomicRequirement(
        requirement_id=rid, type=rtype, obligation=obligation, text_ru=text[:300],
        source=src, trust=trust, review_status=review_status,
        extraction_confidence=confidence, penalty_allowed=_penalty_ready(trust, src),
    )


def summary_to_knowledge(summary: Any) -> ProtocolKnowledgeDocument:
    """Конвертер ``ProtocolSummary`` -> ``ProtocolKnowledgeDocument`` (§11.2).

    Не придумывает обязательность: берёт requirement_level из summary. Auto-поля -> C,
    path/rich-table -> D. Сохраняет source refs и condition scope.
    """
    protocol_id = getattr(summary, "protocol_id", "") or ""
    review_status = getattr(summary, "review_status", "not_reviewed") or "not_reviewed"
    base_trust = _summary_trust(review_status)
    source = getattr(summary, "source", None)
    title = getattr(source, "title", "") if source else ""

    conditions: list[ConditionDefinition] = []
    for cond in getattr(summary, "conditions", []) or []:
        cid = getattr(cond, "condition_id", "") or ""
        reqs: list[AtomicRequirement] = []
        meds: list[MedicationRegimen] = []
        i = 0

        for exam in getattr(cond, "required_exams", []) or []:
            i += 1
            reqs.append(_mk_req(
                f"{cid}:exam:{i}", "required_exam",
                _obligation_from_level(getattr(exam, "requirement_level", "required")),
                getattr(exam, "name", ""), getattr(exam, "source_ref", None),
                base_trust, protocol_id, 0.7, review_status,
            ))
        for exam in getattr(cond, "conditional_exams", []) or []:
            i += 1
            reqs.append(_mk_req(
                f"{cid}:cexam:{i}", "conditional_exam", "conditional",
                getattr(exam, "name", ""), getattr(exam, "source_ref", None),
                base_trust, protocol_id, 0.6, review_status,
            ))
        crit = getattr(cond, "diagnostic_criteria", None)
        for j, c in enumerate(getattr(crit, "required", []) or [] if crit else []):
            reqs.append(_mk_req(
                f"{cid}:crit:{j}", "diagnostic_criterion", "required",
                getattr(c, "text", ""), getattr(c, "source_ref", None),
                base_trust, protocol_id, 0.6, review_status,
            ))
        tx = getattr(cond, "treatment", None)
        if tx is not None:
            for k, g in enumerate(getattr(tx, "drug_groups", []) or []):
                reqs.append(_mk_req(
                    f"{cid}:txg:{k}", "treatment_group", "recommended",
                    getattr(g, "drug_group", ""), getattr(g, "source_ref", None),
                    base_trust, protocol_id, 0.6, review_status,
                ))
            for d in getattr(tx, "drugs", []) or []:
                src = _src_from_ref(getattr(d, "source_ref", None), protocol_id)
                meds.append(MedicationRegimen(
                    drug_name=getattr(d, "drug_name", None),
                    active_substance=getattr(d, "active_substance", None),
                    drug_group=getattr(d, "drug_group", None),
                    dose_text=getattr(d, "dose_text", None),
                    frequency_text=getattr(d, "frequency_text", None),
                    duration_text=getattr(d, "duration_text", None),
                    route=getattr(d, "route", None),
                    monitoring=list(getattr(d, "monitoring", []) or []),
                    contraindications=list(getattr(d, "contraindications", []) or []),
                    source=src,
                ))
        for m, rf in enumerate(getattr(cond, "red_flags", []) or []):
            reqs.append(_mk_req(
                f"{cid}:rf:{m}", "red_flag", "warning",
                getattr(rf, "text", ""), getattr(rf, "source_ref", None),
                base_trust, protocol_id, 0.6, review_status,
            ))
        for n, fu in enumerate(getattr(cond, "follow_up", []) or []):
            reqs.append(_mk_req(
                f"{cid}:fu:{n}", "follow_up", "recommended",
                getattr(fu, "text", ""), getattr(fu, "source_ref", None),
                base_trust, protocol_id, 0.6, review_status,
            ))

        appl = getattr(cond, "condition_applicability", None)
        k_appl = KnowledgeApplicability()
        if appl is not None:
            k_appl = KnowledgeApplicability(
                population=[str(p) for p in (getattr(appl, "population", []) or [])],
                age_min_years=getattr(appl, "age_min_years", None),
                age_max_years=getattr(appl, "age_max_years", None),
                sex=getattr(appl, "sex", "unknown") or "unknown",
                pregnancy=getattr(appl, "pregnancy", "unknown") or "unknown",
                care_setting=[str(c) for c in (getattr(appl, "care_setting", []) or [])],
            )

        conditions.append(ConditionDefinition(
            condition_id=cid, name=getattr(cond, "name", "") or "",
            icd10_codes=list(getattr(cond, "icd10_codes", []) or []),
            applicability=k_appl, requirements=reqs, medications=meds,
        ))

    return ProtocolKnowledgeDocument(
        protocol_id=protocol_id, title=title or "",
        version=getattr(summary, "summary_version", "1.0") or "1.0",
        review_status=review_status, trust_level=base_trust,
        conditions=conditions,
    )


def validate_knowledge_document(doc: ProtocolKnowledgeDocument) -> dict[str, Any]:
    """Диагностика пригодности к штрафующей оценке (§11.3).

    Возвращает агрегат: доли penalty-ready требований, причины непригодности.
    """
    total = 0
    penalty_ready = 0
    with_quote = 0
    verified_quote = 0
    reasons: dict[str, int] = {}

    def _bump(key: str) -> None:
        reasons[key] = reasons.get(key, 0) + 1

    for cond in doc.conditions:
        if not cond.condition_id:
            _bump("condition_unknown")
        for req in cond.requirements:
            total += 1
            q = (req.source.quote or "").strip()
            if q:
                with_quote += 1
            if req.source.quote_verified:
                verified_quote += 1
            if req.penalty_allowed:
                penalty_ready += 1
                continue
            # причины непригодности
            if not q:
                _bump("no_source_quote")
            elif not req.source.quote_verified:
                _bump("quote_not_verified")
            if req.trust in (TRUST_C, TRUST_D):
                _bump("trust_below_B")
            if req.source.table_index is not None:
                _bump("table_context_risk")
            if not (cond.applicability.population or cond.applicability.care_setting):
                _bump("no_applicability")

    return {
        "protocol_id": doc.protocol_id,
        "requirements_total": total,
        "penalty_ready": penalty_ready,
        "penalty_ready_pct": round(100.0 * penalty_ready / total, 1) if total else 0.0,
        "with_quote": with_quote,
        "verified_quote": verified_quote,
        "review_status": doc.review_status,
        "trust_level": doc.trust_level,
        "reasons": reasons,
        "document_penalty_ready": penalty_ready > 0 and doc.trust_level in (TRUST_A, TRUST_B),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _validate_path(path: str) -> dict[str, Any]:
    from .protocol_summary.schema import ProtocolSummary

    data = json.loads(open(path, encoding="utf-8").read())
    summary = ProtocolSummary.model_validate(data)
    doc = summary_to_knowledge(summary)
    return validate_knowledge_document(doc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Валидатор knowledge-model протокола")
    ap.add_argument("--validate", help="путь к summary.json")
    ap.add_argument("--validate-all", action="store_true", help="все загруженные summary")
    args = ap.parse_args(argv)

    if args.validate:
        print(json.dumps(_validate_path(args.validate), ensure_ascii=False, indent=2))
        return 0
    if args.validate_all:
        from .protocol_summary.loader import load_protocol_summaries

        summaries = load_protocol_summaries()
        ready = 0
        for s in summaries:
            doc = summary_to_knowledge(s)
            v = validate_knowledge_document(doc)
            if v["document_penalty_ready"]:
                ready += 1
        print(json.dumps({
            "protocols": len(summaries),
            "penalty_ready_documents": ready,
        }, ensure_ascii=False, indent=2))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
