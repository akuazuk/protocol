"""Тесты knowledge-model протокола + адаптера summary (Workstream G ТЗ overnight-v1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.protocol_knowledge_model import (
    summary_to_knowledge,
    validate_knowledge_document,
)
from clinical_knowledge.protocol_summary.schema import (
    ConditionSummary,
    ExamRequirement,
    ProtocolSource,
    ProtocolSummary,
    SummarySourceRef,
)

_QUOTE = "общий анализ крови развёрнутый выполняется всем пациентам"


def _summary(review_status: str = "not_reviewed", verified_quote: bool = True) -> ProtocolSummary:
    ref = SummarySourceRef(protocol_id="p1", section_title="Обследование", quote=_QUOTE if verified_quote else "")
    exam = ExamRequirement(name="общий анализ крови", requirement_level="required", source_ref=ref)
    cond = ConditionSummary(condition_id="c1", name="Острый фарингит", icd10_codes=["J02.9"], required_exams=[exam])
    return ProtocolSummary(
        protocol_id="p1", review_status=review_status,
        source=ProtocolSource(title="КП острый фарингит"), conditions=[cond],
    )


def test_adapter_preserves_source_and_marks_auto_C():
    doc = summary_to_knowledge(_summary("not_reviewed"))
    assert doc.protocol_id == "p1"
    assert len(doc.conditions) == 1
    reqs = doc.conditions[0].requirements
    assert len(reqs) == 1
    r = reqs[0]
    assert r.type == "required_exam"
    assert r.obligation == "required"
    assert r.source.quote and r.source.quote_verified is True
    assert r.trust == "C"  # auto summary без review
    assert r.penalty_allowed is False


def test_adapter_approved_becomes_A_penalty_ready():
    doc = summary_to_knowledge(_summary("approved"))
    r = doc.conditions[0].requirements[0]
    assert r.trust == "A"
    assert r.penalty_allowed is True
    v = validate_knowledge_document(doc)
    assert v["penalty_ready"] == 1
    assert v["document_penalty_ready"] is True


def test_validator_flags_no_quote():
    doc = summary_to_knowledge(_summary("approved", verified_quote=False))
    r = doc.conditions[0].requirements[0]
    assert r.penalty_allowed is False  # нет цитаты -> не penalty-ready
    v = validate_knowledge_document(doc)
    assert v["penalty_ready"] == 0
    assert "no_source_quote" in v["reasons"]


def test_validator_not_ready_when_trust_below_B():
    doc = summary_to_knowledge(_summary("not_reviewed"))
    v = validate_knowledge_document(doc)
    assert v["document_penalty_ready"] is False
    assert v["verified_quote"] == 1
    assert "trust_below_B" in v["reasons"]
