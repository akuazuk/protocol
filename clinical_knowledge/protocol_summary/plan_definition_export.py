"""Экспорт Protocol Summary Cards в FHIR PlanDefinition (черновик CDS)."""
from __future__ import annotations

import re
from typing import Any

from .schema import ConditionSummary, ProtocolSummary


def _slug(s: str, *, max_len: int = 64) -> str:
    out = re.sub(r"[^a-zA-Z0-9\-]+", "-", (s or "").strip().lower())
    out = re.sub(r"-+", "-", out).strip("-")
    return (out or "item")[:max_len]


def _source_citation(ref: Any) -> dict[str, Any] | None:
    if ref is None:
        return None
    doc = {}
    if getattr(ref, "document_url", None):
        doc["url"] = ref.document_url
    if getattr(ref, "local_path", None):
        doc["title"] = ref.local_path
    if getattr(ref, "page", None) is not None:
        doc["page"] = ref.page
    return doc or None


def _action_from_text(
    *,
    title: str,
    text: str,
    prefix: str,
    idx: int,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    act: dict[str, Any] = {
        "id": f"{prefix}-{_slug(title)[:40]}-{idx}",
        "title": title,
        "description": text.strip(),
        "code": [{"text": title}],
    }
    if source:
        act["relatedArtifact"] = [{"type": "citation", "document": source}]
    return act


def condition_to_plan_actions(condition: ConditionSummary) -> list[dict[str, Any]]:
    """Действия PlanDefinition из одного condition summary."""
    actions: list[dict[str, Any]] = []
    idx = 0

    if condition.diagnosis_structure:
        for comp in (condition.diagnosis_structure.required_components or [])[:8]:
            src = _source_citation(comp.source_ref)
            actions.append(
                _action_from_text(
                    title=f"Диагноз: {comp.name}",
                    text=comp.description or comp.name,
                    prefix="dx",
                    idx=idx,
                    source=src,
                )
            )
            idx += 1

    for ex in (condition.required_exams or [])[:12]:
        txt = ex.name
        if ex.comment:
            txt = f"{txt}. {ex.comment}"
        if ex.timing:
            txt = f"{txt} ({ex.timing})"
        actions.append(
            _action_from_text(
                title=f"Обследование ({ex.requirement_level}): {ex.name}",
                text=txt,
                prefix="exam",
                idx=idx,
                source=_source_citation(ex.source_ref),
            )
        )
        idx += 1

    if condition.treatment:
        for drug in (condition.treatment.drugs or [])[:10]:
            parts = [drug.drug_name or drug.active_substance or drug.drug_group or "Препарат"]
            if drug.indication:
                parts.append(str(drug.indication))
            if drug.dose_text:
                parts.append(str(drug.dose_text))
            actions.append(
                _action_from_text(
                    title=f"Лечение: {parts[0]}",
                    text="; ".join(p for p in parts if p),
                    prefix="rx",
                    idx=idx,
                    source=_source_citation(drug.source_ref),
                )
            )
            idx += 1
        for nd in (condition.treatment.non_drug or [])[:6]:
            actions.append(
                _action_from_text(
                    title="Немедикаментозное лечение",
                    text=nd.text,
                    prefix="tx",
                    idx=idx,
                    source=_source_citation(nd.source_ref),
                )
            )
            idx += 1

    for fu in (condition.follow_up or [])[:6]:
        txt = fu.text
        if fu.timing:
            txt = f"{txt} ({fu.timing})"
        actions.append(
            _action_from_text(
                title="Наблюдение",
                text=txt,
                prefix="fu",
                idx=idx,
                source=_source_citation(fu.source_ref),
            )
        )
        idx += 1

    return actions


def summary_to_plan_definition(
    summary: ProtocolSummary,
    *,
    condition_id: str | None = None,
    status: str = "draft",
    publisher: str = "Protocol RAG / Minzdrav RB",
) -> dict[str, Any]:
    """Собрать FHIR PlanDefinition (JSON) из summary card."""
    cond = None
    if condition_id:
        cond = next((c for c in summary.conditions if c.condition_id == condition_id), None)
    if cond is None and summary.conditions:
        cond = summary.conditions[0]
    if cond is None:
        raise ValueError("summary has no conditions")

    title = cond.name or summary.protocol_id
    icd = ", ".join(cond.icd10_codes[:8])
    actions = condition_to_plan_actions(cond)

    pd: dict[str, Any] = {
        "resourceType": "PlanDefinition",
        "id": _slug(f"{summary.protocol_id}-{cond.condition_id}"),
        "url": f"urn:protocol-summary:{summary.protocol_id}:{cond.condition_id}",
        "version": summary.summary_version,
        "name": _slug(cond.condition_id),
        "title": title,
        "status": status,
        "experimental": True,
        "date": summary.source.approval_date if summary.source else None,
        "publisher": publisher,
        "description": (
            f"Автоэкспорт из Protocol Summary Card ({summary.extraction_status}). "
            f"МКБ-10: {icd or '—'}."
        ),
        "type": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/plan-definition-type",
                    "code": "clinical-protocol",
                    "display": "Clinical Protocol",
                }
            ]
        },
        "subjectCodeableConcept": {
            "coding": [{"system": "urn:icd-10", "code": c} for c in cond.icd10_codes[:6]]
        },
        "action": actions,
        "extension": [
            {
                "url": "urn:protocol-summary:protocol-id",
                "valueString": summary.protocol_id,
            },
            {
                "url": "urn:protocol-summary:condition-id",
                "valueString": cond.condition_id,
            },
            {
                "url": "urn:protocol-summary:review-status",
                "valueString": summary.review_status,
            },
        ],
    }
    if summary.source and summary.source.local_path:
        pd.setdefault("relatedArtifact", []).append(
            {
                "type": "documentation",
                "display": summary.source.title or summary.source.local_path,
                "url": summary.source.local_path,
            }
        )
    return pd


def export_summaries_to_plan_definitions(
    summaries: list[ProtocolSummary],
    *,
    usable_only: bool = True,
    status: str = "draft",
) -> list[dict[str, Any]]:
    """Пакетный экспорт всех conditions из списка summary cards."""
    from .validator import summary_is_usable

    out: list[dict[str, Any]] = []
    for summary in summaries:
        if usable_only and not summary_is_usable(summary):
            continue
        for cond in summary.conditions:
            try:
                out.append(
                    summary_to_plan_definition(
                        summary,
                        condition_id=cond.condition_id,
                        status=status,
                    )
                )
            except ValueError:
                continue
    return out
