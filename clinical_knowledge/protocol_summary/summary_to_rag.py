"""Генерация RAG-чанков из Protocol Summary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import protocol_summary_config
from .schema import ConditionSummary, ProtocolSummary

ROOT = Path(__file__).resolve().parents[2]

SUMMARY_CHUNK_TYPES = (
    "summary_overview",
    "summary_diagnosis_structure",
    "summary_clinical_criteria",
    "summary_diagnostic_criteria",
    "summary_required_exams",
    "summary_treatment",
    "summary_follow_up",
    "summary_red_flags",
    "summary_contraindications",
)


def _chunk_base(summary: ProtocolSummary, cond: ConditionSummary, section_type: str) -> dict[str, Any]:
    return {
        "chunk_id": f"{summary.protocol_id}__{cond.condition_id}__{section_type}",
        "protocol_id": summary.protocol_id,
        "condition_id": cond.condition_id,
        "condition_name": cond.name,
        "icd10_codes": cond.icd10_codes,
        "rubric_name": summary.rubric.name,
        "rubric_slug": summary.rubric.slug,
        "section_type": section_type,
        "generated_from_summary": True,
        "summary_version": summary.summary_version,
    }


def _refs_to_list(refs: list[Any]) -> list[dict[str, Any]]:
    out = []
    for r in refs:
        if hasattr(r, "model_dump"):
            out.append(r.model_dump(mode="json"))
        elif isinstance(r, dict):
            out.append(r)
    return out


def condition_to_summary_chunks(summary: ProtocolSummary, cond: ConditionSummary) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    overview = _chunk_base(summary, cond, "summary_overview")
    overview["text"] = (
        f"Протокол: {summary.source.title}. Нозология: {cond.name}. "
        f"МКБ-10: {', '.join(cond.icd10_codes) or ' - '}. "
        f"Рубрика: {summary.rubric.name}."
    )
    overview["source_refs"] = _refs_to_list(cond.source_refs)
    chunks.append(overview)

    if cond.diagnosis_structure and (
        cond.diagnosis_structure.required_components or cond.diagnosis_structure.examples
    ):
        c = _chunk_base(summary, cond, "summary_diagnosis_structure")
        parts = [x.name for x in cond.diagnosis_structure.required_components]
        c["text"] = "Структура диагноза: " + "; ".join(parts)
        c["source_refs"] = _refs_to_list(cond.diagnosis_structure.source_refs)
        chunks.append(c)

    for block, stype in (
        (cond.clinical_criteria, "summary_clinical_criteria"),
        (cond.diagnostic_criteria, "summary_diagnostic_criteria"),
    ):
        if block and block.required:
            c = _chunk_base(summary, cond, stype)
            c["text"] = "; ".join(x.text for x in block.required[:12])
            c["source_refs"] = [x.source_ref.model_dump(mode="json") for x in block.required[:5]]
            chunks.append(c)

    if cond.required_exams or cond.conditional_exams:
        c = _chunk_base(summary, cond, "summary_required_exams")
        names = [e.name for e in cond.required_exams] + [e.name for e in cond.conditional_exams]
        c["text"] = "Обследования: " + "; ".join(names[:20])
        c["source_refs"] = [
            e.source_ref.model_dump(mode="json")
            for e in (cond.required_exams + cond.conditional_exams)[:5]
        ]
        chunks.append(c)

    if cond.treatment and (
        cond.treatment.drugs or cond.treatment.drug_groups or cond.treatment.non_drug
    ):
        c = _chunk_base(summary, cond, "summary_treatment")
        parts: list[str] = []
        for g in cond.treatment.drug_groups:
            parts.append(g.drug_group)
        for d in cond.treatment.drugs:
            parts.append(d.drug_name or d.active_substance or "")
        for nd in cond.treatment.non_drug:
            parts.append(nd.text)
        c["text"] = "Лечение: " + "; ".join(p for p in parts if p)
        c["source_refs"] = _refs_to_list(cond.treatment.source_refs)
        chunks.append(c)

    if cond.follow_up:
        c = _chunk_base(summary, cond, "summary_follow_up")
        c["text"] = "; ".join(f.text for f in cond.follow_up)
        c["source_refs"] = [f.source_ref.model_dump(mode="json") for f in cond.follow_up[:5]]
        chunks.append(c)

    if cond.red_flags:
        c = _chunk_base(summary, cond, "summary_red_flags")
        c["text"] = "; ".join(rf.text for rf in cond.red_flags)
        c["source_refs"] = [rf.source_ref.model_dump(mode="json") for rf in cond.red_flags[:5]]
        chunks.append(c)

    if cond.contraindications:
        c = _chunk_base(summary, cond, "summary_contraindications")
        c["text"] = "; ".join(x.text for x in cond.contraindications)
        c["source_refs"] = [x.source_ref.model_dump(mode="json") for x in cond.contraindications[:5]]
        chunks.append(c)

    return chunks


def summary_to_rag_chunks(summary: ProtocolSummary) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for cond in summary.conditions:
        chunks.extend(condition_to_summary_chunks(summary, cond))
    return chunks


def write_summary_rag_jsonl(
    summaries: list[ProtocolSummary] | None = None,
    out_path: Path | None = None,
) -> Path:
    summaries = summaries or list(__import__(
        "clinical_knowledge.protocol_summary.loader", fromlist=["load_protocol_summaries"],
    ).load_protocol_summaries())
    root = Path(protocol_summary_config.data_root)
    if not root.is_absolute():
        root = ROOT / root
    out_path = out_path or (root / "summary_chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for summary in summaries:
            for chunk in summary_to_rag_chunks(summary):
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return out_path
