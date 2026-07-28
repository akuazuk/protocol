"""Merge LLM JSON passes into ProtocolSummary."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .schema import (
    ConditionSummary,
    CriteriaBlock,
    CriterionItem,
    DrugTreatmentItem,
    ExamRequirement,
    ExtractionMetadata,
    FollowUpRequirement,
    NonDrugTreatmentItem,
    ProtocolApplicability,
    ProtocolRubric,
    ProtocolSource,
    ProtocolSummary,
    RedFlagItem,
    RoutingRequirement,
    SummarySourceRef,
    TreatmentBlock,
)


def _norm_icd(code: str) -> str:
    return re.sub(r"\s+", "", (code or "").upper().strip())


def _slug(s: str, max_len: int = 48) -> str:
    t = re.sub(r"[^a-z0-9а-яё]+", "_", (s or "").lower()).strip("_")
    return t[:max_len] or "condition"


def _ref(
    protocol_id: str,
    local_path: str,
    *,
    quote: str = "",
    page_start: int | None = None,
    section_title: str | None = None,
) -> SummarySourceRef:
    return SummarySourceRef(
        protocol_id=protocol_id,
        local_path=local_path,
        page_start=page_start,
        quote=(quote or "…")[:400],
        section_title=section_title,
    )


def _merge_condition_block(
    protocol_id: str,
    local_path: str,
    condition: dict[str, Any],
    block: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "required_exams": [],
        "conditional_exams": [],
        "diagnostic_criteria": [],
        "treatment_non_drug": [],
        "drugs": [],
        "red_flags": [],
        "follow_up": [],
        "routing": [],
    }
    for ex in block.get("required_exams") or []:
        if not isinstance(ex, dict):
            continue
        lvl = str(ex.get("level") or "required").lower()
        item = ExamRequirement(
            name=str(ex.get("name") or "")[:200],
            requirement_level="required" if lvl == "required" else "conditional",
            source_ref=_ref(
                protocol_id,
                local_path,
                quote=str(ex.get("quote") or ex.get("name") or ""),
                page_start=ex.get("page_start"),
            ),
        )
        if item.requirement_level == "required":
            out["required_exams"].append(item)
        else:
            out["conditional_exams"].append(item)
    for txt in block.get("diagnostic_criteria") or []:
        if not isinstance(txt, str) or not txt.strip():
            continue
        out["diagnostic_criteria"].append(
            CriterionItem(
                text=txt.strip()[:500],
                source_ref=_ref(protocol_id, local_path, quote=txt.strip()[:200]),
            ),
        )
    for txt in block.get("treatment_non_drug") or []:
        if isinstance(txt, str) and txt.strip():
            out["treatment_non_drug"].append(
                NonDrugTreatmentItem(text=txt.strip()[:500], source_ref=_ref(protocol_id, local_path, quote=txt[:200])),
            )
    for d in block.get("drugs") or []:
        if not isinstance(d, dict):
            continue
        monitoring = [
            str(item).strip()[:240]
            for item in (d.get("monitoring") or [])
            if isinstance(item, str) and item.strip()
        ]
        out["drugs"].append(
            DrugTreatmentItem(
                drug_name=str(d.get("name") or "")[:120] or None,
                dose_text=str(d.get("dose_text") or "")[:200] or None,
                frequency_text=str(d.get("frequency_text") or "")[:200] or None,
                duration_text=str(d.get("duration_text") or "")[:200] or None,
                route=str(d.get("route") or "")[:120] or None,
                monitoring=monitoring[:12],
                source_ref=_ref(
                    protocol_id,
                    local_path,
                    quote=str(d.get("quote") or d.get("name") or ""),
                    page_start=d.get("page_start"),
                ),
            ),
        )
    for rf in block.get("red_flags") or []:
        if not isinstance(rf, dict):
            continue
        sev = str(rf.get("severity") or "medium").lower()
        if sev not in ("low", "medium", "high", "critical"):
            sev = "medium"
        out["red_flags"].append(
            RedFlagItem(
                text=str(rf.get("text") or "")[:400],
                severity=sev,  # type: ignore[arg-type]
                source_ref=_ref(
                    protocol_id,
                    local_path,
                    quote=str(rf.get("quote") or rf.get("text") or ""),
                    page_start=rf.get("page_start"),
                ),
            ),
        )
    for txt in block.get("follow_up") or []:
        if isinstance(txt, str) and txt.strip():
            out["follow_up"].append(
                FollowUpRequirement(text=txt.strip()[:400], source_ref=_ref(protocol_id, local_path, quote=txt[:200])),
            )
    for txt in block.get("routing") or []:
        if isinstance(txt, str) and txt.strip():
            out["routing"].append(
                RoutingRequirement(text=txt.strip()[:400], source_ref=_ref(protocol_id, local_path, quote=txt[:200])),
            )
    return out


def build_condition_summary(
    protocol_id: str,
    local_path: str,
    skeleton_cond: dict[str, Any],
    blocks: list[dict[str, Any]],
) -> ConditionSummary:
    merged: dict[str, Any] = {
        "required_exams": [],
        "conditional_exams": [],
        "diagnostic_criteria": [],
        "treatment_non_drug": [],
        "drugs": [],
        "red_flags": [],
        "follow_up": [],
        "routing": [],
    }
    for block in blocks:
        part = _merge_condition_block(protocol_id, local_path, skeleton_cond, block)
        for k in merged:
            merged[k].extend(part.get(k) or [])

    icd_codes = [_norm_icd(str(c)) for c in (skeleton_cond.get("icd10_codes") or []) if _norm_icd(str(c))]
    cond_id = str(skeleton_cond.get("condition_id") or _slug(str(skeleton_cond.get("name") or "condition")))
    diag_block = None
    if merged["diagnostic_criteria"]:
        diag_block = CriteriaBlock(required=merged["diagnostic_criteria"][:16])

    treatment = None
    if merged["drugs"] or merged["treatment_non_drug"]:
        treatment = TreatmentBlock(
            drugs=merged["drugs"][:24],
            non_drug=merged["treatment_non_drug"][:16],
        )

    return ConditionSummary(
        condition_id=cond_id,
        name=str(skeleton_cond.get("name") or cond_id)[:240],
        icd10_codes=icd_codes[:12],
        diagnostic_criteria=diag_block,
        required_exams=merged["required_exams"][:16],
        conditional_exams=merged["conditional_exams"][:12],
        treatment=treatment,
        follow_up=merged["follow_up"][:8],
        routing=merged["routing"][:6],
        red_flags=merged["red_flags"][:10],
    )


def merge_to_protocol_summary(
    doc: dict[str, Any],
    skeleton: dict[str, Any],
    condition_blocks: dict[str, list[dict[str, Any]]],
    *,
    extractor: str,
    extractor_version: str,
    extraction_status: str = "llm_extracted",
) -> ProtocolSummary:
    protocol_id = str(doc.get("protocol_id") or "")
    local_path = str(doc.get("path") or "")
    title = str(skeleton.get("title_ru") or doc.get("title") or protocol_id)
    pops = skeleton.get("population") or []
    if isinstance(pops, str):
        pops = [pops.strip()] if pops.strip() else []
    elif not isinstance(pops, list):
        pops = [str(pops)] if pops else []
    if not pops:
        aud = str(doc.get("audience") or "").lower()
        if aud == "child":
            pops = ["child"]
        elif aud == "adult":
            pops = ["adult"]
        else:
            pops = ["unknown"]

    conditions: list[ConditionSummary] = []
    for sk in skeleton.get("conditions") or []:
        if not isinstance(sk, dict):
            continue
        cid = str(sk.get("condition_id") or _slug(str(sk.get("name") or "")))
        blocks = condition_blocks.get(cid) or []
        conditions.append(build_condition_summary(protocol_id, local_path, sk, blocks))

    if not conditions:
        name = title[:200]
        conditions.append(
            ConditionSummary(
                condition_id=_slug(name),
                name=name,
                icd10_codes=[_norm_icd(c) for c in (doc.get("icd10_primary") or []) if _norm_icd(str(c))][:8],
                source_refs=[
                    _ref(protocol_id, local_path, quote=title[:120]),
                ],
            ),
        )

    slug = str(doc.get("specialty_slug") or "")
    return ProtocolSummary(
        protocol_id=protocol_id,
        summary_version="2.0",
        extraction_status=extraction_status,  # type: ignore[arg-type]
        review_status="needs_review",
        source=ProtocolSource(title=title, local_path=local_path),
        rubric=ProtocolRubric(name=slug.replace("_", " ").title() or title[:80], slug=slug or None),
        applicability=ProtocolApplicability(population=pops),  # type: ignore[arg-type]
        conditions=conditions,
        extraction_metadata=ExtractionMetadata(
            extracted_at=datetime.now(timezone.utc).isoformat(),
            extractor=extractor,
            extractor_version=extractor_version,
            notes=["llm_pipeline_v1"],
        ),
    )
