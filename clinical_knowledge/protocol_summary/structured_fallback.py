"""Structured extraction from source_text without LLM (fallback)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .builder import _EXAM_RE, _RED_FLAG_RE, _FOLLOW_UP_RE, _DRUG_DOSE_RE, _protocol_id_from_path
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
from .source_text import section_text_blob


def _norm_icd(code: str) -> str:
    return re.sub(r"\s+", "", (code or "").upper().strip())


def _slug(s: str, max_len: int = 48) -> str:
    t = re.sub(r"[^a-z0-9а-яё]+", "_", (s or "").lower()).strip("_")
    return t[:max_len] or "condition"


def _ref(protocol_id: str, path: str, text: str, page: int | None = None) -> SummarySourceRef:
    return SummarySourceRef(
        protocol_id=protocol_id,
        local_path=path,
        page_start=page,
        quote=(text or "…")[:400],
    )


def _extract_exams(blob: str, protocol_id: str, path: str) -> tuple[list[ExamRequirement], list[ExamRequirement]]:
    required: list[ExamRequirement] = []
    conditional: list[ExamRequirement] = []
    seen: set[str] = set()
    for m in _EXAM_RE.finditer(blob):
        name = m.group(0).strip()
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        ctx = blob[max(0, m.start() - 40) : m.end() + 80]
        lvl = "conditional" if re.search(r"по\s+показан|при\s+необходим|может\s+быть", ctx, re.I) else "required"
        item = ExamRequirement(
            name=name,
            requirement_level=lvl,  # type: ignore[arg-type]
            source_ref=_ref(protocol_id, path, ctx.strip()[:200]),
        )
        if lvl == "required":
            required.append(item)
        else:
            conditional.append(item)
        if len(required) + len(conditional) >= 14:
            break
    return required, conditional


def _extract_drugs(blob: str, protocol_id: str, path: str) -> list[DrugTreatmentItem]:
    out: list[DrugTreatmentItem] = []
    seen: set[str] = set()
    for m in _DRUG_DOSE_RE.finditer(blob):
        name = m.group(1).strip()
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        dose = (m.group(2) or "").strip()
        out.append(
            DrugTreatmentItem(
                drug_name=name,
                dose_text=dose or None,
                source_ref=_ref(protocol_id, path, m.group(0)[:200]),
            ),
        )
        if len(out) >= 12:
            break
    return out


def _extract_red_flags(blob: str, protocol_id: str, path: str) -> list[RedFlagItem]:
    out: list[RedFlagItem] = []
    for rx, rtype, sev in _RED_FLAG_RE:
        for m in rx.finditer(blob):
            ctx = blob[max(0, m.start() - 20) : m.end() + 120].strip()
            out.append(
                RedFlagItem(
                    text=ctx[:300],
                    red_flag_type=rtype,  # type: ignore[arg-type]
                    severity=sev,  # type: ignore[arg-type]
                    source_ref=_ref(protocol_id, path, ctx[:200]),
                ),
            )
            if len(out) >= 6:
                return out
    return out


def _extract_follow_up(blob: str, protocol_id: str, path: str) -> list[FollowUpRequirement]:
    out: list[FollowUpRequirement] = []
    for m in _FOLLOW_UP_RE.finditer(blob):
        txt = m.group(0).strip()
        out.append(FollowUpRequirement(text=txt[:300], source_ref=_ref(protocol_id, path, txt[:200])))
        if len(out) >= 5:
            break
    return out


def _icd_from_blob(blob: str, catalog_icd: list[str]) -> list[str]:
    found: list[str] = []
    for m in re.finditer(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b", blob):
        c = _norm_icd(m.group(1))
        if c and c[0] not in ("Y", "T", "X", "Z") and c not in found:
            found.append(c)
    for c in catalog_icd:
        nc = _norm_icd(c)
        if nc and nc not in found:
            found.append(nc)
    return found[:12]


def build_structured_summary(doc: dict[str, Any]) -> ProtocolSummary:
    """Rule-based unified summary from sectioned source_text."""
    protocol_id = str(doc.get("protocol_id") or _protocol_id_from_path(str(doc.get("path") or "")))
    path = str(doc.get("path") or "")
    title = str(doc.get("title") or protocol_id)
    catalog_icd = list(doc.get("icd10_primary") or doc.get("icd10_all") or [])

    diag_blob = section_text_blob(doc, ["classification", "criteria", "diagnostics"])
    treat_blob = section_text_blob(doc, ["treatment", "other"])
    route_blob = section_text_blob(doc, ["routing", "prevention"])
    full_blob = diag_blob + "\n" + treat_blob

    icd_codes = _icd_from_blob(diag_blob, [str(c) for c in catalog_icd])
    cond_name = title[:200]
    if icd_codes:
        cond_name = cond_name.split("(")[0].strip() or cond_name

    req_ex, cond_ex = _extract_exams(diag_blob, protocol_id, path)
    drugs = _extract_drugs(treat_blob, protocol_id, path)
    non_drug: list[NonDrugTreatmentItem] = []
    for line in re.split(r"[;\n•]", treat_blob):
        t = line.strip()
        if 30 < len(t) < 280 and not _DRUG_DOSE_RE.search(t):
            non_drug.append(NonDrugTreatmentItem(text=t[:400], source_ref=_ref(protocol_id, path, t[:200])))
            if len(non_drug) >= 6:
                break

    criteria: list[CriterionItem] = []
    for line in re.split(r"[;\n•]", diag_blob):
        t = line.strip()
        if 25 < len(t) < 220 and re.search(r"критери|диагноз|определен", t, re.I):
            criteria.append(CriterionItem(text=t[:400], source_ref=_ref(protocol_id, path, t[:200])))
            if len(criteria) >= 8:
                break

    routing: list[RoutingRequirement] = []
    for line in re.split(r"[;\n•]", route_blob):
        t = line.strip()
        if 20 < len(t) < 240 and re.search(r"госпитал|амбулатор|направлен", t, re.I):
            routing.append(RoutingRequirement(text=t[:300], source_ref=_ref(protocol_id, path, t[:200])))
            if len(routing) >= 4:
                break

    treatment = None
    if drugs or non_drug:
        treatment = TreatmentBlock(drugs=drugs, non_drug=non_drug)

    condition = ConditionSummary(
        condition_id=_slug(cond_name),
        name=cond_name,
        icd10_codes=icd_codes,
        diagnostic_criteria=CriteriaBlock(required=criteria) if criteria else None,
        required_exams=req_ex,
        conditional_exams=cond_ex,
        treatment=treatment,
        follow_up=_extract_follow_up(full_blob, protocol_id, path),
        routing=routing,
        red_flags=_extract_red_flags(full_blob, protocol_id, path),
        source_refs=[_ref(protocol_id, path, title[:120])],
    )

    aud = str(doc.get("audience") or "").lower()
    pops: list[str] = ["unknown"]
    if aud == "child":
        pops = ["child"]
    elif aud == "adult":
        pops = ["adult"]
    elif aud == "pregnant":
        pops = ["pregnant"]

    slug = str(doc.get("specialty_slug") or "")
    return ProtocolSummary(
        protocol_id=protocol_id,
        summary_version="2.0",
        extraction_status="auto_extracted",
        review_status="needs_review",
        source=ProtocolSource(title=title, local_path=path),
        rubric=ProtocolRubric(name=slug.replace("_", " ").title() or title[:80], slug=slug or None),
        applicability=ProtocolApplicability(population=pops),  # type: ignore[arg-type]
        conditions=[condition],
        extraction_metadata=ExtractionMetadata(
            extracted_at=datetime.now(timezone.utc).isoformat(),
            extractor="structured_fallback",
            extractor_version="1.0",
            notes=["source_text_sections"],
        ),
    )
