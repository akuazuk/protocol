"""Навигация по Protocol Summary для UI поиска протоколов."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .loader import load_protocol_summaries
from .schema import ConditionSummary, ProtocolSummary

ROOT = Path(__file__).resolve().parents[2]

_SECTION_SPECS: tuple[tuple[str, str, str], ...] = (
    ("criteria", "Критерии и диагностика", "investigations"),
    ("exams", "Обследования", "investigations"),
    ("treatment", "Лечение", "medications"),
    ("red_flags", "Красные флаги", "care_algorithms"),
    ("follow_up", "Наблюдение и маршрутизация", "monitoring_frequency"),
)


def _norm_path(p: str | None) -> str:
    if not p:
        return ""
    return p.replace("\\", "/").strip().lower()


def _path_match(catalog_path: str, summary_path: str | None) -> bool:
    na, nb = _norm_path(catalog_path), _norm_path(summary_path)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na.endswith(nb) or nb.endswith(na):
        return True
    return Path(na).name == Path(nb).name


def find_summary_by_catalog_path(catalog_path: str) -> ProtocolSummary | None:
    """Сопоставление пути PDF в выдаче assist с Protocol Summary."""
    if not (catalog_path or "").strip():
        return None
    for summary in load_protocol_summaries(usable_only=False):
        lp = summary.source.local_path or ""
        if _path_match(catalog_path, lp):
            return summary
        # иногда local_path без префикса minzdrav_protocols/
        if lp and not lp.startswith("minzdrav") and _path_match(
            f"minzdrav_protocols/{lp.lstrip('/')}", catalog_path
        ):
            return summary
    return None


def _section_count(cond: ConditionSummary, section_id: str) -> int:
    if section_id == "criteria":
        n = 0
        for block in (cond.clinical_criteria, cond.diagnostic_criteria):
            if block is None:
                continue
            n += len(block.required) + len(block.optional) + len(block.exclusion)
        if cond.diagnosis_structure:
            n += len(cond.diagnosis_structure.required_components)
            n += len(cond.diagnosis_structure.optional_components)
        return n
    if section_id == "exams":
        return len(cond.required_exams) + len(cond.conditional_exams)
    if section_id == "treatment":
        tb = cond.treatment
        if not tb:
            return 0
        return (
            len(tb.non_drug)
            + len(tb.drug_groups)
            + len(tb.drugs)
            + len(tb.procedures)
            + len(tb.surgery)
        )
    if section_id == "red_flags":
        return len(cond.red_flags)
    if section_id == "follow_up":
        return len(cond.follow_up) + len(cond.hospitalization) + len(cond.routing)
    return 0


def _section_preview(cond: ConditionSummary, section_id: str) -> str | None:
    if section_id == "criteria" and cond.diagnostic_criteria and cond.diagnostic_criteria.required:
        return (cond.diagnostic_criteria.required[0].text or "")[:160] or None
    if section_id == "exams" and cond.required_exams:
        return (cond.required_exams[0].name or "")[:160] or None
    if section_id == "treatment" and cond.treatment and cond.treatment.drugs:
        d0 = cond.treatment.drugs[0]
        return (d0.drug_name or d0.active_substance or d0.drug_group or "")[:160] or None
    if section_id == "red_flags" and cond.red_flags:
        return (cond.red_flags[0].text or "")[:160] or None
    if section_id == "follow_up" and cond.follow_up:
        return (cond.follow_up[0].text or "")[:160] or None
    return None


def _condition_nav(cond: ConditionSummary, *, query: str = "", icd_codes: list[str] | None = None) -> dict[str, Any]:
    icd_set = {c.strip().upper() for c in (icd_codes or []) if c}
    icd_match = bool(icd_set & {c.strip().upper() for c in cond.icd10_codes})
    q = (query or "").lower()
    name_match = bool(q and len(q) >= 3 and q in (cond.name or "").lower())
    sections: list[dict[str, Any]] = []
    for sid, label, focus in _SECTION_SPECS:
        count = _section_count(cond, sid)
        if count <= 0:
            continue
        sections.append(
            {
                "id": sid,
                "label": label,
                "count": count,
                "extract_focus": focus,
                "preview": _section_preview(cond, sid),
            }
        )
    return {
        "condition_id": cond.condition_id,
        "name": cond.name,
        "icd10_codes": list(cond.icd10_codes),
        "icd_match": icd_match,
        "name_match": name_match,
        "sections": sections,
    }


def build_protocol_summary_nav(
    catalog_path: str,
    *,
    query: str = "",
    icd_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Payload для GET /api/protocol-summary-nav."""
    summary = find_summary_by_catalog_path(catalog_path)
    if summary is None:
        return {"available": False, "path": catalog_path}

    conditions = [_condition_nav(c, query=query, icd_codes=icd_codes) for c in summary.conditions]
    conditions = [c for c in conditions if c.get("sections")]
    # приоритет: совпадение по МКБ, затем по имени в запросе
    conditions.sort(
        key=lambda c: (
            0 if c.get("icd_match") else (1 if c.get("name_match") else 2),
            c.get("name") or "",
        )
    )

    return {
        "available": bool(conditions),
        "path": catalog_path,
        "protocol_id": summary.protocol_id,
        "title": summary.source.title or "",
        "review_status": summary.review_status,
        "extraction_status": summary.extraction_status,
        "conditions": conditions,
    }
