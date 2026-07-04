"""Развёрнутая сводка протокола и KZ-aligned brief для врача (без LLM)."""
from __future__ import annotations

import time
from typing import Any

from clinical_knowledge.kravira_sop_rules import sop_reference_for_block

KZ_SECTIONS: tuple[str, ...] = (
    "Жалобы и анамнез",
    "Объективный статус",
    "Диагноз и коды МКБ-10",
    "Обследование",
    "Лечение и назначения",
    "Наблюдение и контроль",
    "Направления и консультации специалистов",
)

_SOP_HINTS: dict[str, str] = {
    "Жалобы и анамнез": sop_reference_for_block("complaints")
    + " Уточните давность, динамику, локализацию; отразите анамнез жизни и аллергоанамнез.",
    "Объективный статус": sop_reference_for_block("objective_status")
    + " Зафиксируйте витальные показатели и локальный статус по профилю.",
    "Диагноз и коды МКБ-10": sop_reference_for_block("diagnosis"),
    "Обследование": sop_reference_for_block("exams"),
    "Лечение и назначения": sop_reference_for_block("treatment"),
    "Наблюдение и контроль": sop_reference_for_block("follow_up"),
    "Направления и консультации специалистов": (
        "Госпитализация, консультации специалистов и направления - по red flags и маршрутизации протокола."
    ),
}

_SOP_BLOCK_FOR_KZ: dict[str, str] = {
    "Жалобы и анамнез": "complaints",
    "Объективный статус": "objective_status",
    "Диагноз и коды МКБ-10": "diagnosis",
    "Обследование": "exams",
    "Лечение и назначения": "treatment",
    "Наблюдение и контроль": "follow_up",
}

_BRIEF_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_BRIEF_CACHE_TTL_SEC = 3600
_BRIEF_CACHE_MAX = 48

_SECTION_LABELS: dict[str, str] = {
    "criteria": "Диагноз и критерии",
    "exams": "Обследования",
    "treatment": "Лечение",
    "red_flags": "Настороженность и противопоказания",
    "follow_up": "Наблюдение и маршрут",
}


def clear_protocol_brief_cache() -> None:
    _BRIEF_CACHE.clear()


def _cache_key(path: str, condition_id: str, query: str, icd_codes: list[str] | None) -> str:
    icd_part = ",".join(sorted(c.strip().upper() for c in (icd_codes or []) if c))
    return f"{path.strip()}|{condition_id.strip()}|{(query or '').strip()[:400]}|{icd_part}"


def _source_fields(source_ref: Any) -> dict[str, Any]:
    if source_ref is None:
        return {}
    return {
        "page_start": getattr(source_ref, "page_start", None),
        "page_end": getattr(source_ref, "page_end", None),
        "section_title": getattr(source_ref, "section_title", None) or None,
        "protocol_ref": (getattr(source_ref, "section_title", None) or "")[:120] or None,
        "quote": (getattr(source_ref, "quote", None) or "")[:800] or None,
    }


def _item(
    *,
    text: str,
    obligation: str = "recommended",
    label: str = "",
    icd_related: bool = False,
    source_ref: Any = None,
) -> dict[str, Any]:
    body = (text or "").strip()
    if not body:
        return {}
    out: dict[str, Any] = {
        "text": body[:600],
        "obligation": obligation if obligation in ("required", "recommended", "conditional", "not_applicable") else "recommended",
        "label": label[:80] if label else "",
        "icd_related": bool(icd_related),
    }
    src = _source_fields(source_ref)
    if src.get("quote"):
        out["quote"] = src["quote"]
    if src.get("page_start"):
        out["page_start"] = src["page_start"]
    if src.get("section_title"):
        out["section_title"] = src["section_title"]
    if src.get("protocol_ref"):
        out["protocol_ref"] = src["protocol_ref"]
    return out


def _format_drug(drug: Any) -> str:
    parts: list[str] = []
    name = (getattr(drug, "drug_name", None) or getattr(drug, "active_substance", None) or getattr(drug, "drug_group", None) or "").strip()
    if name:
        parts.append(name)
    dose = (getattr(drug, "dose_text", None) or "").strip()
    freq = (getattr(drug, "frequency_text", None) or "").strip()
    dur = (getattr(drug, "duration_text", None) or "").strip()
    route = (getattr(drug, "route", None) or "").strip()
    if dose:
        parts.append(dose)
    if freq:
        parts.append(freq)
    if dur:
        parts.append(dur)
    if route:
        parts.append(route)
    ind = (getattr(drug, "indication", None) or "").strip()
    line = ", ".join(parts) if parts else "препарат"
    if ind and ind.lower() not in line.lower():
        line += f" ({ind})"
    return line[:500]


def _exam_obligation(exam: Any) -> str:
    level = str(getattr(exam, "requirement_level", None) or "recommended").lower()
    if level == "required":
        return "required"
    if level in ("conditional", "optional"):
        return "conditional"
    return "recommended"


def _collect_expanded_items(cond: Any, section_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
    from clinical_knowledge.protocol_summary.schema import ConditionSummary

    if not isinstance(cond, ConditionSummary):
        return []
    out: list[dict[str, Any]] = []

    def add(raw: dict[str, Any]) -> None:
        if raw and len(out) < limit:
            out.append(raw)

    if section_id == "criteria":
        for block in (cond.clinical_criteria, cond.diagnostic_criteria):
            if block is None:
                continue
            for item in block.required:
                add(_item(text=item.text, obligation="required", label="критерий", source_ref=item.source_ref))
            for item in block.optional:
                add(_item(text=item.text, obligation="recommended", label="критерий", source_ref=item.source_ref))
            for item in block.exclusion:
                add(_item(text=item.text, obligation="conditional", label="исключить", source_ref=item.source_ref))
        if cond.diagnosis_structure:
            for comp in cond.diagnosis_structure.required_components:
                txt = (comp.name or comp.description or "").strip()
                add(_item(text=txt, obligation="required", label="компонент диагноза", source_ref=comp.source_ref))
            for comp in cond.diagnosis_structure.optional_components:
                txt = (comp.name or comp.description or "").strip()
                add(_item(text=txt, obligation="recommended", label="компонент диагноза", source_ref=comp.source_ref))
        if cond.icd10_codes:
            icd_line = "МКБ-10: " + ", ".join(cond.icd10_codes[:8])
            add(_item(text=icd_line, obligation="required", label="МКБ", icd_related=True))

    elif section_id == "exams":
        for exam in cond.required_exams:
            txt = (exam.name or "").strip()
            if exam.comment:
                txt += f" - {exam.comment.strip()}"
            if exam.timing:
                txt += f" ({exam.timing.strip()})"
            add(_item(text=txt, obligation=_exam_obligation(exam), label="обследование", source_ref=exam.source_ref))
        for exam in cond.conditional_exams:
            txt = (exam.name or "").strip()
            if exam.required_if:
                txt += " - при: " + "; ".join(exam.required_if[:3])
            add(_item(text=txt, obligation="conditional", label="обследование", source_ref=exam.source_ref))

    elif section_id == "treatment":
        tb = cond.treatment
        if tb:
            for drug in tb.drugs:
                add(_item(text=_format_drug(drug), obligation="recommended", label="препарат", source_ref=drug.source_ref))
            for grp in tb.drug_groups:
                txt = grp.drug_group
                if grp.indication:
                    txt += f" ({grp.indication})"
                add(_item(text=txt, obligation="recommended", label="группа препаратов", source_ref=grp.source_ref))
            for nd in tb.non_drug:
                add(_item(text=nd.text, obligation="recommended", label="немедикаментозно", source_ref=nd.source_ref))
            for proc in tb.procedures:
                txt = proc.name
                if proc.indication:
                    txt += f" ({proc.indication})"
                add(_item(text=txt, obligation="conditional", label="процедура", source_ref=proc.source_ref))
            for surg in tb.surgery:
                txt = surg.name
                if surg.indication:
                    txt += f" ({surg.indication})"
                add(_item(text=txt, obligation="conditional", label="хирургия", source_ref=surg.source_ref))

    elif section_id == "red_flags":
        for rf in cond.red_flags:
            txt = rf.text
            if rf.expected_actions:
                txt += " → " + "; ".join(rf.expected_actions[:3])
            ob = "required" if rf.severity in ("high", "critical") else "conditional"
            add(_item(text=txt, obligation=ob, label="red flag", source_ref=rf.source_ref))

    elif section_id == "follow_up":
        for fu in cond.follow_up:
            txt = fu.text
            if fu.timing:
                txt = f"{fu.timing}: {txt}"
            add(_item(text=txt, obligation="recommended", label="наблюдение", source_ref=fu.source_ref))
        for hosp in cond.hospitalization:
            txt = hosp.text
            if hosp.timing:
                txt = f"{hosp.timing}: {txt}"
            add(_item(text=txt, obligation="conditional", label="госпитализация", source_ref=hosp.source_ref))
        for route in cond.routing:
            txt = route.text
            if route.timing:
                txt = f"{route.timing}: {txt}"
            add(_item(text=txt, obligation="conditional", label="маршрут", source_ref=route.source_ref))

    return out[:limit]


def _section_total_count(cond: Any, section_id: str) -> int:
    from clinical_knowledge.protocol_summary.nav import _section_count

    return _section_count(cond, section_id)


def build_expanded_brief(
    catalog_path: str,
    *,
    condition_id: str,
    query: str = "",
    icd_codes: list[str] | None = None,
    items_per_section: int = 5,
) -> dict[str, Any]:
    """Развёрнутая сводка по разделам Summary (top-N пунктов с obligation)."""
    from clinical_knowledge.protocol_summary.nav import find_summary_by_catalog_path

    summary = find_summary_by_catalog_path(catalog_path)
    if summary is None:
        return {"available": False, "path": catalog_path}
    cond = next((c for c in summary.conditions if c.condition_id == condition_id), None)
    if cond is None:
        return {"available": False, "path": catalog_path, "error": "condition_not_found"}

    icd_set = {c.strip().upper() for c in (icd_codes or []) if c}
    sections_out: list[dict[str, Any]] = []
    red_flags: list[dict[str, Any]] = []

    for sid in ("red_flags", "criteria", "exams", "treatment", "follow_up"):
        items = _collect_expanded_items(cond, sid, limit=items_per_section)
        total = _section_total_count(cond, sid)
        if sid == "red_flags":
            red_flags = items
            continue
        if not items and total <= 0:
            continue
        sections_out.append(
            {
                "id": sid,
                "label": _SECTION_LABELS.get(sid, sid),
                "items": items,
                "total_count": total,
                "has_more": total > len(items),
            }
        )

    return {
        "available": bool(sections_out or red_flags),
        "path": catalog_path,
        "protocol_id": summary.protocol_id,
        "title": summary.source.title or "",
        "condition_id": condition_id,
        "condition_name": cond.name,
        "icd10_codes": list(cond.icd10_codes),
        "sections": sections_out,
        "red_flags": red_flags,
        "llm_used": False,
    }


def build_kz_brief(
    catalog_path: str,
    *,
    condition_id: str,
    query: str = "",
    icd_codes: list[str] | None = None,
    items_per_section: int = 8,
) -> dict[str, Any]:
    """Сводка, выровненная по рубрикам консультативного заключения."""
    from clinical_knowledge.protocol_summary.nav import find_summary_by_catalog_path

    summary = find_summary_by_catalog_path(catalog_path)
    if summary is None:
        return {"available": False, "path": catalog_path}
    cond = next((c for c in summary.conditions if c.condition_id == condition_id), None)
    if cond is None:
        return {"available": False, "path": catalog_path, "error": "condition_not_found"}

    icd_set = {c.strip().upper() for c in (icd_codes or []) if c}
    cond_icd = {c.strip().upper() for c in cond.icd10_codes if c}
    icd_overlap = bool(icd_set and cond_icd and (icd_set & cond_icd or any(
        a.startswith(b.split(".")[0]) or b.startswith(a.split(".")[0])
        for a in icd_set for b in cond_icd
    )))

    def icd_flag() -> bool:
        return icd_overlap

    kz_sections: list[dict[str, Any]] = []
    required_total = 0
    required_shown = 0

    def push_kz(kz_name: str, items: list[dict[str, Any]]) -> None:
        if not items:
            kz_sections.append({"kz_section": kz_name, "items": [], "sop_hint": _SOP_HINTS.get(kz_name, "")})
            return
        kz_sections.append(
            {
                "kz_section": kz_name,
                "items": items[:items_per_section],
                "sop_hint": _SOP_HINTS.get(kz_name, ""),
                "total_count": len(items),
            }
        )

    # Диагноз
    diag_items: list[dict[str, Any]] = []
    for raw in _collect_expanded_items(cond, "criteria", limit=items_per_section):
        if raw.get("obligation") == "required":
            required_total += 1
            required_shown += 1
        raw = dict(raw)
        raw["icd_related"] = raw.get("icd_related") or icd_flag()
        diag_items.append(raw)
    push_kz("Диагноз и коды МКБ-10", diag_items)

    # Обследование
    exam_items: list[dict[str, Any]] = []
    for raw in _collect_expanded_items(cond, "exams", limit=items_per_section):
        if raw.get("obligation") == "required":
            required_total += 1
            required_shown += 1
        exam_items.append(raw)
    push_kz("Обследование", exam_items)

    # Лечение
    treat_items = _collect_expanded_items(cond, "treatment", limit=items_per_section)
    push_kz("Лечение и назначения", treat_items)

    # Наблюдение
    follow_items = _collect_expanded_items(cond, "follow_up", limit=items_per_section)
    push_kz("Наблюдение и контроль", follow_items)

    # Направления / госпитализация (red flags - отдельно в expanded, не дублируем)
    route_items: list[dict[str, Any]] = []
    for raw in _collect_expanded_items(cond, "follow_up", limit=items_per_section):
        lbl = (raw.get("label") or "").lower()
        if lbl in ("госпитализация", "маршрут"):
            route_items.append(raw)
    push_kz("Направления и консультации специалистов", route_items[:items_per_section])

    # Жалобы / статус - в Summary обычно пусто; не засоряем карточку пустыми блоками.

    order = {s: i for i, s in enumerate(KZ_SECTIONS)}
    kz_sections.sort(key=lambda s: order.get(s["kz_section"], 99))

    blocks_filled = sum(1 for s in kz_sections if (s.get("items") or []))
    blocks_total = len(KZ_SECTIONS)

    summary_ru = ""
    if cond.name:
        summary_ru = f"Нозология: {cond.name}."
        if cond.icd10_codes:
            summary_ru += f" МКБ: {', '.join(cond.icd10_codes[:4])}."

    return {
        "available": blocks_filled > 0,
        "path": catalog_path,
        "title": summary.source.title or "",
        "condition_id": condition_id,
        "condition_name": cond.name,
        "summary_ru": summary_ru,
        "sections": kz_sections,
        "coverage": {
            "required_total": required_total,
            "required_shown": required_shown,
            "blocks_filled": blocks_filled,
            "blocks_total": blocks_total,
            "matrix_items": sum(len(s.get("items") or []) for s in kz_sections),
        },
        "llm_used": False,
    }


def kz_brief_to_heuristic_matrix(
    kz_brief: dict[str, Any],
    *,
    path: str,
    title: str,
    icd_codes: list[str] | None,
) -> dict[str, Any]:
    """Матрица КЗ из KZ-brief без LLM (для prefetch в UI)."""
    sections_out: list[dict[str, Any]] = []
    for sec in kz_brief.get("sections") or []:
        kz_name = str(sec.get("kz_section") or "").strip()
        if not kz_name:
            continue
        items_in: list[dict[str, Any]] = []
        for it in sec.get("items") or []:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            ob = str(it.get("obligation") or "recommended").lower()
            if ob not in ("required", "recommended", "conditional", "not_applicable"):
                ob = "recommended"
            items_in.append(
                {
                    "text": text[:500],
                    "obligation": ob,
                    "protocol_ref": str(it.get("protocol_ref") or it.get("section_title") or "")[:120],
                    "protocol_excerpt": str(it.get("quote") or it.get("text") or "")[:600],
                    "icd_related": bool(it.get("icd_related")),
                }
            )
        if items_in:
            sections_out.append({"kz_section": kz_name, "items": items_in})
    order = {s: i for i, s in enumerate(KZ_SECTIONS)}
    sections_out.sort(key=lambda s: order.get(s["kz_section"], 99))
    return {
        "path": path,
        "protocol_title": title,
        "icd_codes": list(icd_codes or []),
        "summary_ru": str(kz_brief.get("summary_ru") or "").strip(),
        "sections": sections_out,
        "disclaimer_ru": "Ориентир по карточке Summary; проверьте по PDF протокола.",
        "source": "summary_heuristic",
    }


def build_protocol_brief_bundle(
    catalog_path: str,
    *,
    condition_id: str,
    query: str = "",
    icd_codes: list[str] | None = None,
) -> dict[str, Any]:
    """expanded + kz_brief + prefetch matrix."""
    t0 = time.perf_counter()
    expanded = build_expanded_brief(
        catalog_path,
        condition_id=condition_id,
        query=query,
        icd_codes=icd_codes,
    )
    kz = build_kz_brief(
        catalog_path,
        condition_id=condition_id,
        query=query,
        icd_codes=icd_codes,
    )
    title = expanded.get("title") or kz.get("title") or ""
    matrix = kz_brief_to_heuristic_matrix(
        kz,
        path=catalog_path,
        title=title,
        icd_codes=icd_codes,
    )
    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "available": bool(expanded.get("available") or kz.get("available")),
        "path": catalog_path,
        "condition_id": condition_id,
        "expanded": expanded,
        "kz_brief": kz,
        "kz_matrix": matrix,
        "brief_ms": ms,
        "llm_used": False,
    }


def resolve_protocol_brief_bundle_cached(
    path: str,
    *,
    condition_id: str,
    query: str = "",
    icd_codes: list[str] | None = None,
) -> dict[str, Any]:
    key = _cache_key(path, condition_id, query, icd_codes)
    now = time.time()
    cached = _BRIEF_CACHE.get(key)
    if cached and now - cached[0] < _BRIEF_CACHE_TTL_SEC:
        out = dict(cached[1])
        out["cache_hit"] = True
        return out
    out = build_protocol_brief_bundle(
        path,
        condition_id=condition_id,
        query=query,
        icd_codes=icd_codes,
    )
    out = dict(out)
    out["cache_hit"] = False
    if len(_BRIEF_CACHE) >= _BRIEF_CACHE_MAX:
        oldest_key = min(_BRIEF_CACHE, key=lambda k: _BRIEF_CACHE[k][0])
        _BRIEF_CACHE.pop(oldest_key, None)
    _BRIEF_CACHE[key] = (now, dict(out))
    return out


def attach_protocol_brief_map(
    payload: dict[str, Any],
    *,
    query: str,
    icd_codes: list[str] | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """Вложить protocol_brief для top-N протоколов. Не вызывать из /api/assist на Render - только on-demand API."""
    from clinical_knowledge.protocol_nav_cache import resolve_protocol_nav_cached

    protos: list[dict[str, Any]] = []
    llm = payload.get("llm_json")
    if isinstance(llm, dict):
        raw = llm.get("protocols") or []
        protos = [p for p in raw if isinstance(p, dict)]

    codes = list(icd_codes or [])
    if not codes:
        icd_payload = payload.get("icd") or {}
        if isinstance(icd_payload, dict):
            codes = list(icd_payload.get("codes_for_retrieval") or [])

    brief_map: dict[str, Any] = {}
    for pr in protos[: max(0, int(limit))]:
        pth = str(pr.get("path") or "").strip()
        if not pth or pth in brief_map:
            continue
        nav = resolve_protocol_nav_cached(
            pth,
            query=query,
            icd_codes=codes or None,
            allow_rich_fallback=False,
        )
        conds = nav.get("conditions") or []
        if not nav.get("available") or not conds:
            continue
        cid = str(conds[0].get("condition_id") or "").strip()
        if not cid:
            continue
        brief_map[pth] = resolve_protocol_brief_bundle_cached(
            pth,
            condition_id=cid,
            query=query,
            icd_codes=codes or None,
        )

    if brief_map:
        payload["protocol_brief"] = brief_map
    return payload
