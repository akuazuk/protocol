"""Навигация по Protocol Summary для UI поиска протоколов."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .loader import clear_protocol_summary_cache, load_protocol_summaries
from .schema import ConditionSummary, ProtocolSummary

ROOT = Path(__file__).resolve().parents[2]

_SECTION_SPECS: tuple[tuple[str, str, str], ...] = (
    ("criteria", "Критерии и диагностика", "investigations"),
    ("exams", "Обследования", "investigations"),
    ("treatment", "Лечение", "medications"),
    ("red_flags", "Красные флаги", "care_algorithms"),
    ("follow_up", "Наблюдение и маршрутизация", "monitoring_frequency"),
)

_ICD_SHORT_RU: dict[str, str] = {
    "J20": "Острый бронхит",
    "J20.9": "Острый бронхит неуточнённый",
    "J41": "Простой хронический бронхит",
    "J41.0": "Простой хронический бронхит",
    "J41.1": "Слизисто-гнойный хронический бронхит",
    "J41.8": "Смешанный хронический бронхит",
    "J42": "Хронический бронхит неуточнённый",
}

_GENERIC_COND_NAME_RE = re.compile(
    r"^клинический\s+протокол\s+диагностики\s+и\s+лечения\s+",
    re.I,
)


def _icd_family(code: str) -> str:
    c = (code or "").strip().upper()
    if not c:
        return ""
    if "." in c:
        return c.split(".", 1)[0]
    return c


def _icd_codes_overlap(query_codes: set[str], cond_codes: list[str]) -> bool:
    if not query_codes or not cond_codes:
        return False
    cond_u = {c.strip().upper() for c in cond_codes if c}
    for q in query_codes:
        qu = q.strip().upper()
        if not qu:
            continue
        if qu in cond_u:
            return True
        qf = _icd_family(qu)
        for cu in cond_u:
            if cu == qf or cu.startswith(qf) or qu.startswith(_icd_family(cu)):
                return True
    return False


def _icd_from_condition_id(condition_id: str) -> str | None:
    m = re.match(r"^([a-z])(\d{2})(?:_(\d))?", (condition_id or "").lower())
    if not m:
        return None
    code = f"{m.group(1).upper()}{m.group(2)}"
    if m.group(3) is not None:
        code += f".{m.group(3)}"
    return code


def _norm_cond_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())[:96]


def _condition_group_key(cond: dict[str, Any]) -> str:
    icd_list = [str(c).strip().upper() for c in (cond.get("icd10_codes") or []) if c]
    if icd_list:
        return _icd_family(icd_list[0])
    cid_icd = _icd_from_condition_id(str(cond.get("condition_id") or ""))
    if cid_icd:
        return _icd_family(cid_icd)
    return _norm_cond_name(str(cond.get("name") or cond.get("condition_id") or ""))


def _display_label_for_condition(cond: dict[str, Any], *, protocol_title: str = "") -> str:
    icd_list = [str(c).strip().upper() for c in (cond.get("icd10_codes") or []) if c]
    primary_icd = icd_list[0] if icd_list else (_icd_from_condition_id(str(cond.get("condition_id") or "")) or "")
    if primary_icd:
        label = _ICD_SHORT_RU.get(primary_icd) or _ICD_SHORT_RU.get(_icd_family(primary_icd)) or primary_icd
        if len(icd_list) > 1:
            extra = ", ".join(icd_list[1:3])
            if len(icd_list) > 3:
                extra += "…"
            label = f"{label} ({primary_icd}; ещё {extra})"
        else:
            label = f"{label} ({primary_icd})"
    else:
        raw = str(cond.get("name") or "").strip()
        if _GENERIC_COND_NAME_RE.match(raw):
            raw = _GENERIC_COND_NAME_RE.sub("", raw).strip(" ·-")
        label = raw[:72] if raw else str(cond.get("condition_id") or "Нозология")
    n_sec = len(cond.get("sections") or [])
    if n_sec:
        label += f" · {n_sec} разд."
    return label


def _match_reason_ru(cond: dict[str, Any], *, query: str, icd_codes: list[str] | None) -> str:
    if cond.get("icd_match"):
        icd_s = ", ".join((icd_codes or [])[:2])
        return f"Совпадение с кодом МКБ из запроса ({icd_s})" if icd_s else "Совпадение с МКБ из запроса"
    if cond.get("name_match"):
        return "Название совпадает с текстом запроса"
    preview = ""
    for sec in cond.get("sections") or []:
        if sec.get("preview"):
            preview = str(sec["preview"])[:80]
            break
    if preview:
        return f"Разделы карточки: {preview}…"
    return "Блоки из автоматически извлечённой карточки протокола"


def _condition_relevance_score(cond: dict[str, Any], *, query: str, icd_codes: list[str] | None) -> int:
    score = 42
    if cond.get("icd_match"):
        score += 40
    elif _icd_codes_overlap({c.upper() for c in (icd_codes or []) if c}, list(cond.get("icd10_codes") or [])):
        score += 28
    if cond.get("name_match"):
        score += 12
    q = (query or "").lower()
    disp = (cond.get("display_label") or "").lower()
    if q and any(tok in disp for tok in re.findall(r"[а-яёa-z]{5,}", q)[:4]):
        score += 8
    score += min(8, len(cond.get("sections") or []))
    return min(98, score)


def _merge_condition_into(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    if incoming.get("icd_match"):
        target["icd_match"] = True
    if incoming.get("name_match"):
        target["name_match"] = True
    seen_ids = {s.get("id") for s in target.get("sections") or []}
    for sec in incoming.get("sections") or []:
        sid = sec.get("id")
        if sid in seen_ids:
            continue
        target.setdefault("sections", []).append(sec)
        seen_ids.add(sid)
    aliases = target.setdefault("alias_condition_ids", [])
    cid = incoming.get("condition_id")
    if cid and cid not in aliases and cid != target.get("condition_id"):
        aliases.append(cid)


def dedupe_nav_conditions(
    conditions: list[dict[str, Any]],
    *,
    query: str = "",
    icd_codes: list[str] | None = None,
    protocol_title: str = "",
    max_items: int = 8,
) -> list[dict[str, Any]]:
    """Схлопывает дубли автоизвлечённых нозологий (одинаковые имена / семейство МКБ)."""
    if not conditions:
        return []
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for cond in conditions:
        key = _condition_group_key(cond)
        if key not in groups:
            groups[key] = dict(cond)
            groups[key]["sections"] = list(cond.get("sections") or [])
            groups[key]["alias_condition_ids"] = []
            order.append(key)
            continue
        _merge_condition_into(groups[key], cond)

    icd_q = {c.strip().upper() for c in (icd_codes or []) if c}
    out: list[dict[str, Any]] = []
    for key in order:
        c = groups[key]
        c["icd_match"] = bool(c.get("icd_match")) or _icd_codes_overlap(icd_q, list(c.get("icd10_codes") or []))
        c["display_label"] = _display_label_for_condition(c, protocol_title=protocol_title)
        c["match_reason"] = _match_reason_ru(c, query=query, icd_codes=icd_codes)
        c["relevance_score"] = _condition_relevance_score(c, query=query, icd_codes=icd_codes)
        if not c.get("sections"):
            continue
        out.append(c)

    out.sort(
        key=lambda c: (
            0 if c.get("icd_match") else (1 if c.get("name_match") else 2),
            -(c.get("relevance_score") or 0),
            c.get("display_label") or "",
        )
    )
    return out[:max_items]


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


def _index_keys_for_path(catalog_path: str, local_path: str) -> list[str]:
    keys: list[str] = []
    for raw in (catalog_path, local_path):
        p = (raw or "").strip()
        if not p:
            continue
        keys.append(p)
        keys.append(Path(p).name)
        if not p.startswith("minzdrav"):
            keys.append(f"minzdrav_protocols/{p.lstrip('/')}")
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


@lru_cache(maxsize=1)
def _catalog_path_index() -> dict[str, ProtocolSummary]:
    """Индекс path/basename → Summary (пересборка при clear_protocol_summary_cache)."""
    index: dict[str, ProtocolSummary] = {}
    for summary in load_protocol_summaries(usable_only=False):
        lp = summary.source.local_path or ""
        for key in _index_keys_for_path(lp, lp):
            index.setdefault(key, summary)
    return index


def rebuild_catalog_path_index() -> None:
    clear_protocol_summary_cache()
    _catalog_path_index.cache_clear()
    _icd_family_index.cache_clear()


@lru_cache(maxsize=1)
def _icd_family_index() -> dict[str, list[tuple[str, str]]]:
    """Индекс семейство МКБ -> [(local_path, title)] для «см. также» (P6)."""
    index: dict[str, list[tuple[str, str]]] = {}
    seen_per_family: dict[str, set[str]] = {}
    for summary in load_protocol_summaries(usable_only=False):
        lp = summary.source.local_path or ""
        title = summary.source.title or ""
        if not lp:
            continue
        fams: set[str] = set()
        for c in summary.conditions:
            for code in c.icd10_codes or []:
                fam = _icd_family(str(code))
                if fam:
                    fams.add(fam)
        for fam in fams:
            bucket = index.setdefault(fam, [])
            seen = seen_per_family.setdefault(fam, set())
            if lp not in seen:
                seen.add(lp)
                bucket.append((lp, title))
    return index


def related_protocols_by_icd(
    icd_codes: list[str] | None,
    *,
    exclude_path: str = "",
    limit: int = 6,
) -> list[dict[str, str]]:
    """Другие протоколы с тем же семейством МКБ (для блока «см. также»)."""
    fams = {_icd_family(c) for c in (icd_codes or []) if c}
    fams.discard("")
    if not fams:
        return []
    ex = _norm_path(exclude_path)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    idx = _icd_family_index()
    for fam in fams:
        for lp, title in idx.get(fam, []):
            if _norm_path(lp) == ex or lp in seen:
                continue
            seen.add(lp)
            out.append({"path": lp, "title": title})
            if len(out) >= limit:
                return out
    return out


def find_summary_by_catalog_path(catalog_path: str) -> ProtocolSummary | None:
    """Сопоставление пути PDF в выдаче assist с Protocol Summary."""
    if not (catalog_path or "").strip():
        return None
    idx = _catalog_path_index()
    for key in _index_keys_for_path(catalog_path, catalog_path):
        hit = idx.get(key)
        if hit is not None:
            return hit
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
    icd_match = _icd_codes_overlap(icd_set, list(cond.icd10_codes))
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


def _append_excerpt_item(
    out: list[dict[str, Any]],
    *,
    label: str,
    text: str,
    source_ref: Any,
    limit: int,
) -> None:
    if len(out) >= limit:
        return
    sr = source_ref
    quote = ""
    page_start = None
    section_title = None
    if sr is not None:
        quote = str(getattr(sr, "quote", None) or "").strip()
        page_start = getattr(sr, "page_start", None)
        section_title = getattr(sr, "section_title", None)
    body = (quote or text or "").strip()
    if not body:
        return
    out.append(
        {
            "label": label,
            "text": (text or body)[:400],
            "quote": body[:800],
            "page_start": page_start,
            "section_title": section_title,
        }
    )


def _collect_section_excerpt_items(cond: ConditionSummary, section_id: str, *, limit: int = 12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if section_id == "criteria":
        for block in (cond.clinical_criteria, cond.diagnostic_criteria):
            if block is None:
                continue
            for item in block.required + block.optional + block.exclusion:
                _append_excerpt_item(
                    out,
                    label="критерий",
                    text=item.text,
                    source_ref=item.source_ref,
                    limit=limit,
                )
        if cond.diagnosis_structure:
            for comp in cond.diagnosis_structure.required_components + cond.diagnosis_structure.optional_components:
                _append_excerpt_item(
                    out,
                    label="компонент диагноза",
                    text=comp.name or comp.description or "",
                    source_ref=comp.source_ref,
                    limit=limit,
                )
    elif section_id == "exams":
        for exam in cond.required_exams + cond.conditional_exams:
            _append_excerpt_item(
                out,
                label=exam.name,
                text=exam.name,
                source_ref=exam.source_ref,
                limit=limit,
            )
    elif section_id == "treatment":
        tb = cond.treatment
        if tb:
            for item in tb.drugs + tb.drug_groups + tb.non_drug + tb.procedures + tb.surgery:
                txt = getattr(item, "text", None) or getattr(item, "drug_name", None) or getattr(item, "drug_group", None) or ""
                _append_excerpt_item(out, label="лечение", text=str(txt), source_ref=item.source_ref, limit=limit)
    elif section_id == "red_flags":
        for rf in cond.red_flags:
            _append_excerpt_item(out, label="красный флаг", text=rf.text, source_ref=rf.source_ref, limit=limit)
    elif section_id == "follow_up":
        for fu in cond.follow_up + cond.hospitalization + cond.routing:
            _append_excerpt_item(
                out,
                label="наблюдение",
                text=getattr(fu, "text", None) or getattr(fu, "indication", None) or "",
                source_ref=fu.source_ref,
                limit=limit,
            )
    return out


def build_section_excerpt(
    catalog_path: str,
    *,
    condition_id: str,
    section_id: str,
) -> dict[str, Any]:
    """Цитаты из Protocol Summary по разделу (без LLM)."""
    summary = find_summary_by_catalog_path(catalog_path)
    if summary is None:
        return {"available": False, "path": catalog_path, "llm_used": False}
    cond: ConditionSummary | None = None
    for c in summary.conditions:
        if c.condition_id == condition_id:
            cond = c
            break
    if cond is None:
        return {
            "available": False,
            "path": catalog_path,
            "error": "condition_not_found",
            "llm_used": False,
        }
    items = _collect_section_excerpt_items(cond, section_id)
    section_label = next((lbl for sid, lbl, _f in _SECTION_SPECS if sid == section_id), section_id)
    return {
        "available": bool(items),
        "path": catalog_path,
        "protocol_id": summary.protocol_id,
        "title": summary.source.title or "",
        "condition_id": condition_id,
        "condition_name": cond.name,
        "section_id": section_id,
        "section_label": section_label,
        "items": items,
        "llm_used": False,
    }


_CARE_SETTING_RU: dict[str, str] = {
    "outpatient": "Амбулаторно",
    "inpatient": "Стационар",
    "emergency": "Скорая/неотложная",
    "intensive_care": "Реанимация/ИТ",
    "rehabilitation": "Реабилитация",
    "palliative": "Паллиативная помощь",
}


def _condition_obj_by_id(
    summary: ProtocolSummary,
    condition_id: str | None,
    aliases: list[str] | None = None,
) -> ConditionSummary | None:
    ids = {i for i in ([condition_id] + list(aliases or [])) if i}
    if not ids:
        return None
    for c in summary.conditions:
        if c.condition_id in ids:
            return c
    return None


def build_protocol_card_from_summary(
    catalog_path: str,
    *,
    query: str = "",
    icd_codes: list[str] | None = None,
    max_extracts: int = 4,
    max_text_chars: int = 260,
    min_extracts: int = 2,
    page_lookup: Any = None,
) -> dict[str, Any]:
    """Компактная карточка-выдержка протокола из Summary Card.

    Возвращает точное название, наиболее релевантную нозологию (по МКБ/имени) и
    до `max_extracts` структурных выдержек (критерии/обследование/лечение/красные
    флаги/наблюдение) - целые утверждения с цитатой и страницей, а не обрывки.
    Без LLM.

    `page_lookup(catalog_path, quote) -> int | None` (необязателен) дополняет
    страницу выдержки, когда в карточке `page_start` пуст: цитату сопоставляют
    с чанками протокола.
    """
    summary = find_summary_by_catalog_path(catalog_path)
    if summary is None:
        return {"available": False, "path": catalog_path, "source": "summary"}

    nav_conditions = [_condition_nav(c, query=query, icd_codes=icd_codes) for c in summary.conditions]
    nav_conditions = [c for c in nav_conditions if c.get("sections")]
    nav_conditions = dedupe_nav_conditions(
        nav_conditions,
        query=query,
        icd_codes=icd_codes,
        protocol_title=summary.source.title or "",
    )
    title = summary.source.title or ""
    care_codes = [cs for cs in (summary.applicability.care_setting or []) if cs and cs != "unknown"]
    care_labels = [_CARE_SETTING_RU[cs] for cs in care_codes if cs in _CARE_SETTING_RU]

    if not nav_conditions:
        return {
            "available": False,
            "path": catalog_path,
            "source": "summary",
            "protocol_id": summary.protocol_id,
            "title": title,
        }

    from clinical_knowledge.extract_quality import best_meaningful_excerpt

    top = nav_conditions[0]
    cond = _condition_obj_by_id(summary, top.get("condition_id"), top.get("alias_condition_ids"))
    extracts: list[dict[str, Any]] = []
    if cond is not None:
        for sid, label, _focus in _SECTION_SPECS:
            if len(extracts) >= max_extracts:
                break
            items = _collect_section_excerpt_items(cond, sid, limit=8)
            chosen_it: dict[str, Any] | None = None
            disp = ""
            for it in items:
                disp = best_meaningful_excerpt(
                    [it.get("text"), it.get("quote")], limit=max_text_chars
                )
                if disp:
                    chosen_it = it
                    break
            if chosen_it is None or not disp:
                continue
            quote = str(chosen_it.get("quote") or "")[:600] or None
            page = chosen_it.get("page_start")
            page_source = "summary" if page else None
            if not page and page_lookup is not None:
                probe = quote or disp
                try:
                    found = page_lookup(catalog_path, probe)
                except Exception:
                    found = None
                if found:
                    page = found
                    page_source = "matched"
            extracts.append(
                {
                    "section_id": sid,
                    "label": label,
                    "text": disp,
                    "quote": quote,
                    "page_start": page,
                    "page_source": page_source,
                    "section_title": chosen_it.get("section_title"),
                }
            )

    return {
        "available": len(extracts) >= max(1, min_extracts),
        "path": catalog_path,
        "source": "summary",
        "protocol_id": summary.protocol_id,
        "title": title,
        "review_status": summary.review_status,
        "extraction_status": summary.extraction_status,
        "care_setting": care_codes,
        "care_setting_labels": care_labels,
        "condition": {
            "condition_id": top.get("condition_id"),
            "name": top.get("name"),
            "display_label": top.get("display_label"),
            "icd10_codes": top.get("icd10_codes") or [],
            "icd_match": bool(top.get("icd_match")),
            "name_match": bool(top.get("name_match")),
            "match_reason": top.get("match_reason"),
        },
        "conditions_total": len(nav_conditions),
        "extracts": extracts,
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
    conditions = dedupe_nav_conditions(
        conditions,
        query=query,
        icd_codes=icd_codes,
        protocol_title=summary.source.title or "",
    )
    # приоритет: совпадение по МКБ, затем по имени в запросе
    conditions.sort(
        key=lambda c: (
            0 if c.get("icd_match") else (1 if c.get("name_match") else 2),
            -(c.get("relevance_score") or 0),
            c.get("display_label") or c.get("name") or "",
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
