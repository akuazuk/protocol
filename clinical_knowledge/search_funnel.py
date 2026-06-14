"""POST /api/search/funnel — единый контракт шагов 0–7 (C5)."""
from __future__ import annotations

import re
import uuid
from typing import Any

FUNNEL_POPULATION_CHOICES = [
    {"id": "adult", "label": "Взрослые (≥18 лет)"},
    {"id": "pediatric", "label": "Дети / подростки"},
    {"id": "pregnant", "label": "Беременные"},
    {"id": "emergency", "label": "Неотложная помощь"},
]

_ICD_LETTER_RUBRICS: dict[str, list[str]] = {
    "I": ["bolezni-sistemy-krovoobrashcheniya"],
    "J": ["pulmonologiya-ftiziatriya", "infektsionnye-zabolevaniya"],
    "E": ["endokrinologiya-narusheniya-obmena-veshchestv"],
    "K": ["gastroenterologiya"],
    "G": ["nevrologiya-neyrokhirurgiya"],
    "N": ["nefrologiya", "urologiya"],
    "O": ["akusherstvo-ginekologiya"],
    "C": ["novoobrazovaniya"],
    "F": ["psikhiatriya-narkologiya"],
}


def _population_line(population_id: str) -> str:
    mapping = {
        "adult": "Контекст подбора: взрослое население",
        "pediatric": "Контекст подбора: детское население",
        "pregnant": "Контекст подбора: беременные",
        "emergency": "Контекст подбора: неотложная помощь",
    }
    return mapping.get(population_id, "")


def _query_has_population_hint(q: str) -> bool:
    return bool(
        re.search(
            r"(\bдет|\bдети|\bребен|\bребён|\bноворожд|\bберемен|\bгрудн|\bвзросл|\bпожил|\bподрост|\bинфант|контекст подбора)",
            q or "",
            re.I,
        )
    )


def _query_has_icd(q: str) -> bool:
    return bool(re.search(r"\b[A-TV-ZА-ЯЁ]\s*\d{2}(?:\s*[.,/\-]\s*\d{1,4})?\b", q or "", re.I))


def _extract_icd_codes(q: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b", q or "", re.I):
        c = m.group(1).upper()
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _infer_rubric_choices(q: str, icd_codes: list[str]) -> list[dict[str, str]]:
    slugs: list[str] = []
    seen: set[str] = set()
    for code in icd_codes:
        letter = code[:1].upper()
        for slug in _ICD_LETTER_RUBRICS.get(letter, []):
            if slug not in seen:
                seen.add(slug)
                slugs.append(slug)
    ql = (q or "").lower()
    if re.search(r"кашел|температ|лихорад|одыш", ql):
        for slug in ("pulmonologiya-ftiziatriya", "infektsionnye-zabolevaniya", "pediatriya"):
            if slug not in seen:
                seen.add(slug)
                slugs.append(slug)
    if re.search(r"живот|гастр|изжог|тошн", ql):
        for slug in ("gastroenterologiya",):
            if slug not in seen:
                seen.add(slug)
                slugs.append(slug)
    return [{"id": s, "label": s.replace("-", " ")} for s in slugs[:6]]


def handle_search_funnel(
    *,
    query: str,
    step: int,
    context: dict[str, Any] | None,
    category_slugs: list[str] | None,
    session_id: str | None,
) -> dict[str, Any]:
    """Обработка одного шага воронки; возвращает payload для клиента."""
    ctx = dict(context or {})
    q = (query or "").strip()
    sid = (session_id or "").strip() or str(uuid.uuid4())
    out: dict[str, Any] = {
        "session_id": sid,
        "query": q,
        "context": ctx,
    }

    if step <= 0:
        out["step"] = 0
        out["auto_skip"] = len(q) >= 4
        out["valid"] = len(q) >= 2
        if not out["valid"]:
            out["error"] = "Запрос слишком короткий"
        return out

    if step == 1:
        out["step"] = 1
        if ctx.get("population") or _query_has_population_hint(q):
            out["auto_skip"] = True
            out["next_step"] = 2
        else:
            out["auto_skip"] = False
            out["choices"] = FUNNEL_POPULATION_CHOICES
        return out

    if step == 2:
        out["step"] = 2
        icd_codes = list(ctx.get("icd_codes") or []) or _extract_icd_codes(q)
        ctx["icd_codes"] = icd_codes
        out["context"] = ctx
        if icd_codes or _query_has_icd(q):
            out["auto_skip"] = True
            out["next_step"] = 3
            return out
        import rag_server as rs

        rs._require_rag_loaded()
        model = rs.get_gemini()
        icd_analysis, _, _, _, icd_err = rs._infer_icd_pipeline_from_full_query(q, model)
        if icd_err:
            out["error"] = icd_err
            out["choices"] = []
            return out
        choices: list[dict[str, Any]] = []
        seen: set[str] = set()
        icd_payload = icd_analysis or {}
        for bucket in ("detected", "suggested"):
            for row in icd_payload.get(bucket) or []:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("code") or "").strip()
                if not code or code in seen:
                    continue
                seen.add(code)
                title = str(row.get("title_ru") or "").strip()
                choices.append(
                    {
                        "id": code,
                        "label": f"{code} · {title}" if title else code,
                        "confidence": row.get("confidence"),
                        "score": row.get("score"),
                    }
                )
        for code in icd_payload.get("codes_for_retrieval") or []:
            c = str(code).strip()
            if c and c not in seen:
                seen.add(c)
                choices.append({"id": c, "label": c})
        out["auto_skip"] = len(choices) == 0
        out["choices"] = choices[:12]
        out["icd"] = icd_payload
        return out

    if step == 3:
        out["step"] = 3
        slugs = list(category_slugs or ctx.get("rubric_slugs") or [])
        if slugs:
            ctx["rubric_slugs"] = slugs
            out["context"] = ctx
            out["auto_skip"] = True
            out["next_step"] = 4
            return out
        icd_codes = list(ctx.get("icd_codes") or []) or _extract_icd_codes(q)
        choices = _infer_rubric_choices(q, icd_codes)
        out["auto_skip"] = len(choices) <= 1
        out["choices"] = choices + [{"id": "", "label": "Все рубрики"}]
        return out

    if step == 4:
        import rag_server as rs
        from rag_server import AssistIn

        rs._require_rag_loaded()
        work_q = q
        pop = str(ctx.get("population") or "").strip()
        if pop and pop != "skipped":
            line = _population_line(pop)
            if line and line.lower() not in work_q.lower():
                work_q = f"{work_q}\n{line}" if work_q else line
        icd_codes = list(ctx.get("icd_codes") or [])
        if icd_codes:
            icd_line = "МКБ-10: " + ", ".join(icd_codes)
            if icd_line.lower() not in work_q.lower():
                work_q = f"{work_q}\n{icd_line}" if work_q else icd_line
        slugs = list(category_slugs or ctx.get("rubric_slugs") or [])
        payload = rs.api_assist(
            AssistIn(
                query=work_q,
                category_slugs=slugs,
                retrieve_only=True,
            )
        )
        out["step"] = 4
        out["auto_skip"] = False
        out["assist"] = payload
        protos = ((payload.get("llm_json") or {}).get("protocols") or [])[:6]
        out["choices"] = [
            {
                "id": p.get("path"),
                "label": p.get("title") or p.get("path"),
                "confidence": p.get("confidence_score"),
            }
            for p in protos
            if isinstance(p, dict) and p.get("path")
        ]
        return out

    if step == 5:
        from clinical_knowledge.protocol_summary.nav import build_protocol_summary_nav

        path = str(ctx.get("protocol_path") or "").strip()
        if not path:
            out["step"] = 5
            out["error"] = "protocol_path обязателен"
            return out
        icd_codes = list(ctx.get("icd_codes") or [])
        nav = build_protocol_summary_nav(path, query=q, icd_codes=icd_codes or None)
        out["step"] = 5
        if not nav.get("available"):
            out["auto_skip"] = True
            out["next_step"] = 7
            out["nav"] = nav
            return out
        conds = nav.get("conditions") or []
        if len(conds) == 1:
            ctx["condition_id"] = conds[0].get("condition_id")
            out["context"] = ctx
            out["auto_skip"] = True
            out["next_step"] = 6
        else:
            out["auto_skip"] = False
            out["choices"] = [
                {
                    "id": c.get("condition_id"),
                    "label": c.get("name") or c.get("condition_id"),
                }
                for c in conds
            ]
        out["nav"] = nav
        return out

    if step == 6:
        from clinical_knowledge.protocol_summary.nav import build_protocol_summary_nav

        path = str(ctx.get("protocol_path") or "").strip()
        cid = str(ctx.get("condition_id") or "").strip()
        if not path or not cid:
            out["step"] = 6
            out["error"] = "protocol_path и condition_id обязательны"
            return out
        icd_codes = list(ctx.get("icd_codes") or [])
        nav = build_protocol_summary_nav(path, query=q, icd_codes=icd_codes or None)
        cond = next((c for c in nav.get("conditions") or [] if c.get("condition_id") == cid), None)
        sections = (cond or {}).get("sections") or []
        out["step"] = 6
        out["auto_skip"] = len(sections) <= 1
        if len(sections) == 1:
            ctx["section_id"] = sections[0].get("id")
            out["context"] = ctx
            out["next_step"] = 7
        out["choices"] = [
            {"id": s.get("id"), "label": s.get("label"), "count": s.get("count")}
            for s in sections
        ]
        return out

    if step >= 7:
        from clinical_knowledge.protocol_summary.nav import build_section_excerpt

        path = str(ctx.get("protocol_path") or "").strip()
        cid = str(ctx.get("condition_id") or "").strip()
        sid = str(ctx.get("section_id") or "criteria").strip()
        out["step"] = 7
        if path and cid and sid:
            excerpt = build_section_excerpt(path, condition_id=cid, section_id=sid)
            out["excerpt"] = excerpt
            out["llm_used"] = False
            if excerpt.get("items"):
                out["source_ref"] = excerpt["items"][0]
        out["pdf_href"] = f"/api/protocol-pdf?path={path}" if path else None
        return out

    out["step"] = step
    out["error"] = f"Неизвестный шаг: {step}"
    return out
