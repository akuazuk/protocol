"""Сверка плана МО с полями Protocol Summary (не новый балл зон).

Вход: clinical hit suggest + слоты рекомендаций МО + опциональный night
plan_concordance. Выход: строки требование / статус / цитата для UI
«План ↔ КП». Не invent порогов: matched только через resolve_plan_route
(A/B), missing/off_protocol night переиспользуются как есть.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from clinical_knowledge.mo_plan_protocol_score import resolve_plan_route, validate_plan_concordance_result

ENGINE = "mo_case_kp_concordance_v1"
SCHEMA_VERSION = 1
MAX_EXAMS = 8
MAX_TREATMENT = 6
MAX_FOLLOW_UP = 4
MAX_NIGHT_EXTRA = 8
QUOTE_WIDTH = 180

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[«»\"'()\[\].,;:]+")


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _blob(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


def _mentioned(name: str, *texts: str) -> bool:
    n = _blob(name)
    if len(n) < 3:
        return False
    n_soft = _PUNCT_RE.sub(" ", n)
    for raw in texts:
        t = _blob(raw)
        if not t:
            continue
        if n in t or (n_soft and n_soft in t):
            return True
        if 3 <= len(n) <= 5 and re.search(rf"\b{re.escape(n)}\b", t, re.I):
            return True
    return False


def _quote_around(text: str, needle: str, *, width: int = QUOTE_WIDTH) -> str:
    blob = str(text or "").strip()
    n = str(needle or "").strip()
    if not blob or not n:
        return ""
    low = blob.lower()
    i = low.find(n.lower())
    if i < 0:
        return ""
    start = max(0, i - 40)
    end = min(len(blob), i + len(n) + 80)
    frag = blob[start:end].strip()
    if start > 0:
        frag = "…" + frag
    if end < len(blob):
        frag = frag + "…"
    return _clip(frag, width)


def _plan_slots(clinical: Mapping[str, Any] | None) -> dict[str, str]:
    src = clinical if isinstance(clinical, Mapping) else {}
    exam = str(src.get("exam_recommendations") or "").strip()
    treatment = str(src.get("treatment_recommendations") or "").strip()
    exam_data = str(src.get("exam_data") or "").strip()
    return {
        "exam_recommendations": exam,
        "treatment_recommendations": treatment,
        "exam_data": exam_data,
        "exam_blob": " ".join(part for part in (exam, exam_data) if part),
        "treatment_blob": treatment,
        "all_blob": " ".join(part for part in (exam, treatment, exam_data) if part),
    }


def _icd_codes_from_hit(hit: Mapping[str, Any] | None) -> list[str]:
    codes: list[str] = []
    if not isinstance(hit, Mapping):
        return codes
    for item in hit.get("icd_fit") or []:
        if isinstance(item, Mapping) and item.get("code"):
            codes.append(str(item.get("code")).strip().upper())
        elif isinstance(item, str) and item.strip():
            codes.append(item.strip().upper())
    return [c for c in codes if c]


def _icd_overlap(query: list[str], cond_codes: list[str]) -> bool:
    if not query or not cond_codes:
        return False
    cond_u = {str(c).strip().upper() for c in cond_codes if c}
    for q in query:
        qu = str(q).strip().upper()
        if not qu:
            continue
        qf = qu.split(".", 1)[0]
        for cu in cond_u:
            if cu == qu or cu.startswith(qf) or qu.startswith(cu.split(".", 1)[0]):
                return True
    return False


def _requirement_count(cond: Any) -> int:
    n = len(getattr(cond, "required_exams", None) or [])
    n += len(getattr(cond, "conditional_exams", None) or [])
    n += len(getattr(cond, "follow_up", None) or [])
    tb = getattr(cond, "treatment", None)
    if tb is not None:
        n += len(getattr(tb, "drugs", None) or [])
        n += len(getattr(tb, "drug_groups", None) or [])
        n += len(getattr(tb, "procedures", None) or [])
        n += len(getattr(tb, "non_drug", None) or [])
    return n


def _pick_condition(summary: Any, hit: Mapping[str, Any] | None) -> Any:
    conditions = list(getattr(summary, "conditions", None) or [])
    if not conditions:
        return None
    codes = _icd_codes_from_hit(hit)
    ranked: list[tuple[int, int, Any]] = []
    for cond in conditions:
        icd_hit = 1 if _icd_overlap(codes, list(getattr(cond, "icd10_codes", None) or [])) else 0
        ranked.append((icd_hit, _requirement_count(cond), cond))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return ranked[0][2]


def _source_quote(item: Any) -> str:
    sr = getattr(item, "source_ref", None)
    if sr is None:
        return ""
    return _clip(getattr(sr, "quote", None) or "", 240)


def _row(
    *,
    kind: str,
    requirement: str,
    status: str,
    slot: str,
    mo_quote: str = "",
    kp_quote: str = "",
    aliases: list[str] | None = None,
    source: str = "summary",
) -> dict[str, Any]:
    return {
        "kind": kind,
        "requirement": _clip(requirement, 160),
        "status": status,
        "slot": slot,
        "mo_quote": _clip(mo_quote, QUOTE_WIDTH),
        "kp_quote": _clip(kp_quote, 240),
        "aliases": [_clip(a, 80) for a in (aliases or [])[:6] if _clip(a, 80)],
        "source": source,
    }


def _match_needles(name: str, aliases: list[str], *blobs: str) -> tuple[bool, str, str]:
    needles = [name] + [a for a in aliases if a and a != name]
    for needle in needles:
        if _mentioned(needle, *blobs):
            for blob in blobs:
                quote = _quote_around(blob, needle)
                if quote:
                    return True, needle, quote
            return True, needle, ""
    return False, "", ""


def _empty_payload(
    *,
    available: bool,
    reason: str,
    route: dict[str, Any] | None = None,
    protocol_id: str = "",
    protocol_title: str = "",
) -> dict[str, Any]:
    route = route or {}
    return {
        "ok": True,
        "available": available,
        "engine": ENGINE,
        "schema_version": SCHEMA_VERSION,
        "reason": reason,
        "kp_status": str(route.get("kp_status") or "unmatched"),
        "route": str(route.get("route") or "llm_no_kp"),
        "protocol_id": protocol_id,
        "protocol_title": _clip(protocol_title, 220),
        "condition_id": "",
        "condition_name": "",
        "rows": [],
        "counts": {"present": 0, "missing": 0, "off_protocol": 0, "n_a": 0},
    }


def _night_plan(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping) or not raw:
        return None
    try:
        return validate_plan_concordance_result(raw)
    except (TypeError, ValueError):
        missing = raw.get("missing_required") if isinstance(raw.get("missing_required"), list) else []
        off = raw.get("off_protocol") if isinstance(raw.get("off_protocol"), list) else []
        if not missing and not off:
            return None
        return {
            "missing_required": [_clip(x, 240) for x in missing[:MAX_NIGHT_EXTRA] if _clip(x, 240)],
            "off_protocol": [_clip(x, 240) for x in off[:MAX_NIGHT_EXTRA] if _clip(x, 240)],
        }


def _rows_from_condition(cond: Any, slots: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exam_blob = slots["exam_blob"]
    treat_blob = slots["treatment_blob"]
    all_blob = slots["all_blob"]

    for exam in list(getattr(cond, "required_exams", None) or [])[:MAX_EXAMS]:
        name = str(getattr(exam, "name", "") or "").strip()
        if not name:
            continue
        aliases = [str(a).strip() for a in (getattr(exam, "aliases", None) or []) if str(a).strip()]
        found, _needle, quote = _match_needles(name, aliases, exam_blob, all_blob)
        rows.append(
            _row(
                kind="exam",
                requirement=name,
                status="present" if found else "missing",
                slot="exam_recommendations",
                mo_quote=quote,
                kp_quote=_source_quote(exam),
                aliases=aliases,
            )
        )

    for exam in list(getattr(cond, "conditional_exams", None) or [])[:4]:
        name = str(getattr(exam, "name", "") or "").strip()
        if not name:
            continue
        aliases = [str(a).strip() for a in (getattr(exam, "aliases", None) or []) if str(a).strip()]
        found, _needle, quote = _match_needles(name, aliases, exam_blob, all_blob)
        rows.append(
            _row(
                kind="exam",
                requirement=name,
                status="present" if found else "n_a",
                slot="exam_recommendations",
                mo_quote=quote,
                kp_quote=_source_quote(exam),
                aliases=aliases,
            )
        )

    tb = getattr(cond, "treatment", None)
    treatment_items: list[tuple[str, list[str], str]] = []
    if tb is not None:
        for drug in list(getattr(tb, "drugs", None) or []):
            name = str(
                getattr(drug, "drug_name", None)
                or getattr(drug, "active_substance", None)
                or getattr(drug, "drug_group", None)
                or ""
            ).strip()
            if not name:
                continue
            aliases = [
                str(x).strip()
                for x in (
                    getattr(drug, "active_substance", None),
                    getattr(drug, "drug_group", None),
                )
                if str(x or "").strip() and str(x).strip() != name
            ]
            treatment_items.append((name, aliases, _source_quote(drug)))
        for group in list(getattr(tb, "drug_groups", None) or []):
            name = str(getattr(group, "drug_group", "") or "").strip()
            if name:
                treatment_items.append((name, [], _source_quote(group)))
        for proc in list(getattr(tb, "procedures", None) or []):
            name = str(getattr(proc, "name", "") or "").strip()
            if name:
                treatment_items.append((name, [], _source_quote(proc)))
        for item in list(getattr(tb, "non_drug", None) or []):
            name = str(getattr(item, "text", "") or "").strip()
            if name:
                treatment_items.append((name, [], _source_quote(item)))

    seen_t: set[str] = set()
    for name, aliases, kp_quote in treatment_items:
        key = _blob(name)
        if not key or key in seen_t:
            continue
        seen_t.add(key)
        if len([r for r in rows if r["kind"] == "treatment"]) >= MAX_TREATMENT:
            break
        found, _needle, quote = _match_needles(name, aliases, treat_blob, all_blob)
        rows.append(
            _row(
                kind="treatment",
                requirement=name,
                status="present" if found else "missing",
                slot="treatment_recommendations",
                mo_quote=quote,
                kp_quote=kp_quote,
                aliases=aliases,
            )
        )

    for follow in list(getattr(cond, "follow_up", None) or [])[:MAX_FOLLOW_UP]:
        name = str(getattr(follow, "text", "") or "").strip()
        if not name:
            continue
        actions = [str(a).strip() for a in (getattr(follow, "expected_actions", None) or []) if str(a).strip()]
        found, _needle, quote = _match_needles(name, actions, treat_blob, exam_blob, all_blob)
        rows.append(
            _row(
                kind="follow_up",
                requirement=name,
                status="present" if found else "missing",
                slot="treatment_recommendations",
                mo_quote=quote,
                kp_quote=_source_quote(follow),
                aliases=actions,
            )
        )
    return rows


def _already_covered(rows: list[dict[str, Any]], text: str) -> bool:
    blob = _blob(text)
    if not blob:
        return False
    for row in rows:
        req = _blob(row.get("requirement") or "")
        if req and (req in blob or blob in req):
            return True
        for alias in row.get("aliases") or []:
            a = _blob(alias)
            if a and (a in blob or blob in a):
                return True
    return False


def _merge_night_rows(rows: list[dict[str, Any]], night: Mapping[str, Any], slots: dict[str, str]) -> None:
    for item in list(night.get("missing_required") or [])[:MAX_NIGHT_EXTRA]:
        text = _clip(item, 160)
        if not text or _already_covered(rows, text):
            continue
        found, _needle, quote = _match_needles(text, [], slots["all_blob"])
        rows.append(
            _row(
                kind="exam",
                requirement=text,
                status="present" if found else "missing",
                slot="exam_recommendations",
                mo_quote=quote,
                source="night",
            )
        )
    for item in list(night.get("off_protocol") or [])[:MAX_NIGHT_EXTRA]:
        text = _clip(item, 160)
        if not text or _already_covered(rows, text):
            continue
        found, _needle, quote = _match_needles(text, [], slots["all_blob"])
        rows.append(
            _row(
                kind="off_protocol",
                requirement=text,
                status="off_protocol",
                slot="treatment_recommendations",
                mo_quote=quote,
                source="night",
            )
        )


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out = {"present": 0, "missing": 0, "off_protocol": 0, "n_a": 0}
    for row in rows:
        key = str(row.get("status") or "")
        if key in out:
            out[key] += 1
    return out


def build_kp_concordance(
    *,
    protocol_suggest: Mapping[str, Any] | None,
    clinical: Mapping[str, Any] | None = None,
    night_plan: Mapping[str, Any] | None = None,
    summary: Any = None,
    summary_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Строки сверки плана с КП. Пусто при unmatched - это не fail плана."""
    route = resolve_plan_route(protocol_suggest)
    hit = route.get("hit") if isinstance(route.get("hit"), Mapping) else None
    if str(route.get("kp_status") or "") != "matched" or not hit:
        reason = "протокол не подобран - сверка плана с КП недоступна"
        if str(route.get("fallback_reason") or "") == "kp_trust_below_threshold":
            reason = "доверие к протоколу ниже A/B - сверка плана с КП недоступна"
        return _empty_payload(available=False, reason=reason, route=route)

    protocol_id = str(hit.get("protocol_id") or "")
    protocol_title = str(hit.get("title") or "")
    source_path = str(hit.get("source_path") or hit.get("local_path") or "")
    loaded = summary
    if loaded is None and source_path:
        loader = summary_loader
        if loader is None:
            from clinical_knowledge.protocol_summary.nav import find_summary_by_catalog_path

            loader = find_summary_by_catalog_path
        try:
            loaded = loader(source_path)
        except Exception:
            loaded = None

    slots = _plan_slots(clinical)
    night = _night_plan(night_plan)
    rows: list[dict[str, Any]] = []
    condition_id = ""
    condition_name = ""
    if loaded is not None:
        cond = _pick_condition(loaded, hit)
        if cond is not None:
            condition_id = str(getattr(cond, "condition_id", "") or "")
            condition_name = str(getattr(cond, "name", "") or "")
            rows.extend(_rows_from_condition(cond, slots))
    if night:
        _merge_night_rows(rows, night, slots)

    if not rows:
        if loaded is None:
            reason = "карточка КП не найдена - сверка полей недоступна"
        else:
            reason = "в карточке КП нет полей обследования / лечения / наблюдения"
        payload = _empty_payload(
            available=True,
            reason=reason,
            route=route,
            protocol_id=protocol_id,
            protocol_title=protocol_title,
        )
        payload["condition_id"] = condition_id
        payload["condition_name"] = _clip(condition_name, 160)
        return payload

    return {
        "ok": True,
        "available": True,
        "engine": ENGINE,
        "schema_version": SCHEMA_VERSION,
        "reason": None,
        "kp_status": "matched",
        "route": str(route.get("route") or "kp_grounded"),
        "protocol_id": protocol_id,
        "protocol_title": _clip(protocol_title, 220),
        "condition_id": condition_id,
        "condition_name": _clip(condition_name, 160),
        "rows": rows,
        "counts": _counts(rows),
    }


def attach_kp_concordance(
    suggest: dict[str, Any] | None,
    *,
    clinical: Mapping[str, Any] | None = None,
    night_plan: Mapping[str, Any] | None = None,
    summary: Any = None,
    summary_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Дописывает kp_concordance в ответ protocol-suggest. Не ломает suggest."""
    payload = suggest if isinstance(suggest, dict) else {}
    try:
        payload["kp_concordance"] = build_kp_concordance(
            protocol_suggest=payload,
            clinical=clinical,
            night_plan=night_plan,
            summary=summary,
            summary_loader=summary_loader,
        )
    except Exception:
        payload["kp_concordance"] = _empty_payload(
            available=False,
            reason="сверка плана с КП недоступна",
        )
    return payload
