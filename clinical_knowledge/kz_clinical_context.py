"""Клинический контекст амбулаторного КЗ для подбора КП и оценки (внутренний слой)."""
from __future__ import annotations

import re
from typing import Any

from clinical_knowledge.consult_schema import ConsultationDocument

_INPATIENT_ONLY = re.compile(
    r"стационарн|госпитализац|круглосуточн|реанимац|интенсивн\s+терап|"
    r"оперативн\w*\s+(?:лечен|вмешательств)(?!.*амбулатор)",
    re.I,
)
_AMBULATORY_OK = re.compile(r"амбулатор|поликлиник|дневн\w*\s+стационар|на\s+дому", re.I)
_SYMPTOM_NOISE = frozenset({
    "жалоб", "болит", "боли", "болью", "давно", "давненько", "период",
    "время", "месяц", "недел", "день", "года", "лет", "давность",
})


def _norm_tokens(s: str) -> set[str]:
    return {
        t for t in re.findall(r"[а-яёa-z]{4,}", (s or "").lower())
        if t not in _SYMPTOM_NOISE
    }


def split_anamnesis_parts(doc: ConsultationDocument) -> dict[str, str]:
    """Анамнез заболевания vs анамнез жизни."""
    disease = (doc.sections.anamnesis or "").strip()
    life = (doc.sections.life_history or "").strip()
    if disease and not life:
        m = re.search(
            r"(?i)(анамнез\s+жизни|перен[её]с|хроническ\w*\s+заболеван|аллерг)",
            disease,
        )
        if m and m.start() > 20:
            life = disease[m.start() :].strip()
            disease = disease[: m.start()].strip(" \n:;-")
    return {"disease": disease, "life": life}


def _missing_for_protocol_pick(
    doc: ConsultationDocument,
    icd_codes: list[str],
    *,
    specialty_label: str | None,
) -> list[str]:
    missing: list[str] = []
    if not (doc.sections.complaints or "").strip():
        missing.append("жалобы")
    anam = split_anamnesis_parts(doc)
    if not anam["disease"]:
        missing.append("анамнез заболевания")
    if not icd_codes:
        missing.append("код МКБ-10")
    if doc.patient.age_years is None and doc.patient.adult_or_child == "unknown":
        missing.append("возраст")
    if not (specialty_label or doc.doctor_specialty or "").strip():
        missing.append("специальность врача")
    return missing


def build_clinical_context(
    doc: ConsultationDocument,
    icd_codes: list[str],
    *,
    specialty_slug: str | None = None,
    specialty_label: str | None = None,
) -> dict[str, Any]:
    """Контекст для подбора КП и оценки обследования/лечения."""
    anam = split_anamnesis_parts(doc)
    complaints = (doc.sections.complaints or "").strip()
    icd = [str(c).upper() for c in (icd_codes or []) if c]
    spec_label = (specialty_label or doc.doctor_specialty or "").strip()
    query_parts = [
        complaints,
        anam["disease"],
        anam["life"][:200] if anam["life"] else "",
        " ".join(icd[:4]),
        spec_label,
    ]
    if doc.patient.age_years is not None:
        query_parts.append(f"{doc.patient.age_years} лет")
    elif doc.patient.adult_or_child in ("adult", "child"):
        query_parts.append("взрослый" if doc.patient.adult_or_child == "adult" else "ребёнок")

    missing = _missing_for_protocol_pick(doc, icd, specialty_label=spec_label)

    return {
        "setting": "ambulatory",
        "complaints": complaints,
        "anamnesis_disease": anam["disease"],
        "anamnesis_life": anam["life"],
        "icd_codes": icd,
        "specialty_slug": specialty_slug,
        "specialty_label": spec_label,
        "age_years": doc.patient.age_years,
        "adult_or_child": doc.patient.adult_or_child,
        "clinical_query": " ".join(p for p in query_parts if p).strip()[:1200],
        "symptom_tokens": sorted(_norm_tokens(" ".join([complaints, anam["disease"]]))),
        "missing_for_protocol_pick": missing,
    }


def format_anamnesis_excerpt(ctx: dict[str, Any], *, limit: int = 320) -> str:
    parts: list[str] = []
    if ctx.get("anamnesis_disease"):
        parts.append(f"Заболевание: {ctx['anamnesis_disease'][:limit]}")
    if ctx.get("anamnesis_life"):
        parts.append(f"Жизни: {ctx['anamnesis_life'][:limit]}")
    return "\n".join(parts)[: limit * 2]


def format_evaluation_basis(
    ctx: dict[str, Any],
    protocol_matches: list[dict[str, Any]] | None = None,
) -> str:
    """Краткая строка: на чём основан подбор КП (для UI, без методологии)."""
    parts: list[str] = []
    if ctx.get("icd_codes"):
        parts.append("МКБ " + ", ".join(ctx["icd_codes"][:3]))
    if ctx.get("specialty_label"):
        parts.append(ctx["specialty_label"])
    if ctx.get("age_years") is not None:
        parts.append(f"{ctx['age_years']} лет")
    if ctx.get("complaints"):
        parts.append("жалобы: " + ctx["complaints"][:60])
    for m in protocol_matches or []:
        title = (m.get("title") or "").strip()
        if title:
            parts.append(f"КП «{title[:70]}»")
            break
    return " · ".join(parts)


def protocol_pick_comment(
    ctx: dict[str, Any],
    protocol_matches: list[dict[str, Any]] | None,
) -> str:
    """Сообщение о качестве/достаточности подбора КП."""
    missing = list(ctx.get("missing_for_protocol_pick") or [])
    matches = protocol_matches or []
    if not ctx.get("icd_codes"):
        return "КП не подобран: в КЗ не указан код МКБ-10."
    if not matches:
        icd_s = ", ".join(ctx["icd_codes"][:2])
        if missing:
            return (
                f"КП по МКБ {icd_s} не найден; для подбора не хватает: "
                + ", ".join(missing) + "."
            )
        return f"КП по МКБ {icd_s} не найден в каталоге."
    top = matches[0]
    score = float(top.get("match_score") or 0)
    title = (top.get("title") or "").strip()
    if score < 22:
        base = f"Соответствие КП сомнительно (балл {score:.0f})"
        if title:
            base += f" для «{title[:60]}»"
        if missing:
            base += f"; не указаны: {', '.join(missing)}."
        else:
            base += "."
        return base
    if missing:
        return f"КП подобран с ограничениями: не указаны {', '.join(missing)}."
    return ""


def filter_ambulatory_kp_items(items: list[str]) -> list[str]:
    """Убрать явно стационарные рекомендации."""
    out: list[str] = []
    for it in items:
        t = (it or "").strip()
        if not t:
            continue
        if _INPATIENT_ONLY.search(t) and not _AMBULATORY_OK.search(t):
            continue
        out.append(t)
    return out


def rank_kp_items_by_context(
    items: list[str],
    context: dict[str, Any],
    *,
    meta: list[dict[str, Any]] | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Ранжировать пункты КП по жалобам, анамнезу и МКБ."""
    blob = (context.get("clinical_query") or "").lower()
    sym = set(context.get("symptom_tokens") or [])
    icd_set = set(context.get("icd_codes") or [])
    meta_by: dict[str, dict[str, Any]] = {}
    for m in meta or []:
        if isinstance(m, dict):
            meta_by[str(m.get("text") or "")[:80].lower()] = m

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for raw in filter_ambulatory_kp_items(items):
        low = raw.lower()
        score = 0.0
        overlap = sym & _norm_tokens(low)
        score += min(3.0, len(overlap) * 0.9)
        for code in icd_set:
            if code.lower() in low or code[:3].lower() in low:
                score += 1.2
        for tok in re.findall(r"[а-яёa-z]{5,}", blob)[:16]:
            if tok in low:
                score += 0.25
        m = meta_by.get(raw[:80].lower(), {})
        obligation = str(m.get("obligation") or "recommended")
        if obligation == "required":
            score += 0.8
        scored.append((score, raw, m))

    scored.sort(key=lambda x: (-x[0], x[1]))
    out: list[dict[str, Any]] = []
    for sc, text, m in scored[:limit]:
        out.append({
            "text": text,
            "score": round(sc, 2),
            "obligation": m.get("obligation") or "recommended",
        })
    return out
