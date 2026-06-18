"""Клинический контекст амбулаторного КЗ: жалобы, анамнез, запрос к КП."""
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
    """Чётко: анамнез заболевания vs анамнез жизни."""
    disease = (doc.sections.anamnesis or "").strip()
    life = (doc.sections.life_history or "").strip()
    if disease and not life:
        # Эвристика: «анамнез жизни» внутри одного блока
        m = re.search(
            r"(?i)(анамнез\s+жизни|перен[её]с|хроническ\w*\s+заболеван|аллерг)",
            disease,
        )
        if m and m.start() > 20:
            life = disease[m.start() :].strip()
            disease = disease[: m.start()].strip(" \n:;-")
    return {"disease": disease, "life": life}


def build_clinical_context(
    doc: ConsultationDocument,
    icd_codes: list[str],
) -> dict[str, Any]:
    """Контекст для подбора и оценки обследования/лечения по КП."""
    anam = split_anamnesis_parts(doc)
    complaints = (doc.sections.complaints or "").strip()
    icd = [str(c).upper() for c in (icd_codes or []) if c]
    query_parts = [complaints, anam["disease"], " ".join(icd[:4]), "амбулатор"]
    if anam["life"]:
        query_parts.append(anam["life"][:200])
    return {
        "setting": "ambulatory",
        "setting_label": "Амбулаторное консультативное заключение",
        "complaints": complaints,
        "anamnesis_disease": anam["disease"],
        "anamnesis_life": anam["life"],
        "icd_codes": icd,
        "clinical_query": " ".join(p for p in query_parts if p).strip()[:1200],
        "symptom_tokens": sorted(_norm_tokens(" ".join([complaints, anam["disease"]]))),
    }


def format_anamnesis_excerpt(ctx: dict[str, Any], *, limit: int = 320) -> str:
    parts: list[str] = []
    if ctx.get("anamnesis_disease"):
        parts.append(f"Заболевание: {ctx['anamnesis_disease'][:limit]}")
    if ctx.get("anamnesis_life"):
        parts.append(f"Жизни: {ctx['anamnesis_life'][:limit]}")
    return "\n".join(parts)[: limit * 2]


def filter_ambulatory_kp_items(items: list[str]) -> list[str]:
    """Убрать явно стационарные рекомендации для амбулаторного КЗ."""
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
    """Ранжировать пункты КП по релевантности жалобам, анамнезу и МКБ."""
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
