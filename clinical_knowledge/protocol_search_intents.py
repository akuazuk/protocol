"""Единый источник intent-спеков навигации по протоколу (backend + sync-тест с UI)."""
from __future__ import annotations

import re
from typing import Any

# Синхронизировать с PROTO_SEARCH_INTENTS в index.html / INTENTS в proto-viewer.html.
INTENT_SPECS: dict[str, dict[str, Any]] = {
    "treatment": {
        "sections": ("treatment",),
        "tags": ("treatment", "drugs"),
        "label": "Лечение",
        "terms": (
            "лекарств",
            "препарат",
            "назнач",
            "терапи",
            "дозиров",
            "флп",
            "таблет",
            "капсул",
            "сироп",
            "суспенз",
            "инъекц",
            "укол",
            "мг",
            "внутрь",
            "пероральн",
            "муколитик",
            "антибиотик",
            "ингаляц",
            "небулайз",
            "компрессион",
            "склеротерап",
            "хирург",
            "операц",
            "лечени",
            "фармакотерап",
            "монотерап",
            "схем",
        ),
        "phrases": (
            "какие лекарства",
            "что назначить",
            "чем лечить",
            "схема лечения",
            "какие препараты",
            "назначение препаратов",
        ),
    },
    "diagnostics": {
        "sections": ("diagnostics",),
        "tags": ("diagnostics", "exams"),
        "label": "Обследования",
        "terms": (
            "обследован",
            "анализ",
            "уздс",
            "узи",
            "диагност",
            "лаборатор",
            "инструментал",
            "мрт",
            "кт",
            "экг",
            "оак",
            "бак",
            "исследован",
        ),
        "phrases": (
            "какие обследования",
            "что сдать",
            "какие анализы",
            "как диагностировать",
            "какие исследования",
        ),
    },
    "diagnosis": {
        "sections": ("diagnosis",),
        "tags": ("diagnosis", "criteria"),
        "label": "Диагноз",
        "terms": ("диагноз", "классификац", "критери", "мкб", "нозолог", "стади", "степень"),
        "phrases": ("как ставить диагноз", "критерии диагноза", "классификация"),
    },
    "followup": {
        "sections": ("followup",),
        "tags": ("followup", "prevention", "routing"),
        "label": "Наблюдение",
        "terms": (
            "наблюден",
            "диспансер",
            "контроль",
            "профилактик",
            "маршрут",
            "направлен",
            "госпитализац",
            "консультац",
        ),
        "phrases": ("как наблюдать", "куда направлять", "когда госпитализировать"),
    },
    "contraindications": {
        "sections": ("treatment", "diagnosis", "diagnostics"),
        "tags": ("treatment", "criteria", "diagnosis"),
        "label": "Противопоказания",
        "terms": ("противопоказан", "нельзя", "не применя", "абсолютн", "относительн"),
        "phrases": ("противопоказания", "когда нельзя"),
    },
}

DRUG_FOCUS_TERMS: tuple[str, ...] = (
    "таблет",
    "капсул",
    "препарат",
    "лекарств",
    "мг",
    "внутрь",
    "сироп",
    "суспенз",
    "ингаляц",
    "небулайз",
    "антибиотик",
    "муколитик",
    "фармакотерап",
    "дозиров",
)

DRUG_CHUNK_TYPES = frozenset({"treatment", "drug_list", "pharmacotherapy"})

TABLE_NOISE_RE = re.compile(
    r"мкб[\-\s]?10|объемы\s+оказания|наименование\s+нозологическ|"
    r"исход\s+заболевания\s+\d|(?:\b\d\s+){5,}\d|"
    r"диагностика\s+и\s+лечение\s+острого\s+бронхита\s+наименование",
    re.I,
)

_DRUG_SENTENCE_RE = re.compile(
    r"(?:\d+[\.,]?\d*\s*(?:мг|г|мл|мкг)|внутрь|перорально|"
    r"муколитическ|антибактериальн|ингаляц|небулайз|даи\b|таблет)",
    re.I,
)


def _norm_q(query: str) -> str:
    return (query or "").lower().replace("ё", "е").strip()


def detect_query_intents(query: str) -> list[str]:
    q = _norm_q(query)
    if len(q) < 2:
        return []
    found: list[str] = []
    for key, spec in INTENT_SPECS.items():
        hit = any(p in q for p in spec.get("phrases") or ())
        if not hit:
            hit = any(t in q for t in spec.get("terms") or ())
        if hit:
            found.append(key)
    return found


def is_drug_focus_query(query: str, intents: list[str] | None = None) -> bool:
    intents = intents if intents is not None else detect_query_intents(query)
    if "treatment" not in intents:
        return False
    q = _norm_q(query)
    return any(t in q for t in DRUG_FOCUS_TERMS)


def expand_terms_for_intents(query: str, intents: list[str]) -> list[str]:
    q = _norm_q(query)
    terms: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        t = _norm_q(raw)
        if len(t) < 3 or t in seen:
            return
        seen.add(t)
        terms.append(t)

    for tok in re.split(r"[^a-zа-я0-9.]+", q):
        if len(tok) >= 3:
            add(tok)
    if len(q) >= 2:
        add(q)
    for key in intents:
        spec = INTENT_SPECS.get(key) or {}
        for t in spec.get("terms") or ():
            if len(t) >= 3:
                add(t)
    return terms


def allowed_sections_for_intents(
    intents: list[str],
    *,
    query: str = "",
) -> set[str] | None:
    if not intents:
        return None
    if is_drug_focus_query(query, intents):
        return {"treatment"}
    allowed: set[str] = set()
    for key in intents:
        spec = INTENT_SPECS.get(key) or {}
        for sec in spec.get("sections") or ():
            allowed.add(str(sec))
    return allowed or None


def intent_result_limits(intents: list[str], *, query: str = "") -> tuple[int, int]:
    """top_k, max_per_group."""
    if is_drug_focus_query(query, intents):
        return 6, 4
    if intents:
        return 8, 5
    return 12, 6


def is_table_noise_text(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) < 24:
        return False
    if TABLE_NOISE_RE.search(t[:320]):
        return True
    digits = sum(1 for c in t[:200] if c.isdigit())
    if digits >= 12 and "мкб" in t.lower():
        return True
    return False


def sentence_drug_score(sentence: str) -> float:
    s = _norm_q(sentence)
    if not s:
        return 0.0
    score = 0.0
    if _DRUG_SENTENCE_RE.search(s):
        score += 4.0
    for term in DRUG_FOCUS_TERMS:
        if term in s:
            score += 1.5
    if re.search(r"\d+[\.,]?\d*\s*(?:мг|г|мл|мкг)", s):
        score += 2.0
    return score


def specs_for_api() -> dict[str, Any]:
    """Сериализуемые спеки для /api/protocol-search-intents."""
    out: dict[str, Any] = {}
    for key, spec in INTENT_SPECS.items():
        out[key] = {
            "sections": list(spec.get("sections") or ()),
            "tags": list(spec.get("tags") or ()),
            "label": spec.get("label") or key,
            "terms": list(spec.get("terms") or ()),
            "phrases": list(spec.get("phrases") or ()),
        }
    return out
