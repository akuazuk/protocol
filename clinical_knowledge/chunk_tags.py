"""Метки протоколов и чанков для подбора КП и оценки КЗ."""
from __future__ import annotations

import re
from typing import Any

from clinical_knowledge.protocol_pick_filters import (
    _SPINE_ICD_ROOTS,
    _SPINE_NEEDLES,
    _card_blob,
    is_administrative_protocol,
)

_PREAMBLE_MARKERS = (
    "постановление министерства",
    "об утверждении",
    "постановляет:",
    "на основании абзаца",
    "министр здравоохранения",
)

# Высокоточные маркеры административного текста (не клиника): список утверждаемых
# протоколов, ссылки на приложения приказа, вступление в силу, подпись министра.
# Сканируется ВЕСЬ текст (маркеры бывают в конце куска), поэтому набор строгий.
_ADMIN_MARKERS = (
    "утвердить:",
    "(прилагается)",
    "прилагается);",
    "вступает в силу",
    "настоящее постановление",
    "настоящий приказ",
    "официального опубликования",
    "к приказу министерства",
    "к постановлению министерства",
    "приложению к приказу",
    "приложению к постановлению",
    # Футер правового портала и подписи/согласования постановлений Минздрава РБ.
    "национальный правовой интернет-портал",
    "признать утратившим силу",
    "признать утратившими силу",
    "областной исполнительный комитет",
    "городской исполнительный комитет",
    "совет министров республики беларусь",
    "зарегистрировано в национальном реестре",
)
# Подпись министра вида «Министр Д.Л.Пиневич».
_MINISTER_SIG = re.compile(r"министр\s+[а-яё]\.\s?[а-яё]\.", re.I)


def is_administrative_text(text: str) -> bool:
    """True для явно административных кусков (преамбула/приложение приказа, подпись)."""
    low = (text or "").lower()
    if any(m in low for m in _ADMIN_MARKERS):
        return True
    return bool(_MINISTER_SIG.search(low))

_OBLIGATION_REQUIRED = re.compile(r"обязательн|must|необходим|минимальн", re.I)
_OBLIGATION_OPTIONAL = re.compile(r"по\s+показан|при\s+необходим|может\s+быть", re.I)
_OBLIGATION_CONTRA = re.compile(r"противопоказан|не\s+рекоменду", re.I)

_INPATIENT = re.compile(r"стационарн|круглосуточн|госпитализац", re.I)
_AMBULATORY = re.compile(r"амбулатор|поликлиник|на\s+дому|дневн\w*\s+стационар", re.I)

_CLINICAL_INTENT_MAP = {
    "diagnostics": "diagnose",
    "criteria_block": "diagnose",
    "table": "diagnose",
    "treatment": "treat",
    "pharmacotherapy": "treat",
    "drug_list": "treat",
    "dispensary": "monitor",
    "prevention": "monitor",
    "rehabilitation": "treat",
    "routing": "refer",
}

_LOW_SIGNAL_TYPES = frozenset({"body", "terms"})

_CONDITION_CLUSTERS: list[tuple[str, tuple[str, ...]]] = [
    ("spine", ("ишиас", "радикул", "люмбо", "позвоноч", "вертеброген", "м54", "спондил", "остеохондр")),
    ("venous", ("тромбоз", "флеб", "варикоз", "тгв", "венозн")),
    ("epilepsy", ("эпилепс", "судорож")),
    ("urti", ("орви", "орз", "респиратор", "кашел", "насморк")),
    ("cardio", ("ишем", "стенокард", "инфаркт", "фибрилл")),
    ("gi", ("гастро", "язв", "рефлюкс", "кишечн")),
]


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


def infer_obligation(text: str, chunk_type: str = "") -> str:
    t = (text or "").strip()
    if _OBLIGATION_CONTRA.search(t):
        return "contraindicated"
    if _OBLIGATION_REQUIRED.search(t):
        return "required"
    if _OBLIGATION_OPTIONAL.search(t):
        return "optional"
    if chunk_type in ("diagnostics", "criteria_block", "table"):
        return "recommended"
    if chunk_type in ("pharmacotherapy", "drug_list", "treatment"):
        return "recommended"
    return "recommended"


def infer_signal(
    text: str,
    *,
    chunk_type: str = "body",
    is_preamble: bool = False,
    icd_codes: list[str] | None = None,
) -> str:
    if is_preamble:
        return "low"
    ctype = (chunk_type or "body").lower()
    if ctype in _LOW_SIGNAL_TYPES and not (icd_codes or []):
        head = (text or "")[:120].lower()
        if any(m in head for m in _PREAMBLE_MARKERS):
            return "low"
    if len((text or "").strip()) < 50:
        return "low"
    if ctype in ("diagnostics", "treatment", "pharmacotherapy", "criteria_block", "table", "dispensary"):
        return "high"
    if ctype == "protocol_overview":
        return "high"
    return "medium"


def infer_care_settings(text: str, existing: list[str] | None = None) -> list[str]:
    low = (text or "").lower()
    out = list(existing or [])
    seen = set(out)
    if _INPATIENT.search(low) and "inpatient" not in seen:
        out.append("inpatient")
        seen.add("inpatient")
    if _AMBULATORY.search(low) and "ambulatory" not in seen:
        out.append("ambulatory")
        seen.add("ambulatory")
    if not out:
        out.append("any")
    return out


def infer_clinical_intent(chunk_type: str) -> str:
    return _CLINICAL_INTENT_MAP.get((chunk_type or "").lower(), "other")


def is_chunk_preamble(text: str) -> bool:
    head = (text or "")[:200].lower()
    return any(m in head for m in _PREAMBLE_MARKERS)


def icd_weights_for_chunk(
    icd_codes: list[str],
    protocol_weights: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Веса МКБ 0-1 на чанке."""
    out: dict[str, float] = {}
    pw = protocol_weights or {}
    for raw in icd_codes or []:
        code = str(raw).upper().strip()
        if not code:
            continue
        if code in pw:
            try:
                v = float(pw[code])
                out[code] = round(min(1.0, v / 100.0 if v > 1 else v), 2)
            except (TypeError, ValueError):
                out[code] = 1.0
        else:
            out[code] = 1.0
    return out


def infer_condition_cluster(blob: str, icd_codes: list[str] | None = None) -> str:
    low = (blob or "").lower()
    for name, needles in _CONDITION_CLUSTERS:
        if any(n in low for n in needles):
            return name
    roots = {_icd_root(c) for c in (icd_codes or [])}
    if roots & _SPINE_ICD_ROOTS:
        return "spine"
    if roots & frozenset({"I80", "I81", "I82", "I83"}):
        return "venous"
    if roots & frozenset({"J06", "J00", "J02", "J03"}):
        return "urti"
    return "general"


def build_chunk_tags(
    *,
    text: str,
    chunk_type: str,
    icd_codes: list[str] | None = None,
    care_setting: list[str] | None = None,
    protocol_weights: dict[str, Any] | None = None,
    drugs: list[str] | None = None,
    imaging: list[str] | None = None,
    lab_tests: list[str] | None = None,
) -> dict[str, Any]:
    preamble = is_chunk_preamble(text)
    obligation = infer_obligation(text, chunk_type)
    signal = infer_signal(text, chunk_type=chunk_type, is_preamble=preamble, icd_codes=icd_codes)
    care = infer_care_settings(text, care_setting)
    entities: dict[str, list[str]] = {}
    if imaging:
        entities["exam"] = list(imaging)[:12]
    if lab_tests:
        entities.setdefault("exam", []).extend(lab_tests[:8])
    if drugs:
        entities["drug"] = list(drugs)[:12]
    return {
        "signal": signal,
        "is_preamble": preamble,
        "obligation": obligation,
        "clinical_intent": infer_clinical_intent(chunk_type),
        "care_setting": care,
        "icd10_weights": icd_weights_for_chunk(list(icd_codes or []), protocol_weights),
        "entities": entities,
        "extractability": "rule_ready" if chunk_type in ("table", "drug_list", "criteria_block") else "narrative",
    }


def build_protocol_tags(
    *,
    title: str,
    source_path: str,
    protocol_kind: str,
    icd_codes: list[str] | None,
    chunk_type_counts: dict[str, int] | None,
    catalog_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    card = {
        "title": title,
        "source_path": source_path,
        "protocol_kind": protocol_kind,
        "icd10_primary": list(icd_codes or [])[:8],
        "icd10_all": list(icd_codes or []),
    }
    if catalog_row:
        card.update(catalog_row)
    admin = is_administrative_protocol(card)
    blob = _card_blob(card)
    counts = chunk_type_counts or {}
    has_dx = counts.get("diagnostics", 0) + counts.get("criteria_block", 0) + counts.get("table", 0) > 0
    has_tx = counts.get("treatment", 0) + counts.get("pharmacotherapy", 0) + counts.get("drug_list", 0) > 0
    has_mon = counts.get("dispensary", 0) + counts.get("prevention", 0) > 0
    clinical_chunks = sum(v for k, v in counts.items() if k not in ("body", "terms"))
    total_chunks = sum(counts.values()) or 1
    richness = round(min(1.0, clinical_chunks / max(total_chunks, 1)), 2)
    cluster = infer_condition_cluster(blob, icd_codes)
    spine_markers = any(n in blob for n in _SPINE_NEEDLES)
    return {
        "admin_order": admin,
        "usable_for_kz_review": not admin and richness >= 0.15,
        "condition_cluster": cluster,
        "content_richness_score": richness,
        "has_actionable_diagnostics": has_dx,
        "has_actionable_treatment": has_tx,
        "has_dispensary": has_mon,
        "care_setting": infer_care_settings(blob),
        "spine_related": spine_markers or cluster == "spine",
    }


def chunk_usable_for_retrieval(chunk: dict[str, Any], *, ambulatory: bool = True) -> bool:
    if chunk.get("indexable") is False:
        return False
    tags = chunk.get("tags") or {}
    if tags.get("is_preamble") or tags.get("signal") == "low":
        return False
    ctype = (chunk.get("chunk_type") or "").lower()
    text = chunk.get("text") or chunk.get("lex_text") or ""
    # Административные куски (список «Утвердить», подпись министра, ссылки на
    # приложения приказа) не годятся как клинические выдержки - отсекаем даже при
    # пустых tags (частый случай для appendix/body без обогащения).
    if is_administrative_text(text):
        return False
    if ctype == "appendix" and is_chunk_preamble(text):
        return False
    if ctype == "terms":
        return bool(chunk.get("icd10_codes"))
    if ambulatory:
        care = tags.get("care_setting") or chunk.get("care_setting") or []
        if "inpatient" in care and "ambulatory" not in care:
            return False
    return True
