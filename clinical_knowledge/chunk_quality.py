"""Детекция проблем качества rich-чанков и флаги indexable."""
from __future__ import annotations

import re
from typing import Any

from clinical_knowledge.chunk_tags import is_chunk_preamble
from clinical_knowledge.chunk_type_infer import infer_chunk_type, resolve_section_title

ISSUE_ICD_INFLATION = "icd_inflation"
ISSUE_WEAK_SECTION_TITLE = "weak_section_title"
ISSUE_TYPE_BODY_BUT_CLINICAL = "type_body_but_clinical"
ISSUE_TOO_SHORT = "too_short"
ISSUE_TOO_LONG = "too_long"
ISSUE_PREAMBLE_LEAK = "preamble_leak"
ISSUE_EMPTY_ENTITIES = "empty_entities"
ISSUE_TRUNCATED_LIST = "truncated_list"
ISSUE_NO_TAGS = "no_tags"

_CLINICAL_WORDS = re.compile(
    r"рекоменду|назнач|показан|диагност|лечени|анализ|узи|мрт|препарат|"
    r"обязательн|противопоказ|доз(?:а|ировк)|исследован",
    re.I,
)

_WEAK_TITLES = frozenset({
    "таблица",
    "постановляет:",
    "утверждено",
    "согласовано",
    "document",
    "документ",
    "№ 2435-xii",
    "глава 1",
    "глава 2",
    "глава 3",
    "глава 4",
    "глава 5",
    "глава 6",
    "глава 7",
})


def is_weak_section_title(title: str) -> bool:
    st = (title or "").strip()
    low = st.lower()
    return not st or low in _WEAK_TITLES or (len(st) < 8 and low not in ("показания",))


def build_section_title_map(chunks: list[dict[str, Any]]) -> dict[str, str]:
    """Лучший section_title по section_number внутри одного doc."""
    out: dict[str, str] = {}
    for ch in chunks:
        sn = str(ch.get("section_number") or "").strip()
        if not sn:
            continue
        candidates: list[str] = []
        for label in reversed(list(ch.get("section_path") or [])):
            candidates.append(str(label))
        candidates.append(str(ch.get("section_title") or ""))
        for t in candidates:
            t = t.strip()
            if t and not is_weak_section_title(t):
                if sn not in out or len(t) > len(out[sn]):
                    out[sn] = t
                break
    return out


def resolve_section_title_with_map(
    chunk: dict[str, Any],
    section_map: dict[str, str] | None = None,
) -> str:
    st = fix_weak_section_title(chunk)
    if is_weak_section_title(st) and section_map:
        sn = str(chunk.get("section_number") or "").strip()
        if sn and sn in section_map:
            return section_map[sn]
    return st


def is_truncated_text(text: str) -> bool:
    t = (text or "").rstrip()
    if not t:
        return False
    return t[-1] in (";", ",", "–", "-", ":") and len(t) < 1100

_NOISE_LINE_RE = re.compile(
    r"^(?:\s*(?:утверждено|согласовано|документ|форма\s+\d+"
    r"|клинический\s+протокол\s*$|стр\.\s*\d+)\s*)$",
    re.I,
)

_FOOTER_RE = re.compile(
    r"^(?:\d+\s*/\s*\d+\s*$|стр\.\s*\d+\s*$|–\s*\d+\s*–\s*$)",
    re.I,
)

_CLINICAL_TYPES = frozenset({
    "diagnostics",
    "treatment",
    "criteria_block",
    "pharmacotherapy",
    "drug_list",
    "prevention",
    "dispensary",
})

_MIN_INDEXABLE_LEN = 40
_MIN_CLINICAL_LEN = 80
_MAX_CHARS = 1200
_ICD_INFLATION_MIN = 15
_ICD_INFLATION_MAX_TEXT = 200


def strip_noise_lines(text: str) -> str:
    """Убрать однострочный служебный шум и колонтитулы."""
    lines = (text or "").splitlines()
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append(line)
            continue
        if _NOISE_LINE_RE.match(s) or _FOOTER_RE.match(s):
            continue
        kept.append(line)
    out = "\n".join(kept).strip()
    return re.sub(r"\n{3,}", "\n\n", out)


def is_icd_inflation(chunk: dict[str, Any]) -> bool:
    icds = chunk.get("icd10_codes") or []
    text = (chunk.get("text") or "").strip()
    return len(icds) >= _ICD_INFLATION_MIN and len(text) < _ICD_INFLATION_MAX_TEXT


def fix_weak_section_title(chunk: dict[str, Any]) -> str:
    return resolve_section_title(
        str(chunk.get("section_title") or ""),
        list(chunk.get("section_path") or []),
    )


def detect_issues(chunk: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    text = (chunk.get("text") or "").strip()
    ctype = (chunk.get("chunk_type") or "body").strip().lower()
    st_raw = (chunk.get("section_title") or "").strip()

    if not chunk.get("tags"):
        issues.append(ISSUE_NO_TAGS)

    if len(text) < _MIN_CLINICAL_LEN and ctype not in ("protocol_overview", "table"):
        issues.append(ISSUE_TOO_SHORT)
    if len(text) > _MAX_CHARS:
        issues.append(ISSUE_TOO_LONG)

    if is_chunk_preamble(text) or chunk.get("tags", {}).get("is_preamble"):
        issues.append(ISSUE_PREAMBLE_LEAK)

    if is_weak_section_title(st_raw):
        issues.append(ISSUE_WEAK_SECTION_TITLE)

    if ctype == "body" and _CLINICAL_WORDS.search(text):
        issues.append(ISSUE_TYPE_BODY_BUT_CLINICAL)

    if is_icd_inflation(chunk):
        issues.append(ISSUE_ICD_INFLATION)

    if ctype in _CLINICAL_TYPES and len(text) > 150:
        has_ent = bool(
            chunk.get("lab_tests")
            or chunk.get("imaging")
            or chunk.get("drugs")
            or chunk.get("dosages")
        )
        if not has_ent and not _CLINICAL_WORDS.search(text):
            issues.append(ISSUE_EMPTY_ENTITIES)

    if text.endswith(";") or text.endswith(",") or text.endswith("–"):
        issues.append(ISSUE_TRUNCATED_LIST)

    return issues


def quality_score(chunk: dict[str, Any]) -> float:
    """0.0-1.0: выше = лучше."""
    issues = detect_issues(chunk)
    score = 1.0
    weights = {
        ISSUE_PREAMBLE_LEAK: 0.45,
        ISSUE_TYPE_BODY_BUT_CLINICAL: 0.25,
        ISSUE_ICD_INFLATION: 0.15,
        ISSUE_WEAK_SECTION_TITLE: 0.1,
        ISSUE_TOO_SHORT: 0.12,
        ISSUE_TOO_LONG: 0.08,
        ISSUE_EMPTY_ENTITIES: 0.08,
        ISSUE_TRUNCATED_LIST: 0.05,
        ISSUE_NO_TAGS: 0.05,
    }
    for iss in issues:
        score -= weights.get(iss, 0.05)
    tags = chunk.get("tags") or {}
    if tags.get("signal") == "high":
        score += 0.05
    if chunk.get("indexable") is False:
        score -= 0.2
    return round(max(0.0, min(1.0, score)), 3)


def should_index(chunk: dict[str, Any]) -> bool:
    """Можно ли индексировать чанк в RAG."""
    if chunk.get("indexable") is False:
        return False
    if chunk.get("chunk_is_empty"):
        return False
    text = (chunk.get("text") or "").strip()
    ctype = (chunk.get("chunk_type") or "body").strip().lower()
    if ctype != "protocol_overview" and len(text) < _MIN_INDEXABLE_LEN:
        return False
    if is_chunk_preamble(text):
        return False
    tags = chunk.get("tags") or {}
    if tags.get("is_preamble") or tags.get("signal") == "low":
        if ctype in ("body", "terms") and not (chunk.get("icd10_codes") or []):
            return False
    noise = chunk.get("noise_flags") or []
    if "preamble" in noise or "legal_header" in noise:
        return False
    return True


def apply_indexable_flags(chunk: dict[str, Any]) -> dict[str, Any]:
    """Выставить indexable и noise_flags на чанке."""
    text = (chunk.get("text") or "").strip()
    noise: list[str] = list(chunk.get("noise_flags") or [])
    st = (chunk.get("section_title") or "").strip().lower()

    if is_chunk_preamble(text):
        if "preamble" not in noise:
            noise.append("preamble")
    if st in ("постановляет:", "утверждено", "согласовано"):
        if "legal_header" not in noise:
            noise.append("legal_header")

    chunk["noise_flags"] = noise
    chunk["indexable"] = should_index({**chunk, "noise_flags": noise})
    chunk["quality_score"] = quality_score(chunk)
    return chunk


def suggest_chunk_type(chunk: dict[str, Any]) -> str | None:
    current = (chunk.get("chunk_type") or "body").strip().lower()
    suggested = infer_chunk_type(
        section_title=str(chunk.get("section_title") or ""),
        section_number=str(chunk.get("section_number") or ""),
        section_path=list(chunk.get("section_path") or []),
        text=str(chunk.get("text") or ""),
    )
    if suggested != current:
        return suggested
    return None
