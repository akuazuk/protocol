"""Подбор целевых PDF для воронки поиска протоколов (жалобы + МКБ + аудитория).

В отличие от consult_retrieval (КЗ), здесь:
  - симптом-коды R* разворачиваются в коды болезней (J02/J06…) по тексту запроса;
  - при уверенном совпадении по заголовку/реестру включается strict path_allowlist;
  - отсекаются явно чужие домены (анестезия, ГЭРБ, аллергический ринит при «горло»).
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from clinical_knowledge.consult_retrieval import consult_target_protocol_paths
from clinical_knowledge.diagnosis_icd import (
    enrich_diagnosis_codes,
    is_symptom_code,
    normalize_code,
    prioritize_codes,
)
from clinical_knowledge.rule_family_gates import (
    expand_specialty_slugs_for_clinical_text,
    expand_specialty_slugs_for_icd,
)

ROOT = Path(__file__).resolve().parents[1]

_SYMPTOM_ICD_EXPAND: dict[str, list[str]] = {
    "R07": ["J02.9", "J06.9"],
    "R07.0": ["J02.9", "J06.9"],
    "R07.2": ["J03.9", "J02.9"],
    "R07.3": ["J02.9"],
    "R05": ["J06.9", "J20.9"],
    "R50": ["J06.9", "J18.9"],
    "R51": ["J06.9", "G43.9"],
}

_THROAT_MARKERS = (
    "горл",
    "глот",
    "фаринг",
    "ангин",
    "тонзилл",
    "ларинг",
    "дисфаг",
    "глотать",
    "глотан",
)
_URI_TITLE_MARKERS = (
    "орви",
    "орз",
    "респиратор",
    "вирусн",
    "фаринг",
    "ангин",
    "тонзилл",
    "ларинг",
    "оторин",
    "лор",
    "уха горла носа",
    "глот",
)
_URI_TITLE_STRONG = (
    "орви",
    "орз",
    "фаринг",
    "ангин",
    "тонзилл",
    "ларинг",
    "оторин",
    "лор",
    "уха горла носа",
)
_WRONG_DOMAIN_FOR_THROAT = (
    "анестезиolog",
    "анестези",
    "хирургическ",
    "пищевод",
    "желудк",
    "двенадцатипер",
    "гастроэзофаг",
    "рефлюкс",
    "гэрб",
    "аллерг",
    "ринит",
    "срыгив",
    "гепатит",
    "кожи и подкож",
    "мягких тканей",
)
_GI_QUERY_MARKERS = ("живот", "изжог", "тошн", "рвот", "желуд", "кишеч", "стул", "диар")


def _path_norm(sp: str) -> str:
    return (sp or "").replace("\\", "/").strip()


def _query_blob(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").lower()).strip()


def _extract_icd_from_query(query: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b", query or "", re.I):
        c = normalize_code(m.group(1))
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _infer_audience(query: str) -> str | None:
    q = _query_blob(query)
    if "контекст подбора: детское" in q or "детское население" in q:
        return "child"
    if "контекст подбора: взрослое" in q or "взрослое население" in q:
        return "adult"
    return None


def _audience_from_title(path: str, title: str) -> str | None:
    s = f"{path} {title}".lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    ped = ("дет нас", "детск", "детс", "детское", "дет ", " дет", "новорожд", "грудн")
    adult = ("взр", "взросл", "в-нас", "в нас")
    has_p = any(p in s for p in ped)
    has_a = any(a in s for a in adult)
    if has_p and not has_a:
        return "child"
    if has_a and not has_p:
        return "adult"
    if has_p and has_a:
        return "mixed"
    return None


def _audience_mismatch(audience: str | None, path: str, title: str) -> bool:
    if audience not in ("adult", "child"):
        return False
    hint = _audience_from_title(path, title)
    if hint is None or hint == "mixed":
        return False
    if audience == "adult" and hint == "child":
        return True
    if audience == "child" and hint == "adult":
        return True
    return False


def _has_throat_context(query: str, icd_codes: list[str]) -> bool:
    ql = _query_blob(query)
    if any(m in ql for m in _THROAT_MARKERS):
        return True
    return any(normalize_code(c).startswith("R07") for c in icd_codes)


def _has_gi_context(query: str) -> bool:
    ql = _query_blob(query)
    return any(m in ql for m in _GI_QUERY_MARKERS)


def _domain_mismatch_for_query(path: str, title: str, *, throat: bool, gi: bool) -> bool:
    blob = f"{path} {title}".lower()
    if throat and not gi:
        return any(m in blob for m in _WRONG_DOMAIN_FOR_THROAT)
    return False


def expand_icd_for_protocol_search(
    query: str,
    icd_codes: list[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    """Коды для retrieve: болезни вперёд, симптом-коды дополнены по словарю и R→J."""
    raw = list(dict.fromkeys(_extract_icd_from_query(query) + list(icd_codes or [])))
    ordered, meta = enrich_diagnosis_codes(query, raw)
    expanded = list(ordered)
    seen = {normalize_code(c) for c in expanded}
    for c in list(raw):
        nc = normalize_code(c)
        root = nc[:3] if len(nc) >= 3 else nc
        for add in _SYMPTOM_ICD_EXPAND.get(nc, _SYMPTOM_ICD_EXPAND.get(root, [])):
            if add not in seen:
                seen.add(add)
                expanded.append(add)
    expanded = prioritize_codes(expanded)
    meta = dict(meta)
    meta["expanded_from_symptom"] = [
        c for c in expanded if c not in {normalize_code(x) for x in raw}
    ]
    meta["had_symptom_only"] = bool(raw) and all(is_symptom_code(c) for c in raw)
    return expanded, meta


@lru_cache(maxsize=1)
def _load_protocol_meta() -> dict[str, dict[str, Any]]:
    path = ROOT / "protocol_meta.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _title_match_paths(
    query: str,
    *,
    audience: str | None,
    throat: bool,
    limit: int = 8,
) -> list[tuple[str, float, str]]:
    """Пути PDF по заголовку/имени файла (protocol_meta + cards registry)."""
    scored: dict[str, tuple[float, str]] = {}
    gi = _has_gi_context(query)

    def consider(sp: str, title: str, base: float, src: str) -> None:
        sp = _path_norm(sp)
        if not sp:
            return
        if _audience_mismatch(audience, sp, title):
            return
        if _domain_mismatch_for_query(sp, title, throat=throat, gi=gi):
            return
        blob = f"{sp} {title}".lower()
        score = 0.0
        if throat:
            strong_hits = sum(1 for m in _URI_TITLE_STRONG if m in blob)
            if strong_hits == 0:
                return
            weak_hits = sum(1 for m in _URI_TITLE_MARKERS if m in blob)
            score = base + 12.0 + 6.0 * strong_hits + 2.0 * max(0, weak_hits - strong_hits)
            aud = _audience_from_title(sp, title)
            if audience == "adult" and aud == "adult":
                score += 10.0
            elif audience == "child" and aud == "child":
                score += 10.0
        else:
            score = base
        prev = scored.get(sp)
        if prev is None or score > prev[0]:
            scored[sp] = (score, src)

    meta = _load_protocol_meta()
    for sp, row in meta.items():
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or sp)
        consider(sp, title, 8.0, "protocol_meta")

    if throat:
        try:
            from clinical_knowledge.loader import load_protocol_cards_registry

            for card in load_protocol_cards_registry():
                sp = _path_norm(str(card.get("source_path") or ""))
                title = str(card.get("title") or sp)
                consider(sp, title, 6.0, "protocol_card")
        except Exception:
            pass

    ranked = sorted(scored.items(), key=lambda x: (-x[1][0], x[0]))
    return [(sp, sc, src) for sp, (sc, src) in ranked[:limit]]


def search_target_protocol_paths(
    *,
    query: str,
    icd_codes: list[str] | None = None,
    category_slugs: list[str] | None = None,
    max_paths: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Целевые PDF для strict/boost режима воронки поиска."""
    limit = max_paths
    if limit is None:
        limit = max(2, min(10, int(os.environ.get("SEARCH_MAX_PROTOCOL_PATHS", "6"))))

    expanded, expand_meta = expand_icd_for_protocol_search(query, icd_codes)
    audience = _infer_audience(query)
    throat = _has_throat_context(query, expanded)
    slugs = sorted(
        expand_specialty_slugs_for_clinical_text(
            expand_specialty_slugs_for_icd(set(category_slugs or []), expanded),
            query,
        )
    )
    if throat and not slugs:
        slugs = sorted(
            {
                "otorinolaringologiya",
                "pulmonologiya-ftiziatriya",
                "infektsionnye-zabolevaniya",
                "terapiya",
            }
        )

    paths: list[str] = []
    seen: set[str] = set()
    sources: dict[str, str] = {}

    def add(sp: str, src: str) -> None:
        n = _path_norm(sp)
        if not n or n in seen:
            return
        seen.add(n)
        paths.append(n)
        sources[n] = src

    registry_paths, registry_meta = consult_target_protocol_paths(
        merged_icd=expanded,
        diag_icd=icd_codes,
        clinical_rules=None,
        specialty_slugs=slugs,
        consult_text=query,
        max_paths=limit,
    )
    for sp in registry_paths:
        blob = sp.lower()
        if throat and _domain_mismatch_for_query(sp, sp, throat=throat, gi=_has_gi_context(query)):
            continue
        add(sp, registry_meta.get("path_sources", {}).get(sp, "icd_registry"))

    title_hits = _title_match_paths(query, audience=audience, throat=throat, limit=limit)
    for sp, sc, src in title_hits:
        if sc >= 18.0:
            add(sp, src)

    strict = bool(paths) and (
        expand_meta.get("had_symptom_only")
        or throat
        or bool(registry_paths)
    )

    meta: dict[str, Any] = {
        "expanded_icd": expanded[:12],
        "expand_meta": expand_meta,
        "specialty_slugs": slugs,
        "path_sources": sources,
        "title_scores": {sp: sc for sp, sc, _ in title_hits[:8]},
        "strict": strict,
        "audience": audience,
        "throat_context": throat,
        "registry_meta": registry_meta,
    }
    deduped: list[str] = []
    seen_names: set[str] = set()
    for sp in paths:
        name = Path(sp).name.lower()
        if name in seen_names:
            continue
        seen_names.add(name)
        deduped.append(sp)
    return deduped[:limit], meta


def build_protocol_search_context(
    *,
    query: str,
    icd_codes: list[str] | None = None,
    category_slugs: list[str] | None = None,
) -> dict[str, Any]:
    """Контекст для api_assist: expanded ICD, path_boost, optional strict allowlist."""
    target_paths, meta = search_target_protocol_paths(
        query=query,
        icd_codes=icd_codes,
        category_slugs=category_slugs,
    )
    from clinical_knowledge.protocol_summary.icd_index import find_catalog_paths_by_icd_codes

    expanded = meta.get("expanded_icd") or list(icd_codes or [])
    summary_paths = find_catalog_paths_by_icd_codes(expanded, limit=8) if expanded else []
    path_boost = list(
        dict.fromkeys(_path_norm(p) for p in (target_paths + summary_paths) if p)
    )
    path_allowlist: list[str] | None = None
    if meta.get("strict") and target_paths:
        path_allowlist = list(target_paths)
    return {
        "expanded_icd_codes": expanded,
        "path_boost": path_boost or None,
        "path_allowlist": path_allowlist,
        "search_meta": meta,
    }
