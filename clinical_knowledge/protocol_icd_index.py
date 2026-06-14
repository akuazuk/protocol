"""Быстрый lookup протоколов по МКБ + аудитория (инвертированный индекс)."""
from __future__ import annotations

import importlib.util
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pc = _load_module("_protocol_catalog", "clinical_knowledge/protocol_catalog.py")
load_protocol_catalog = _pc.load_protocol_catalog
_diag = _load_module("_diag_icd", "clinical_knowledge/diagnosis_icd.py")
normalize_code = _diag.normalize_code
prioritize_codes = _diag.prioritize_codes
enrich_diagnosis_codes = _diag.enrich_diagnosis_codes

_ICD_RUBRIC: dict[str, list[str]] = {
    "I": ["bolezni-sistemy-krovoobrashcheniya"],
    "J": ["pulmonologiya-ftiziatriya", "infektsionnye-zabolevaniya"],
    "E": ["endokrinologiya-narusheniya-obmena-veshchestv"],
    "K": ["gastroenterologiya"],
    "G": ["nevrologiya-neyrokhirurgiya"],
    "N": ["nefrologiya", "urologiya"],
    "O": ["akusherstvo-ginekologiya"],
    "C": ["novoobrazovaniya"],
    "F": ["psikhiatriya-narkologiya"],
    "L": ["dermatovenerologiya"],
    "H": ["oftalmologiya"],
    "M": ["ortopediya-travmatologiya"],
}


def _icd_root(code: str) -> str:
    c = normalize_code(code)
    return c[:3] if len(c) >= 3 else c


def _population_to_audience(population: str | None) -> str | None:
    p = (population or "").strip().lower()
    if p in ("adult", "взросл"):
        return "adult"
    if p in ("pediatric", "child", "ped", "дет"):
        return "child"
    if p in ("pregnant", "беремен"):
        return "adult"
    return None


def _audience_mismatch(request_aud: str | None, entry_aud: str) -> bool:
    if not request_aud or request_aud not in ("adult", "child"):
        return False
    if entry_aud in ("any", "mixed", ""):
        return False
    if request_aud == "adult" and entry_aud == "pediatric":
        return True
    if request_aud == "child" and entry_aud == "adult":
        return True
    return False


def _title_tokens(text: str) -> set[str]:
    return set(re.findall(r"[а-яёa-z0-9]{4,}", (text or "").lower(), re.I))


@lru_cache(maxsize=1)
def _inverted_index() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """exact code -> paths, icd root -> paths."""
    exact: dict[str, list[str]] = {}
    prefix: dict[str, list[str]] = {}
    cat = load_protocol_catalog()
    for path, row in cat.items():
        codes = list(row.get("icd10_all") or []) + list(row.get("icd10_primary") or [])
        seen_local: set[str] = set()
        for raw in codes:
            c = normalize_code(str(raw))
            if not c or c in seen_local:
                continue
            seen_local.add(c)
            exact.setdefault(c, [])
            if path not in exact[c]:
                exact[c].append(path)
            root = _icd_root(c)
            if root:
                prefix.setdefault(root, [])
                if path not in prefix[root]:
                    prefix[root].append(path)
    return exact, prefix


def prewarm_protocol_icd_index() -> dict[str, int]:
    cat = load_protocol_catalog()
    exact, prefix = _inverted_index()
    return {
        "catalog_entries": len(cat),
        "icd_exact_keys": len(exact),
        "icd_prefix_keys": len(prefix),
    }


def _score_entry(
    row: dict[str, Any],
    *,
    query_icd: set[str],
    query_roots: set[str],
    query_tokens: set[str],
    rubric_slugs: set[str],
    request_aud: str | None,
) -> tuple[float, list[str]]:
    path = str(row.get("path") or "")
    primary = {_norm_icd(x) for x in (row.get("icd10_primary") or [])}
    all_icd = {_norm_icd(x) for x in (row.get("icd10_all") or [])}
    reasons: list[str] = []
    score = 0.0

    for code in query_icd:
        if code in primary:
            score += 100.0
            reasons.append(f"exact {code} in icd10_primary")
        elif code in all_icd:
            score += 80.0
            reasons.append(f"exact {code} in icd10_all")
        else:
            root = _icd_root(code)
            for ic in all_icd | primary:
                if ic.startswith(root) or code.startswith(_icd_root(ic)):
                    score += 60.0
                    reasons.append(f"prefix {code}↔{ic}")
                    break

    for root in query_roots:
        for ic in all_icd | primary:
            if _icd_root(ic) == root:
                score += 25.0
                reasons.append(f"same root {root}")
                break

    slug = str(row.get("specialty_slug") or "")
    if slug and slug in rubric_slugs:
        score += 20.0
        reasons.append(f"rubric {slug}")

    title = str(row.get("display_title") or Path(path).name)
    title_tok = _title_tokens(title)
    overlap = len(query_tokens & title_tok)
    if overlap:
        score += min(15.0, 3.0 * overlap)
        reasons.append(f"title overlap ×{overlap}")

    aud = str(row.get("audience") or "any")
    if request_aud == "adult" and aud == "adult":
        score += 15.0
        reasons.append("audience adult")
    elif request_aud == "child" and aud == "pediatric":
        score += 15.0
        reasons.append("audience pediatric")
    elif _audience_mismatch(request_aud, aud):
        score -= 50.0
        reasons.append("audience mismatch")

    return score, reasons


def _norm_icd(code: str) -> str:
    return normalize_code(code)


def _expand_icd_for_lookup(query: str, icd_codes: list[str] | None) -> list[str]:
    raw: list[str] = list(icd_codes or [])
    for m in re.finditer(r"\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b", query or "", re.I):
        raw.append(m.group(1).upper())
    ordered, _ = enrich_diagnosis_codes(query, raw)
    return prioritize_codes(list(ordered))


def _infer_audience_from_query(query: str) -> str | None:
    ql = (query or "").lower()
    if re.search(r"контекст подбора:\s*взросл|взрослое население", ql):
        return "adult"
    if re.search(r"контекст подбора:\s*дет|детское население", ql):
        return "child"
    if re.search(r"\bдет|\bребен|\bребён|\bноворожд", ql):
        return "child"
    if re.search(r"\bвзросл|\bпожил", ql):
        return "adult"
    return None


def lookup_protocols_by_icd(
    *,
    icd_codes: list[str] | None,
    query: str = "",
    population: str | None = None,
    rubric_slugs: list[str] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Детерминированный подбор PDF по МКБ без RAG."""
    t0 = time.perf_counter()
    cat = load_protocol_catalog()
    if not cat:
        return {
            "protocols": [],
            "ambiguous": True,
            "lookup_ms": 0,
            "match_reasons": {},
            "expanded_icd": [],
        }

    expanded = _expand_icd_for_lookup(query, icd_codes)
    query_icd = {normalize_code(c) for c in expanded if c}
    query_roots = {_icd_root(c) for c in query_icd}
    rubric_set = {s.strip() for s in (rubric_slugs or []) if s}
    if not rubric_set:
        for code in list(query_icd)[:3]:
            letter = code[:1].upper()
            rubric_set.update(_ICD_RUBRIC.get(letter, []))

    request_aud = _population_to_audience(population)
    if request_aud is None and query:
        request_aud = _infer_audience_from_query(query)

    exact_idx, prefix_idx = _inverted_index()
    candidate_paths: set[str] = set()
    for code in query_icd:
        for p in exact_idx.get(code, []):
            candidate_paths.add(p)
        for p in prefix_idx.get(_icd_root(code), []):
            candidate_paths.add(p)

    if not candidate_paths:
        for slug in rubric_set:
            for row in cat.values():
                if row.get("specialty_slug") == slug:
                    candidate_paths.add(str(row["path"]))
    if not candidate_paths:
        for row in cat.values():
            if row.get("general_scope"):
                candidate_paths.add(str(row["path"]))

    query_tokens = _title_tokens(query)
    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    for path in candidate_paths:
        row = cat.get(path)
        if not row:
            continue
        sc, reasons = _score_entry(
            row,
            query_icd=query_icd,
            query_roots=query_roots,
            query_tokens=query_tokens,
            rubric_slugs=rubric_set,
            request_aud=request_aud,
        )
        if sc <= 0:
            continue
        scored.append((sc, row, reasons))

    scored.sort(key=lambda x: (-x[0], x[1].get("path") or ""))
    top = scored[: max(limit, 1)]
    protocols: list[dict[str, Any]] = []
    match_reasons: dict[str, list[str]] = {}
    max_sc = top[0][0] if top else 1.0
    for sc, row, reasons in top[:limit]:
        path = str(row.get("path") or "")
        title = str(row.get("display_title") or Path(path).name)
        conf = min(0.97, max(0.42, 0.38 + 0.55 * (sc / max(max_sc, 1.0))))
        protocols.append(
            {
                "path": path,
                "title": title,
                "confidence_score": round(conf, 4),
                "rag_support": round(conf * 0.9, 4),
                "icd_lookup_score": round(sc, 2),
                "audience": row.get("audience"),
                "protocol_kind": row.get("protocol_kind"),
                "scope_label_ru": row.get("scope_label_ru"),
                "general_scope": bool(row.get("general_scope")),
                "matched_icd_codes": list(query_icd)[:6],
            }
        )
        match_reasons[path] = reasons[:4]

    ambiguous = False
    if len(top) >= 2:
        gap = top[0][0] - top[1][0]
        ambiguous = gap < 12.0 or top[0][0] < 60.0
    elif not top:
        ambiguous = True

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "protocols": protocols,
        "ambiguous": ambiguous,
        "lookup_ms": elapsed_ms,
        "match_reasons": match_reasons,
        "expanded_icd": expanded[:12],
        "path_allowlist": [p["path"] for p in protocols[:15]],
    }


def format_assist_payload(
    *,
    query: str,
    lookup_result: dict[str, Any],
    icd_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ответ в формате api_assist для UI."""
    protos = lookup_result.get("protocols") or []
    icd = icd_analysis or {}
    return {
        "query": query,
        "retrieve_only": True,
        "assist_lite": True,
        "icd_fast_lookup": True,
        "lookup_ms": lookup_result.get("lookup_ms", 0),
        "icd_lookup_ambiguous": lookup_result.get("ambiguous", False),
        "match_reasons": lookup_result.get("match_reasons") or {},
        "llm_json": {"protocols": protos, "icd_codes": icd.get("detected") or []},
        "icd": icd,
        "expanded_icd_codes": lookup_result.get("expanded_icd") or [],
        "finish_reason": "ICD_LOOKUP",
    }
