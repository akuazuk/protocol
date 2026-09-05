"""Семантический поиск внутри протокола + AI Overview (Фаза 1-2)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from clinical_knowledge.protocol_search_intents import (
    DRUG_CHUNK_TYPES,
    INTENT_SPECS,
    allowed_sections_for_intents,
    detect_query_intents,
    expand_terms_for_intents,
    intent_result_limits,
    is_drug_focus_query,
    is_table_noise_text,
)
from clinical_knowledge.protocol_source_view import (
    _GROUP_ORDER,
    _TYPE_INTENT_TAGS,
    _TYPE_TO_GROUP,
    build_view_from_items,
    format_rich_chunk_nav_item,
)

ROOT = Path(__file__).resolve().parent.parent

_OVERVIEW_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_OVERVIEW_CACHE_TTL_SEC = int(os.environ.get("PROTOCOL_OVERVIEW_CACHE_TTL_SEC", "3600"))


def _norm_q(q: str) -> str:
    return (q or "").lower().replace("ё", "е").strip()


def _query_tokens(query: str, intents: list[str] | None = None) -> set[str]:
    intents = intents if intents is not None else detect_query_intents(query)
    return set(expand_terms_for_intents(query, intents))


def _lex_score(chunk: dict[str, Any], query: str, tokens: set[str]) -> float:
    parts = [
        str(chunk.get("embedding_ready_text") or ""),
        str(chunk.get("text") or ""),
        str(chunk.get("section_title") or ""),
    ]
    for field in ("drugs", "imaging", "lab_tests", "procedures"):
        vals = chunk.get(field) or []
        if isinstance(vals, list):
            parts.extend(str(v) for v in vals[:8])
    blob = _norm_q(" ".join(parts))
    q = _norm_q(query)
    score = 0.0
    if q and q in blob:
        score += 0.35
    for tok in tokens:
        if tok in blob:
            score += 0.08 if len(tok) >= 5 else 0.05
    return min(score, 1.0)


def _intent_boost(chunk: dict[str, Any], intents: list[str]) -> float:
    if not intents:
        return 0.0
    ctype = str(chunk.get("chunk_type") or "").strip().lower()
    group = _TYPE_TO_GROUP.get(ctype, "")
    tags = set(_TYPE_INTENT_TAGS.get(ctype, ()))
    boost = 0.0
    for key in intents:
        spec = INTENT_SPECS.get(key) or {}
        if group in (spec.get("sections") or ()):
            boost = max(boost, 0.55)
        if tags.intersection(spec.get("tags") or ()):
            boost = max(boost, 0.35)
    return boost


def _merge_score(
    *,
    cosine: float,
    lex: float,
    intent: float,
) -> float:
    w_vec = float(os.environ.get("PROTOCOL_SEMANTIC_VEC_W", "0.6"))
    w_intent = float(os.environ.get("PROTOCOL_SEMANTIC_INTENT_W", "0.2"))
    w_lex = float(os.environ.get("PROTOCOL_SEMANTIC_LEX_W", "0.2"))
    total = w_vec + w_intent + w_lex
    if total <= 0:
        total = 1.0
    return (w_vec * cosine + w_intent * intent + w_lex * lex) / total


def _embed_query(query: str) -> list[float] | None:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    import rag_server as rs

    model = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview").strip()
    try:
        return rs._gemini_embed_one(model, query[:8000], "retrieval_query")
    except Exception:
        return None


def _cosine_score(chunk: dict[str, Any], q_vec: list[float]) -> float:
    emb = chunk.get("embedding")
    if not isinstance(emb, list) or len(emb) < 8:
        return 0.0
    try:
        import numpy as np

        q = np.asarray([float(x) for x in q_vec], dtype=np.float32)
        v = np.asarray([float(x) for x in emb], dtype=np.float32)
        qn = float(np.linalg.norm(q))
        vn = float(np.linalg.norm(v))
        if qn < 1e-9 or vn < 1e-9:
            return 0.0
        return float(np.dot(q / qn, v / vn))
    except Exception:
        return 0.0


def _attach_global_indices(chunks: list[dict[str, Any]]) -> None:
    from clinical_knowledge.vector_index import global_index_for_chunk_id

    for ch in chunks:
        if ch.get("_global_index") is not None:
            continue
        gid = global_index_for_chunk_id(ch.get("chunk_id"))
        if gid is not None:
            ch["_global_index"] = gid


def _build_vector_hits(
    protocol_chunks: list[dict[str, Any]],
    q_vec: list[float] | None,
    *,
    top_k: int,
) -> dict[int, float]:
    from clinical_knowledge.vector_index import (
        cosine_for_global_index,
        index_stats,
        search_scoped_with_scores,
    )

    vector_hits: dict[int, float] = {}
    if not q_vec:
        return vector_hits

    for i, ch in enumerate(protocol_chunks):
        cos = 0.0
        if isinstance(ch.get("embedding"), list):
            cos = _cosine_score(ch, q_vec)
        elif ch.get("_global_index") is not None:
            got = cosine_for_global_index(int(ch["_global_index"]), q_vec)
            if got is not None:
                cos = got
        if cos > 0:
            vector_hits[i] = max(vector_hits.get(i, 0.0), cos)

    if vector_hits or not index_stats().get("loaded"):
        return vector_hits

    allowed = {
        int(ch["_global_index"])
        for ch in protocol_chunks
        if ch.get("_global_index") is not None
    }
    if not allowed:
        return vector_hits

    global_to_local = {
        int(ch["_global_index"]): i
        for i, ch in enumerate(protocol_chunks)
        if ch.get("_global_index") is not None
    }
    for global_i, score in search_scoped_with_scores(
        q_vec,
        allowed,
        top_k=max(top_k * 3, 24),
    ):
        local_i = global_to_local.get(global_i)
        if local_i is not None:
            vector_hits[local_i] = max(vector_hits.get(local_i, 0.0), score)
    return vector_hits


def _load_protocol_chunks(path: str) -> list[dict[str, Any]]:
    """Чанки одного протокола: lazy store (semantic) или RAM."""
    import rag_server as rs
    from clinical_knowledge.lazy_rag_config import lazy_chunk_store_enabled

    norm = path.replace("\\", "/").strip()
    max_c = int(os.environ.get("PROTOCOL_SEMANTIC_MAX_CHUNKS", "256"))
    if lazy_chunk_store_enabled() or not rs._chunks:
        store = rs._ensure_lazy_chunk_store()
        if store:
            rows = store.get_chunks_for_path(norm, max_chunks=max_c, semantic=True)
            if rows:
                _attach_global_indices(rows)
                return rows
    rows = rs.get_rich_chunks_for_path(norm)
    if rows:
        out = rows[:max_c]
        _attach_global_indices(out)
        return out
    if rs._chunks:
        allowed = rs._chunk_indices_for_path_allowlist(frozenset({norm}))
        out: list[dict[str, Any]] = []
        for global_i in sorted(allowed):
            if 0 <= global_i < len(rs._chunks):
                ch = dict(rs._chunks[global_i])
                ch["_global_index"] = global_i
                out.append(ch)
            if len(out) >= max_c:
                break
        return out
    return []


def _chunk_type_boost(chunk_type: str, intents: list[str], *, query: str = "") -> float:
    ctype = str(chunk_type or "").strip().lower()
    boost = 0.0
    if is_drug_focus_query(query, intents) and ctype in DRUG_CHUNK_TYPES:
        boost += 0.22
    elif "treatment" in intents and ctype in DRUG_CHUNK_TYPES:
        boost += 0.12
    return boost


def _should_drop_nav_item(item: dict[str, Any], *, query: str, intents: list[str]) -> bool:
    lead = str(item.get("lead") or "")
    body = str(item.get("body") or "")
    combined = f"{lead} {body}".strip()
    if is_table_noise_text(combined):
        return True
    if is_drug_focus_query(query, intents) and str(item.get("section_id") or "") != "treatment":
        return True
    if intents and not is_drug_focus_query(query, intents):
        allowed = allowed_sections_for_intents(intents, query=query)
        if allowed and str(item.get("section_id") or "") not in allowed:
            return True
    return False


def search_protocol_semantic(
    path: str,
    query: str,
    *,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Семантический поиск по чанкам одного протокола."""
    from clinical_knowledge.protocol_links import normalize_protocol_path
    from clinical_knowledge.vector_index import (
        ensure_index_loaded,
        index_stats,
        vector_index_enabled,
    )

    pth = normalize_protocol_path(path.strip())
    q = (query or "").strip()
    if not pth or len(q) < 2:
        return {"ok": False, "reason": "empty_query_or_path", "path": pth, "query": q}

    import rag_server as rs

    rs._require_rag_loaded(max_wait_sec=max(3.0, float(os.environ.get("RAG_LOAD_WAIT_LITE_SEC", "28"))))

    if vector_index_enabled():
        ensure_index_loaded(rs._chunks)

    protocol_chunks = _load_protocol_chunks(pth)
    if not protocol_chunks:
        return {
            "ok": False,
            "reason": "no_chunks_for_path",
            "path": pth,
            "query": q,
            "vector_enabled": vector_index_enabled(),
        }

    k = top_k or int(os.environ.get("PROTOCOL_SEMANTIC_TOP_K", "12"))
    intents = detect_query_intents(q)
    intent_k, max_per_group = intent_result_limits(intents, query=q)
    if top_k is None:
        k = intent_k
    tokens = _query_tokens(q, intents)

    q_vec: list[float] | None = None
    has_inline_emb = any(isinstance(c.get("embedding"), list) for c in protocol_chunks)
    mapped_n = sum(1 for c in protocol_chunks if c.get("_global_index") is not None)
    use_vector = vector_index_enabled() and (
        has_inline_emb or mapped_n > 0 or index_stats().get("loaded")
    )
    if use_vector:
        q_vec = _embed_query(q)

    vector_hits = _build_vector_hits(protocol_chunks, q_vec, top_k=k)

    ranked: list[tuple[float, dict[str, Any]]] = []
    seen_fp: set[str] = set()
    for i, ch in enumerate(protocol_chunks):
        cosine = vector_hits.get(i, 0.0)
        lex = _lex_score(ch, q, tokens)
        intent = _intent_boost(ch, intents) + _chunk_type_boost(
            str(ch.get("chunk_type") or ch.get("kind") or ""), intents, query=q
        )
        if cosine <= 0 and lex <= 0 and intent <= 0:
            continue
        score = _merge_score(cosine=cosine, lex=lex, intent=intent)
        item = format_rich_chunk_nav_item(ch, query=q, intents=intents)
        if not item:
            continue
        if _should_drop_nav_item(item, query=q, intents=intents):
            continue
        fp = re.sub(r"\s+", " ", (item.get("lead") or "").lower())[:160]
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        item["score"] = round(score, 4)
        item["score_parts"] = {
            "cosine": round(cosine, 4),
            "lex": round(lex, 4),
            "intent": round(intent, 4),
        }
        ranked.append((score, item))

    ranked.sort(key=lambda row: row[0], reverse=True)
    items = [item for _, item in ranked[:k]]
    view = build_view_from_items(items, max_per_group=max_per_group)
    labels = {gid: label for gid, label in _GROUP_ORDER}
    focus = "drugs" if is_drug_focus_query(q, intents) else (intents[0] if intents else "")
    return {
        "ok": True,
        "path": pth,
        "query": q,
        "mode": "semantic" if any(s > 0 for s in vector_hits.values()) else "lexical",
        "vector_enabled": bool(use_vector and index_stats().get("loaded")),
        "mapped_chunks": mapped_n,
        "has_inline_emb": has_inline_emb,
        "intents": intents,
        "focus": focus,
        "match_count": len(items),
        "items": items,
        "view": {**view, "section_labels": labels},
    }


def _overview_cache_dir() -> Path:
    raw = (os.environ.get("PROTOCOL_OVERVIEW_CACHE_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return ROOT / "data" / "ml" / "protocol_overview_cache"


def _overview_cache_key(path: str, query: str) -> str:
    raw = f"{path.strip()}|{_norm_q(query)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _load_overview_disk_cache(key: str) -> dict[str, Any] | None:
    p = _overview_cache_dir() / f"{key}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _save_overview_disk_cache(key: str, payload: dict[str, Any]) -> None:
    try:
        d = _overview_cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _parse_overview_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return data


def _build_overview_prompt(
    *,
    query: str,
    title: str,
    path: str,
    sources: list[dict[str, Any]],
) -> str:
    lines = [
        ("Ты клинический ассистент. По фрагментам клинического протокола Минздрава РБ "
        "дай краткий ответ врачу на приёме."),
        "Правила:",
        "- Только факты из фрагментов ниже, без домыслов.",
        "- 3-5 коротких пунктов, каждый - одно предложение.",
        "- Укажи source_idx (0-based) для каждого пункта.",
        "- Если данных недостаточно, напиши это в summary.",
        "- Ответ строго JSON без markdown.",
        "",
        f"Вопрос: {query}",
        f"Протокол: {title or path}",
        "",
        "Фрагменты:",
    ]
    for i, src in enumerate(sources):
        page = src.get("page")
        lead = str(src.get("lead") or "").strip()
        body = str(src.get("body") or "").strip()
        text = lead if not body or body == lead else f"{lead} {body}"
        text = re.sub(r"\s+", " ", text)[:900]
        lines.append(f"[{i}] стр.{page or '?'}: {text}")
    lines.extend(
        [
            "",
            'Формат JSON: {"summary":"1-2 предложения","points":[{"text":"...","source_idx":0}]}',
        ]
    )
    return "\n".join(lines)


def build_protocol_overview(
    path: str,
    query: str,
    *,
    search_payload: dict[str, Any] | None = None,
    title: str = "",
) -> dict[str, Any]:
    """AI Overview по top-K чанкам протокола (Gemini Flash)."""
    from clinical_knowledge.protocol_links import normalize_protocol_path

    pth = normalize_protocol_path(path.strip())
    q = (query or "").strip()
    if not pth or len(q) < 2:
        return {"ok": False, "reason": "empty_query_or_path", "path": pth, "query": q}

    cache_key = _overview_cache_key(pth, q)
    now = time.time()
    mem = _OVERVIEW_CACHE.get(cache_key)
    if mem and now - mem[0] < _OVERVIEW_CACHE_TTL_SEC:
        out = dict(mem[1])
        out["cache_hit"] = True
        return out

    disk = _load_overview_disk_cache(cache_key)
    if disk and disk.get("ok"):
        _OVERVIEW_CACHE[cache_key] = (now, disk)
        out = dict(disk)
        out["cache_hit"] = True
        return out

    search_out = search_payload or search_protocol_semantic(pth, q, top_k=8)
    if not search_out.get("ok"):
        return {
            "ok": False,
            "reason": search_out.get("reason") or "search_failed",
            "path": pth,
            "query": q,
            "search": search_out,
        }

    sources = list(search_out.get("items") or [])[:8]
    if not sources:
        return {
            "ok": False,
            "reason": "no_sources",
            "path": pth,
            "query": q,
            "search": search_out,
        }

    import rag_server as rs

    model = rs.get_gemini()
    if model is None:
        return {
            "ok": False,
            "reason": "gemini_unavailable",
            "path": pth,
            "query": q,
            "search": search_out,
            "sources": sources,
        }

    prompt = _build_overview_prompt(query=q, title=title, path=pth, sources=sources)
    # gemini-2.5-flash расходует часть бюджета на «мышление»; при 1200 на длинном промпте
    # (summary + до 6 points) вывод обнулялся -> поднимаем дефолт, чтобы хватало на ответ.
    max_out = int(os.environ.get("PROTOCOL_OVERVIEW_MAX_TOKENS", "3000"))
    try:
        resp = rs.generate_gemini_consult_review_synthesize(model, prompt, max_out=max_out)
        txt = rs._extract_gemini_text(resp)
    except Exception as exc:
        return {
            "ok": False,
            "reason": "synthesis_error",
            "error": str(exc)[:300],
            "path": pth,
            "query": q,
            "search": search_out,
            "sources": sources,
        }

    parsed = _parse_overview_json(txt)
    points: list[dict[str, Any]] = []
    summary = ""
    if parsed:
        summary = str(parsed.get("summary") or "").strip()
        raw_points = parsed.get("points") or []
        if isinstance(raw_points, list):
            for row in raw_points[:6]:
                if not isinstance(row, dict):
                    continue
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                idx = row.get("source_idx")
                src = None
                if isinstance(idx, int) and 0 <= idx < len(sources):
                    src = sources[idx]
                points.append(
                    {
                        "text": text,
                        "source_idx": idx,
                        "page": (src or {}).get("page"),
                        "section_id": (src or {}).get("section_id"),
                        "lead": (src or {}).get("lead"),
                    }
                )

    out = {
        "ok": bool(points or summary),
        "path": pth,
        "query": q,
        "summary": summary,
        "points": points,
        "sources": [
            {
                "lead": s.get("lead"),
                "page": s.get("page"),
                "section_id": s.get("section_id"),
                "score": s.get("score"),
            }
            for s in sources
        ],
        "search": {
            "mode": search_out.get("mode"),
            "match_count": search_out.get("match_count"),
            "vector_enabled": search_out.get("vector_enabled"),
        },
        "synthesis_ok": bool(points or summary),
        "cache_hit": False,
    }
    if out["ok"]:
        _OVERVIEW_CACHE[cache_key] = (now, out)
        _save_overview_disk_cache(cache_key, out)
    return out
