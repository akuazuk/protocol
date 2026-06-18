"""Подбор релевантных PDF протоколов для анализа КЗ (только МКБ + matched cards + рубрики)."""
from __future__ import annotations

from typing import Any

from clinical_knowledge.rule_family_gates import expand_specialty_slugs_for_icd, expand_specialty_slugs_for_clinical_text


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


def _path_norm(sp: str) -> str:
    return (sp or "").replace("\\", "/").strip()


def _card_icd_roots(card: dict[str, Any]) -> set[str]:
    icd = list(card.get("icd10_all") or card.get("icd10_primary") or [])
    return {_icd_root(str(x)) for x in icd if x}


def _icd_overlap_score(card: dict[str, Any], icd_roots: set[str], icd_full: set[str]) -> float:
    if not icd_roots and not icd_full:
        return 0.0
    card_icd = [str(x).upper() for x in (card.get("icd10_all") or card.get("icd10_primary") or [])]
    card_roots = {_icd_root(c) for c in card_icd}
    overlap = icd_roots & card_roots
    score = 0.0
    if overlap:
        score += 40.0 + 8.0 * len(overlap)
    for c in icd_full:
        for cc in card_icd:
            if c.startswith(_icd_root(cc)) or cc.startswith(_icd_root(c)):
                score += 18.0
                break
    return score


_SPINE_ICD_ROOTS = frozenset({"M51", "M53", "M54"})


def _path_spine_domain_mismatch(sp: str, icd_roots: set[str]) -> bool:
    """M54* + путь КП мочевого пузыря или опухолей без позвоночника - чужой протокол."""
    if not icd_roots & _SPINE_ICD_ROOTS:
        return False
    low = sp.lower()
    bladder = any(n in low for n in ("мочевого", "мочев", "пузыр"))
    neoplasm = any(n in low for n in ("опухол", "новообраз", "онколог", "злокач"))
    spine = any(n in low for n in ("позвоноч", "радикул", "люмбо", "нейрохирург", "ишиас"))
    return (bladder or neoplasm) and not spine


def consult_target_protocol_paths(
    *,
    merged_icd: list[str] | None,
    diag_icd: list[str] | None,
    clinical_rules: dict[str, Any] | None,
    specialty_slugs: list[str] | None,
    consult_text: str | None = None,
    max_paths: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Список source_path PDF, по которым разрешён RAG для КЗ."""
    from .loader import load_protocol_cards_registry

    limit = max_paths
    if limit is None:
        import os

        limit = max(2, min(10, int(os.environ.get("CONSULT_REVIEW_MAX_PROTOCOL_PATHS", "6"))))

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

    if isinstance(clinical_rules, dict):
        for mp in clinical_rules.get("matched_protocols") or []:
            if not isinstance(mp, dict):
                continue
            sp = mp.get("source_path")
            if sp:
                add(str(sp), "matched_protocol_card")

    diag = [str(x).upper() for x in (diag_icd or []) if x]
    merged = [str(x).upper() for x in (merged_icd or []) if x]
    primary_icd = diag or merged
    icd_roots = {_icd_root(c) for c in primary_icd}
    icd_full = set(primary_icd)
    slugs = expand_specialty_slugs_for_icd(set(specialty_slugs or []), primary_icd)
    slugs = expand_specialty_slugs_for_clinical_text(slugs, consult_text or "")

    if primary_icd:
        # Лучший балл на КАЖДЫЙ PDF (а не на каждую секцию), иначе PDF с многими
        # секциями вытесняет другие релевантные протоколы из топа.
        best_by_path: dict[str, float] = {}
        for card in load_protocol_cards_registry():
            sp = _path_norm(str(card.get("source_path") or ""))
            if not sp:
                continue
            if slugs and card.get("specialty_slug") not in slugs:
                continue
            sc = _icd_overlap_score(card, icd_roots, icd_full)
            if sc <= 0:
                continue
            if _path_spine_domain_mismatch(sp, icd_roots):
                continue
            if slugs and card.get("specialty_slug") in slugs:
                sc += 12.0
            if sc > best_by_path.get(sp, 0.0):
                best_by_path[sp] = sc
        scored = sorted(best_by_path.items(), key=lambda x: (-x[1], x[0]))
        for sp, sc in scored:
            if sc >= 18.0:
                add(sp, "icd_registry_match")
            if len(paths) >= limit:
                break

    if not paths and primary_icd:
        icd_roots = {_icd_root(c) for c in primary_icd}
        if icd_roots & {"J06", "J00", "J02", "J03", "J04", "J05"} or any(
            r.startswith("J06") for r in icd_roots
        ):
            urti_markers = ("орви", "респиратор", "вирусн", "орз", "орви", "инфекц")
            for card in load_protocol_cards_registry():
                if slugs and card.get("specialty_slug") not in slugs:
                    continue
                sp = _path_norm(str(card.get("source_path") or ""))
                if not sp or sp in seen:
                    continue
                blob = f"{card.get('title') or ''} {sp}".lower()
                if any(m in blob for m in urti_markers):
                    add(sp, "icd_title_urti_fallback")

    meta: dict[str, Any] = {
        "primary_icd": primary_icd[:12],
        "specialty_slugs": sorted(slugs),
        "path_sources": sources,
        "strict": bool(paths),
    }
    return paths[:limit], meta


def filter_retrieval_rows_by_paths(
    rows: list[dict[str, Any]],
    allowed_paths: list[str] | None,
) -> list[dict[str, Any]]:
    if not allowed_paths:
        return rows
    allow = {_path_norm(p) for p in allowed_paths}
    if not allow:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        p = _path_norm(str(row.get("path") or ""))
        if p in allow:
            out.append(row)
    return out


def filter_retrieval_by_category_slugs(
    rows: list[dict[str, Any]],
    allowed_slugs: list[str] | None,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Отбрасывает чанки вне рубрики врача КЗ (напр. акушерство при неврологии)."""
    if not allowed_slugs or not strict:
        return rows
    allow = {s.strip() for s in allowed_slugs if s and s.strip()}
    if not allow:
        return rows
    out = [r for r in rows if (r.get("category") or "").strip() in allow]
    return out if out else rows


def unify_consult_protocol_paths(
    *,
    target_paths: list[str] | None,
    rules_paths: list[str] | None,
    rag_paths: list[str] | None,
    max_paths: int | None = None,
) -> list[str]:
    """Единый список PDF: target (МКБ/cards) + rules + RAG без дублей."""
    import os

    limit = max_paths or max(2, min(10, int(os.environ.get("CONSULT_REVIEW_MAX_PROTOCOL_PATHS", "6"))))
    out: list[str] = []
    seen: set[str] = set()
    for group in (target_paths or [], rules_paths or [], rag_paths or []):
        for p in group:
            n = _path_norm(str(p))
            if n and n not in seen:
                seen.add(n)
                out.append(n)
            if len(out) >= limit:
                return out
    return out


def supplement_retrieval_from_rich_chunks(
    retrieved: list[dict[str, Any]],
    *,
    paths: list[str],
    icd_codes: list[str],
    get_chunks: Any,
    query: str = "",
) -> list[dict[str, Any]]:
    """Typed-дополнение RAG: diagnostics / treatment / dispensary из rich-чанков."""
    if not paths or not get_chunks:
        return retrieved
    from clinical_knowledge.protocol_practical_lite import _chunk_type, _pick_chunks

    seen_keys: set[str] = {
        f"{r.get('path')}|{r.get('page_from')}|{(r.get('text') or '')[:60]}"
        for r in retrieved
    }
    extra: list[dict[str, Any]] = []
    q = query or " ".join(icd_codes or [])

    type_passes: tuple[tuple[str, tuple[str, ...], int], ...] = (
        ("diagnostics", ("diagnostics", "criteria_block", "table"), 4),
        ("treatment", ("treatment", "pharmacotherapy", "drug_list"), 4),
        ("monitoring", ("dispensary", "prevention"), 2),
    )

    for path in paths[:6]:
        chunks = get_chunks(path) or []
        if not chunks:
            continue
        for label, ctypes, lim in type_passes:
            picked = _pick_chunks(chunks, q, icd_codes, limit=lim, chunk_types=ctypes)
            for ch in picked:
                text = (ch.get("text") or "").strip()
                if len(text) < 40:
                    continue
                key = f"{path}|{ch.get('page_from')}|{text[:60]}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                extra.append({
                    "path": path,
                    "text": text[:1200],
                    "score": 0.72,
                    "chunk_type": _chunk_type(ch),
                    "section_title": ch.get("section_title") or label,
                    "page_from": ch.get("page_from"),
                    "page_to": ch.get("page_to"),
                    "typed_retrieve": True,
                })

    if not extra:
        return retrieved
    return list(retrieved) + extra
