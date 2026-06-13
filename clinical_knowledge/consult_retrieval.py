"""Подбор релевантных PDF протоколов для анализа КЗ (только МКБ + matched cards + рубрики)."""
from __future__ import annotations

from typing import Any


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


def _path_spine_bladder_mismatch(sp: str, icd_roots: set[str]) -> bool:
    """M54* + путь КП мочевого пузыря без позвоночника - чужой протокол."""
    if not icd_roots & _SPINE_ICD_ROOTS:
        return False
    low = sp.lower()
    bladder = any(n in low for n in ("мочевого", "мочев", "пузыр"))
    spine = any(n in low for n in ("позвоноч", "радикул", "люмбо", "нейрохирург", "ишиас"))
    return bladder and not spine


def consult_target_protocol_paths(
    *,
    merged_icd: list[str] | None,
    diag_icd: list[str] | None,
    clinical_rules: dict[str, Any] | None,
    specialty_slugs: list[str] | None,
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
    slugs = set(specialty_slugs or [])

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
            if _path_spine_bladder_mismatch(sp, icd_roots):
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
