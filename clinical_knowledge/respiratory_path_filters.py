"""Фильтры путей PDF для ОРВИ / кашель+лихорадка (устраняют ложные КП из ICD lookup)."""
from __future__ import annotations

import re
from pathlib import Path

_URTI_ICD_PREFIXES = ("J00", "J01", "J02", "J03", "J04", "J05", "J06", "J11")

_URTI_PATH_GOOD = (
    "орви",
    "орз",
    "респиратор",
    "вирусн",
    "бронхит",
    "грипп",
    "гриpp",
    "инфекц",
    "простуд",
    "трахеит",
    "фаринг",
    "ларинг",
    "тонзилл",
    "риносинус",
)

_URTI_PATH_WRONG = (
    "саркоид",
    "астм",
    "хобл",
    "обструктивн",
    "бронхоэктат",
    "интерстициаль",
    "кожи и подкож",
    "кожи_и_подкож",
    "гепатит",
    "вич",
    "сифил",
    "почек",
    "паллиат",
    "иммунодефиц",
    "туберкул",
    "микобактер",
    "онколог",
    "трансплант",
    "реабилит",
    "дистресс-синдром",
    "ардс",
)


def _blob(path: str, title: str = "") -> str:
    stem = Path(path).stem.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", f"{path} {title} {stem}".lower()).strip()


def is_urti_icd_only(codes: list[str] | None) -> bool:
    norm = [str(c or "").upper().strip() for c in (codes or []) if c]
    if not norm:
        return False
    return all(any(c.startswith(p) for p in _URTI_ICD_PREFIXES) for c in norm)


def path_has_urti_markers(path: str, title: str = "") -> bool:
    b = _blob(path, title)
    return any(m in b for m in _URTI_PATH_GOOD)


def path_has_respiratory_wrong_markers(path: str, title: str = "") -> bool:
    b = _blob(path, title)
    return any(m in b for m in _URTI_PATH_WRONG)


def urti_path_rank(path: str, title: str = "") -> float:
    """Выше - лучше для ОРВИ/кашель+лихорадка."""
    b = _blob(path, title)
    if path_has_respiratory_wrong_markers(path, title):
        return -100.0
    score = 0.0
    for m in _URTI_PATH_GOOD:
        if m in b:
            score += 12.0 if m in ("орви", "орз", "респиратор", "вирусн") else 6.0
    if "в-нас" in b or "взр" in b:
        score += 4.0
    if "д-нас" in b or "дет" in b:
        score -= 2.0
    return score


def filter_paths_for_respiratory_context(
    paths: list[str],
    *,
    limit: int,
    titles: dict[str, str] | None = None,
) -> list[str]:
    """Оставить релевантные респираторные КП; отсечь саркоид/астму/ХОБЛ и т.п."""
    titles = titles or {}
    if not paths:
        return []
    ranked = sorted(
        paths,
        key=lambda p: (-urti_path_rank(p, titles.get(p, "")), p),
    )
    good = [p for p in ranked if urti_path_rank(p, titles.get(p, "")) > 0]
    if good:
        return good[:limit]
    soft = [p for p in ranked if not path_has_respiratory_wrong_markers(p, titles.get(p, ""))]
    return soft[:limit]


def scan_registry_urti_paths(
    *,
    specialty_slugs: set[str] | list[str] | None,
    limit: int = 4,
) -> list[str]:
    """Резерв: КП по заголовку с маркерами ОРВИ/респиратор."""
    try:
        from clinical_knowledge.loader import load_protocol_cards_registry
    except Exception:
        return []
    slugs = {str(s).strip() for s in (specialty_slugs or []) if s}
    out: list[tuple[float, str]] = []
    for card in load_protocol_cards_registry():
        if slugs and card.get("specialty_slug") not in slugs:
            continue
        sp = str(card.get("source_path") or "").strip()
        if not sp:
            continue
        title = str(card.get("title") or "")
        if not path_has_urti_markers(sp, title):
            continue
        if path_has_respiratory_wrong_markers(sp, title):
            continue
        out.append((urti_path_rank(sp, title), sp))
    out.sort(key=lambda x: (-x[0], x[1]))
    return [p for _s, p in out[:limit]]
