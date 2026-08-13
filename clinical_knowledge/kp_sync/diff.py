"""Сверка сайта МЗ с локальным манифестом. Файлы с диска не удаляем."""
from __future__ import annotations

import hashlib
import re
from typing import Any

POST_RE = re.compile(
    r"(?:от\s+)?(\d{1,2}\.\d{1,2}\.\d{4})\s*(?:г\.?\s*)?№\s*(\d+)",
    re.I,
)
POST_ALT_RE = re.compile(
    r"пост[.\s_]*МЗ[.\s_]*РБ[.\s_]*от[.\s_]*(\d{1,2}\.\d{1,2}\.\d{4}).*?№\s*(\d+)",
    re.I,
)


def parse_post_from_name(name: str) -> tuple[str, str] | None:
    blob = name or ""
    m = POST_RE.search(blob) or POST_ALT_RE.search(blob)
    if not m:
        return None
    day, month, year = m.group(1).split(".")
    date = f"{int(day):02d}.{int(month):02d}.{year}"
    return date, m.group(2)


def _norm_name(name: str) -> str:
    n = (name or "").lower().replace("ё", "е")
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"\.(pdf|doc|docx|zip|rar)$", "", n, flags=re.I)
    return n.strip()


def _local_index(local: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for row in local:
        rel = str(row.get("relative_path") or row.get("source_path") or "").replace("\\", "/")
        fn = str(row.get("filename") or "")
        if not rel and fn:
            slug = str(row.get("slug") or row.get("category") or "unknown")
            rel = f"minzdrav_protocols/{slug}/{fn}"
        if not rel:
            continue
        rec = dict(row)
        rec["relative_path"] = rel
        rec.setdefault("filename", rel.rsplit("/", 1)[-1])
        by_path[rel] = rec
    return by_path


def _match_local(
    site: dict[str, str],
    by_path: dict[str, dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    by_post: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    rel = site["relative_path"]
    if rel in by_path:
        return by_path[rel]
    fn = site["filename"]
    if fn in by_name:
        return by_name[fn]
    nn = _norm_name(fn)
    for loc in by_path.values():
        if _norm_name(str(loc.get("filename") or "")) == nn:
            return loc
    post = parse_post_from_name(fn) or parse_post_from_name(site.get("url") or "")
    if post and post in by_post:
        return by_post[post]
    return None


def diff_catalog(
    site_docs: list[dict[str, Any]],
    local_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    """added / updated / unchanged / superseded / needs_sha.

    updated: тот же path/имя/пост, но sha256 отличается (если оба известны).
    superseded: локальный файл, которого больше нет на сайте (не удалять).
    """
    by_path = _local_index(local_docs)
    by_name: dict[str, dict[str, Any]] = {}
    by_post: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in by_path.values():
        fn = str(rec.get("filename") or "")
        if fn:
            by_name.setdefault(fn, rec)
        post = parse_post_from_name(fn) or parse_post_from_name(
            str(rec.get("display_title") or "")
        )
        if post:
            by_post.setdefault(post, rec)

    added: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    matched_local: set[str] = set()

    for site in site_docs:
        loc = _match_local(site, by_path, by_name, by_post)
        if loc is None:
            added.append(dict(site))
            continue
        loc_path = str(loc.get("relative_path") or "")
        matched_local.add(loc_path)
        site_sha = str(site.get("sha256") or "")
        loc_sha = str(loc.get("sha256") or "")
        renamed = loc_path != site["relative_path"]
        if site_sha and loc_sha and site_sha != loc_sha:
            rec = {**site, "previous_path": loc_path, "action": "updated"}
            updated.append(rec)
        elif renamed:
            rec = {**site, "previous_path": loc_path, "action": "renamed"}
            updated.append(rec)
        elif site_sha and loc_sha and site_sha == loc_sha:
            unchanged.append({**site, "local_path": loc_path, "action": "unchanged"})
        else:
            unchanged.append({**site, "local_path": loc_path, "action": "kept"})

    superseded: list[dict[str, Any]] = []
    for rel, loc in by_path.items():
        if rel in matched_local:
            continue
        rec = dict(loc)
        rec["action"] = "superseded"
        rec["status"] = "superseded"
        superseded.append(rec)

    return {
        "site_count": len(site_docs),
        "local_count": len(by_path),
        "added": added,
        "updated": updated,
        "unchanged": unchanged,
        "superseded": superseded,
        "changed_paths": [
            str(r.get("relative_path") or "")
            for r in added + updated
            if r.get("relative_path")
        ],
        "kp_corpus_generation": hashlib.sha256(
            "|".join(
                sorted(
                    str(r.get("relative_path") or "")
                    for r in added + updated
                    if r.get("relative_path")
                )
            ).encode("utf-8")
        ).hexdigest()[:16],
    }
