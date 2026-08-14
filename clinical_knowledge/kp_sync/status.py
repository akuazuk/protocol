"""Свежий kp_sync_*.json и сводки корпуса для вкладки Протоколы МЗ (без ПДн)."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from clinical_knowledge.kp_sync.diff import parse_post_from_name

ROOT = Path(__file__).resolve().parents[2]
SYNC_FILE_RE = re.compile(r"kp_sync_(\d{4}-\d{2}-\d{2})\.json$")
YEAR_RE = re.compile(r"(?:^|[^\d])((?:19|20)\d{2})(?:[^\d]|$)")

SLUG_RU = {
    "akusherstvo-ginekologiya": "Акушерство и гинекология",
    "allergologiya-immunologiya": "Аллергология и иммунология",
    "anesteziologiya-reanimatologiya": "Анестезиология",
    "bolezni-sistemy-krovoobrashcheniya": "Кардиология",
    "dermatovenerologiya": "Дерматовенерология",
    "endokrinologiya-narusheniya-obmena-veshchestv": "Эндокринология",
    "gastroenterologiya": "Гастроэнтерология",
    "gematologiya": "Гематология",
    "infektsionnye-zabolevaniya": "Инфекции",
    "khirurgiya": "Хирургия",
    "nefrologiya": "Нефрология",
    "nevrologiya-neyrokhirurgiya": "Неврология",
    "novoobrazovaniya": "Онкология",
    "oftalmologiya": "Офтальмология",
    "otorinolaringologiya": "ЛОР",
    "palliativnaya-pomoshch": "Паллиатив",
    "psikhiatriya-narkologiya": "Психиатрия",
    "pulmonologiya-ftiziatriya": "Пульмонология",
    "revmatologiya": "Ревматология",
    "stomatologiya": "Стоматология",
    "transplantatsiya-organov-i-tkaney": "Трансплантация",
    "travmatologiya-ortopediya": "Травматология",
    "urologiya": "Урология",
    "zabolevaniya-perinatalnogo-perioda": "Перинатология",
}

KIND_RU = {
    "clinical": "Клинические",
    "rehab": "Реабилитация",
    "algorithm": "Алгоритмы",
    "admin": "Административные",
}


def sync_dirs() -> list[Path]:
    raw = [
        os.environ.get("KP_SYNC_DIR") or "",
        os.environ.get("PROTOCOL_CORPUS_ROOT") or "",
        "/var/data/protocol_corpus/_sync",
        str(ROOT / "data" / "kp_sync"),
    ]
    out: list[Path] = []
    for item in raw:
        item = str(item).strip()
        if not item:
            continue
        p = Path(item)
        if p.name != "_sync" and (p / "_sync").is_dir():
            p = p / "_sync"
        out.append(p)
    return out


def catalog_paths() -> list[Path]:
    raw = [
        os.environ.get("PROTOCOL_CATALOG_PATH") or "",
        str(Path(os.environ.get("PROTOCOL_CORPUS_ROOT") or "") / "protocol_catalog.jsonl"),
        "/var/data/protocol_corpus/protocol_catalog.jsonl",
        str(ROOT / "data" / "protocol_catalog.jsonl"),
    ]
    return [Path(p) for p in raw if p and p not in {"/protocol_catalog.jsonl"}]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_latest_kp_sync(sync_dir: Path | None = None) -> dict[str, Any] | None:
    files = _sync_files(sync_dir)
    if not files:
        return None
    path = max(files, key=lambda p: p.stat().st_mtime)
    data = _read_json(path)
    if not data:
        return None
    data["_source_file"] = path.name
    data["_sync_day"] = _day_from_name(path.name) or ""
    return data


def load_all_kp_syncs(sync_dir: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _sync_files(sync_dir):
        data = _read_json(path)
        if not data:
            continue
        day = _day_from_name(path.name) or ""
        if not day or day in seen:
            continue
        seen.add(day)
        data["_source_file"] = path.name
        data["_sync_day"] = day
        rows.append(data)
    rows.sort(key=lambda r: str(r.get("_sync_day") or ""))
    return rows


def load_catalog_rows(catalog_path: Path | None = None) -> list[dict[str, Any]]:
    paths = [catalog_path] if catalog_path is not None else catalog_paths()
    for path in paths:
        if path is None or not path.is_file():
            continue
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    rows.append(rec)
        except OSError:
            continue
        if rows:
            return rows
    return []


def post_meta(*blobs: str) -> dict[str, str]:
    """Дата/номер поста МЗ из имени или заголовка."""
    for blob in blobs:
        parsed = parse_post_from_name(blob or "")
        if parsed:
            dotted, number = parsed
            iso = _dotted_to_iso(dotted)
            return {
                "post_date": dotted,
                "post_date_iso": iso or "",
                "post_number": number,
                "post_year": iso[:4] if iso else "",
            }
    year = ""
    for blob in blobs:
        m = YEAR_RE.search(blob or "")
        if m:
            year = m.group(1)
            break
    return {
        "post_date": "",
        "post_date_iso": f"{year}-01-01" if year else "",
        "post_number": "",
        "post_year": year,
    }


def public_kp_sync_payload(
    raw: dict[str, Any] | None,
    *,
    days: int = 30,
    history: list[dict[str, Any]] | None = None,
    catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    today = datetime.now(timezone.utc).date()
    hist = history if history is not None else load_all_kp_syncs()
    cat = catalog if catalog is not None else load_catalog_rows()
    if raw is None and hist:
        raw = hist[-1]

    if not raw:
        empty = _empty_payload(days)
        empty.update(_catalog_block(cat, today))
        empty["history"] = []
        empty["sync_periods"] = _zero_periods()
        return empty

    sync_day = str(raw.get("_sync_day") or "")
    added = _rows(raw.get("added") or [], sync_day)
    updated = _rows(raw.get("updated") or [], sync_day)
    superseded = _rows(raw.get("superseded") or [], sync_day)
    changed_n = len(raw.get("changed_paths") or added + updated)
    payload = {
        "ok": True,
        "status": str(raw.get("status") or "success"),
        "crawled_utc": raw.get("crawled_utc") or raw.get("applied_utc") or "",
        "source_file": raw.get("_source_file") or "",
        "sync_day": sync_day,
        "site_count": int(raw.get("site_count") or 0),
        "local_count": int(raw.get("local_count") or 0),
        "changed_n": changed_n,
        "added_n": len(raw.get("added") or []),
        "updated_n": len(raw.get("updated") or []),
        "superseded_n": len(raw.get("superseded") or []),
        "added": added,
        "updated": updated,
        "superseded": superseded,
        "pending_summaries": int(raw.get("pending_summaries") or 0),
        "kp_corpus_generation": raw.get("kp_corpus_generation") or "",
        "rescored_n": int(raw.get("rescored_n") or 0),
        "days": days,
    }
    payload["history"] = _history_series(hist)
    payload["sync_periods"] = _sync_periods(payload["history"], today)
    payload.update(_catalog_block(cat, today))
    return payload


def _sync_files(sync_dir: Path | None) -> list[Path]:
    dirs = [sync_dir] if sync_dir is not None else sync_dirs()
    files: list[Path] = []
    for d in dirs:
        if d is None or not d.is_dir():
            continue
        files.extend(sorted(d.glob("kp_sync_*.json")))
    return files


def _day_from_name(name: str) -> str:
    m = SYNC_FILE_RE.search(name or "")
    return m.group(1) if m else ""


def _dotted_to_iso(dotted: str) -> str:
    try:
        day, month, year = dotted.split(".")
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    except (ValueError, AttributeError):
        return ""


def _iso_to_dotted(iso: str) -> str:
    if not iso or len(iso) < 10:
        return iso or ""
    return f"{iso[8:10]}.{iso[5:7]}.{iso[:4]}"


def _parse_iso(iso: str) -> date | None:
    try:
        return date.fromisoformat((iso or "")[:10])
    except ValueError:
        return None


def _rows(items: list, sync_day: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in items or []:
        if not isinstance(rec, dict):
            continue
        filename = str(rec.get("filename") or "")
        rel = str(rec.get("relative_path") or rec.get("path") or "")
        meta = post_meta(filename, rel, str(rec.get("display_title") or ""))
        out.append(
            {
                "filename": filename,
                "slug": rec.get("slug") or rec.get("category") or rec.get("specialty_slug") or "",
                "slug_ru": SLUG_RU.get(
                    str(rec.get("slug") or rec.get("category") or rec.get("specialty_slug") or ""),
                    "",
                ),
                "relative_path": rel,
                "action": rec.get("action") or "",
                "alias_of": rec.get("alias_of") or "",
                "post_date": meta["post_date"],
                "post_date_iso": meta["post_date_iso"],
                "post_number": meta["post_number"],
                "post_year": meta["post_year"],
                "synced_on": sync_day,
            }
        )
    return out[:300]


def _history_series(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for raw in history:
        day = str(raw.get("_sync_day") or "")
        added = raw.get("added") or []
        updated = raw.get("updated") or []
        superseded = raw.get("superseded") or []
        series.append(
            {
                "date": day,
                "added": len(added),
                "updated": len(updated),
                "superseded": len(superseded),
                "changed": len(raw.get("changed_paths") or []) or (len(added) + len(updated)),
                "site_count": int(raw.get("site_count") or 0),
                "local_count": int(raw.get("local_count") or 0),
            }
        )
    return series


def _zero_periods() -> dict[str, dict[str, int]]:
    return {
        key: {"added": 0, "updated": 0, "changed": 0, "superseded": 0, "nights": 0}
        for key in ("d7", "d30", "d90", "ytd")
    }


def _sync_periods(history: list[dict[str, Any]], today: date) -> dict[str, dict[str, int]]:
    out = _zero_periods()
    cutoffs = {
        "d7": today - timedelta(days=7),
        "d30": today - timedelta(days=30),
        "d90": today - timedelta(days=90),
        "ytd": date(today.year, 1, 1),
    }
    for row in history:
        day = _parse_iso(str(row.get("date") or ""))
        if not day:
            continue
        for key, start in cutoffs.items():
            if day < start:
                continue
            bucket = out[key]
            bucket["added"] += int(row.get("added") or 0)
            bucket["updated"] += int(row.get("updated") or 0)
            bucket["changed"] += int(row.get("changed") or 0)
            bucket["superseded"] += int(row.get("superseded") or 0)
            bucket["nights"] += 1
    return out


def _catalog_block(rows: list[dict[str, Any]], today: date) -> dict[str, Any]:
    years: Counter[str] = Counter()
    months: Counter[str] = Counter()
    slugs: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    dated = 0
    recent: list[dict[str, Any]] = []
    period_posts = {"d7": 0, "d30": 0, "d90": 0, "y2026": 0, "ytd": 0}
    cutoffs = {
        "d7": today - timedelta(days=7),
        "d30": today - timedelta(days=30),
        "d90": today - timedelta(days=90),
        "ytd": date(today.year, 1, 1),
    }
    for rec in rows:
        path = str(rec.get("path") or rec.get("relative_path") or "")
        title = str(rec.get("display_title") or rec.get("title") or "")
        filename = path.rsplit("/", 1)[-1] if path else ""
        slug = str(rec.get("specialty_slug") or rec.get("category") or rec.get("slug") or "")
        if not slug and path.startswith("minzdrav_protocols/"):
            parts = path.split("/")
            if len(parts) >= 2:
                slug = parts[1]
        kind = str(rec.get("protocol_kind") or "unknown")
        meta = post_meta(filename, title, path)
        year = meta["post_year"] or "нет даты"
        years[year] += 1
        slugs[slug or "без рубрики"] += 1
        kinds[kind] += 1
        iso = meta["post_date_iso"]
        day = _parse_iso(iso) if iso and meta["post_date"] else None
        if iso and len(iso) >= 7:
            months[iso[:7]] += 1
        if meta["post_date"]:
            dated += 1
            recent.append(
                {
                    "filename": filename,
                    "slug": slug,
                    "slug_ru": SLUG_RU.get(slug, slug),
                    "relative_path": path,
                    "post_date": meta["post_date"],
                    "post_date_iso": iso,
                    "post_number": meta["post_number"],
                    "post_year": year,
                    "title": title[:180],
                }
            )
            if day:
                if day >= cutoffs["d7"]:
                    period_posts["d7"] += 1
                if day >= cutoffs["d30"]:
                    period_posts["d30"] += 1
                if day >= cutoffs["d90"]:
                    period_posts["d90"] += 1
                if day >= cutoffs["ytd"]:
                    period_posts["ytd"] += 1
        if year == "2026":
            period_posts["y2026"] += 1

    recent.sort(key=lambda r: r.get("post_date_iso") or "", reverse=True)
    year_rows = [
        {"year": year, "n": n}
        for year, n in sorted(years.items(), key=lambda kv: (kv[0] == "нет даты", kv[0]))
    ]
    month_rows = [{"month": month, "n": n} for month, n in sorted(months.items())]
    slug_rows = [
        {"slug": slug, "label": SLUG_RU.get(slug, slug), "n": n}
        for slug, n in slugs.most_common(16)
    ]
    kind_rows = [
        {"kind": kind, "label": KIND_RU.get(kind, kind), "n": n}
        for kind, n in kinds.most_common()
    ]
    return {
        "catalog_n": len(rows),
        "catalog_dated_n": dated,
        "post_periods": period_posts,
        "by_year": year_rows,
        "by_month": month_rows,
        "by_slug": slug_rows,
        "by_kind": kind_rows,
        "recent_posts": recent[:80],
    }


def _empty_payload(days: int) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "missing",
        "detail": "Сверки КП ещё не было",
        "added": [],
        "updated": [],
        "superseded": [],
        "changed_n": 0,
        "added_n": 0,
        "updated_n": 0,
        "superseded_n": 0,
        "site_count": 0,
        "local_count": 0,
        "days": days,
    }
