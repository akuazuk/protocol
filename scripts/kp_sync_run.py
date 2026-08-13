#!/usr/bin/env python3
"""Сверка каталога КП МЗ с локальным манифестом (без обязательной сети)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.kp_sync.diff import diff_catalog
from clinical_knowledge.kp_sync.parse import site_docs_from_pages


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json_list(path: Path) -> list:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("docs"), list):
        return raw["docs"]
    raise SystemExit(f"ожидался список в {path}")


def _load_manifest(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return _load_json_list(path)
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def local_docs_from_pdf_root(pdf_root: Path) -> list[dict]:
    """Манифест с диска, если _manifest.jsonl ещё нет."""
    rows: list[dict] = []
    if not pdf_root.is_dir():
        return rows
    for p in sorted(pdf_root.rglob("*.pdf")):
        try:
            rel_inside = p.relative_to(pdf_root).as_posix()
        except ValueError:
            continue
        slug = rel_inside.split("/", 1)[0] if "/" in rel_inside else "unknown"
        rows.append(
            {
                "relative_path": f"minzdrav_protocols/{rel_inside}",
                "filename": p.name,
                "slug": slug,
            }
        )
    return rows


def cmd_diff(args: argparse.Namespace) -> int:
    site = _load_json_list(Path(args.site))
    local = _load_manifest(Path(args.local))
    result = diff_catalog(site, local)
    result["crawled_utc"] = _now()
    out = Path(args.out) if args.out else None
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text)
    print(
        f"site={result['site_count']} local={result['local_count']} "
        f"added={len(result['added'])} updated={len(result['updated'])} "
        f"superseded={len(result['superseded'])} unchanged={len(result['unchanged'])}",
        file=sys.stderr,
    )
    return 0


def cmd_from_html(args: argparse.Namespace) -> int:
    index_html = Path(args.index_html).read_text(encoding="utf-8")
    from clinical_knowledge.kp_sync.parse import category_pages

    cats = category_pages(index_html)
    pages: list[tuple[str, str, str]] = []
    html_dir = Path(args.html_dir) if args.html_dir else Path(args.index_html).parent
    for slug, url in cats:
        f = html_dir / f"{slug}.html"
        if not f.is_file():
            continue
        pages.append((slug, url, f.read_text(encoding="utf-8")))
    site = site_docs_from_pages(pages)
    local = _load_manifest(Path(args.local)) if args.local else []
    result = diff_catalog(site, local)
    result["crawled_utc"] = _now()
    result["rubrics"] = [s for s, _ in cats]
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def cmd_crawl(args: argparse.Namespace) -> int:
    import ssl

    import download_minzdrav_protocols as dl

    if os.environ.get("KP_SYNC_SSL_INSECURE", "").strip() in {"1", "true", "yes"}:
        dl.CTX = ssl._create_unverified_context()
    index_html = dl.fetch(dl.BASE + dl.INDEX)
    cats = dl.category_pages(index_html)
    pages: list[tuple[str, str, str]] = []
    errors: list[dict] = []
    for slug, url in cats:
        try:
            pages.append((slug, url, dl.fetch(url)))
        except Exception as exc:  # noqa: BLE001
            errors.append({"slug": slug, "url": url, "error": str(exc)})
        time.sleep(float(os.environ.get("KP_SYNC_CRAWL_DELAY_SEC", "0.25")))
    site = site_docs_from_pages(pages)
    payload = {"docs": site, "rubrics": [s for s, _ in cats], "errors": errors, "crawled_utc": _now()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"crawl docs={len(site)} rubrics={len(cats)} errors={len(errors)} → {out}", file=sys.stderr)
    return 1 if errors and not site else 0


def cmd_apply(args: argparse.Namespace) -> int:
    import download_minzdrav_protocols as dl

    diff = json.loads(Path(args.diff).read_text(encoding="utf-8"))
    dest_root = Path(args.dest)
    cap = int(args.max_downloads)
    raw_todo = list(diff.get("added") or []) + list(diff.get("updated") or [])
    todo: list[dict] = []
    skipped_exist = 0
    for rec in raw_todo:
        dest = dest_root / str(rec.get("slug") or "") / str(rec.get("filename") or "")
        if dest.is_file() and str(rec.get("action") or "") != "updated":
            skipped_exist += 1
            continue
        todo.append(rec)
    downloaded: list[str] = []
    errors: list[dict] = []
    for rec in todo[:cap]:
        url = str(rec.get("url") or "")
        rel = str(rec.get("relative_path") or "")
        if not url or not rel:
            continue
        dest = dest_root / rec["slug"] / rec["filename"]
        ok, status, err = dl.download_file(url, dest)
        if ok:
            downloaded.append(str(dest))
        else:
            errors.append({"url": url, "error": err, "http_status": status})
        time.sleep(0.12)
    leftover = max(0, len(todo) - cap)
    summary = {
        "downloaded": downloaded,
        "errors": errors,
        "leftover": leftover,
        "skipped_exist": skipped_exist,
        "applied_utc": _now(),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"downloaded={len(downloaded)} skipped_exist={skipped_exist} errors={len(errors)} leftover={leftover}",
        file=sys.stderr,
    )
    return 1 if errors and not downloaded else 0


def cmd_scan_local(args: argparse.Namespace) -> int:
    rows = local_docs_from_pdf_root(Path(args.dest))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"local_pdfs={len(rows)} → {out}", file=sys.stderr)
    return 0


def cmd_merge_indexes(args: argparse.Namespace) -> int:
    from clinical_knowledge.kp_sync.indexes import merge_catalog_for_paths, merge_icd_profiles_for_paths
    from clinical_knowledge.kp_sync.jsonl_merge import load_jsonl

    paths = [
        ln.strip()
        for ln in Path(args.paths).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    chunks = load_jsonl(Path(args.chunks))
    by_path: dict[str, list] = {}
    for ch in chunks:
        p = str(ch.get("source_path") or "").replace("\\", "/")
        if p:
            by_path.setdefault(p, []).append(ch)
    icd = merge_icd_profiles_for_paths(
        paths, chunks_by_path=by_path, out_path=Path(args.icd_index) if args.icd_index else None
    )
    cat = merge_catalog_for_paths(
        paths, chunks_by_path=by_path, out_path=Path(args.catalog) if args.catalog else None
    )
    print(json.dumps({"icd_profiles": icd, "catalog": cat}, ensure_ascii=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="КП МЗ: diff сайта и локального манифеста")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diff", help="JSON сайта vs манифест")
    d.add_argument("--site", required=True, help="JSON список документов сайта")
    d.add_argument("--local", required=True, help="manifest.jsonl или JSON список")
    d.add_argument("--out", default="")
    d.set_defaults(func=cmd_diff)

    h = sub.add_parser("from-html", help="diff из сохранённого HTML (фикстуры)")
    h.add_argument("--index-html", required=True)
    h.add_argument("--html-dir", default="")
    h.add_argument("--local", default="")
    h.add_argument("--out", default="")
    h.set_defaults(func=cmd_from_html)

    c = sub.add_parser("crawl", help="Живой обход сайта МЗ → JSON списка PDF")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_crawl)

    a = sub.add_parser("apply", help="Скачать added/updated из diff JSON (без unlink)")
    a.add_argument("--diff", required=True)
    a.add_argument("--dest", required=True, help="Каталог minzdrav_protocols")
    a.add_argument("--max-downloads", type=int, default=int(os.environ.get("KP_SYNC_MAX_DOWNLOADS", "8")))
    a.add_argument("--out", default="")
    a.set_defaults(func=cmd_apply)

    s = sub.add_parser("scan-local", help="Манифест из PDF на диске")
    s.add_argument("--dest", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_scan_local)

    m = sub.add_parser("merge-indexes", help="Влить ICD-профили и catalog только для changed paths")
    m.add_argument("--paths", required=True, help="Файл со списком relative_path")
    m.add_argument("--chunks", required=True, help="chunks.jsonl")
    m.add_argument("--icd-index", default="")
    m.add_argument("--catalog", default="")
    m.set_defaults(func=cmd_merge_indexes)

    args = p.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
