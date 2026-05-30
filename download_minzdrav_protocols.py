#!/usr/bin/env python3
"""
Скачивает клинические протоколы с minzdrav.gov.by по рубрикам.
Только стандартная библиотека Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse, unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

BASE = "https://minzdrav.gov.by"
INDEX = "/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/"
OUT_ROOT = Path(__file__).resolve().parent / "minzdrav_protocols"
MANIFEST_PATH = OUT_ROOT / "_manifest.jsonl"
ERRORS_PATH = OUT_ROOT / "_download_errors.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

DOC_EXT = re.compile(r"\.(pdf|doc|docx|zip|rar)(\?.*)?$", re.I)
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)

CTX = ssl.create_default_context()


def encode_iri(url: str) -> str:
    """Кодирует путь и query для не-ASCII символов (пробелы, кириллица, № и т.д.)."""
    p = urlsplit(url)
    if not p.scheme:
        return url
    segs = p.path.split("/")
    enc_path = "/".join(quote(s, safe="") for s in segs)
    return urlunsplit((p.scheme, p.netloc, enc_path, p.query, p.fragment))


def fetch(url: str) -> str:
    req = Request(encode_iri(url), headers=HEADERS)
    with urlopen(req, timeout=120, context=CTX) as r:
        raw = r.read()
    enc = r.headers.get_content_charset() or "utf-8"
    try:
        return raw.decode(enc)
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def category_pages(html: str) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for m in re.finditer(
        r'href=["\'](/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/([^"\']+\.php))["\']',
        html,
    ):
        path, fname = m.group(1), m.group(2)
        if fname == "index.php":
            continue
        if path in seen:
            continue
        seen.add(path)
        slug = fname.replace(".php", "")
        out.append((slug, urljoin(BASE, path)))
    return out


def document_hrefs(html: str, page_url: str) -> set[str]:
    urls: set[str] = set()
    for m in HREF_RE.finditer(html):
        href = m.group(1).strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        if not DOC_EXT.search(href):
            continue
        full = urljoin(page_url, href)
        netloc = urlparse(full).netloc
        if netloc and "minzdrav.gov.by" not in netloc:
            continue
        urls.add(full.split("#")[0])
    return urls


def safe_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = unquote(path.rsplit("/", 1)[-1])
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name or "download.bin"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def download_file(url: str, dest: Path) -> tuple[bool, int, str | None]:
    """Скачивает файл. Возвращает (успех, http_status, текст_ошибки)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = Request(encode_iri(url), headers=HEADERS)
        with urlopen(req, timeout=300, context=CTX) as r:
            status = int(getattr(r, "status", 0) or 200)
            data = r.read()
        with open(tmp, "wb") as f:
            f.write(data)
        tmp.replace(dest)
        return True, status, None
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print(f"  Ошибка: {dest.name}: {e}", file=sys.stderr)
        return False, getattr(e, "code", 0) or 0, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Скачивание клинических протоколов с minzdrav.gov.by с манифестом."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Перекачивать файлы заново и обновлять, если sha256 изменился.",
    )
    args = parser.parse_args()

    print("Загрузка индекса…")
    index_html = fetch(urljoin(BASE, INDEX))
    cats = category_pages(index_html)
    print(f"Найдено рубрик: {len(cats)}")

    all_docs: dict[str, set[str]] = {}
    for slug, cat_url in sorted(cats, key=lambda x: x[0]):
        try:
            html = fetch(cat_url)
        except Exception as e:
            print(f"Рубрика {slug}: не удалось открыть страницу: {e}", file=sys.stderr)
            continue
        hrefs = document_hrefs(html, cat_url)
        all_docs[slug] = hrefs
        print(f"  {slug}: {len(hrefs)} файлов")

    total = sum(len(s) for s in all_docs.values())
    print(f"Всего ссылок на документы: {total}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict] = []
    errors: list[dict] = []
    done = 0
    ok = 0
    skipped = 0
    updated = 0
    crawl_ts = _now_utc_iso()

    for slug, urls in sorted(all_docs.items()):
        folder = OUT_ROOT / slug
        for u in sorted(urls):
            name = safe_filename_from_url(u)
            dest = folder / name
            done += 1
            existed = dest.exists() and dest.stat().st_size > 0
            prev_hash = _sha256_file(dest) if existed else None

            if existed and not args.refresh:
                ok += 1
                skipped += 1
                manifest_records.append(
                    {
                        "url": u,
                        "slug": slug,
                        "filename": name,
                        "relative_path": f"minzdrav_protocols/{slug}/{name}",
                        "sha256": prev_hash,
                        "bytes": dest.stat().st_size,
                        "downloaded_utc": crawl_ts,
                        "http_status": 0,
                        "action": "kept",
                    }
                )
                continue

            success, status, err = download_file(u, dest)
            if success:
                ok += 1
                new_hash = _sha256_file(dest)
                action = "downloaded"
                if existed:
                    action = "updated" if new_hash != prev_hash else "unchanged"
                    if action == "updated":
                        updated += 1
                manifest_records.append(
                    {
                        "url": u,
                        "slug": slug,
                        "filename": name,
                        "relative_path": f"minzdrav_protocols/{slug}/{name}",
                        "sha256": new_hash,
                        "bytes": dest.stat().st_size,
                        "downloaded_utc": crawl_ts,
                        "http_status": status,
                        "action": action,
                    }
                )
            else:
                errors.append(
                    {
                        "url": u,
                        "slug": slug,
                        "filename": name,
                        "http_status": status,
                        "error": err,
                        "downloaded_utc": crawl_ts,
                    }
                )
            if done % 25 == 0:
                print(f"  … обработано {done}/{total}")
            time.sleep(0.12)

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ERRORS_PATH.write_text(
        json.dumps(
            {"crawl_utc": crawl_ts, "errors": errors, "error_count": len(errors)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Готово. Сохранено в: {OUT_ROOT}")
    print(f"Успешно (или уже были на диске): {ok} из {total}")
    print(f"Пропущено (уже на диске): {skipped}; обновлено: {updated}; ошибок: {len(errors)}")
    print(f"Манифест: {MANIFEST_PATH}")
    if errors:
        print(f"Журнал ошибок: {ERRORS_PATH}")


if __name__ == "__main__":
    main()
