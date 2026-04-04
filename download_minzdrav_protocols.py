#!/usr/bin/env python3
"""
Скачивает клинические протоколы с minzdrav.gov.by по рубрикам.
Только стандартная библиотека Python.
"""
from __future__ import annotations

import re
import sys
import time
import ssl
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse, unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

BASE = "https://minzdrav.gov.by"
INDEX = "/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/"
OUT_ROOT = Path(__file__).resolve().parent / "minzdrav_protocols"

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


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = Request(encode_iri(url), headers=HEADERS)
        with urlopen(req, timeout=300, context=CTX) as r:
            data = r.read()
        with open(tmp, "wb") as f:
            f.write(data)
        tmp.replace(dest)
        return True
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print(f"  Ошибка: {dest.name}: {e}", file=sys.stderr)
        return False


def main() -> None:
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

    done = 0
    ok = 0
    for slug, urls in sorted(all_docs.items()):
        folder = OUT_ROOT / slug
        for u in sorted(urls):
            name = safe_filename_from_url(u)
            dest = folder / name
            if dest.exists() and dest.stat().st_size > 0:
                ok += 1
                done += 1
                continue
            if download_file(u, dest):
                ok += 1
            done += 1
            if done % 25 == 0:
                print(f"  … обработано {done}/{total}")
            time.sleep(0.12)

    print(f"Готово. Сохранено в: {OUT_ROOT}")
    print(f"Успешно (или уже были на диске): {ok} из {total}")


if __name__ == "__main__":
    main()
