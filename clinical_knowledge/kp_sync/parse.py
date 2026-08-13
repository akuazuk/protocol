"""Разбор HTML каталога Минздрава (без сети)."""
from __future__ import annotations

import re
from urllib.parse import unquote, urljoin, urlparse

BASE = "https://minzdrav.gov.by"
INDEX_PATH = "/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/"

DOC_EXT = re.compile(r"\.(pdf|doc|docx|zip|rar)(\?.*)?$", re.I)
HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.I)
CAT_RE = re.compile(
    r"""href=["'](/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/([^"']+\.php))["']""",
    re.I,
)


def category_pages(html: str, *, base: str = BASE) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for m in CAT_RE.finditer(html or ""):
        path, fname = m.group(1), m.group(2)
        if fname == "index.php" or path in seen:
            continue
        seen.add(path)
        slug = fname.replace(".php", "")
        out.append((slug, urljoin(base, path)))
    return out


def document_hrefs(html: str, page_url: str) -> set[str]:
    urls: set[str] = set()
    for m in HREF_RE.finditer(html or ""):
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


def site_docs_from_pages(
    pages: list[tuple[str, str, str]],
) -> list[dict[str, str]]:
    """pages: (slug, page_url, html) -> unique site document records."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for slug, page_url, html in pages:
        for url in sorted(document_hrefs(html, page_url)):
            name = safe_filename_from_url(url)
            rel = f"minzdrav_protocols/{slug}/{name}"
            if rel in seen:
                continue
            seen.add(rel)
            out.append(
                {
                    "slug": slug,
                    "url": url,
                    "filename": name,
                    "relative_path": rel,
                }
            )
    return out
