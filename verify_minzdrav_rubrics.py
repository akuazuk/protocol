#!/usr/bin/env python3
"""
Сравнивает список рубрик на странице Минздрава с набором slug в rag_server.py
(без импорта rag_server — чтобы не грузить чанки).

Запуск: python3 verify_minzdrav_rubrics.py
"""
from __future__ import annotations

import re
import ssl
import sys
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

BASE = "https://minzdrav.gov.by"
INDEX_PATH = "/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/"

# Синхронизировать с ALLOWED_SPECIALTY_SLUGS в rag_server.py
EXPECTED_SLUGS = frozenset(
    [
        "akusherstvo-ginekologiya",
        "allergologiya-immunologiya",
        "anesteziologiya-reanimatologiya",
        "bolezni-sistemy-krovoobrashcheniya",
        "dermatovenerologiya",
        "endokrinologiya-narusheniya-obmena-veshchestv",
        "gastroenterologiya",
        "gematologiya",
        "infektsionnye-zabolevaniya",
        "khirurgiya",
        "nefrologiya",
        "nevrologiya-neyrokhirurgiya",
        "novoobrazovaniya",
        "oftalmologiya",
        "otorinolaringologiya",
        "palliativnaya-pomoshch",
        "psikhiatriya-narkologiya",
        "pulmonologiya-ftiziatriya",
        "revmatologiya",
        "stomatologiya",
        "transplantatsiya-organov-i-tkaney",
        "travmatologiya-ortopediya",
        "urologiya",
        "zabolevaniya-perinatalnogo-perioda",
    ]
)

CTX = ssl.create_default_context()
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; Protocol-verify/1.0; +https://github.com/)"
    ),
    "Accept": "text/html",
}


def _encode_iri(url: str) -> str:
    p = urlsplit(url)
    if not p.scheme:
        return url
    segs = p.path.split("/")
    enc_path = "/".join(quote(s, safe="") for s in segs)
    return urlunsplit((p.scheme, p.netloc, enc_path, p.query, p.fragment))


def fetch(url: str) -> str:
    req = Request(_encode_iri(url), headers=HEADERS)
    with urlopen(req, timeout=60, context=CTX) as r:
        raw = r.read()
    enc = r.headers.get_content_charset() or "utf-8"
    return raw.decode(enc, errors="replace")


def slugs_from_index_html(html: str) -> set[str]:
    found: set[str] = set()
    for m in re.finditer(
        r'href=["\'](/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/([^"\']+\.php))["\']',
        html,
    ):
        fname = m.group(2)
        slug = fname.replace(".php", "")
        if slug != "index":
            found.add(slug)
    return found


def main() -> int:
    print("Загрузка индекса Минздрава…")
    try:
        html = fetch(BASE + INDEX_PATH)
    except Exception as e:
        print(f"Ошибка запроса: {e}", file=sys.stderr)
        return 1
    site = slugs_from_index_html(html)
    only_site = site - EXPECTED_SLUGS
    only_code = EXPECTED_SLUGS - site
    print(f"На сайте (кроме index): {len(site)} рубрик")
    print(f"В коде (EXPECTED_SLUGS): {len(EXPECTED_SLUGS)}")
    if not only_site and not only_code:
        print("OK: наборы совпадают.")
        print(
            'Примечание: отдельной рубрики «Терапия» на minzdrav.gov.by нет — '
            "общетерапевтические КП распределены по разделам (ССС, гастроэнтерология и т.д.)."
        )
        return 0
    if only_site:
        print("Только на сайте (добавить в rag_server?):", sorted(only_site))
    if only_code:
        print("Только в коде (устарели на сайте?):", sorted(only_code))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
