"""Crawl действующих ЛС → manifest.jsonl."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from clinical_knowledge.rceth_sync.http_client import (
    RefbankClient,
    active_search_pairs,
    page_pairs_from_html,
)
from clinical_knowledge.rceth_sync.parse import (
    merge_manifest_rows,
    parse_result_counts,
    parse_search_results,
)
from clinical_knowledge.rceth_sync.paths import manifest_path
from clinical_knowledge.rceth_sync.status import write_status

# Буквы для Start-обхода (кириллица + латиница + цифры).
DEFAULT_LETTERS = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя") + list("abcdefghijklmnopqrstuvwxyz") + list(
    "0123456789"
)


ProgressCb = Callable[[dict[str, Any]], None]


def write_manifest(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
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
    return rows


def crawl_letter(
    client: RefbankClient,
    letter: str,
    *,
    max_pages: int | None = None,
) -> tuple[list[dict[str, Any]], int | None, int | None]:
    code, html = client.post_search(active_search_pairs(letter, "Start"))
    if code >= 400:
        raise RuntimeError(f"search letter={letter!r} HTTP {code}")
    records, pages = parse_result_counts(html)
    rows = parse_search_results(html)
    page_n = 1
    total_pages = pages or 1
    if max_pages is not None:
        total_pages = min(total_pages, max_pages)
    while page_n < total_pages:
        page_n += 1
        code, html = client.post_search(page_pairs_from_html(html, page_n))
        if code >= 400:
            break
        # refresh QueryStringFind carrier from latest html
        more = parse_search_results(html)
        if not more:
            break
        # защита от зацикливания, если сервер вернул ту же страницу
        seen = {r.get("reg_id") for r in rows if r.get("reg_id")}
        fresh = [r for r in more if r.get("reg_id") and r["reg_id"] not in seen]
        if not fresh:
            break
        rows.extend(fresh)
    return rows, records, pages


def crawl_active_manifest(
    *,
    root: Path | None = None,
    letters: list[str] | None = None,
    throttle_sec: float = 0.6,
    insecure_ssl: bool = False,
    max_letters: int | None = None,
    max_pages_per_letter: int | None = None,
    client: RefbankClient | None = None,
) -> dict[str, Any]:
    """Обойти действующие ЛС и записать manifest.jsonl."""
    letters = list(letters or DEFAULT_LETTERS)
    if max_letters is not None:
        letters = letters[: max(0, max_letters)]
    client = client or RefbankClient(throttle_sec=throttle_sec, insecure_ssl=insecure_ssl)
    write_status(
        phase="crawl",
        status="running",
        done=0,
        total=len(letters),
        message="session",
        root=root,
    )
    client.ensure_session()
    all_rows: list[dict[str, Any]] = []
    letter_stats: list[dict[str, Any]] = []
    errors = 0
    for i, letter in enumerate(letters, start=1):
        write_status(
            phase="crawl",
            status="running",
            done=i - 1,
            total=len(letters),
            message=f"letter={letter}",
            current_reg_id="",
            errors=errors,
            root=root,
        )
        try:
            rows, rec, pages = crawl_letter(client, letter, max_pages=max_pages_per_letter)
            all_rows.extend(rows)
            letter_stats.append(
                {"letter": letter, "rows": len(rows), "records": rec, "pages": pages}
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            letter_stats.append({"letter": letter, "error": str(exc)[:200]})
    merged = merge_manifest_rows(all_rows)
    # оставить только active в v1-корпусе (на всякий случай)
    active = [r for r in merged if r.get("status") == "active"]
    path = write_manifest(active, manifest_path(root))
    with_s = sum(1 for r in active if r.get("has_s_pdf") or r.get("url_s"))
    summary = {
        "manifest_path": str(path),
        "manifest_count": len(active),
        "with_s_pdf": with_s,
        "letters": letter_stats,
        "errors": errors,
        "raw_rows_before_dedup": len(all_rows),
    }
    write_status(
        phase="crawl",
        status="done" if errors == 0 else "done",
        done=len(letters),
        total=len(letters),
        message=f"manifest={len(active)} with_s={with_s}",
        errors=errors,
        root=root,
        extra={"summary": {"manifest_count": len(active), "with_s_pdf": with_s}},
    )
    return summary
