#!/usr/bin/env python3
"""CLI: rceth Refbank manifest / preflight / download (GCE-first)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.rceth_sync.crawl import crawl_active_manifest, load_manifest
from clinical_knowledge.rceth_sync.download import download_s_pdfs, preflight_ndfiles
from clinical_knowledge.rceth_sync.http_client import RefbankClient
from clinical_knowledge.rceth_sync.labels import parse_downloaded_labels
from clinical_knowledge.rceth_sync.paths import data_root, manifest_path, status_path
from clinical_knowledge.rceth_sync.status import public_rceth_sync_payload, read_status


def _print(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_preflight(args: argparse.Namespace) -> int:
    root = data_root(args.data_root)
    client = RefbankClient(
        throttle_sec=args.throttle,
        insecure_ssl=args.insecure_ssl,
        timeout=args.timeout,
        retries=args.retries,
    )
    result = preflight_ndfiles(
        client,
        insecure_ssl=args.insecure_ssl,
        throttle_sec=args.throttle,
        root=root,
    )
    _print(result)
    return 0 if result.get("ok") else 2


def cmd_crawl(args: argparse.Namespace) -> int:
    root = data_root(args.data_root)
    client = RefbankClient(
        throttle_sec=args.throttle,
        insecure_ssl=args.insecure_ssl,
        timeout=args.timeout,
        retries=args.retries,
    )
    summary = crawl_active_manifest(
        root=root,
        throttle_sec=args.throttle,
        insecure_ssl=args.insecure_ssl,
        max_letters=args.max_letters,
        max_pages_per_letter=args.max_pages,
        client=client,
    )
    _print(summary)
    print(f"manifest -> {manifest_path(root)}", file=sys.stderr)
    print(f"status   -> {status_path(root)}", file=sys.stderr)
    return 0 if int(summary.get("errors") or 0) == 0 else 1


def cmd_download(args: argparse.Namespace) -> int:
    root = data_root(args.data_root)
    if not manifest_path(root).is_file():
        print(f"нет манифеста: {manifest_path(root)} (сначала crawl)", file=sys.stderr)
        return 2
    client = RefbankClient(
        throttle_sec=args.throttle,
        insecure_ssl=args.insecure_ssl,
        timeout=args.timeout,
        retries=args.retries,
    )
    summary = download_s_pdfs(
        root=root,
        limit=args.limit,
        throttle_sec=args.throttle,
        insecure_ssl=args.insecure_ssl,
        require_preflight=not args.skip_preflight,
        client=client,
        retries=args.retries,
    )
    _print(summary)
    return 0 if summary.get("ok") else 1


def cmd_parse(args: argparse.Namespace) -> int:
    root = data_root(args.data_root)
    if not manifest_path(root).is_file():
        print(f"нет манифеста: {manifest_path(root)}", file=sys.stderr)
        return 2
    reg_ids = [x.strip() for x in (args.reg_ids or "").split(",") if x.strip()] or None
    summary = parse_downloaded_labels(root=root, limit=args.limit, reg_ids=reg_ids)
    _print(summary)
    return 0 if summary.get("ok") else 1


def cmd_status(args: argparse.Namespace) -> int:
    root = data_root(args.data_root)
    live = read_status(root)
    rows = load_manifest(manifest_path(root))
    latest = {
        "sync_day": "",
        "manifest_count": len(rows),
        "with_s_pdf": sum(1 for r in rows if r.get("url_s") or r.get("has_s_pdf")),
        "downloaded": sum(1 for r in rows if r.get("pdf_s_sha256")),
        "failed": sum(1 for r in rows if r.get("download_error")),
        "no_pdf": sum(1 for r in rows if not (r.get("url_s") or r.get("has_s_pdf"))),
        "written_at": (live or {}).get("updated_at") or "",
    }
    _print(public_rceth_sync_payload(latest, live))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rceth ЛС sync (manifest / download / status)")
    p.add_argument(
        "--data-root",
        default=os.environ.get("RCETH_DATA_ROOT") or "",
        help="корень данных (default: $RCETH_DATA_ROOT или /var/data/rceth или data/rceth)",
    )
    p.add_argument("--throttle", type=float, default=0.6, help="пауза между HTTP запросами, сек")
    p.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("RCETH_HTTP_TIMEOUT") or 30),
        help="HTTP timeout сек (default 30 / $RCETH_HTTP_TIMEOUT)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=int(os.environ.get("RCETH_HTTP_RETRIES") or 3),
        help="повторы при timeout/503 (default 3 / $RCETH_HTTP_RETRIES)",
    )
    p.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="отключить проверку TLS (только если на GCE/Mac нет CA)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preflight", help="проверка /NDfiles/instr/*.pdf")
    sp.set_defaults(func=cmd_preflight)

    sc = sub.add_parser("crawl", help="обход действующих ЛС → manifest.jsonl")
    sc.add_argument("--max-letters", type=int, default=None, help="лимит букв (пилот)")
    sc.add_argument("--max-pages", type=int, default=None, help="лимит страниц на букву")
    sc.set_defaults(func=cmd_crawl)

    sd = sub.add_parser("download", help="скачать _s.pdf по манифесту")
    sd.add_argument("--limit", type=int, default=None, help="макс. файлов")
    sd.add_argument("--skip-preflight", action="store_true")
    sd.set_defaults(func=cmd_download)

    sp_parse = sub.add_parser("parse", help="разметить скачанные _s.pdf → labels/*.json")
    sp_parse.add_argument("--limit", type=int, default=None, help="макс. карточек")
    sp_parse.add_argument("--reg-ids", default="", help="список reg_id через запятую")
    sp_parse.set_defaults(func=cmd_parse)

    ss = sub.add_parser("status", help="live status.json + снимок манифеста")
    ss.set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    if not args.data_root:
        args.data_root = None
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
