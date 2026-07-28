#!/usr/bin/env python3
"""Batch LLM/structured extraction with --resume checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.protocol_summary.builder import publish_summaries  # noqa: E402
from clinical_knowledge.protocol_summary.llm_extractor import extract_protocol_summary_llm  # noqa: E402
from clinical_knowledge.protocol_summary.loader import (  # noqa: E402
    clear_protocol_summary_cache,
    load_protocol_summaries,
)
from clinical_knowledge.protocol_summary.source_text import (  # noqa: E402
    SOURCE_DIR,
    build_source_text_document,
    load_source_text,
    save_source_text,
)
from clinical_knowledge.protocol_summary.summary_to_rag import write_summary_rag_jsonl  # noqa: E402
from clinical_knowledge.protocol_summary.validator import validate_protocol_summary, write_validation_report  # noqa: E402

STATE = Path(
    os.environ.get("REEXTRACT_STATE")
    or ROOT / "data" / "protocol_summaries" / "llm_batch_state.jsonl",
)
CATALOG = ROOT / "data" / "protocol_catalog.jsonl"


def _load_state() -> dict[str, str]:
    out: dict[str, str] = {}
    if not STATE.is_file():
        return out
    with STATE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = str(row.get("protocol_id") or "")
            st = str(row.get("status") or "")
            if pid:
                out[pid] = st
    return out


def _append_state(protocol_id: str, status: str, *, detail: str = "") -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"protocol_id": protocol_id, "status": status, "detail": detail, "ts": time.time()},
                ensure_ascii=False,
            )
            + "\n",
        )


def _catalog_rows(limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with CATALOG.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("path"):
                rows.append(row)
    rows.sort(key=lambda r: str(r.get("path") or ""))
    if limit:
        rows = rows[:limit]
    return rows


def main() -> int:
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="повторно обработать записи со status=ok")
    ap.add_argument("--state", type=Path, default=STATE)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0, help="pause between LLM calls")
    args = ap.parse_args()
    STATE = args.state.expanduser().resolve()

    done_state = _load_state() if args.resume and not args.force else {}
    summaries = []
    invalid = 0
    skipped = 0
    source_by_path: dict[str, tuple[dict, Path]] = {}
    for source_path in SOURCE_DIR.glob("*.json"):
        try:
            source_doc = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_key = str(source_doc.get("path") or "").replace("\\", "/")
        if source_key:
            source_by_path[source_key] = (source_doc, source_path)

    for row in _catalog_rows(args.limit):
        path = str(row.get("path") or "").replace("\\", "/")
        pid = None
        doc = None
        source_match = source_by_path.get(path)
        if source_match:
            doc, source_file = source_match
            pid = str(doc.get("protocol_id") or source_file.stem)
        if doc is None:
            doc = build_source_text_document(path)
            save_source_text(doc)
            pid = str(doc.get("protocol_id") or "")
        else:
            pid = str(doc.get("protocol_id") or pid or "")

        if args.resume and done_state.get(pid) == "ok":
            skipped += 1
            continue

        summary = extract_protocol_summary_llm(doc, use_llm=not args.no_llm)
        blob = json.dumps(doc.get("sections") or {}, ensure_ascii=False)
        result = validate_protocol_summary(summary, source_blob=blob)
        summary.validation = result
        write_validation_report(summary, result)
        summaries.append(summary)
        status = "ok" if result.status != "invalid" else "invalid"
        _append_state(pid, status, detail=result.status)
        if result.status == "invalid":
            invalid += 1
        print(f"{pid}: {result.status} ({summary.extraction_metadata.extractor})")
        if args.sleep > 0:
            time.sleep(args.sleep)

    if summaries:
        stats = publish_summaries(summaries)
        clear_protocol_summary_cache()
        # При resume в текущем запуске summaries содержит только хвост. Индекс должен
        # включать весь корпус, иначе частичный restart незаметно обрезает поиск.
        write_summary_rag_jsonl(load_protocol_summaries())
        print(f"Published: {stats}")
    print(f"Batch done: {len(summaries)} processed, {skipped} skipped, {invalid} invalid")
    return 1 if invalid and not summaries else 0


if __name__ == "__main__":
    raise SystemExit(main())
