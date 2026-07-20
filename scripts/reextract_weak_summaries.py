#!/usr/bin/env python3
"""Переизвлечение слабых Protocol Summary через LLM (Фаза 2).

Очередь: data/protocol_summaries/reextract_queue.json (или --protocol-id).
Приоритет: empty (0 выдержек) → sparse (≤1) → no_exams → all.

Usage:
  set -a && source .env && set +a
  python3 scripts/reextract_weak_summaries.py --tier empty --publish
  python3 scripts/reextract_weak_summaries.py --tier all --limit 50 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.consult_evidence_pack import (  # noqa: E402
    EVIDENCE_BLOCK_IDS,
    _emit_condition_excerpts,
)
from clinical_knowledge.protocol_summary.builder import publish_summaries  # noqa: E402
from clinical_knowledge.protocol_summary.llm_extractor import extract_protocol_summary_llm  # noqa: E402
from clinical_knowledge.protocol_summary.loader import (  # noqa: E402
    clear_protocol_summary_cache,
    load_protocol_summaries,
)
from clinical_knowledge.protocol_summary.source_text import (  # noqa: E402
    build_source_text_document,
    load_source_text,
    save_source_text,
)
from clinical_knowledge.protocol_summary.summary_to_rag import write_summary_rag_jsonl  # noqa: E402
from clinical_knowledge.protocol_summary.validator import validate_protocol_summary, write_validation_report  # noqa: E402

QUEUE = ROOT / "data" / "protocol_summaries" / "reextract_queue.json"
# Прогресс можно вынести на persistent disk (Render): REEXTRACT_STATE=/var/data/...jsonl,
# чтобы --resume переживал рестарт эфемерного диска приложения.
STATE = Path(os.environ.get("REEXTRACT_STATE") or (ROOT / "data" / "protocol_summaries" / "reextract_state.jsonl"))
CATALOG = ROOT / "data" / "protocol_catalog.jsonl"


def _load_state() -> dict[str, str]:
    out: dict[str, str] = {}
    if not STATE.is_file():
        return out
    for line in STATE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = str(row.get("protocol_id") or "")
        if pid:
            out[pid] = str(row.get("status") or "")
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


def _path_for_protocol_id(protocol_id: str) -> str | None:
    for d in (
        ROOT / "data" / "protocol_summaries" / "source_text",
        ROOT / "data" / "protocol_summaries" / "json",
    ):
        if not d.is_dir():
            continue
        for p in d.glob(f"{protocol_id}.*"):
            if p.suffix == ".json":
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    path = data.get("path") or (data.get("source") or {}).get("local_path")
                    if path:
                        return str(path).replace("\\", "/")
                except (OSError, json.JSONDecodeError):
                    pass
    # catalog fallback
    if CATALOG.is_file():
        for line in CATALOG.read_text(encoding="utf-8").splitlines():
            if protocol_id[:20] not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = str(row.get("path") or "")
            if path and protocol_id.split("_")[0] in path.lower().replace("-", "_"):
                return path.replace("\\", "/")
    return None


def _excerpt_stats(protocol_id: str) -> dict[str, int]:
    from clinical_knowledge.protocol_summary.loader import load_summary_by_protocol_id

    s = load_summary_by_protocol_id(protocol_id)
    if s is None:
        return {"total": 0, "exams": 0}
    out = {b: 0 for b in EVIDENCE_BLOCK_IDS}
    for cond in s.conditions:
        tmp = {b: [] for b in EVIDENCE_BLOCK_IDS}
        _emit_condition_excerpts(tmp, src_path="x", cond=cond, max_per_block=6)
        for b in EVIDENCE_BLOCK_IDS:
            out[b] += len(tmp[b])
    return {"total": sum(out.values()), "exams": out["exams"], **out}


def _tier_filter(tier: str) -> list[str]:
    if tier == "no_exams":
        out: list[str] = []
        for s in load_protocol_summaries(usable_only=False):
            has = any(
                (getattr(c, "required_exams", None) or getattr(c, "conditional_exams", None))
                for c in s.conditions
            )
            if not has:
                out.append(s.protocol_id)
        return out
    summaries = load_protocol_summaries(usable_only=False)
    empty: list[str] = []
    sparse: list[str] = []
    for s in summaries:
        st = _excerpt_stats(s.protocol_id)
        if st["total"] == 0:
            empty.append(s.protocol_id)
        if st["total"] <= 1:
            sparse.append(s.protocol_id)
    if tier == "empty":
        return empty
    if tier == "sparse":
        return sparse
    if tier == "all":
        if QUEUE.is_file():
            return json.loads(QUEUE.read_text(encoding="utf-8"))
        return sorted(set(empty + _tier_filter("no_exams")))
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-file", type=str, default=None, help="JSON list of protocol_id")
    ap.add_argument("--tier", choices=("empty", "sparse", "no_exams", "all"), default="empty")
    ap.add_argument("--protocol-id", action="append", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    if args.protocol_id:
        queue = list(args.protocol_id)
    elif args.queue_file:
        queue = json.loads(Path(args.queue_file).read_text(encoding="utf-8"))
    else:
        queue = _tier_filter(args.tier)
    if args.limit:
        queue = queue[: args.limit]

    done = _load_state() if args.resume else {}
    summaries = []
    ok = fail = skip = 0

    print(f"Queue: {len(queue)} protocols (tier={args.tier})")
    for pid in queue:
        if args.resume and done.get(pid) == "ok":
            skip += 1
            continue
        doc = load_source_text(pid)
        if doc is None:
            path = _path_for_protocol_id(pid)
            if not path:
                print(f"{pid}: SKIP no source_text/path", file=sys.stderr)
                _append_state(pid, "skip", detail="no_source")
                skip += 1
                continue
            doc = build_source_text_document(path)
            save_source_text(doc)

        before = _excerpt_stats(pid)
        summary = extract_protocol_summary_llm(doc, use_llm=not args.no_llm)
        blob = json.dumps(doc.get("sections") or {}, ensure_ascii=False)
        result = validate_protocol_summary(summary, source_blob=blob)
        summary.validation = result
        write_validation_report(summary, result)

        from clinical_knowledge.protocol_summary.loader import export_summary_json

        export_summary_json(summary)
        clear_protocol_summary_cache()
        summaries.append(summary)
        after = _excerpt_stats(pid)
        status = "ok" if result.status != "invalid" else "invalid"
        _append_state(pid, status, detail=f"{before['total']}->{after['total']} excerpts")
        if status == "ok":
            ok += 1
        else:
            fail += 1
        print(
            f"{pid}: {result.status} excerpts {before['total']}->{after['total']} "
            f"exams {before['exams']}->{after['exams']} ({summary.extraction_metadata.extractor})"
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    if summaries and args.publish:
        stats = publish_summaries(summaries)
        clear_protocol_summary_cache()
        # ВАЖНО: перестраиваем RAG-jsonl из ВСЕХ опубликованных сводок, а не только
        # из переизвлечённой партии - иначе summary_chunks.jsonl затрётся и потеряет
        # не тронутые в этом прогоне протоколы (write открывает файл в режиме "w").
        out = write_summary_rag_jsonl()
        print(f"Published: {stats}")
        print(f"Rebuilt RAG chunks (all summaries): {out}")

    print(f"Done: ok={ok} fail={fail} skip={skip}")
    return 1 if fail and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
