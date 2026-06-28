#!/usr/bin/env python3
"""Targeted chunk QA queue из слабых KZ (L1/L2 batch) + missing + review."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.chunk_quality import detect_issues, quality_score

DEFAULT_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.final.jsonl"
FALLBACK_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.v2.jsonl"
DEFAULT_L2 = ROOT / "ml" / "experiments" / "batch_clients_consult_2026-06-28" / "l2_weak_report.json"
DEFAULT_L1 = ROOT / "ml" / "experiments" / "batch_clients_consult_2026-06-28" / "report.json"
DEFAULT_MISSING = ROOT / "data" / "ml" / "chunk_qa_missing.json"
DEFAULT_REVIEW = ROOT / "data" / "ml" / "chunk_qa_review.jsonl"
DEFAULT_OUT = ROOT / "data" / "ml" / "chunk_qa_queue_kz_targeted.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "ml" / "chunk_qa_queue_kz_targeted_manifest.json"

# clients_consult/a_*.pdf — B2C анализы, не заключения (КЗ)
B2C_ANALYSIS_PREFIX = "a_"


def _is_b2c_analysis_case(case_id: str) -> bool:
    return str(case_id or "").startswith(B2C_ANALYSIS_PREFIX)


def _norm_path(p: str) -> str:
    return str(p or "").replace("\\", "/").strip().lower()


def _load_json(path: Path) -> dict | list:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol_paths_from_reports(l2: dict, l1: dict) -> dict[str, set[str]]:
    """case_id -> set of protocol paths (L2 primary, L1 fallback)."""
    out: dict[str, set[str]] = {}
    for row in l2.get("results") or []:
        cid = str(row.get("case_id") or "")
        paths = set(_norm_path(p) for p in (row.get("protocol_paths") or []) if p)
        if cid and paths:
            out[cid] = paths
    for row in (l1.get("reports") if isinstance(l1, dict) else []) or []:
        cid = str(row.get("case_id") or "")
        if cid in out:
            continue
        paths = set(_norm_path(p) for p in (row.get("matched_protocols") or []) if p)
        if cid and paths:
            out[cid] = paths
    return out


def build_targeted_queue(
    chunks_path: Path,
    *,
    case_paths: dict[str, set[str]],
    missing_ids: list[str],
    review_ids: set[str],
    max_per_protocol: int = 200,
) -> list[dict]:
    all_paths = set()
    for paths in case_paths.values():
        all_paths.update(paths)

    by_path_chunks: dict[str, list[dict]] = {}
    missing_set = set(missing_ids)
    queued: dict[str, dict] = {}

    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cid = str(row.get("chunk_id") or "")
            sp = _norm_path(str(row.get("source_path") or ""))
            if cid in missing_set and cid not in queued:
                queued[cid] = {
                    "chunk_id": cid,
                    "doc_id": row.get("doc_id"),
                    "source_path": row.get("source_path"),
                    "chunk_type": row.get("chunk_type"),
                    "quality_score": quality_score(row),
                    "issues": detect_issues(row) + ["gemini_missing"],
                    "priority": 130,
                    "reason": "gemini_missing_tail",
                }
            if cid in review_ids and cid not in queued:
                queued[cid] = {
                    "chunk_id": cid,
                    "doc_id": row.get("doc_id"),
                    "source_path": row.get("source_path"),
                    "chunk_type": row.get("chunk_type"),
                    "quality_score": quality_score(row),
                    "issues": detect_issues(row) + ["manual_review"],
                    "priority": 125,
                    "reason": "chunk_qa_review",
                }
            if not sp:
                continue
            matched_case = None
            for case_id, paths in case_paths.items():
                if any(sp.endswith(p.split("/")[-1].lower()) or p in sp or sp.endswith(p) for p in paths):
                    matched_case = case_id
                    break
                # substring match on path tail
                for p in paths:
                    tail = p.split("/")[-1]
                    if tail and tail in sp:
                        matched_case = case_id
                        break
                if matched_case:
                    break
            if matched_case:
                by_path_chunks.setdefault(sp, []).append((matched_case, row))

    for sp, items in by_path_chunks.items():
        # sort: lower quality first, clinical types
        def sort_key(item: tuple[str, dict]) -> tuple:
            case_id, row = item
            score = quality_score(row)
            issues = detect_issues(row)
            pri = 0
            if "preamble_leak" in issues:
                pri += 30
            if "type_body_but_clinical" in issues:
                pri += 20
            if score < 0.7:
                pri += 15
            return (-pri, score, str(row.get("chunk_id") or ""))

        items.sort(key=sort_key)
        seen = 0
        for case_id, row in items[:max_per_protocol]:
            cid = str(row.get("chunk_id") or "")
            if not cid or cid in queued:
                continue
            issues = detect_issues(row)
            score = quality_score(row)
            priority = 120
            if "preamble_leak" in issues:
                priority = 128
            elif score < 0.6:
                priority = 126
            queued[cid] = {
                "chunk_id": cid,
                "doc_id": row.get("doc_id"),
                "source_path": row.get("source_path"),
                "chunk_type": row.get("chunk_type"),
                "quality_score": score,
                "issues": issues,
                "priority": priority,
                "reason": "kz_weak_l2",
                "kz_case_id": case_id,
            }
            seen += 1

    out = list(queued.values())
    out.sort(key=lambda x: (-int(x.get("priority") or 0), str(x.get("chunk_id") or "")))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build targeted KZ-weak chunk QA queue")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--l2-report", type=Path, default=DEFAULT_L2)
    parser.add_argument("--l1-report", type=Path, default=DEFAULT_L1)
    parser.add_argument("--missing", type=Path, default=DEFAULT_MISSING)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-per-protocol", type=int, default=200)
    args = parser.parse_args()

    chunks_path = args.chunks or (DEFAULT_CHUNKS if DEFAULT_CHUNKS.is_file() else FALLBACK_CHUNKS)
    if not chunks_path.is_file():
        print(f"Нет {chunks_path}", file=sys.stderr)
        return 1

    l2 = _load_json(args.l2_report)
    l1 = _load_json(args.l1_report)
    case_paths = _protocol_paths_from_reports(l2 if isinstance(l2, dict) else {}, l1 if isinstance(l1, dict) else {})
    skipped_b2c = sorted(k for k in case_paths if _is_b2c_analysis_case(k))
    case_paths = {k: v for k, v in case_paths.items() if not _is_b2c_analysis_case(k)}

    missing_ids: list[str] = []
    if args.missing.is_file():
        raw = json.loads(args.missing.read_text(encoding="utf-8"))
        missing_ids = raw if isinstance(raw, list) else []

    review_ids: set[str] = set()
    if args.review.is_file():
        for line in args.review.open(encoding="utf-8"):
            try:
                row = json.loads(line)
                fix = row.get("fix") or {}
                review_ids.add(str(row.get("chunk_id") or fix.get("chunk_id") or ""))
            except json.JSONDecodeError:
                pass
        review_ids.discard("")

    queue = build_targeted_queue(
        chunks_path,
        case_paths=case_paths,
        missing_ids=missing_ids,
        review_ids=review_ids,
        max_per_protocol=args.max_per_protocol,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for item in queue:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "queue_size": len(queue),
        "cases": list(case_paths.keys()),
        "skipped_b2c_analysis_cases": skipped_b2c,
        "protocol_paths": {k: sorted(v) for k, v in case_paths.items()},
        "priority_counts": {},
        "out": str(args.out),
    }
    from collections import Counter
    manifest["priority_counts"] = dict(Counter(int(x.get("priority") or 0) for x in queue))

    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
