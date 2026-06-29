#!/usr/bin/env python3
"""Export chunk QA labels for classifier training (Wave A + rule issues).

Reads:
  - data/ml/chunk_qa_issues.jsonl       (rule-based issues, ~25k)
  - data/ml/chunk_qa_fixes_wave_a.jsonl (Gemini verdicts, ~10k)
  - output/rich_chunks/rich_chunks.jsonl (full text by chunk_id)

Writes:
  - ml/datasets/chunk_qa_sft.jsonl       (input/output schema for docs)
  - ml/datasets/chunk_qa_classifier.jsonl (flat rows for sklearn)
  - ml/datasets/chunk_qa_export_manifest.json

Example:
  python3 scripts/export_chunk_qa_dataset.py
  python3 scripts/export_chunk_qa_dataset.py --max-rows 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ISSUES = ROOT / "data/ml/chunk_qa_issues.jsonl"
DEFAULT_FIXES = ROOT / "data/ml/chunk_qa_fixes_wave_a.jsonl"
DEFAULT_RICH = ROOT / "output/rich_chunks/rich_chunks.jsonl"
OUT_DIR = ROOT / "ml/datasets"

ACTION_VERDICTS = frozenset({"fix", "drop", "merge_with_next"})
P0_ISSUES = ("preamble_leak", "icd_inflation")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _doc_id(chunk_id: str) -> str:
    cid = (chunk_id or "").strip()
    if "_s" in cid:
        return cid.split("_s", 1)[0]
    return cid.split("_", 1)[0] if "_" in cid else cid


def load_issues(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cid = str(row.get("chunk_id") or "").strip()
        if cid:
            out[cid] = row
    return out


def load_fixes(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cid = str(row.get("chunk_id") or "").strip()
        if cid:
            out[cid] = row
    return out


def load_rich_chunks(path: Path, want: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if not want or not path.is_file():
        return found
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row.get("chunk_id") or "").strip()
            if cid in want:
                found[cid] = row
                if len(found) >= len(want):
                    break
    return found


def _issues_from_fix(fix: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    verdict = str(fix.get("verdict") or "").strip()
    if verdict == "drop":
        issues.append("preamble_leak")
    noise = fix.get("noise_reasons") or []
    for n in noise:
        s = str(n).strip().lower()
        if "preamble" in s or "шапк" in s or "утвержден" in s:
            if "preamble_leak" not in issues:
                issues.append("preamble_leak")
        if "icd" in s or "мкб" in s:
            if "icd_inflation" not in issues:
                issues.append("icd_inflation")
    return issues


def merge_row(
    chunk_id: str,
    issue_row: dict[str, Any] | None,
    fix_row: dict[str, Any] | None,
    rich_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    text = ""
    chunk_type = ""
    section_title = ""
    doc_id = _doc_id(chunk_id)

    if rich_row:
        text = str(rich_row.get("text") or "").strip()
        chunk_type = str(rich_row.get("chunk_type") or "").strip()
        section_title = str(rich_row.get("section_title") or "").strip()
        doc_id = str(rich_row.get("doc_id") or doc_id)
    if issue_row:
        chunk_type = chunk_type or str(issue_row.get("chunk_type") or "").strip()
        section_title = section_title or str(issue_row.get("section_title") or "").strip()
        if not text:
            text = str(issue_row.get("text_preview") or "").strip()
        doc_id = str(issue_row.get("doc_id") or doc_id)

    if not text or len(text) < 20:
        return None

    issues = list(issue_row.get("issues") or []) if issue_row else []
    issues = [str(i).strip() for i in issues if str(i).strip()]
    if fix_row:
        issues.extend(_issues_from_fix(fix_row))
    issues = sorted(set(issues))

    verdict = str((fix_row or {}).get("verdict") or "").strip()
    if not verdict:
        verdict = "fix" if issues else "ok"

    confidence = float(fix_row.get("confidence") or 0.0) if fix_row else 0.0
    needs_action = verdict in ACTION_VERDICTS or bool(issues)

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "input": {
            "text": text[:4000],
            "chunk_type": chunk_type,
            "section_title": section_title[:280],
            "text_len": len(text),
        },
        "output": {
            "verdict": verdict,
            "issues": issues,
            "needs_action": needs_action,
            "confidence": confidence,
        },
        "source": "issues+fixes_wave_a",
    }


def export_dataset(
    *,
    issues_path: Path,
    fixes_path: Path,
    rich_path: Path,
    out_dir: Path,
    max_rows: int = 0,
) -> dict[str, Any]:
    issues = load_issues(issues_path)
    fixes = load_fixes(fixes_path)
    all_ids = set(issues) | set(fixes)
    rich = load_rich_chunks(rich_path, all_ids)

    rows: list[dict[str, Any]] = []
    skipped_short = 0
    for cid in sorted(all_ids):
        row = merge_row(cid, issues.get(cid), fixes.get(cid), rich.get(cid))
        if not row:
            skipped_short += 1
            continue
        rows.append(row)
        if max_rows > 0 and len(rows) >= max_rows:
            break

    issue_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    for r in rows:
        verdict_counts[r["output"]["verdict"]] += 1
        for i in r["output"]["issues"]:
            issue_counts[i] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "chunk_qa_sft.jsonl"
    clf_path = out_dir / "chunk_qa_classifier.jsonl"
    with sft_path.open("w", encoding="utf-8") as sf, clf_path.open("w", encoding="utf-8") as cf:
        for r in rows:
            sf.write(json.dumps(r, ensure_ascii=False) + "\n")
            flat = {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "text": r["input"]["text"],
                "chunk_type": r["input"]["chunk_type"],
                "section_title": r["input"]["section_title"],
                "text_len": r["input"]["text_len"],
                "verdict": r["output"]["verdict"],
                "issues": r["output"]["issues"],
                "needs_action": int(r["output"]["needs_action"]),
            }
            cf.write(json.dumps(flat, ensure_ascii=False) + "\n")

    manifest = {
        "exported_at": _utc_now(),
        "sources": {
            "issues": str(issues_path),
            "fixes": str(fixes_path),
            "rich_chunks": str(rich_path),
        },
        "counts": {
            "issue_rows": len(issues),
            "fix_rows": len(fixes),
            "unique_chunk_ids": len(all_ids),
            "rich_resolved": len(rich),
            "exported": len(rows),
            "skipped_short_text": skipped_short,
        },
        "verdict_distribution": dict(verdict_counts.most_common()),
        "issue_distribution": dict(issue_counts.most_common(20)),
        "p0_issue_counts": {k: issue_counts.get(k, 0) for k in P0_ISSUES},
        "outputs": [
            str(sft_path.relative_to(ROOT)),
            str(clf_path.relative_to(ROOT)),
        ],
    }
    manifest_path = out_dir / "chunk_qa_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export chunk QA classifier dataset")
    parser.add_argument("--issues", type=Path, default=DEFAULT_ISSUES)
    parser.add_argument("--fixes", type=Path, default=DEFAULT_FIXES)
    parser.add_argument("--rich-chunks", type=Path, default=DEFAULT_RICH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows (0 = all)")
    args = parser.parse_args()

    manifest = export_dataset(
        issues_path=args.issues,
        fixes_path=args.fixes,
        rich_path=args.rich_chunks,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
