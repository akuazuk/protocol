#!/usr/bin/env python3
"""Tier-очередь chunk QA: P0/P1/P2, покрытие рубрик, ICD из clients_consult KZ."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.chunk_quality import detect_issues, quality_score
from clinical_knowledge.patient_upload_classifier import is_b2c_lab_filename

DEFAULT_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.final.jsonl"
FALLBACK_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.v2.jsonl"
DEFAULT_FIXES = ROOT / "data" / "ml" / "chunk_qa_fixes_merged.jsonl"
DEFAULT_OUT = ROOT / "data" / "ml" / "chunk_qa_queue_tiered.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "ml" / "chunk_qa_queue_tiered_manifest.json"
CLIENTS = ROOT / "clients_consult"
FEEDBACK_DIR = ROOT / "data" / "ml" / "feedback"
SECTION_MAP_DIR = ROOT / "data" / "ml" / "protocol_section_map"

P0_ISSUES = frozenset({"preamble_leak", "icd_inflation", "type_body_but_clinical"})
P1_CLINICAL = frozenset({
    "diagnostics", "treatment", "criteria_block", "pharmacotherapy", "drug_list",
})
SKIP_FIX_CONF = 0.85


def _load_fixes(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
            cid = str(row.get("chunk_id") or "")
            if cid:
                out[cid] = row
        except json.JSONDecodeError:
            pass
    return out


def _section_mapped_doc_ids() -> set[str]:
    if not SECTION_MAP_DIR.is_dir():
        return set()
    out: set[str] = set()
    for fp in SECTION_MAP_DIR.glob("*.json"):
        try:
            row = json.loads(fp.read_text(encoding="utf-8"))
            if row.get("error"):
                continue
            did = str(row.get("doc_id") or fp.stem)
            if did:
                out.add(did)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _feedback_paths() -> set[str]:
    paths: set[str] = set()
    if not FEEDBACK_DIR.is_dir():
        return paths
    for fp in FEEDBACK_DIR.glob("*.jsonl"):
        for line in fp.open(encoding="utf-8"):
            try:
                row = json.loads(line)
                for key in ("source_path", "path", "protocol_path", "chosen_path"):
                    sp = row.get(key)
                    if sp:
                        paths.add(str(sp))
            except json.JSONDecodeError:
                pass
    return paths


def _rubric(source_path: str) -> str:
    sp = str(source_path or "").replace("\\", "/")
    parts = sp.split("/")
    return parts[1] if len(parts) > 2 else "unknown"


def _kz_protocol_paths(folder: Path) -> set[str]:
    """PDF paths протоколов, релевантных KZ из clients_consult (все специальности)."""
    if not folder.is_dir():
        return set()
    from clinical_knowledge.consult_analysis import analyze_consultation_text
    from clinical_knowledge.text_extract import extract_text_from_path

    out: set[str] = set()
    for pdf in sorted(folder.glob("*.pdf")):
        if is_b2c_lab_filename(pdf.stem):
            continue
        try:
            text = extract_text_from_path(pdf).strip()
            if len(text) < 80:
                continue
            sa = analyze_consultation_text(text, consultation_id=pdf.stem, with_markdown=False)
            for m in sa.get("matches") or []:
                sp = str(m.get("source_path") or "")
                if sp:
                    out.add(sp.replace("\\", "/"))
        except Exception:
            continue
    return out


def _should_skip(row: dict, issues: list[str], score: float, fixes: dict[str, dict]) -> bool:
    cid = str(row.get("chunk_id") or "")
    fix = fixes.get(cid)
    if fix and float(fix.get("confidence") or 0) >= SKIP_FIX_CONF:
        return True

    ctype = str(row.get("chunk_type") or "body").lower()
    p0 = set(issues) & P0_ISSUES

    if not p0 and issues == ["weak_section_title"] and score >= 0.88:
        return True
    if not p0 and ctype == "body" and score >= 0.9:
        return True
    if not p0 and ctype in ("terms", "appendix") and "too_short" in issues and score >= 0.8:
        return True
    return False


def _priority(
    row: dict,
    issues: list[str],
    score: float,
    *,
    kz_paths: set[str],
    feedback: set[str],
    section_mapped: set[str],
) -> tuple[int, str]:
    sp = str(row.get("source_path") or "").replace("\\", "/")
    doc_id = str(row.get("doc_id") or "")
    ctype = str(row.get("chunk_type") or "body").lower()
    iss = set(issues)

    if sp in feedback:
        return 95, "methodist_feedback"

    # B2C crosscheck: diagnostics без entities на протоколах из KZ funnel
    if (
        sp in kz_paths
        and ctype == "diagnostics"
        and "empty_entities" in iss
        and score < 0.8
    ):
        return 93, "b2c_crosscheck"

    if sp in kz_paths and (iss & P0_ISSUES or score < 0.75):
        return 92, "kz_linked_protocol"

    if iss & P0_ISSUES:
        if "type_body_but_clinical" in iss and doc_id and doc_id not in section_mapped:
            return 100, "p0_body_clinical_no_section_map"
        return 100, "p0_critical"

    if "truncated_list" in iss and ctype in P1_CLINICAL:
        return 85, "p1_truncated_clinical"
    if "too_long" in iss and ctype in P1_CLINICAL:
        return 82, "p1_too_long_clinical"
    if "empty_entities" in iss and ctype in P1_CLINICAL and score < 0.75:
        return 80, "p1_empty_entities"

    if score < 0.65:
        return 70, "p2_low_score"

    return 0, ""


def build_tiered_queue(
    chunks_path: Path,
    *,
    fixes: dict[str, dict],
    kz_paths: set[str],
    feedback: set[str],
    section_mapped: set[str],
    max_total: int = 10000,
    min_per_rubric: int = 5,
) -> tuple[list[dict], dict]:
    candidates: dict[str, dict] = {}
    rubric_pool: dict[str, list[tuple[int, dict]]] = defaultdict(list)

    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cid = str(row.get("chunk_id") or "")
            if not cid:
                continue
            issues = detect_issues(row)
            score = quality_score(row)
            if _should_skip(row, issues, score, fixes):
                continue
            pri, reason = _priority(
                row,
                issues,
                score,
                kz_paths=kz_paths,
                feedback=feedback,
                section_mapped=section_mapped,
            )
            if pri <= 0:
                continue
            item = {
                "chunk_id": cid,
                "doc_id": row.get("doc_id"),
                "source_path": row.get("source_path"),
                "chunk_type": row.get("chunk_type"),
                "quality_score": score,
                "issues": issues,
                "priority": pri,
                "reason": reason,
                "rubric": _rubric(str(row.get("source_path") or "")),
            }
            candidates[cid] = item
            rubric_pool[item["rubric"]].append((pri, item))

    # стратификация: min_per_rubric лучших P0/P1
    selected: dict[str, dict] = {}
    for rub, items in rubric_pool.items():
        items.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))
        for _, item in items[:min_per_rubric]:
            selected[item["chunk_id"]] = item

    # остальное по priority до max_total
    rest = sorted(candidates.values(), key=lambda x: (-int(x["priority"]), x["chunk_id"]))
    for item in rest:
        selected[item["chunk_id"]] = item
        if len(selected) >= max_total:
            break

    out = sorted(selected.values(), key=lambda x: (-int(x["priority"]), x["chunk_id"]))
    if len(out) > max_total:
        out = out[:max_total]

    manifest = {
        "queue_size": len(out),
        "candidates_before_cap": len(candidates),
        "kz_protocol_paths_n": len(kz_paths),
        "kz_protocol_paths": sorted(kz_paths)[:30],
        "section_map_docs_n": len(section_mapped),
        "priority_counts": dict(Counter(int(x["priority"]) for x in out)),
        "reason_counts": dict(Counter(x.get("reason") for x in out)),
        "rubric_counts": dict(Counter(x.get("rubric") for x in out)),
        "rubrics_covered": len(set(x.get("rubric") for x in out)),
    }
    return out, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build tiered corpus-wide chunk QA queue")
    ap.add_argument("--chunks", type=Path, default=None)
    ap.add_argument("--fixes", type=Path, default=DEFAULT_FIXES)
    ap.add_argument("--kz-folder", type=Path, default=CLIENTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--max-total", type=int, default=10000)
    ap.add_argument("--min-per-rubric", type=int, default=5)
    args = ap.parse_args()

    chunks_path = args.chunks or (DEFAULT_CHUNKS if DEFAULT_CHUNKS.is_file() else FALLBACK_CHUNKS)
    if not chunks_path.is_file():
        print(f"Нет {chunks_path}", file=sys.stderr)
        return 1

    fixes = _load_fixes(args.fixes)
    feedback = _feedback_paths()
    section_mapped = _section_mapped_doc_ids()
    print(f"KZ protocol mapping from {args.kz_folder} ...", flush=True)
    kz_paths = _kz_protocol_paths(args.kz_folder.resolve())
    print(f"  kz-linked protocols: {len(kz_paths)}", flush=True)
    print(f"  section_map docs: {len(section_mapped)}", flush=True)

    queue, manifest = build_tiered_queue(
        chunks_path,
        fixes=fixes,
        kz_paths=kz_paths,
        feedback=feedback,
        section_mapped=section_mapped,
        max_total=args.max_total,
        min_per_rubric=args.min_per_rubric,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for item in queue:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest["out"] = str(args.out)
    manifest["chunks"] = str(chunks_path)
    manifest["fixes_skipped"] = len(fixes)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
