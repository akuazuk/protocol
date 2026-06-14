#!/usr/bin/env python3
"""Batch L1 (+ optional AI-review) для папки КЗ — методистский Tier T1.

Примеры:
  python3 scripts/run_methodist_batch.py --folder tests/fixtures/consultations
  python3 scripts/run_methodist_batch.py --folder tests/fixtures/consultations --tier L1 --workers 2
  python3 scripts/run_methodist_batch.py --folder tests/fixtures/consultations --ai-review auto
  python3 scripts/run_methodist_batch.py --glob 'pl_*.txt' --list-only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import env_load

    env_load.load_project_env(ROOT)
except ImportError:
    pass

SUPPORTED = {".pdf", ".txt", ".md", ".docx", ".rtf", ".odt", ".html"}


def _load_text(path: Path) -> str:
    from clinical_knowledge.text_extract import extract_text_from_path

    return extract_text_from_path(path).strip()


def _discover(folder: Path, glob_pat: str | None) -> list[Path]:
    if glob_pat:
        return sorted(p for p in folder.glob(glob_pat) if p.is_file() and p.suffix.lower() in SUPPORTED)
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def _needs_ai_review(rep: dict, *, mode: str) -> bool:
    if mode == "off":
        return False
    if mode == "all":
        return True
    rules_pct = rep.get("rules_pct")
    failed_n = rep.get("failed_rules_count") or 0
    overall = rep.get("overall_pct")
    if failed_n >= 3:
        return True
    if rules_pct is not None and float(rules_pct) < 50:
        return True
    if overall is not None and float(overall) >= 88 and failed_n > 0:
        return True
    return False


def run_one(path: Path, *, tier: str, ai_mode: str) -> dict:
    from clinical_knowledge.consult_tiering import run_consult_by_tier
    from clinical_knowledge.feedback_store import (
        build_kz_analysis_event,
        save_analysis_snapshot,
        store_secure_kz_text,
    )
    from clinical_knowledge.methodist_enrich import enrich_methodist_tier_payload

    case_id = path.stem
    text = _load_text(path)
    t0 = time.perf_counter()
    result = run_consult_by_tier(
        tier=tier,
        text=text,
        consultation_id=case_id,
        category_slugs="",
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    result = enrich_methodist_tier_payload(
        result,
        tier=tier,
        full_text=text,
        latency_ms=latency_ms,
    )
    analysis_id = str(uuid.uuid4())
    result["analysis_id"] = analysis_id
    result["text_hash"] = store_secure_kz_text(text)
    event = build_kz_analysis_event(
        result=result,
        tier=tier,
        full_text=text,
        consultation_id=case_id,
        latency_ms=latency_ms,
    )
    event["analysis_id"] = analysis_id
    save_analysis_snapshot(
        analysis_id,
        {
            "analysis_id": analysis_id,
            "text_hash": result["text_hash"],
            "tier": tier,
            "text_excerpt": text[:500],
            "api_result": result,
        },
    )

    rev = result.get("review") or {}
    sa = (result.get("structured_analysis") or {}).get("compliance") or {}
    cr = (result.get("clinical_rules") or {}).get("rules_check") or {}
    failed = [
        f for f in (cr.get("findings") or [])
        if not f.get("passed") and not f.get("skipped")
    ]

    rep: dict = {
        "file": path.name,
        "case_id": case_id,
        "analysis_id": analysis_id,
        "text_hash": result.get("text_hash"),
        "tier": tier,
        "analysis_ms": latency_ms,
        "overall_pct": rev.get("overall_compliance_pct") or sa.get("overall_score"),
        "structured_pct": sa.get("overall_score"),
        "rules_pct": cr.get("rules_compliance_pct"),
        "status": sa.get("overall_status"),
        "failed_rules_count": len(failed),
        "failed_rule_ids": [str(f.get("rule_id") or "") for f in failed[:8]],
        "matched_protocols": [
            (m.get("path") or m.get("source_path") or "")[:100]
            for m in (result.get("clinical_rules") or {}).get("matched_protocols") or []
        ][:5],
        "text_len": len(text),
    }

    ai_review = None
    ai_error = None
    rep["ai_review_ms"] = 0
    if _needs_ai_review(rep, mode=ai_mode):
        try:
            from clinical_knowledge.methodist_ai_review import run_methodist_ai_review

            ai_t0 = time.perf_counter()
            ai_review = run_methodist_ai_review(result, text)
            rep["ai_review_ms"] = int((time.perf_counter() - ai_t0) * 1000)
        except Exception as exc:
            ai_error = str(exc)[:400]
    rep["ai_review"] = ai_review
    rep["ai_error"] = ai_error
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch Methodist L1 analysis")
    ap.add_argument("--folder", type=Path, default=ROOT / "tests" / "fixtures" / "consultations")
    ap.add_argument("--glob", type=str, default=None, help="Glob внутри folder, напр. 'pl_*.txt'")
    ap.add_argument("--tier", choices=("L0", "L1", "L2"), default="L1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--ai-review",
        choices=("off", "auto", "all"),
        default="off",
        help="auto: спорные кейсы (rules<50%%, много failed)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        print(f"Папка не найдена: {folder}", file=sys.stderr)
        return 1

    paths = _discover(folder, args.glob)
    if not paths:
        print(f"Нет файлов КЗ в {folder}", file=sys.stderr)
        return 1

    if args.list_only:
        for p in paths:
            print(p.name)
        return 0

    out_dir = args.out or (ROOT / "ml" / "experiments" / f"methodist_batch_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    errors: list[dict] = []

    def _job(p: Path) -> dict:
        return run_one(p, tier=args.tier, ai_mode=args.ai_review)

    if args.workers <= 1:
        for p in paths:
            print(f"=== {p.name} ===", flush=True)
            try:
                rep = _job(p)
                reports.append(rep)
                print(
                    f"  overall={rep.get('overall_pct')}% rules={rep.get('rules_pct')}% "
                    f"failed={rep.get('failed_rules_count')}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({"file": p.name, "error": str(exc)})
                print(f"  ERROR: {exc}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_job, p): p for p in paths}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    reports.append(fut.result())
                    print(f"OK {p.name}", flush=True)
                except Exception as exc:
                    errors.append({"file": p.name, "error": str(exc)})
                    print(f"ERROR {p.name}: {exc}", file=sys.stderr)

    reports.sort(key=lambda r: r.get("file") or "")
    summary = {
        "folder": str(folder),
        "tier": args.tier,
        "ai_review": args.ai_review,
        "total": len(paths),
        "ok": len(reports),
        "errors": len(errors),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (out_dir / "report.json").write_text(
        json.dumps({"summary": summary, "reports": reports, "errors": errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = out_dir / "batch_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "case_id",
                "analysis_id",
                "overall_pct",
                "rules_pct",
                "status",
                "failed_rules_count",
                "analysis_ms",
                "ai_review_ms",
            ],
        )
        w.writeheader()
        for r in reports:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    print(f"\nSaved: {out_dir / 'report.json'}")
    print(f"CSV: {csv_path}")
    print(f"OK {summary['ok']}/{summary['total']}, errors {summary['errors']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
