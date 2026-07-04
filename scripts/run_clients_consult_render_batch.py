#!/usr/bin/env python3
"""Batch L1 (+ optional AI-review) для clients_consult/ через Render API."""
from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import env_load

    env_load.load_project_env(ROOT)
except ImportError:
    pass

CLIENTS = ROOT / "clients_consult"
from clinical_knowledge.patient_upload_classifier import is_b2c_lab_filename

SUPPORTED = {".pdf", ".txt", ".md", ".docx", ".rtf", ".odt", ".html"}


def _is_b2c_lab_analysis(case_id: str) -> bool:
    """Файлы A/a* - B2C анализы, не заключения (КЗ)."""
    return is_b2c_lab_filename(case_id)


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


def _post(base: str, path: str, body: dict, *, token: str = "", timeout: int = 360) -> dict:
    headers = {"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}
    if token:
        headers["X-Methodist-Token"] = token
        headers["X-Methodist-Reviewer"] = os.environ.get("METHODIST_REVIEWER", "render-batch")
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{base.rstrip('/')}{path}", data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_text(path: Path) -> str:
    from clinical_knowledge.text_extract import extract_text_from_path

    return extract_text_from_path(path).strip()


def _discover(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED and p.name.lower() != "readme.md"
    )


def _needs_ai(rep: dict) -> bool:
    failed = int(rep.get("failed_rules_count") or 0)
    rules = rep.get("rules_pct")
    overall = rep.get("overall_pct")
    if failed >= 2:
        return True
    if rules is not None and float(rules) < 60:
        return True
    if overall is not None and float(overall) < 70:
        return True
    if overall is not None and float(overall) >= 85 and failed > 0:
        return True
    return False


def run_case(
    path: Path,
    *,
    base: str,
    token: str,
    ai_review: str,
    tier: str = "L1",
) -> dict:
    case_id = path.stem
    text = _load_text(path)
    t0 = time.perf_counter()
    result = _post(
        base,
        "/api/consult-review/tier",
        {
            "tier": tier,
            "text": text,
            "consultation_id": case_id,
            "methodist_mode": True,
            "category_slugs": "",
        },
        timeout=360,
    )
    ms = int((time.perf_counter() - t0) * 1000)
    rev = result.get("review") or {}
    sa = (result.get("structured_analysis") or {}).get("compliance") or {}
    cr = (result.get("clinical_rules") or {}).get("rules_check") or {}
    failed = [f for f in (cr.get("findings") or []) if not f.get("passed") and not f.get("skipped")]

    def _paths(items: object) -> list[str]:
        out: list[str] = []
        for it in items or []:  # type: ignore[union-attr]
            p = it.get("path") if isinstance(it, dict) else str(it)
            if p:
                out.append(str(p))
        return out

    # Актуальные ключи ответа /api/consult-review/tier (с fallback на старые имена).
    matched = _paths(result.get("protocol_paths_used")) or _paths(
        (result.get("clinical_rules") or {}).get("matched_protocols")
    )
    retrieval = _paths(result.get("retrieval_paths")) or _paths(
        result.get("retrieval_top_paths") or result.get("matched_protocol_paths")
    )
    fragments = result.get("consult_protocol_fragments")
    rag_chunks_n = len(fragments) if isinstance(fragments, list) else 0

    rep: dict = {
        "file": path.name,
        "case_id": case_id,
        "doc_kind": "b2c_analysis" if _is_b2c_lab_analysis(case_id) else "kz",
        "upload_mismatch": bool(result.get("upload_mismatch")),
        "wrong_document_kind": result.get("wrong_document_kind"),
        "review_tier": tier,
        "analysis_id": result.get("analysis_id"),
        "analysis_ms": ms,
        "overall_pct": rev.get("overall_compliance_pct") or sa.get("overall_score"),
        "rules_pct": cr.get("rules_compliance_pct"),
        "status": sa.get("overall_status"),
        "failed_rules_count": len(failed),
        "failed_rule_ids": [str(f.get("rule_id") or "") for f in failed[:10]],
        "failed_rule_titles": [str(f.get("title_ru") or f.get("message_ru") or "")[:80] for f in failed[:5]],
        "matched_protocols": [p[-70:] for p in matched[:3]],
        "matched_protocols_full": matched[:3],
        "retrieval_top": [p[-70:] for p in retrieval[:3]],
        "retrieval_top_full": retrieval[:3],
        "rag_chunks_n": rag_chunks_n,
        "text_len": len(text),
        "server_version": result.get("server_version"),
    }

    if ai_review != "off" and token and (ai_review == "all" or _needs_ai(rep)):
        try:
            ai_t0 = time.perf_counter()
            ai = _post(
                base,
                "/api/methodist/ai-review",
                {"analysis_id": rep["analysis_id"]},
                token=token,
                timeout=180,
            )
            rep["ai_review_ms"] = int((time.perf_counter() - ai_t0) * 1000)
            rep["ai_rating"] = (ai.get("ai_review") or ai).get("rating") if isinstance(ai, dict) else None
            ar = ai.get("ai_review") if isinstance(ai, dict) else ai
            if isinstance(ar, dict):
                rep["ai_tags"] = ar.get("tags") or []
                rep["ai_summary"] = (ar.get("summary_ru") or "")[:300]
                rep["ai_improvements"] = (ar.get("engine_improvements_ru") or [])[:5]
        except Exception as exc:
            rep["ai_error"] = str(exc)[:300]

    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", type=Path, default=CLIENTS)
    ap.add_argument("--base", default=os.environ.get("RENDER_URL", "https://protocol-bimy.onrender.com"))
    ap.add_argument("--ai-review", choices=("off", "auto", "all"), default="auto")
    ap.add_argument("--tier", choices=("L1", "L2"), default="L1")
    ap.add_argument("--kz-only", action="store_true", help="Пропустить A/a* B2C анализы")
    ap.add_argument("--cases", default="", help="Список case_id через запятую")
    ap.add_argument("--out", type=Path, default=ROOT / "ml" / "experiments" / f"batch_clients_consult_{time.strftime('%Y-%m-%d')}")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    token = (os.environ.get("METHODIST_TOKEN") or os.environ.get("METHODIST_PIN") or "").strip()
    if args.ai_review != "off" and not token:
        print("WARN: нет METHODIST_TOKEN - AI-review пропущен", file=sys.stderr)
        args.ai_review = "off"

    paths = _discover(args.folder.resolve())
    if args.cases.strip():
        allow = {c.strip() for c in args.cases.split(",") if c.strip()}
        paths = [p for p in paths if p.stem in allow]
    if args.kz_only:
        paths = [p for p in paths if not _is_b2c_lab_analysis(p.stem)]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("Нет файлов", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    errors: list[dict] = []

    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {p.name} ...", flush=True)
        try:
            rep = run_case(p, base=args.base, token=token, ai_review=args.ai_review, tier=args.tier)
            reports.append(rep)
            print(
                f"  overall={rep.get('overall_pct')}% rules={rep.get('rules_pct')}% "
                f"failed={rep.get('failed_rules_count')} ms={rep.get('analysis_ms')}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"file": p.name, "error": str(exc)[:400]})
            print(f"  ERROR: {exc}", file=sys.stderr)

    summary = {
        "base": args.base,
        "folder": str(args.folder),
        "total": len(paths),
        "ok": len(reports),
        "errors": len(errors),
        "ai_review": args.ai_review,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if reports:
        ovs = [float(r["overall_pct"]) for r in reports if r.get("overall_pct") is not None]
        summary["overall_avg"] = round(sum(ovs) / len(ovs), 1) if ovs else None
        summary["failed_ge3"] = sum(1 for r in reports if (r.get("failed_rules_count") or 0) >= 3)
        summary["overall_lt70"] = sum(1 for r in reports if r.get("overall_pct") is not None and float(r["overall_pct"]) < 70)
        kz = [r for r in reports if r.get("doc_kind") == "kz" and not r.get("upload_mismatch")]
        if kz:
            kz_ovs = [float(r["overall_pct"]) for r in kz if r.get("overall_pct") is not None]
            summary["kz_only"] = {
                "count": len(kz),
                "overall_avg": round(sum(kz_ovs) / len(kz_ovs), 1) if kz_ovs else None,
                "overall_lt70": sum(
                    1 for r in kz if r.get("overall_pct") is not None and float(r["overall_pct"]) < 70
                ),
            }
        summary["b2c_analysis_count"] = sum(1 for r in reports if r.get("doc_kind") == "b2c_analysis")

    payload = {"summary": summary, "reports": reports, "errors": errors}
    (args.out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.out / "batch_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fields = ["file", "case_id", "overall_pct", "rules_pct", "failed_rules_count", "analysis_ms", "ai_rating", "ai_tags"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in reports:
            row = dict(r)
            if isinstance(row.get("ai_tags"), list):
                row["ai_tags"] = ",".join(str(x) for x in row["ai_tags"])
            w.writerow({k: row.get(k) for k in fields})

    print(f"\nSaved: {args.out / 'report.json'}")
    print(f"OK {summary['ok']}/{summary['total']}, avg overall={summary.get('overall_avg')}%")
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
