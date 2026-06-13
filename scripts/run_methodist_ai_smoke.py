#!/usr/bin/env python3
"""Прогон L1 + AI-оценка методиста для pl_1_f и report_n_2."""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import env_load

    env_load.load_project_env(ROOT)
except ImportError:
    pass

FIXTURES = ROOT / "tests" / "fixtures" / "consultations"
CLIENTS = ROOT / "clients_consult"


def _load_pdf_text(path: Path) -> str:
    import fitz

    return "\n".join(page.get_text() for page in fitz.open(path)).strip()


def _load(name: str) -> str:
    """name: pl_1_f.pdf или pl_1_f.txt - ищет в clients_consult, затем fixtures."""
    candidates = [CLIENTS / name, FIXTURES / name]
    stem = Path(name).stem
    candidates.extend([CLIENTS / f"{stem}.pdf", FIXTURES / f"{stem}.txt"])
    for p in candidates:
        if p.is_file():
            if p.suffix.lower() == ".pdf":
                return _load_pdf_text(p)
            return p.read_text(encoding="utf-8", errors="replace").strip()
    raise FileNotFoundError(f"Не найден кейс {name!r} в {CLIENTS} или {FIXTURES}")


def run_case(case_id: str, text: str, tier: str = "L1") -> dict:
    from clinical_knowledge.consult_tiering import run_consult_by_tier
    from clinical_knowledge.feedback_store import (
        build_kz_analysis_event,
        save_analysis_snapshot,
        store_secure_kz_text,
    )
    from clinical_knowledge.methodist_ai_review import run_methodist_ai_review
    from clinical_knowledge.methodist_enrich import enrich_methodist_tier_payload

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

    ai_review = None
    ai_error = None
    ai_ms = 0
    try:
        ai_t0 = time.perf_counter()
        ai_review = run_methodist_ai_review(result, text)
        ai_ms = int((time.perf_counter() - ai_t0) * 1000)
    except Exception as exc:
        ai_error = str(exc)[:400]

    rev = result.get("review") or {}
    sa = (result.get("structured_analysis") or {}).get("compliance") or {}
    cr = (result.get("clinical_rules") or {}).get("rules_check") or {}

    return {
        "case_id": case_id,
        "tier": tier,
        "analysis_ms": latency_ms,
        "ai_review_ms": ai_ms,
        "overall_pct": rev.get("overall_compliance_pct") or sa.get("overall_score"),
        "structured_pct": sa.get("overall_score"),
        "rules_pct": cr.get("rules_compliance_pct"),
        "status": sa.get("overall_status"),
        "matched_protocols": [
            (m.get("path") or m.get("source_path") or "")[:80]
            for m in (result.get("clinical_rules") or {}).get("matched_protocols") or []
        ][:5],
        "ai_review": ai_review,
        "ai_error": ai_error,
    }


def main() -> int:
    cases = [
        ("pl_1_f", "pl_1_f.pdf", "L1"),
        ("report_n_2", "report_n_2.pdf", "L1"),
    ]
    out_dir = ROOT / "ml" / "experiments" / "methodist_ai_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    for case_id, filename, tier in cases:
        print(f"\n=== {case_id} ({tier}) ← {filename} ===", flush=True)
        try:
            text = _load(filename)
            print(f"Текст КЗ: {len(text)} символов", flush=True)
            rep = run_case(case_id, text, tier=tier)
            reports.append(rep)
            ai = rep.get("ai_review")
            print(f"Система: overall={rep['overall_pct']}% struct={rep['structured_pct']}% rules={rep['rules_pct']}%")
            print(f"Протоколы: {rep['matched_protocols']}")
            if ai:
                print(f"AI gold: {ai.get('kz_compliance_gold')} | rating={ai.get('system_accuracy_rating')}/5 | {ai.get('system_accuracy_verdict')}")
                print(f"AI summary: {ai.get('summary_ru', '')[:280]}")
                print("Правки движка (engine_improvements_ru):")
                for i, imp in enumerate(
                    ai.get("engine_improvements_ru") or ai.get("improvements_ru") or [], 1
                ):
                    print(f"  {i}. {imp}")
                print(f"System notes: {(ai.get('system_notes_ru') or '')[:200]}")
                print(f"Tags: {ai.get('tags')} | confidence: {ai.get('confidence')} | model: {ai.get('model_used')}")
            elif rep.get("ai_error"):
                print(f"AI-этап: ОШИБКА - {rep['ai_error']}")
            print(f"Timing: analysis {rep['analysis_ms']}ms, AI {rep['ai_review_ms']}ms")
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            reports.append({"case_id": case_id, "error": str(exc)})

    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0 if all(r.get("ai_review") or r.get("ai_error") for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
