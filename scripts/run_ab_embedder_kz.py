#!/usr/bin/env python3
"""A/B: baseline e5-small vs fine-tuned embedder на RAG для КЗ и golden queries.

  python3 scripts/run_ab_embedder_kz.py
  python3 scripts/run_ab_embedder_kz.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLD_KZ = ROOT / "data" / "gastro_mvp" / "consult_gold.jsonl"
GOLDEN_RAG = ROOT / "eval" / "golden_queries.prod.jsonl"
BASE_MODEL = "intfloat/multilingual-e5-small"
FINETUNED = ROOT / "ml" / "experiments" / "embedder_exp_001" / "checkpoint_final"
OUT_DIR = ROOT / "ml" / "experiments" / "ab_kz_embedder"

# Ожидаемые фрагменты пути КП в топ-5 retrieve по тексту КЗ
CONDITION_PATH_NEEDLES: dict[str, list[str]] = {
    "gerd": ["рефлюкс", "гэрб", "k21"],
    "peptic_ulcer": ["язв", "k26", "k27", "двенадцат"],
    "gastritis": ["гастрит", "k29"],
    "celiac": ["целиак", "k90"],
    "ulcerative_colitis": ["колит", "k51", "язвенн"],
    "crohn": ["крон", "k50"],
    "acute_pancreatitis": ["панкреатит", "k85"],
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _setup_rag_env() -> None:
    os.environ.setdefault("RAG_GEMINI_EMBED_RERANK", "1")
    os.environ.setdefault("RAG_CHUNKS_SOURCE", "jsonl")
    os.environ.pop("RAG_CHUNKS_JSONL", None)
    os.environ.setdefault("GOOGLE_API_KEY", "local-ab-test")
    os.environ.setdefault("GEMINI_API_KEY", "local-ab-test")


def _wait_rag_ready() -> None:
    import rag_server as rs

    if not rs._chunks_load_done.wait(timeout=300):
        raise SystemExit("rag_server: timeout loading corpus")
    if rs._chunks_load_error:
        raise SystemExit(rs._chunks_load_error)


def _patch_embed(model_path: str) -> None:
    import rag_server as rs
    from ml.local_embedder import make_embed_fn

    rs._gemini_embed_one = make_embed_fn(model_path)  # type: ignore[method-assign]


def _retrieve_top_paths(query: str, *, max_chunks: int = 5) -> list[str]:
    import rag_server as rs

    rows = rs.retrieve(query, max_chunks=max_chunks, max_per_path=2, embed_rerank=True)
    return [str(r.get("path") or "") for r in rows]


def _path_hit(paths: list[str], needles: list[str]) -> bool:
    blob = " ".join(paths).lower()
    return any(n.lower() in blob for n in needles)


def _eval_kz_rag(model_path: str) -> dict[str, Any]:
    _patch_embed(model_path)
    import rag_server as rs

    rs._chunks_load_done.wait(timeout=1)
    cases = _load_jsonl(GOLD_KZ)
    rows: list[dict[str, Any]] = []
    for case in cases:
        cid = str(case.get("consultation_id") or "")
        cond = str(case.get("target_condition") or "")
        query = str(case.get("text") or "")[:1200]
        needles = CONDITION_PATH_NEEDLES.get(cond, [cond.replace("_", " ")])
        paths = _retrieve_top_paths(query)
        hit = _path_hit(paths, needles)
        rows.append(
            {
                "consultation_id": cid,
                "target_condition": cond,
                "rag_protocol_hit_top5": hit,
                "top_path": paths[0] if paths else "",
            }
        )
    passed = sum(1 for r in rows if r["rag_protocol_hit_top5"])
    return {
        "cases_total": len(rows),
        "cases_passed": passed,
        "pass_rate_pct": round(100.0 * passed / len(rows), 1) if rows else 0,
        "cases": rows,
    }


def _eval_golden_rag(model_path: str) -> dict[str, Any]:
    if not GOLDEN_RAG.is_file():
        return {"skipped": True, "reason": "no golden_queries.prod.jsonl"}
    _patch_embed(model_path)
    from eval.search_quality_eval import evaluate_one, load_golden_lines

    import rag_server as rs

    retrieve = rs.retrieve
    api_key = True
    rows = load_golden_lines(GOLDEN_RAG)
    ok = 0
    details: list[dict[str, Any]] = []
    for i, case in enumerate(rows):
        rep = evaluate_one(
            i,
            case,
            retrieve,
            max_chunks=6,
            max_per_path=2,
            gemini_advice=False,
            api_key_present=api_key,
            embed_requested=True,
        )
        if rep.ok:
            ok += 1
        details.append({"query": case.get("query"), "ok": rep.ok, "notes": case.get("notes")})
    return {
        "queries_total": len(rows),
        "queries_passed": ok,
        "pass_rate_pct": round(100.0 * ok / len(rows), 1) if rows else 0,
        "cases": details,
    }


def _gastro_rules_benchmark() -> dict[str, Any]:
    from clinical_knowledge.benchmark import run_gastro_gold_benchmark

    return run_gastro_gold_benchmark()


def run_ab() -> dict[str, Any]:
    if not FINETUNED.is_dir():
        raise SystemExit(f"Fine-tuned checkpoint not found: {FINETUNED}. Run run_embedder_experiment.py first.")

    _setup_rag_env()
    import rag_server as rs  # noqa: F401 — triggers corpus load

    _wait_rag_ready()

    t0 = time.time()
    rules = _gastro_rules_benchmark()

    print("A: baseline e5-small …", flush=True)
    kz_a = _eval_kz_rag(BASE_MODEL)
    rag_a = _eval_golden_rag(BASE_MODEL)

    print("B: fine-tuned …", flush=True)
    kz_b = _eval_kz_rag(str(FINETUNED))
    rag_b = _eval_golden_rag(str(FINETUNED))

    report = {
        "experiment": "ab_kz_embedder",
        "base_model": BASE_MODEL,
        "finetuned_checkpoint": str(FINETUNED.relative_to(ROOT)),
        "methodology_ru": (
            "A/B локального embedder в retrieve(): текст КЗ как запрос RAG, успех - релевантный КП в топ-5. "
            "Rule checker (send_gate) не использует embedder - одинаков на обоих плечах."
        ),
        "rules_layer": {
            "note": "Детерминированный rule_checker на consult_gold - не зависит от embedder",
            "baseline": rules,
            "finetuned": rules,
        },
        "arm_a_baseline": {
            "kz_rag_top5": kz_a,
            "golden_rag": rag_a,
        },
        "arm_b_finetuned": {
            "kz_rag_top5": kz_b,
            "golden_rag": rag_b,
        },
        "delta": {
            "kz_rag_pass_rate_pct": round(
                kz_b["pass_rate_pct"] - kz_a["pass_rate_pct"], 1
            ),
            "golden_rag_pass_rate_pct": round(
                (rag_b.get("pass_rate_pct") or 0) - (rag_a.get("pass_rate_pct") or 0), 1
            ),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = run_ab()
    out = OUT_DIR / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        d = report["delta"]
        ka = report["arm_a_baseline"]["kz_rag_top5"]
        kb = report["arm_b_finetuned"]["kz_rag_top5"]
        ga = report["arm_a_baseline"]["golden_rag"]
        gb = report["arm_b_finetuned"]["golden_rag"]
        print(f"KZ RAG top-5: A={ka['pass_rate_pct']}% B={kb['pass_rate_pct']}% Δ={d['kz_rag_pass_rate_pct']}%")
        print(f"Golden RAG: A={ga.get('pass_rate_pct')}% B={gb.get('pass_rate_pct')}% Δ={d['golden_rag_pass_rate_pct']}%")
        print(f"Rules (unchanged): {report['rules_layer']['baseline']['pass_rate_pct']}%")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
