#!/usr/bin/env python3
"""Benchmark поиска протоколов: ICD lookup vs retrieve_only assist (S5).

Пример:
  python3 scripts/benchmark_search_speed.py
  python3 scripts/benchmark_search_speed.py --base https://protocol-bimy.onrender.com
"""
from __future__ import annotations

import argparse
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

CASES = [
    {"id": "icd_lookup", "kind": "protocols-by-icd", "query": "ОРВИ кашель", "icd_codes": ["J06.9"]},
    {"id": "assist_icd_text", "kind": "assist", "query": "J06.9 ОРВИ кашель насморк", "retrieve_only": True},
    {"id": "assist_symptom", "kind": "assist", "query": "кашель и температура 38", "retrieve_only": True},
    {"id": "assist_icd_I10", "kind": "assist", "query": "I10 гипертония давление", "retrieve_only": True},
]


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


def _post_json(url: str, body: dict, *, timeout: int = 120) -> tuple[int, dict, float]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
        raw = resp.read().decode("utf-8")
        ms = (time.perf_counter() - t0) * 1000
        return resp.status, json.loads(raw) if raw.strip() else {}, ms


def _run_local(case: dict) -> dict:
    from rag_server import AssistIn, api_assist, api_search_protocols_by_icd
    from rag_server import ProtocolsByIcdIn

    t0 = time.perf_counter()
    if case["kind"] == "protocols-by-icd":
        out = api_search_protocols_by_icd(
            ProtocolsByIcdIn(query=case["query"], icd_codes=case["icd_codes"])
        )
    else:
        out = api_assist(
            AssistIn(
                query=case["query"],
                retrieve_only=bool(case.get("retrieve_only")),
                icd_fast_path=bool(case.get("icd_codes")),
                icd_codes=list(case.get("icd_codes") or []),
            )
        )
    wall_ms = (time.perf_counter() - t0) * 1000
    timing = out.get("search_timing") or {}
    protos = ((out.get("llm_json") or {}).get("protocols") or [])
    return {
        "id": case["id"],
        "wall_ms": round(wall_ms, 1),
        "path": timing.get("path") or out.get("finish_reason"),
        "lookup_ms": timing.get("lookup_ms"),
        "total_ms": timing.get("total_ms"),
        "icd_fast": out.get("icd_fast_lookup"),
        "n_protocols": len(protos),
        "top1": (protos[0].get("path") or "")[:80] if protos else "",
    }


def _run_remote(base: str, case: dict) -> dict:
    if case["kind"] == "protocols-by-icd":
        path = "/api/search/protocols-by-icd"
        body = {"query": case["query"], "icd_codes": case["icd_codes"]}
    else:
        path = "/api/assist"
        body = {
            "query": case["query"],
            "retrieve_only": bool(case.get("retrieve_only")),
            "icd_fast_path": True if case.get("icd_codes") else bool("J" in case["query"] or "I" in case["query"]),
            "icd_codes": list(case.get("icd_codes") or []),
        }
    try:
        status, out, wall_ms = _post_json(base.rstrip("/") + path, body)
    except urllib.error.HTTPError as e:
        return {"id": case["id"], "error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"id": case["id"], "error": str(e)[:200]}
    if status != 200:
        return {"id": case["id"], "error": out.get("detail") or f"HTTP {status}"}
    timing = out.get("search_timing") or {}
    protos = ((out.get("llm_json") or {}).get("protocols") or [])
    return {
        "id": case["id"],
        "wall_ms": round(wall_ms, 1),
        "path": timing.get("path") or out.get("finish_reason"),
        "lookup_ms": timing.get("lookup_ms"),
        "total_ms": timing.get("total_ms"),
        "icd_fast": out.get("icd_fast_lookup"),
        "n_protocols": len(protos),
        "top1": (protos[0].get("path") or "")[:80] if protos else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark protocol search speed")
    ap.add_argument("--base", default=os.environ.get("RENDER_URL", "").strip())
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--skip-rag-load", action="store_true", help="Только protocols-by-icd (без загрузки RAG)")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.base and not args.local_only:
        print(f"Remote: {args.base}\n")
        for case in CASES:
            if args.skip_rag_load and case["kind"] == "assist":
                continue
            row = _run_remote(args.base, case)
            rows.append(row)
            print(row)
    else:
        if not args.skip_rag_load:
            import rag_server as rs

            if not rs._chunks_load_done.is_set():
                print("Загрузка RAG…", flush=True)
                rs._run_load_data_background()
                rs._require_rag_loaded()
        for case in CASES:
            if args.skip_rag_load and case["kind"] == "assist":
                continue
            row = _run_local(case)
            rows.append(row)
            print(row)

    out_path = ROOT / "data" / "ml" / "reports" / "search_speed_benchmark_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
