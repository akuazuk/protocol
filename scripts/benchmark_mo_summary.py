#!/usr/bin/env python3
"""Измерить тёплый p95 SQL-endpoint /summary на локальной витрине."""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=ROOT / "data" / "medical_exams" / "warehouse" / "mo_analytics.sqlite",
    )
    parser.add_argument("--month", default="2026-07")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    if not args.warehouse.is_file():
        parser.error(f"витрина не найдена: {args.warehouse}")
    os.environ["MO_ANALYTICS_DB"] = str(args.warehouse)
    os.environ["MO_BACKEND_SOURCE"] = "warehouse"
    os.environ["METHODIST_TOKEN"] = "local-benchmark-token"
    from fastapi.testclient import TestClient

    import rag_server

    client = TestClient(rag_server.app)
    url = f"/api/methodist/mo/summary?period=month&month={args.month}"
    headers = {"X-Methodist-Token": "local-benchmark-token"}
    for _ in range(5):
        response = client.get(url, headers=headers)
        response.raise_for_status()
    timings = []
    for _ in range(max(1, args.runs)):
        started = time.perf_counter()
        response = client.get(url, headers=headers)
        response.raise_for_status()
        timings.append((time.perf_counter() - started) * 1000)
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))]
    print(
        f"HTTP /summary month={args.month} runs={len(timings)} "
        f"median_ms={statistics.median(timings):.2f} p95_ms={p95:.2f}"
    )
    return 0 if p95 < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
