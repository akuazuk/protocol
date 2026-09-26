#!/usr/bin/env python3
"""Тайминги API МО Аналитики одной командой (план 2026-09-26, волна T).

Меряет каждый вызов `/api/methodist/mo/*`, которым пользуется кабинет,
дважды: первый запрос («холодный» для этого процесса probe) и повтор
(«тёплый»). Тела ответов не печатаются и не сохраняются - только статус,
размер и миллисекунды. Токен берётся из окружения и в вывод не попадает.

Запуск на GCE внутри контейнера (канон) или с Mac по HTTPS:

    METHODIST_TOKEN=... python3 scripts/ops/mo_api_latency_probe.py \
        --base http://127.0.0.1:8000 --month 2026-09 --out /tmp/probe.json

    python3 scripts/ops/mo_api_latency_probe.py --compare before.json after.json

Пороги из плана: первый экран ≤ 1500 мс, любой фильтр ≤ 800 мс, разбор ≤ 1200 мс.
`--compare` печатает регресс > 20 % как FAIL и выходит с кодом 1.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:8000"
API = "/api/methodist/mo"

# (имя, путь, параметры, порог мс). Параметры с {from}/{to}/{month} подставляются.
PROBES: list[tuple[str, str, dict[str, str], int]] = [
    ("cases_month_p1", "/cases", {"date_from": "{from}", "date_to": "{to}", "page": "1", "page_size": "50"}, 1500),
    ("cases_sort_score", "/cases", {"date_from": "{from}", "date_to": "{to}", "sort_by": "score", "page_size": "50"}, 800),
    ("cases_queue_only", "/cases", {"date_from": "{from}", "date_to": "{to}", "queue_only": "1", "page_size": "50"}, 1000),
    ("cases_finding_lab", "/cases", {"date_from": "{from}", "date_to": "{to}", "finding_family": "lab", "page_size": "50"}, 800),
    ("cases_grade_poor", "/cases", {"date_from": "{from}", "date_to": "{to}", "overall_grade": "poor", "page_size": "50"}, 800),
    ("cases_q_icd", "/cases", {"date_from": "{from}", "date_to": "{to}", "q": "I10", "page_size": "50"}, 800),
    ("cases_q_word", "/cases", {"date_from": "{from}", "date_to": "{to}", "q": "гипертензия", "page_size": "50"}, 800),
    ("facets", "/facets", {"date_from": "{from}", "date_to": "{to}"}, 500),
    ("score_dashboard", "/score-dashboard", {"date_from": "{from}", "date_to": "{to}", "period": "custom"}, 1500),
    ("daily_report", "/daily-report", {"date": "{to}"}, 1000),
    ("freshness", "/freshness", {}, 300),
    ("drugs_labs_kpis_drug", "/drugs-labs-kpis", {"date_from": "{from}", "date_to": "{to}", "family": "drug"}, 1500),
    ("drugs_labs_kpis_lab", "/drugs-labs-kpis", {"date_from": "{from}", "date_to": "{to}", "family": "lab"}, 1500),
    ("dimensions_doctors", "/dimensions/doctors", {"date_from": "{from}", "date_to": "{to}"}, 1000),
    ("reports", "/reports", {}, 1000),
    ("kp_sync", "/kp-sync", {}, 800),
    ("rceth_sync", "/rceth-sync", {}, 800),
    ("mis_coverage", "/mis/coverage", {"date_from": "{from}", "date_to": "{to}"}, 800),
]


def _yesterday_minsk() -> dt.date:
    # API отвергает date_to позже «вчера» по Минску (UTC+3): для текущего
    # месяца конец окна нужно обрезать, иначе все зондирования дадут 422.
    now_minsk = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)
    return now_minsk.date() - dt.timedelta(days=1)


def _month_bounds(month: str) -> tuple[str, str]:
    year, mon = (int(x) for x in month.split("-"))
    first = dt.date(year, mon, 1)
    nxt = dt.date(year + (mon == 12), 1 if mon == 12 else mon + 1, 1)
    last = min(nxt - dt.timedelta(days=1), _yesterday_minsk())
    if last < first:
        last = first
    return first.isoformat(), last.isoformat()


def _request(base: str, path: str, params: dict[str, str], token: str, timeout: float) -> dict[str, Any]:
    url = base.rstrip("/") + API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"X-Methodist-Token": token, "Accept": "application/json"})
    started = time.perf_counter()
    status: int | str
    size = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - адрес задаёт оператор
            body = resp.read()
            status = resp.status
            size = len(body)
    except urllib.error.HTTPError as err:
        status = err.code
        try:
            size = len(err.read())
        except Exception:  # noqa: BLE001
            size = 0
    except Exception as err:  # noqa: BLE001
        status = type(err).__name__
    return {"status": status, "ms": round((time.perf_counter() - started) * 1000), "bytes": size}


def run_probe(base: str, month: str, token: str, timeout: float, case_id: str | None) -> dict[str, Any]:
    date_from, date_to = _month_bounds(month)
    subst = {"{from}": date_from, "{to}": date_to, "{month}": month}

    def fill(value: str) -> str:
        for key, val in subst.items():
            value = value.replace(key, val)
        return value

    probes = list(PROBES)
    if case_id:
        probes.append(("case_detail", f"/cases/{case_id}", {}, 1200))
        probes.append(("case_protocol_suggest", f"/cases/{case_id}/protocol-suggest", {}, 1200))
    results: list[dict[str, Any]] = []
    for name, path, params, threshold in probes:
        filled = {k: fill(v) for k, v in params.items()}
        if "date_from" in filled and "period" not in filled:
            filled["period"] = "custom"  # иначе часть endpoint берёт period=month и игнорирует даты
        first = _request(base, path, filled, token, timeout)
        second = _request(base, path, filled, token, timeout)
        warm = second["ms"]
        results.append({
            "name": name,
            "path": path,
            "params": {k: v for k, v in filled.items() if k != "q"} | ({"q": "<hidden>"} if "q" in filled else {}),
            "first_ms": first["ms"],
            "warm_ms": warm,
            "status": second["status"],
            "bytes": second["bytes"],
            "threshold_ms": threshold,
            "ok": second["status"] == 200 and warm <= threshold,
        })
        print(f"{name:26s} {str(second['status']):>5s} first {first['ms']:>6d} ms  warm {warm:>6d} ms  thr {threshold}"
              f"  {'ok' if results[-1]['ok'] else 'SLOW/ERR'}", file=sys.stderr)
    return {
        "taken_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "base": base,
        "month": month,
        "results": results,
        "failures": [r["name"] for r in results if not r["ok"]],
    }


def compare(before_path: str, after_path: str, regress_pct: float) -> int:
    before = {r["name"]: r for r in json.load(open(before_path, encoding="utf-8"))["results"]}
    after_doc = json.load(open(after_path, encoding="utf-8"))
    failed = False
    for row in after_doc["results"]:
        prev = before.get(row["name"])
        if not prev:
            print(f"{row['name']:26s} new  warm {row['warm_ms']} ms")
            continue
        delta = row["warm_ms"] - prev["warm_ms"]
        pct = (delta / prev["warm_ms"] * 100) if prev["warm_ms"] else 0.0
        flag = "FAIL" if (pct > regress_pct and delta > 100) else "ok"
        failed = failed or flag == "FAIL"
        print(f"{row['name']:26s} {prev['warm_ms']:>6d} -> {row['warm_ms']:>6d} ms  {pct:+6.1f}%  {flag}")
    if after_doc.get("failures"):
        print("над порогом:", ", ".join(after_doc["failures"]))
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=os.environ.get("MO_PROBE_BASE", DEFAULT_BASE))
    parser.add_argument("--month", default=dt.date.today().strftime("%Y-%m"))
    parser.add_argument("--case-id", default=None, help="mis_id/visit_id для замера разбора (не печатается)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", default=None, help="куда сохранить JSON")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--regress-pct", type=float, default=20.0)
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1], args.regress_pct)

    token = (os.environ.get("METHODIST_TOKEN") or os.environ.get("MO_METHODIST_TOKEN") or "").strip()
    if not token:
        print("METHODIST_TOKEN не задан в окружении", file=sys.stderr)
        return 2
    doc = run_probe(args.base, args.month, token, args.timeout, args.case_id)
    if args.case_id:
        for row in doc["results"]:
            row["path"] = row["path"].replace(args.case_id, "<case>")
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"saved {args.out}; над порогом: {len(doc['failures'])}", file=sys.stderr)
    else:
        print(text)
    return 1 if doc["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
