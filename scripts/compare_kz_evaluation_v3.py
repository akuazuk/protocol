"""Shadow benchmark: legacy deep scorer vs scorer v3 (§17 ТЗ overnight-v1).

Считает распределения score/coverage/confidence, изменения статусов, cap-и, случаи
где legacy высок, а v3 низок (и наоборот), долю C/D findings исключённых из штрафа.

Запуск:
    python -m scripts.compare_kz_evaluation_v3 \
      --fixtures tests/fixtures/kz_v3_cases.jsonl \
      --json data/ml/reports/kz_evaluation_v3_shadow_latest.json \
      --markdown data/ml/reports/kz_evaluation_v3_shadow_latest.md

``--cases`` может указывать на безопасный агрегированный JSONL без ПДн (не коммитить raw).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from clinical_knowledge.kz_evaluation_engine import evaluate_kz_v3  # noqa: E402


def _legacy_deep(case: dict, protocol: Any) -> dict:
    try:
        from clinical_knowledge.kz_deep_eval import evaluate_kz_deep
    except Exception:  # noqa: BLE001
        return {}
    return evaluate_kz_deep(case, protocol_ctx=protocol)


def _iter_fixtures(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        yield row.get("id", "?"), row.get("case", {}), row.get("protocol")


def _dist(values: list[float]) -> dict[str, float | None]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 1),
        "median": round(statistics.median(vals), 1),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
    }


def run(fixtures: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    legacy_scores: list[float] = []
    v3_scores: list[float] = []
    coverages: list[float] = []
    confidences: list[float] = []
    caps = 0
    status_changes = 0
    legacy_high_v3_low = 0
    cd_findings_excluded = 0
    protocol_mismatch = 0

    for cid, case, protocol in _iter_fixtures(fixtures):
        legacy = _legacy_deep(case, protocol)
        r = evaluate_kz_v3(case, protocol_ctx=protocol, legacy={
            "deep_overall_pct": legacy.get("overall_pct"),
            "deep_status": legacy.get("overall_status"),
        })
        ls = legacy.get("overall_pct")
        vs = r.score_pct
        if isinstance(ls, (int, float)):
            legacy_scores.append(ls)
        if isinstance(vs, (int, float)):
            v3_scores.append(vs)
        if r.coverage.overall is not None:
            coverages.append(r.coverage.overall)
        if r.confidence.overall is not None:
            confidences.append(r.confidence.overall)
        if r.risk.cap_applied:
            caps += 1
        if legacy.get("overall_status") != r.status:
            status_changes += 1
        if isinstance(ls, (int, float)) and isinstance(vs, (int, float)) and ls >= 75 and vs < 60:
            legacy_high_v3_low += 1
        excl = sum(
            1 for f in r.findings
            if f.trust_level in ("C", "D") and not f.penalty_applied
        )
        cd_findings_excluded += excl
        if r.protocols and not r.protocols[0].penalty_eligible:
            protocol_mismatch += 1

        rows.append({
            "id": cid,
            "legacy_score": ls,
            "legacy_status": legacy.get("overall_status"),
            "v3_score": vs,
            "v3_status": r.status,
            "coverage": r.coverage.overall,
            "confidence": r.confidence.overall,
            "cap_applied": r.risk.cap_applied,
            "protocol_penalty_eligible": (r.protocols[0].penalty_eligible if r.protocols else None),
            "cd_advisory_findings": excl,
        })

    return {
        "n": len(rows),
        "legacy_score": _dist(legacy_scores),
        "v3_score": _dist(v3_scores),
        "coverage": _dist(coverages),
        "confidence": _dist(confidences),
        "caps_applied": caps,
        "status_changes": status_changes,
        "legacy_high_v3_low": legacy_high_v3_low,
        "cd_findings_excluded_from_penalty": cd_findings_excluded,
        "protocol_mismatch_advisory": protocol_mismatch,
        "rows": rows,
    }


def _write_md(res: dict[str, Any], path: Path) -> None:
    lines = [
        "# Shadow benchmark: legacy deep vs scorer v3",
        "",
        f"- N кейсов: **{res['n']}**",
        f"- Legacy score: mean {res['legacy_score']['mean']} / median {res['legacy_score']['median']}",
        f"- V3 score: mean {res['v3_score']['mean']} / median {res['v3_score']['median']}",
        f"- Coverage: mean {res['coverage']['mean']}",
        f"- Confidence: mean {res['confidence']['mean']}",
        f"- Cap применён: **{res['caps_applied']}**",
        f"- Смен статуса: **{res['status_changes']}**",
        f"- Legacy высок, v3 низок: **{res['legacy_high_v3_low']}**",
        f"- C/D findings исключены из штрафа: **{res['cd_findings_excluded_from_penalty']}**",
        f"- Протокол advisory (не penalty-eligible): **{res['protocol_mismatch_advisory']}**",
        "",
        "## По кейсам",
        "",
        "| id | legacy | v3 | статус v3 | cov | conf | cap | proto pen | C/D adv |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        lines.append(
            f"| {r['id']} | {r['legacy_score']} | {r['v3_score']} | {r['v3_status']} | "
            f"{r['coverage']} | {r['confidence']} | {r['cap_applied']} | "
            f"{r['protocol_penalty_eligible']} | {r['cd_advisory_findings']} |",
        )
    lines += [
        "",
        "> Shadow-режим: production score/gate не переключаются. C/D findings всегда",
        "> исключены из штрафа (advisory), что устраняет архитектурный источник ложных",
        "> штрафов по недоверенным правилам (ТЗ §2.2, §6).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shadow benchmark scorer v3")
    ap.add_argument("--fixtures", default="tests/fixtures/kz_v3_cases.jsonl")
    ap.add_argument("--json", default="data/ml/reports/kz_evaluation_v3_shadow_latest.json")
    ap.add_argument("--markdown", default="data/ml/reports/kz_evaluation_v3_shadow_latest.md")
    args = ap.parse_args(argv)

    res = run(ROOT / args.fixtures)
    jp = ROOT / args.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(res, ROOT / args.markdown)
    print(json.dumps({k: res[k] for k in (
        "n", "legacy_score", "v3_score", "caps_applied", "status_changes",
        "cd_findings_excluded_from_penalty", "protocol_mismatch_advisory",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
