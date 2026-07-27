"""Аудит и приоритизация корпуса протоколов для scorer v3 (Workstream F ТЗ overnight-v1).

Считает по каждому протоколу пригодность к штрафующей оценке через
``protocol_knowledge_model`` (canonical knowledge model) и агрегирует метрики §10.1-10.2.
Формирует очередь методиста (§10.3) по приоритету
``real_mis_frequency * clinical_risk * missing_structure_factor``.

Запуск:
    python -m scripts.audit_kz_protocol_knowledge \
      --json data/ml/reports/kz_protocol_knowledge_audit_latest.json \
      --markdown data/ml/reports/kz_protocol_knowledge_audit_latest.md \
      --queue-json data/ml/reports/kz_protocol_methodist_queue_latest.json \
      --queue-markdown data/ml/reports/kz_protocol_methodist_queue_latest.md \
      --mis-summary data/ml/reports/kz_dx_frequency_2026-07.json

Не коммитить огромные файлы: очередь ограничена ``--queue-top`` (по умолчанию 120).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from clinical_knowledge.protocol_knowledge_model import (  # noqa: E402
    summary_to_knowledge,
    validate_knowledge_document,
)
from clinical_knowledge.protocol_summary.loader import load_protocol_summaries  # noqa: E402

_HIGH_RISK_MARKERS = (
    "новообразован", "злокачеств", "онко", "тромб", "инфаркт", "инсульт",
    "сепсис", "кровотеч", "суицид", "неотложн", "жизнеугрож",
)


def _mis_frequency(mis_summary_path: str | None) -> dict[str, int]:
    if not mis_summary_path:
        return {}
    p = Path(mis_summary_path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    freq: dict[str, int] = {}
    for row in data.get("top_roots", []) or []:
        root = str(row.get("icd_root") or "").upper()
        if root:
            freq[root] = int(row.get("n") or 0)
    return freq


def _icd_root(code: str) -> str:
    code = (code or "").strip().upper()
    return code[:3]


def _clinical_risk(name: str, has_red_flags: bool) -> float:
    blob = (name or "").lower()
    if any(m in blob for m in _HIGH_RISK_MARKERS):
        return 3.0
    if has_red_flags:
        return 2.0
    return 1.0


def run_audit(mis_summary_path: str | None, queue_top: int) -> dict[str, Any]:
    summaries = load_protocol_summaries()
    freq = _mis_frequency(mis_summary_path)

    agg: dict[str, Any] = {
        "protocols_total": len(summaries),
        "review_status": {},
        "extraction_status": {},
        "trust_levels": {"A": 0, "B": 0, "C": 0, "D": 0},
        "requirements_total": 0,
        "penalty_eligible_rules": 0,
        "advisory_rules": 0,
        "verified_quote_rules": 0,
        "protocols_with_any_penalty_rule": 0,
        "protocols_penalty_ready": 0,
        "coverage": {
            "diagnosis_criteria": 0,
            "required_exams": 0,
            "conditional_exams": 0,
            "treatment": 0,
            "dose": 0,
            "route": 0,
            "frequency": 0,
            "duration": 0,
            "red_flags": 0,
            "monitoring": 0,
            "follow_up": 0,
        },
        "reasons": {},
    }
    per_protocol: list[dict[str, Any]] = []

    for s in summaries:
        rs = getattr(s, "review_status", "not_reviewed") or "not_reviewed"
        es = getattr(s, "extraction_status", "draft") or "draft"
        agg["review_status"][rs] = agg["review_status"].get(rs, 0) + 1
        agg["extraction_status"][es] = agg["extraction_status"].get(es, 0) + 1

        doc = summary_to_knowledge(s)
        agg["trust_levels"][doc.trust_level] = agg["trust_levels"].get(doc.trust_level, 0) + 1
        v = validate_knowledge_document(doc)

        agg["requirements_total"] += v["requirements_total"]
        agg["penalty_eligible_rules"] += v["penalty_ready"]
        agg["verified_quote_rules"] += v["verified_quote"]
        agg["advisory_rules"] += max(0, v["requirements_total"] - v["penalty_ready"])
        for key, cnt in (v.get("reasons") or {}).items():
            agg["reasons"][key] = agg["reasons"].get(key, 0) + cnt
        if v["penalty_ready"] > 0:
            agg["protocols_with_any_penalty_rule"] += 1
        if v["document_penalty_ready"]:
            agg["protocols_penalty_ready"] += 1

        # coverage-флаги по наличию структур
        has_red = False
        max_freq = 0
        icd_all: list[str] = []
        for cond in doc.conditions:
            icd_all.extend(cond.icd10_codes)
            types = {r.type for r in cond.requirements}
            if "diagnostic_criterion" in types:
                agg["coverage"]["diagnosis_criteria"] += 0  # накопим ниже по протоколу
            if "red_flag" in types:
                has_red = True
        # покрытие на уровне протокола (есть хотя бы 1)
        cov_flags = _protocol_coverage_flags(s)
        for k, val in cov_flags.items():
            agg["coverage"][k] += 1 if val else 0

        for code in icd_all:
            max_freq = max(max_freq, freq.get(_icd_root(code), 0))

        name = getattr(getattr(s, "source", None), "title", "") or (
            doc.conditions[0].name if doc.conditions else ""
        )
        missing_structure = 1.0 - (v["penalty_ready_pct"] / 100.0)
        risk = _clinical_risk(name, has_red)
        priority = (max_freq or 1) * risk * (0.2 + missing_structure)

        per_protocol.append({
            "protocol_id": doc.protocol_id,
            "title": (name or "")[:90],
            "review_status": rs,
            "trust_level": doc.trust_level,
            "requirements_total": v["requirements_total"],
            "penalty_ready": v["penalty_ready"],
            "penalty_ready_pct": v["penalty_ready_pct"],
            "mis_frequency": max_freq,
            "clinical_risk": risk,
            "missing_structure_factor": round(missing_structure, 3),
            "priority": round(priority, 1),
        })

    # производные метрики §10.2
    n = max(1, agg["protocols_total"])
    reqs = max(1, agg["requirements_total"])
    agg["metrics"] = {
        "protocol_structured_coverage_pct": round(
            100.0 * agg["protocols_with_any_penalty_rule"] / n, 1,
        ),
        "penalty_eligible_coverage_pct": round(100.0 * agg["penalty_eligible_rules"] / reqs, 1),
        "source_verified_coverage_pct": round(100.0 * agg["verified_quote_rules"] / reqs, 1),
        "methodist_approved_coverage_pct": round(
            100.0 * agg["review_status"].get("approved", 0) / n, 1,
        ),
        "protocols_without_safe_penalty_rule": n - agg["protocols_with_any_penalty_rule"],
    }

    per_protocol.sort(key=lambda r: r["priority"], reverse=True)
    queue = per_protocol[:queue_top]
    return {"audit": agg, "queue": queue}


def _protocol_coverage_flags(summary: Any) -> dict[str, bool]:
    flags = {
        "diagnosis_criteria": False,
        "required_exams": False,
        "conditional_exams": False,
        "treatment": False,
        "dose": False,
        "route": False,
        "frequency": False,
        "duration": False,
        "red_flags": False,
        "monitoring": False,
        "follow_up": False,
    }
    for cond in getattr(summary, "conditions", []) or []:
        crit = getattr(cond, "diagnostic_criteria", None)
        if crit and (getattr(crit, "required", []) or getattr(crit, "optional", [])):
            flags["diagnosis_criteria"] = True
        if getattr(cond, "required_exams", []):
            flags["required_exams"] = True
        if getattr(cond, "conditional_exams", []):
            flags["conditional_exams"] = True
        if getattr(cond, "red_flags", []):
            flags["red_flags"] = True
        if getattr(cond, "follow_up", []):
            flags["follow_up"] = True
        tx = getattr(cond, "treatment", None)
        if tx and (getattr(tx, "drugs", []) or getattr(tx, "drug_groups", []) or getattr(tx, "non_drug", [])):
            flags["treatment"] = True
        for d in (getattr(tx, "drugs", []) if tx else []) or []:
            if getattr(d, "dose_text", None):
                flags["dose"] = True
            if getattr(d, "route", None):
                flags["route"] = True
            if getattr(d, "frequency_text", None):
                flags["frequency"] = True
            if getattr(d, "duration_text", None):
                flags["duration"] = True
            if getattr(d, "monitoring", None):
                flags["monitoring"] = True
    return flags


def _write_markdown(audit: dict[str, Any], path: Path) -> None:
    a = audit["audit"]
    m = a["metrics"]
    lines = [
        "# Аудит knowledge-корпуса протоколов (scorer v3)",
        "",
        f"- Протоколов: **{a['protocols_total']}**",
        f"- Требований (атомарных): **{a['requirements_total']}**",
        f"- Penalty-eligible правил: **{a['penalty_eligible_rules']}** "
        f"({m['penalty_eligible_coverage_pct']}%)",
        f"- Advisory правил: **{a['advisory_rules']}**",
        f"- С подтверждённой цитатой: **{a['verified_quote_rules']}** "
        f"({m['source_verified_coverage_pct']}%)",
        "",
        "## Ключевые метрики покрытия (§10.2)",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| protocol_structured_coverage_pct | {m['protocol_structured_coverage_pct']}% |",
        f"| penalty_eligible_coverage_pct | {m['penalty_eligible_coverage_pct']}% |",
        f"| source_verified_coverage_pct | {m['source_verified_coverage_pct']}% |",
        f"| methodist_approved_coverage_pct | {m['methodist_approved_coverage_pct']}% |",
        f"| protocols_without_safe_penalty_rule | {m['protocols_without_safe_penalty_rule']} |",
        "",
        "## Trust levels",
        "",
        "| Level | N |",
        "|---|---|",
    ]
    for lvl in ("A", "B", "C", "D"):
        lines.append(f"| {lvl} | {a['trust_levels'].get(lvl, 0)} |")
    lines += ["", "## Review status", "", "| Status | N |", "|---|---|"]
    for k, val in sorted(a["review_status"].items()):
        lines.append(f"| {k} | {val} |")
    lines += ["", "## Покрытие структур (протоколов с ≥1 элементом)", "", "| Поле | N |", "|---|---|"]
    for k, val in a["coverage"].items():
        lines.append(f"| {k} | {val} |")
    lines += ["", "## Причины непригодности к штрафу", "", "| Причина | N |", "|---|---|"]
    for k, val in sorted(a["reasons"].items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {val} |")
    lines += [
        "",
        "> Наличие правила != пригодность к штрафу. Штраф допустим только для trust A/B",
        "> с подтверждённой цитатой и применимостью (см. ТЗ §6, §10.2).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_queue_markdown(queue: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Очередь методиста: приоритет протоколов (scorer v3)",
        "",
        "priority = mis_frequency × clinical_risk × (0.2 + missing_structure_factor)",
        "",
        "| # | Протокол | MIS freq | Risk | Penalty-ready % | Priority |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(queue[:60], 1):
        lines.append(
            f"| {i} | {r['title']} | {r['mis_frequency']} | {r['clinical_risk']} | "
            f"{r['penalty_ready_pct']}% | {r['priority']} |",
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Аудит knowledge-корпуса протоколов")
    ap.add_argument("--json", default="data/ml/reports/kz_protocol_knowledge_audit_latest.json")
    ap.add_argument("--markdown", default="data/ml/reports/kz_protocol_knowledge_audit_latest.md")
    ap.add_argument("--queue-json", default="data/ml/reports/kz_protocol_methodist_queue_latest.json")
    ap.add_argument("--queue-markdown", default="data/ml/reports/kz_protocol_methodist_queue_latest.md")
    ap.add_argument("--mis-summary", default="data/ml/reports/kz_dx_frequency_2026-07.json")
    ap.add_argument("--queue-top", type=int, default=120)
    args = ap.parse_args(argv)

    out = run_audit(args.mis_summary, args.queue_top)

    jp = ROOT / args.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out["audit"], ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out, ROOT / args.markdown)

    qp = ROOT / args.queue_json
    qp.parent.mkdir(parents=True, exist_ok=True)
    qp.write_text(json.dumps(out["queue"], ensure_ascii=False, indent=2), encoding="utf-8")
    _write_queue_markdown(out["queue"], ROOT / args.queue_markdown)

    a = out["audit"]
    print("protocols:", a["protocols_total"], "reqs:", a["requirements_total"])
    print("metrics:", json.dumps(a["metrics"], ensure_ascii=False))
    print("queue top-1:", out["queue"][0]["title"] if out["queue"] else "-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
