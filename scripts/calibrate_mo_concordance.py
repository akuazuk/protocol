#!/usr/bin/env python3
"""E2: калибровка shadow concordance findings на выборке МО (CSV/JSONL).

Считает частоты кодов / severity / audience без записи PHI в отчёт.
Опционально пишет обезличенные positive/negative fixtures в eval/mo_concordance/.

  PYTHONPATH=. python3 scripts/calibrate_mo_concordance.py \\
    --csv /path/to/mis_protocol_2026-07_complete.csv \\
    --limit 5000 \\
    --report docs/reports/2026-08-05-mo-concordance-calibration-e2.md \\
    --fixtures-dir eval/mo_concordance
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_concordance_findings import (  # noqa: E402
    evaluate_mo_concordance,
)

CASE_FIELDS = (
    "complaints",
    "anamnesis_doctor",
    "anamnesis_auto",
    "objective_status",
    "exam_data",
    "clinical_diagnosis",
    "diagnosis_main_text",
    "diagnosis_list",
    "mkb_code_main",
    "treatment_recommendations",
    "exam_recommendations",
    "patient_age_years",
)

# Минимум clinical text, чтобы не считать пустые строки.
_MIN_TEXT = 40


def _age_years(row: dict[str, Any]) -> float | None:
    raw = row.get("patient_age_years")
    if raw is None or raw == "":
        bdate = str(row.get("patient_bdate") or "").strip()
        if bdate and len(bdate) >= 4:
            try:
                y = int(bdate[:4])
                return float(date.today().year - y)
            except ValueError:
                return None
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def row_to_case(row: dict[str, Any]) -> dict[str, Any] | None:
    case = {k: (row.get(k) or "") for k in CASE_FIELDS if k != "patient_age_years"}
    age = _age_years(row)
    if age is not None:
        case["patient_age_years"] = age
    blob = " ".join(str(case.get(k) or "") for k in (
        "complaints", "anamnesis_doctor", "objective_status", "clinical_diagnosis"
    ))
    if len(blob.strip()) < _MIN_TEXT:
        return None
    return case


def iter_csv_cases(path: Path, *, limit: int | None, seed: int):
    # csv с многострочными полями; DictReader справляется.
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for i, row in enumerate(reader):
            case = row_to_case(row)
            if case is None:
                continue
            rows.append(case)
            if limit and len(rows) >= max(limit * 4, limit):
                # набрали с запасом до семпла
                break
            if not limit and i > 200_000:
                break
    if limit and len(rows) > limit:
        rng = random.Random(seed)
        rows = rng.sample(rows, limit)
    return rows


def anonymize(case: dict[str, Any]) -> dict[str, Any]:
    out = {k: case.get(k) for k in CASE_FIELDS if case.get(k) not in (None, "")}
    # Клинические поля: не обрезать слишком агрессивно (иначе теряется отёк в конце status).
    limits = {
        "objective_status": 1400,
        "exam_data": 1000,
        "anamnesis_doctor": 800,
        "anamnesis_auto": 800,
        "treatment_recommendations": 900,
        "exam_recommendations": 700,
        "clinical_diagnosis": 500,
        "complaints": 400,
    }
    for k, v in list(out.items()):
        if isinstance(v, str):
            lim = limits.get(k, 600)
            if len(v) > lim:
                out[k] = v[:lim].rstrip() + "…"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--fixtures-dir", type=Path, default=None)
    ap.add_argument("--n-pos", type=int, default=5)
    ap.add_argument("--n-neg", type=int, default=5)
    args = ap.parse_args()

    cases = iter_csv_cases(args.csv, limit=args.limit or None, seed=args.seed)
    n = len(cases)
    if n == 0:
        print("no cases", file=sys.stderr)
        return 1

    code_counts: Counter[str] = Counter()
    sev_counts: Counter[str] = Counter()
    aud_counts: Counter[str] = Counter()
    code_by_aud: dict[str, Counter[str]] = defaultdict(Counter)
    any_finding = 0
    p1_plus = 0
    underworkup_ped = 0
    underworkup_adult = 0
    positives: list[tuple[dict, list]] = []
    negatives: list[dict] = []

    from clinical_knowledge.mo_case_signals import extract_mo_case_signals

    for case in cases:
        sig = extract_mo_case_signals(case)
        aud = str(sig.get("audience") or "unknown")
        aud_counts[aud] += 1
        findings = evaluate_mo_concordance(case)
        if findings:
            any_finding += 1
            codes = {f["code"] for f in findings}
            sevs = {f["severity"] for f in findings}
            if sevs & {"P0", "P1"}:
                p1_plus += 1
            for f in findings:
                code_counts[f["code"]] += 1
                sev_counts[f["severity"]] += 1
                code_by_aud[aud][f["code"]] += 1
                if f["code"] == "underworkup_chronic_red_flag":
                    if aud == "pediatric":
                        underworkup_ped += 1
                    else:
                        underworkup_adult += 1
            # positive: at least one of the Smirnova core P1/P2
            core = codes & {
                "finding_not_in_diagnosis",
                "underworkup_chronic_red_flag",
                "anamnesis_thin_for_duration",
            }
            if core and len(positives) < max(args.n_pos * 20, args.n_pos):
                positives.append((case, findings))
        else:
            if len(negatives) < max(args.n_neg * 20, args.n_neg):
                negatives.append(case)

    def pct(x: int) -> str:
        return f"{100.0 * x / n:.1f}%"

    lines: list[str] = []
    lines.append("# Калибровка MO concordance (E2)")
    lines.append("")
    lines.append(f"Дата: {date.today().isoformat()}")
    lines.append(f"Источник: `{args.csv.name}` (локально, PHI не коммитится)")
    lines.append(f"Выборка: **{n}** МО (seed={args.seed}, limit={args.limit})")
    lines.append("")
    lines.append("## Audience")
    lines.append("")
    lines.append("| audience | n | share |")
    lines.append("|--|--|--|")
    for k, v in aud_counts.most_common():
        lines.append(f"| {k} | {v} | {pct(v)} |")
    lines.append("")
    lines.append("## Trigger rates")
    lines.append("")
    lines.append(f"- any shadow finding: **{any_finding}** ({pct(any_finding)})")
    lines.append(f"- any P0/P1: **{p1_plus}** ({pct(p1_plus)})")
    lines.append("")
    lines.append("| code | n | rate | pediatric | adult |")
    lines.append("|--|--|--|--|--|")
    for code, cnt in code_counts.most_common():
        ped = code_by_aud["pediatric"][code]
        ad = code_by_aud["adult"][code]
        lines.append(f"| `{code}` | {cnt} | {pct(cnt)} | {ped} | {ad} |")
    if not code_counts:
        lines.append("| (none) | 0 | 0% | 0 | 0 |")
    lines.append("")
    lines.append("## Severity mix (finding instances)")
    lines.append("")
    lines.append("| severity | n |")
    lines.append("|--|--|")
    for k in ("P0", "P1", "P2", "P3"):
        lines.append(f"| {k} | {sev_counts.get(k, 0)} |")
    lines.append("")
    lines.append("## Decision notes")
    lines.append("")
    lines.append(
        f"- `underworkup_chronic_red_flag`: pediatric={underworkup_ped}, "
        f"adult={underworkup_adult}"
    )
    lines.append(
        "- Рекомендация E2: держать underworkup **P1 только pediatric**, "
        "adult → P2 (если adult rate высокий / шумный)."
        if underworkup_adult > underworkup_ped
        else "- Рекомендация E2: underworkup оставить P1 для pediatric; adult P2."
    )
    lines.append(
        "- Primary (`MO_CONCORDANCE_PRIMARY`) **не включать**, пока P1+ rate "
        f"не объяснён методистом (сейчас {pct(p1_plus)})."
    )
    lines.append("- В прод-дашборд blocking не публиковать.")
    lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")
    print(f"n={n} any={any_finding} ({pct(any_finding)}) p1+={p1_plus} ({pct(p1_plus)})")
    for code, cnt in code_counts.most_common():
        print(f"  {code}: {cnt} ({pct(cnt)})")

    if args.fixtures_dir:
        args.fixtures_dir.mkdir(parents=True, exist_ok=True)
        pos_out = []
        for case, _findings in positives:
            if len(pos_out) >= args.n_pos:
                break
            anon = anonymize(case)
            anon_findings = evaluate_mo_concordance(anon)
            if not anon_findings:
                continue
            pos_out.append(
                {
                    "label": "positive",
                    "expected_codes": sorted({f["code"] for f in anon_findings}),
                    "case": anon,
                }
            )
        neg_out = []
        for c in negatives:
            if len(neg_out) >= args.n_neg:
                break
            anon = anonymize(c)
            if evaluate_mo_concordance(anon):
                continue
            neg_out.append({"label": "negative", "expected_codes": [], "case": anon})
        (args.fixtures_dir / "positives.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in pos_out) + ("\n" if pos_out else ""),
            encoding="utf-8",
        )
        (args.fixtures_dir / "negatives.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in neg_out) + ("\n" if neg_out else ""),
            encoding="utf-8",
        )
        meta = {
            "source_file": args.csv.name,
            "n_sample": n,
            "n_pos": len(pos_out),
            "n_neg": len(neg_out),
            "note": "Обезличено: без visit_id/patient_id/ФИО. expected_codes после anonymize.",
        }
        readme = (
            "# eval/mo_concordance\n\n"
            "Positive/negative fixtures для concordance (E0/E2).\n\n"
            "```json\n"
            + json.dumps(meta, ensure_ascii=False, indent=2)
            + "\n```\n"
        )
        (args.fixtures_dir / "README.md").write_text(readme, encoding="utf-8")
        print(f"fixtures → {args.fixtures_dir} pos={len(pos_out)} neg={len(neg_out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
