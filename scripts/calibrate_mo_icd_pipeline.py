#!/usr/bin/env python3
"""Фаза 3: калибровка ICD match pipeline на дне warehouse + эталонах.

Без PHI в отчёте: только агрегаты, id эталонов, hashed visit_id.
Confusion: predicted chip vs expected_chip из etalon_labels JSONL
(прокси «мнение методиста» до ручной разметки дня).

  PYTHONPATH=. python3 scripts/calibrate_mo_icd_pipeline.py \\
    --etalons eval/mo_icd_pipeline/etalon_labels_v1.jsonl \\
    --data-root /var/data/medical_exams --day 2026-08-04 --limit 200 \\
    --report docs/reports/2026-08-08-mo-icd-pipeline-calibration.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

from clinical_knowledge.mo_icd_match_pipeline import evaluate_mo_icd_match  # noqa: E402
from clinical_knowledge.mo_icd_thresholds import snapshot  # noqa: E402

_CLINICAL_KEYS = (
    "complaints",
    "anamnesis_doctor",
    "anamnesis_auto",
    "objective_status",
    "clinical_diagnosis",
    "exam_data",
    "exam_recommendations",
    "treatment_recommendations",
    "manipulations",
    "mis_diagnos",
    "mkb_code_main",
    "mkb_code_agreement",
    "mkb_code_mis",
    "diagnosis_structured_raw",
)


def _hash_id(raw: str) -> str:
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:12]


def _load_etalons(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _load_csv_by_visit(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            vid = str(row.get("visit_id") or "").strip()
            if vid:
                out[vid] = row
    return out


def _case_from_csv(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _CLINICAL_KEYS:
        val = row.get(key)
        if val not in (None, ""):
            out[key] = val
    for key in ("visit_id", "mis_id", "patient_id"):
        if row.get(key):
            out[key] = row[key]
    return out


def _iter_day_cases(
    data_root: Path,
    day: date,
    *,
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    secure = data_root / "secure_cases" / f"{day:%Y}" / f"{day:%m}"
    csv_path = secure / f"mo_{day.isoformat()}.csv"
    by_visit = _load_csv_by_visit(csv_path)
    cases = [_case_from_csv(r) for r in by_visit.values()]
    # fallback: jsonl meta only if csv missing clinical (rare)
    if not cases:
        jsonl = secure / f"kz_l1_{day.isoformat()}_cases.jsonl"
        if jsonl.is_file():
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if limit and len(cases) > limit:
        rng = random.Random(seed)
        cases = rng.sample(cases, limit)
    return cases


def _score_bins(values: list[float]) -> dict[str, int]:
    bins = {"<0.28": 0, "0.28-0.42": 0, "0.42-0.60": 0, ">=0.60": 0, "na": 0}
    for v in values:
        if v is None:
            bins["na"] += 1
        elif v < 0.28:
            bins["<0.28"] += 1
        elif v < 0.42:
            bins["0.28-0.42"] += 1
        elif v < 0.60:
            bins["0.42-0.60"] += 1
        else:
            bins[">=0.60"] += 1
    return bins


def run_etalons(path: Path) -> dict[str, Any]:
    rows = _load_etalons(path)
    confusion: dict[str, Counter] = defaultdict(Counter)
    mismatches: list[dict[str, str]] = []
    chip_hist: Counter = Counter()
    finding_hist: Counter = Counter()
    ok_n = 0
    for row in rows:
        expected = str(row.get("expected_chip") or "").strip()
        case = row.get("case") if isinstance(row.get("case"), dict) else {}
        pipe = evaluate_mo_icd_match(case)
        pred = str((pipe.get("chip") or {}).get("status") or "unknown")
        chip_hist[pred] += 1
        confusion[expected][pred] += 1
        for f in pipe.get("findings") or []:
            if isinstance(f, dict) and f.get("code"):
                finding_hist[str(f["code"])] += 1
        if pred == expected:
            ok_n += 1
        else:
            mismatches.append(
                {
                    "id": str(row.get("id") or ""),
                    "expected": expected,
                    "predicted": pred,
                    "verdict": str(pipe.get("pipeline_verdict") or ""),
                    "note": str(row.get("note") or "")[:80],
                }
            )
    n = len(rows) or 1
    # precision/recall for not_in_directory as binary class
    tp = confusion["not_in_directory"]["not_in_directory"]
    fp = sum(confusion[e]["not_in_directory"] for e in confusion if e != "not_in_directory")
    fn = sum(
        v
        for pred, v in confusion["not_in_directory"].items()
        if pred != "not_in_directory"
    )
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return {
        "n": len(rows),
        "accuracy": round(ok_n / n, 3),
        "chip_hist": dict(chip_hist),
        "finding_hist": dict(finding_hist),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "mismatches": mismatches,
        "not_in_directory_precision": None if prec is None else round(prec, 3),
        "not_in_directory_recall": None if rec is None else round(rec, 3),
    }


def run_day(cases: list[dict[str, Any]]) -> dict[str, Any]:
    chip_hist: Counter = Counter()
    verdict_hist: Counter = Counter()
    finding_hist: Counter = Counter()
    name_fits: list[float] = []
    text_fits: list[float] = []
    needs_llm = 0
    sample_hashes: list[str] = []
    for i, case in enumerate(cases):
        pipe = evaluate_mo_icd_match(case)
        chip = str((pipe.get("chip") or {}).get("status") or "unknown")
        chip_hist[chip] += 1
        verdict_hist[str(pipe.get("pipeline_verdict") or "")] += 1
        if pipe.get("needs_llm_review"):
            needs_llm += 1
        for f in pipe.get("findings") or []:
            if isinstance(f, dict) and f.get("code"):
                finding_hist[str(f["code"])] += 1
        nf = (pipe.get("name_only") or {}).get("name_fit")
        if nf is not None:
            name_fits.append(float(nf))
        tf = (pipe.get("directory") or {}).get("text_rubric_fit")
        if tf is not None:
            text_fits.append(float(tf))
        if i < 8:
            vid = str(case.get("visit_id") or case.get("mis_id") or i)
            sample_hashes.append(_hash_id(vid))
    n = len(cases) or 1
    return {
        "n": len(cases),
        "chip_hist": dict(chip_hist),
        "chip_share": {k: round(v / n, 3) for k, v in chip_hist.items()},
        "verdict_hist": dict(verdict_hist),
        "finding_hist": dict(finding_hist.most_common(20)),
        "needs_llm_review_rate": round(needs_llm / n, 3),
        "name_fit_bins": _score_bins(name_fits),
        "text_fit_bins": _score_bins(text_fits),
        "sample_visit_hashes": sample_hashes,
    }


def _md_table(counter: dict[str, Any], *, headers: tuple[str, str] = ("key", "n")) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "|--|--|"]
    for k, v in sorted(counter.items(), key=lambda kv: (-int(kv[1]) if str(kv[1]).isdigit() or isinstance(kv[1], int) else 0, str(kv[0]))):
        lines.append(f"| `{k}` | {v} |")
    if len(lines) == 2:
        lines.append("| _(empty)_ | 0 |")
    return "\n".join(lines)


def render_report(
    *,
    etalon: dict[str, Any] | None,
    day_stats: dict[str, Any] | None,
    day: str | None,
    thresholds: dict[str, float],
    decision: dict[str, Any],
) -> str:
    lines = [
        "# Калибровка MO ICD match pipeline (фаза 3)",
        "",
        f"Дата отчёта: {date.today().isoformat()}",
        "Скрипт: `scripts/calibrate_mo_icd_pipeline.py`",
        "PHI: не включён (агрегаты + hashed visit_id + id эталонов).",
        "",
        "## Пороги (snapshot)",
        "",
        _md_table({k: str(v) for k, v in thresholds.items()}, headers=("threshold", "value")),
        "",
    ]
    if etalon:
        lines += [
            "## Эталоны (прокси методиста)",
            "",
            f"- n = **{etalon['n']}**",
            f"- accuracy (chip) = **{etalon['accuracy']}**",
            f"- precision `not_in_directory` = **{etalon.get('not_in_directory_precision')}**",
            f"- recall `not_in_directory` = **{etalon.get('not_in_directory_recall')}**",
            "",
            "### Chip histogram (predicted)",
            "",
            _md_table(etalon.get("chip_hist") or {}),
            "",
            "### Confusion expected → predicted",
            "",
        ]
        conf = etalon.get("confusion") or {}
        lines.append("| expected \\ predicted | " + " | ".join(sorted({p for c in conf.values() for p in c})) + " |")
        preds = sorted({p for c in conf.values() for p in c})
        lines.append("|--|" + "|".join(["--"] * len(preds)) + "|")
        for exp in sorted(conf):
            row = [str(conf[exp].get(p, 0)) for p in preds]
            lines.append(f"| `{exp}` | " + " | ".join(row) + " |")
        lines += ["", "### Mismatches (etalon id only)", ""]
        if etalon.get("mismatches"):
            lines.append("| id | expected | predicted | verdict | note |")
            lines.append("|--|--|--|--|--|")
            for m in etalon["mismatches"]:
                lines.append(
                    f"| `{m['id']}` | `{m['expected']}` | `{m['predicted']}` | "
                    f"`{m['verdict']}` | {m['note']} |"
                )
        else:
            lines.append("_нет расхождений_")
        lines += ["", "### Finding codes (etalons)", "", _md_table(etalon.get("finding_hist") or {}), ""]

    if day_stats:
        lines += [
            f"## Выборка дня `{day}`",
            "",
            f"- n = **{day_stats['n']}**",
            f"- needs_llm_review rate = **{day_stats['needs_llm_review_rate']}**",
            f"- sample visit hashes: {', '.join('`'+h+'`' for h in day_stats.get('sample_visit_hashes') or [])}",
            "",
            "### Chip share",
            "",
            _md_table({k: f"{v} ({day_stats['chip_share'].get(k, 0)})" for k, v in (day_stats.get('chip_hist') or {}).items()}),
            "",
            "### Pipeline verdict",
            "",
            _md_table(day_stats.get("verdict_hist") or {}),
            "",
            "### Top finding codes",
            "",
            _md_table(day_stats.get("finding_hist") or {}),
            "",
            "### name_fit bins",
            "",
            _md_table(day_stats.get("name_fit_bins") or {}),
            "",
            "### text_rubric_fit bins",
            "",
            _md_table(day_stats.get("text_fit_bins") or {}),
            "",
        ]

    lines += [
        "## Решение по primary",
        "",
        f"- `MO_ICD_NAME_IN_PRIMARY` → **{decision.get('name_primary')}**",
        f"- `MO_ICD_DIR_IN_PRIMARY` → **{decision.get('dir_primary')}**",
        f"- `MO_ICD_PIPELINE_IN_PRIMARY` → **{decision.get('pipeline_primary')}**",
        f"- пороги менять: **{decision.get('tune_thresholds')}**",
        "",
        decision.get("rationale") or "",
        "",
        "## Следующий шаг",
        "",
        "- Фаза 4: LLM review только для `needs_llm_review` (GCE, флаг off).",
        "- После ручной разметки ≥20 живых визитов - пересмотреть DIR primary.",
        "",
    ]
    return "\n".join(lines)


def decide(etalon: dict[str, Any] | None, day_stats: dict[str, Any] | None) -> dict[str, Any]:
    """Правила фазы 3: NAME primary при хороших эталонах; DIR/pipeline пока off."""
    acc = float((etalon or {}).get("accuracy") or 0)
    prec = (etalon or {}).get("not_in_directory_precision")
    rec = (etalon or {}).get("not_in_directory_recall")
    missing_share = float(((day_stats or {}).get("chip_share") or {}).get("missing_dx") or 0)
    not_dir_share = float(((day_stats or {}).get("chip_share") or {}).get("not_in_directory") or 0)

    name_ok_flag = acc >= 0.85 and (prec is None or prec >= 0.85) and (rec is None or rec >= 0.80)
    # DIR primary жёстче: не включать если день шумит missing/not_in_directory > 40%
    dir_ok_flag = name_ok_flag and missing_share + not_dir_share < 0.40

    rationale_parts = [
        f"Эталон accuracy={acc}, not_in_directory P/R={prec}/{rec}.",
        f"День: missing_dx share={missing_share}, not_in_directory share={not_dir_share}.",
    ]
    if name_ok_flag:
        rationale_parts.append(
            "Включаем **только** `MO_ICD_NAME_IN_PRIMARY=1` (мягкая ось). "
            "`MO_ICD_DIR_IN_PRIMARY` и `MO_ICD_PIPELINE_IN_PRIMARY` оставляем 0 "
            "до ручной разметки дня и контроля overall."
        )
    else:
        rationale_parts.append(
            "Эталоны ниже цели - primary флаги остаются 0; нужна подкрутка порогов/алиасов."
        )
    if not dir_ok_flag and name_ok_flag:
        rationale_parts.append(
            "DIR primary отложен: высокая доля chip fail на дне или не хватает ручных labels."
        )

    return {
        "name_primary": "1" if name_ok_flag else "0",
        "dir_primary": "0",
        "pipeline_primary": "0",
        "tune_thresholds": "нет (дефолты v3)" if name_ok_flag else "да - разобрать mismatches",
        "rationale": " ".join(rationale_parts),
        "enable_name_primary": name_ok_flag,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--etalons",
        type=Path,
        default=ROOT / "eval/mo_icd_pipeline/etalon_labels_v1.jsonl",
    )
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--day", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/reports/2026-08-08-mo-icd-pipeline-calibration.md",
    )
    ap.add_argument("--skip-etalons", action="store_true")
    args = ap.parse_args()

    etalon_stats = None if args.skip_etalons else run_etalons(args.etalons)
    day_stats = None
    if args.data_root and args.day:
        day = date.fromisoformat(args.day)
        cases = _iter_day_cases(args.data_root, day, limit=args.limit, seed=args.seed)
        day_stats = run_day(cases)

    decision = decide(etalon_stats, day_stats)
    thr = snapshot()
    report = render_report(
        etalon=etalon_stats,
        day_stats=day_stats,
        day=args.day,
        thresholds=thr,
        decision=decision,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote {args.report}")
    print(json.dumps({"etalon_accuracy": (etalon_stats or {}).get("accuracy"), "decision": decision}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
