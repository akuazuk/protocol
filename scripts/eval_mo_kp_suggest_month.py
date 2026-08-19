#!/usr/bin/env python3
"""Агрегированный прогон подбора КП по месяцу МО. В отчёт не пишем PHI."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.applicability import is_child_only_kp_name  # noqa: E402

def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def _resolve_partition(data_root: Path, day: date) -> Path | None:
    secure_dir = data_root / "secure_cases" / f"{day:%Y}" / f"{day:%m}"
    for path in (
        secure_dir / f"mo_{day.isoformat()}.csv",
        secure_dir / f"mo_{day.isoformat()}.parquet",
        data_root / "raw" / f"{day:%Y}" / f"{day:%m}" / f"mo_{day.isoformat()}.parquet",
        data_root / "raw" / f"{day:%Y}" / f"{day:%m}" / f"mo_{day.isoformat()}.csv",
    ):
        if path.is_file():
            return path
    return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    from clinical_knowledge.mo_daily import classify_document_kind

    suffix = path.suffix.lower()
    if suffix == ".csv":
        import csv

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    elif suffix == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "result_raw"}
        kind, reason = classify_document_kind(item)
        item["document_kind"] = kind
        item["document_kind_reason"] = reason
        out.append(item)
    return out


def _icd_root(raw: str) -> str:
    token = re.search(r"[A-TV-ZА-Яа-я]\d{2}", (raw or "").upper())
    if not token:
        return ""
    return token.group(0).replace("А", "A").replace("В", "B").replace("С", "C")


def _filename(path: str) -> str:
    name = (path or "").replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    return name[:160]


def _specialty(row: dict[str, Any]) -> str:
    return str(
        row.get("doctor_specialization")
        or row.get("specialty")
        or row.get("specialization")
        or ""
    ).strip().lower()[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", default="2026-07-01")
    parser.add_argument("--to", dest="date_to", default="2026-07-31")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("MO_DATA_ROOT") or "/var/data/medical_exams",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = все clinical_visit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    first = date.fromisoformat(args.date_from)
    last = date.fromisoformat(args.date_to)
    data_root = Path(args.data_root)

    from clinical_knowledge.case_protocol_suggest import suggest_protocols_for_case
    from clinical_knowledge.kp_validity import looks_omnibus
    from clinical_knowledge.mo_daily import is_scored_document_kind
    from clinical_knowledge.patient_age import resolve_patient_age

    os.environ.setdefault("CASE_PROTOCOL_SUGGEST", "1")

    n_rows = 0
    n_clinical = 0
    n_available = 0
    n_empty = 0
    n_adult_child_kp = 0
    n_age_resolved = 0
    n_omnibus_top1 = 0
    sources: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    empty_icd: Counter[str] = Counter()
    empty_spec: Counter[str] = Counter()
    top1_file: Counter[str] = Counter()
    score_buckets: Counter[str] = Counter()

    started = datetime.now(timezone.utc)
    for day in _days(first, last):
        partition = _resolve_partition(data_root, day)
        if partition is None:
            continue
        for row in _load_rows(partition):
            n_rows += 1
            if not is_scored_document_kind(str(row.get("document_kind") or "")):
                continue
            n_clinical += 1
            if args.limit and n_clinical > args.limit:
                break
            clinical = {
                "clinical_diagnosis": row.get("clinical_diagnosis"),
                "diagnosis_main_text": row.get("diagnosis_main_text") or row.get("diagnosis_text"),
                "diagnosis_short": row.get("diagnosis_short"),
                "mis_diagnos": row.get("mis_diagnos") or row.get("diagnosis_code") or row.get("mkb_code_main"),
                "complaints": row.get("complaints"),
                "anamnesis_doctor": row.get("anamnesis_doctor"),
                "anamnesis_auto": row.get("anamnesis_auto"),
                "patient_age_years": row.get("patient_age_years") or row.get("age_years"),
                "patient_bdate": row.get("patient_bdate"),
                "visit_date": row.get("date") or row.get("visit_date") or day.isoformat(),
                "doctor_specialization": row.get("doctor_specialization"),
            }
            record = {
                "visit_id": "x",
                "specialty": row.get("doctor_specialization") or row.get("specialty"),
                "visit_date": row.get("date") or row.get("visit_date") or day.isoformat(),
                "date": row.get("date") or row.get("visit_date") or day.isoformat(),
                "patient_bdate": row.get("patient_bdate"),
            }
            age_meta = resolve_patient_age(clinical, record)
            if age_meta.get("audience") in {"adult", "child"}:
                n_age_resolved += 1
            result = suggest_protocols_for_case(
                clinical=clinical,
                record=record,
                limit=3,
            )
            source = str(result.get("query_source") or "unknown")
            sources[source] += 1
            modes[str(result.get("mode") or "")] += 1
            items = result.get("items") or []
            if result.get("available") and items:
                n_available += 1
                top = items[0]
                fname = _filename(str(top.get("source_path") or ""))
                if fname:
                    top1_file[fname] += 1
                score = float(top.get("score") or 0)
                if score < 40:
                    score_buckets["lt40"] += 1
                elif score < 70:
                    score_buckets["40_69"] += 1
                else:
                    score_buckets["ge70"] += 1
                age = clinical.get("patient_age_years")
                try:
                    adult = float(str(age).replace(",", ".")) >= 18
                except (TypeError, ValueError):
                    adult = False
                if adult and is_child_only_kp_name(fname):
                    n_adult_child_kp += 1
                if looks_omnibus(
                    {
                        "title": top.get("title"),
                        "source_path": top.get("source_path"),
                    }
                ):
                    n_omnibus_top1 += 1
            else:
                n_empty += 1
                reasons[str(result.get("reason") or "empty")] += 1
                root = _icd_root(str(clinical.get("mis_diagnos") or ""))
                if root:
                    empty_icd[root] += 1
                spec = _specialty(row)
                if spec:
                    empty_spec[spec] += 1
        else:
            continue
        break

    payload = {
        "ok": True,
        "engine": "case_protocol_suggest_v5",
        "period": {"from": first.isoformat(), "to": last.isoformat()},
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "elapsed_sec": int((datetime.now(timezone.utc) - started).total_seconds()),
        "n_rows": n_rows,
        "n_clinical": n_clinical if not args.limit else min(n_clinical, args.limit),
        "n_available": n_available,
        "n_empty": n_empty,
        "available_pct": round(100.0 * n_available / max(1, n_available + n_empty), 1),
        "empty_pct": round(100.0 * n_empty / max(1, n_available + n_empty), 1),
        "adult_with_child_kp": n_adult_child_kp,
        "age_resolved": n_age_resolved,
        "age_resolved_pct": round(100.0 * n_age_resolved / max(1, n_available + n_empty), 1),
        "omnibus_top1": n_omnibus_top1,
        "query_source": dict(sources),
        "mode": dict(modes),
        "empty_reason": reasons.most_common(8),
        "empty_icd_root": empty_icd.most_common(20),
        "empty_specialty": empty_spec.most_common(15),
        "top1_filename": top1_file.most_common(25),
        "score_buckets": dict(score_buckets),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "period", "n_clinical", "n_available", "n_empty", "available_pct",
        "empty_pct", "query_source", "elapsed_sec",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
