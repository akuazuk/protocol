"""Совместимый backend аналитики МО и локального CRM для методиста."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import uuid
from collections import Counter
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .mis_kz_quality import (
    _facets,
    _filtered_agg,
    _flat_case,
    _load_csv_by_visit_cached,
    _match_filters,
    build_kz_case_detail,
    build_kz_dynamics,
    load_kz_cases,
)

ROOT = Path(__file__).resolve().parent.parent
SUPPRESSION_N = max(2, int(os.environ.get("MO_SUPPRESSION_N", "5")))
DOCUMENT_KINDS = frozenset(
    {"medical_exam", "consultation", "certificate", "diagnostic", "non_clinical", "empty", "unknown"}
)
CRM_STATUSES = frozenset(
    {
        "new",
        "assigned",
        "in_review",
        "confirmed_issue",
        "false_positive",
        "needs_more_data",
        "sent_to_doctor",
        "resolved",
        "closed",
    }
)
CRM_ROLES = frozenset({"methodist", "lead", "admin"})
_SYNCED_MONTHS: set[str] = set()


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _db_path() -> Path:
    configured = (os.environ.get("MO_ANALYTICS_DB") or "").strip()
    if configured:
        return Path(configured)
    persistent = Path("/var/data/medical_exams/warehouse")
    if persistent.is_dir() and os.access(persistent, os.W_OK):
        # Операционный CRM хранится отдельно от аналитического warehouse pipeline:
        # у таблиц разные жизненные циклы и миграции, общий filename приводил к
        # несовместимой схеме fact_mo_case.
        return persistent / "mo_methodist.sqlite"
    return ROOT / "data" / "ml" / "secure" / "mo_methodist.sqlite"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS fact_mo_case (
          case_id TEXT PRIMARY KEY, visit_date TEXT, document_kind TEXT, payload_hash TEXT,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fact_mo_finding (
          case_id TEXT NOT NULL, finding_code TEXT NOT NULL, severity TEXT,
          PRIMARY KEY(case_id, finding_code)
        );
        CREATE TABLE IF NOT EXISTS fact_mo_score_axis (
          case_id TEXT NOT NULL, axis TEXT NOT NULL, score REAL,
          PRIMARY KEY(case_id, axis)
        );
        CREATE TABLE IF NOT EXISTS fact_mo_daily (
          report_date TEXT PRIMARY KEY, n_cases INTEGER, avg_score REAL, revision INTEGER
        );
        CREATE TABLE IF NOT EXISTS dim_date (date_key TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS dim_doctor (doctor_key TEXT PRIMARY KEY, display_name TEXT);
        CREATE TABLE IF NOT EXISTS dim_specialty (specialty_key TEXT PRIMARY KEY, display_name TEXT);
        CREATE TABLE IF NOT EXISTS dim_branch (branch_key TEXT PRIMARY KEY, display_name TEXT);
        CREATE TABLE IF NOT EXISTS dim_diagnosis (diagnosis_key TEXT PRIMARY KEY, display_name TEXT);
        CREATE TABLE IF NOT EXISTS dim_service (service_key TEXT PRIMARY KEY, display_name TEXT);
        CREATE TABLE IF NOT EXISTS dim_document_kind (
          document_kind TEXT PRIMARY KEY, display_name TEXT
        );
        CREATE TABLE IF NOT EXISTS crm_case_state (
          case_id TEXT PRIMARY KEY,
          status TEXT NOT NULL DEFAULT 'new',
          assignee TEXT,
          tags_json TEXT NOT NULL DEFAULT '[]',
          due_date TEXT,
          finding_decisions_json TEXT NOT NULL DEFAULT '{}',
          updated_at TEXT NOT NULL,
          updated_by TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS crm_case_event (
          event_id TEXT PRIMARY KEY,
          case_id TEXT NOT NULL,
          event_type TEXT NOT NULL,
          actor TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS saved_view (
          view_id TEXT PRIMARY KEY,
          owner TEXT NOT NULL,
          scope TEXT NOT NULL,
          name TEXT NOT NULL,
          filters_json TEXT NOT NULL,
          config_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS export_job (
          job_id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
          kind TEXT NOT NULL, filters_json TEXT NOT NULL, result_path TEXT,
          created_at TEXT NOT NULL, expires_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_crm_state_status ON crm_case_state(status);
        CREATE INDEX IF NOT EXISTS idx_crm_state_assignee ON crm_case_state(assignee);
        CREATE INDEX IF NOT EXISTS idx_crm_event_case_time ON crm_case_event(case_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_saved_view_owner ON saved_view(owner, scope);
        """
    )
    conn.commit()
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return conn


def _month_for_date(value: str) -> str:
    value = (value or "").strip()
    return value[:7] if len(value) >= 7 else datetime.now().strftime("%Y-%m")


def _selected_months(params: dict[str, Any]) -> list[str]:
    periods = _values(params.get("periods"))
    if periods:
        return sorted({_month_for_date(v) for v in periods})
    start = str(params.get("date_from") or "")
    end = str(params.get("date_to") or "")
    if start and end:
        try:
            cur = date.fromisoformat(start[:10]).replace(day=1)
            stop = date.fromisoformat(end[:10]).replace(day=1)
            out: list[str] = []
            while cur <= stop and len(out) < 36:
                out.append(cur.strftime("%Y-%m"))
                cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
            return out
        except ValueError:
            pass
    return [_month_for_date(str(params.get("month") or start or end))]


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple, set)) else str(value).split(",")
    return [str(v).strip() for v in raw if str(v).strip()]


def _medical_exam_roots() -> list[Path]:
    configured = (os.environ.get("MO_DATA_ROOT") or "").strip()
    candidates = (
        [Path(configured).expanduser()]
        if configured
        else [Path("/var/data/medical_exams"), ROOT / "data" / "medical_exams"]
    )
    return list(dict.fromkeys(path.resolve() for path in candidates))


@lru_cache(maxsize=24)
def _pipeline_records_for_month(month: str) -> tuple[dict[str, Any], ...]:
    """Загрузить ежедневные результаты МО, отдавая приоритет последней ревизии."""
    by_case: dict[str, dict[str, Any]] = {}
    year, month_number = month.split("-", 1)
    for root in _medical_exam_roots():
        secure_dir = root / "secure_cases" / year / month_number
        if not secure_dir.is_dir():
            continue
        for case_path in sorted(secure_dir.glob(f"kz_l1_{month}-??_cases.jsonl")):
            day = case_path.stem.removeprefix("kz_l1_").removesuffix("_cases")
            raw_path = secure_dir / f"mo_{day}.csv"
            csv_by_visit: dict[str, dict[str, Any]] = {}
            if raw_path.is_file():
                with raw_path.open(encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        visit_id = str(row.get("visit_id") or "")
                        if visit_id:
                            csv_by_visit[visit_id] = row
            with case_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        case = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    visit_id = str(case.get("visit_id") or "")
                    if not visit_id:
                        continue
                    rec = _flat_case(case, csv_by_visit.get(visit_id))
                    rec["case_id"] = visit_id
                    rec["_month"] = month
                    rec["_source"] = "daily_pipeline"
                    deep = case.get("deep") if isinstance(case.get("deep"), dict) else {}
                    v3 = case.get("evaluation_v3") if isinstance(case.get("evaluation_v3"), dict) else {}
                    rec["_findings"] = v3.get("findings") or deep.get("findings") or []
                    by_case[visit_id] = rec
    return tuple(by_case.values())


def _records(params: dict[str, Any]) -> list[dict[str, Any]]:
    records_by_key: dict[str, dict[str, Any]] = {}
    months = _selected_months(params)
    for month in months:
        cases, _ = load_kz_cases(month=month)
        csv = _load_csv_by_visit_cached(month)
        for case in cases:
            case_id = str(case.get("visit_id") or "")
            dedupe = f"{month}:{case_id}"
            if not case_id:
                continue
            rec = _flat_case(case, csv.get(case_id))
            rec["case_id"] = case_id
            rec["_month"] = month
            rec["_source"] = "monthly_legacy"
            records_by_key[dedupe] = rec
        for rec in _pipeline_records_for_month(month):
            records_by_key[f"{month}:{rec['case_id']}"] = dict(rec)
    records = list(records_by_key.values())
    pending_months = set(months) - _SYNCED_MONTHS
    if pending_months:
        _sync_warehouse([r for r in records if r["_month"] in pending_months])
        _SYNCED_MONTHS.update(pending_months)
    return records


def _sync_warehouse(records: list[dict[str, Any]]) -> None:
    """Идемпотентно переносит обезличенные факты legacy-витрины в star schema."""
    if not records:
        return
    now = _utc()
    labels = {
        "medical_exam": "Медицинский осмотр",
        "consultation": "Консультация",
        "certificate": "Справка",
        "diagnostic": "Диагностика",
        "non_clinical": "Неклиническая запись",
        "empty": "Пустая запись",
        "unknown": "Требует уточнения",
    }
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "INSERT OR IGNORE INTO dim_document_kind(document_kind,display_name) VALUES(?,?)",
            labels.items(),
        )
        for rec in records:
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "date": rec.get("date"),
                        "kind": rec.get("document_kind"),
                        "score": rec.get("overall_pct"),
                        "status": rec.get("status"),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            conn.execute(
                "INSERT OR REPLACE INTO fact_mo_case(case_id,visit_date,document_kind,payload_hash,updated_at) "
                "VALUES(?,?,?,?,?)",
                (rec["case_id"], rec.get("date"), rec.get("document_kind"), fingerprint, now),
            )
            for axis, score in (
                ("documentation", rec.get("axis_documentation")),
                ("clinical_concordance", rec.get("axis_concordance")),
                ("safety", rec.get("axis_safety")),
                ("regulatory", rec.get("axis_regulatory")),
            ):
                conn.execute(
                    "INSERT OR REPLACE INTO fact_mo_score_axis(case_id,axis,score) VALUES(?,?,?)",
                    (rec["case_id"], axis, score),
                )
            for table, key, value in (
                ("dim_date", rec.get("date"), None),
                ("dim_doctor", rec.get("doctor_fio"), rec.get("doctor_fio")),
                ("dim_specialty", rec.get("specialization"), rec.get("specialization")),
                ("dim_branch", rec.get("filial"), rec.get("filial")),
                ("dim_diagnosis", rec.get("mkb_code_main"), rec.get("mkb_code_main")),
            ):
                if not key:
                    continue
                if table == "dim_date":
                    conn.execute("INSERT OR IGNORE INTO dim_date(date_key) VALUES(?)", (key,))
                else:
                    column = {
                        "dim_doctor": "doctor_key",
                        "dim_specialty": "specialty_key",
                        "dim_branch": "branch_key",
                        "dim_diagnosis": "diagnosis_key",
                    }[table]
                    conn.execute(
                        f"INSERT OR IGNORE INTO {table}({column},display_name) VALUES(?,?)",
                        (key, value),
                    )
        conn.commit()


_MULTI_FILTERS = {
    "specializations": "specialization",
    "filials": "filial",
    "doctors": "doctor_fio",
    "document_kinds": "document_kind",
    "kz_kinds": "kz_kind",
    "statuses": "status",
    "mkb_chapters": "icd_chapter",
}


def _needs_review(rec: dict[str, Any]) -> bool:
    score = rec.get("overall_pct")
    return bool(
        int(rec.get("p0") or 0) > 0
        or int(rec.get("p1") or 0) > 0
        or (isinstance(score, (int, float)) and score < 75)
        or rec.get("status") == "manual_review_required"
    )


def _filter_records(records: Iterable[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    single = {
        key: params.get(key)
        for key in (
            "date_from",
            "date_to",
            "specialization",
            "filial",
            "doctor",
            "document_kind",
            "kz_kind",
            "mkb_chapter",
            "mkb_agreement",
            "age_group",
            "status",
            "score_band",
            "finding_axis",
            "min_severity",
            "q",
        )
        if params.get(key) not in (None, "")
    }
    selected = {field: set(_values(params.get(key))) for key, field in _MULTI_FILTERS.items()}
    excludes = {
        field: set(_values(params.get(f"exclude_{key}"))) for key, field in _MULTI_FILTERS.items()
    }
    out = []
    for rec in records:
        if not _match_filters(rec, single):
            continue
        if any(vals and str(rec.get(field) or "") not in vals for field, vals in selected.items()):
            continue
        if any(vals and str(rec.get(field) or "") in vals for field, vals in excludes.items()):
            continue
        if str(params.get("queue_only") or "").lower() in {"1", "true", "yes"}:
            if not _needs_review(rec):
                continue
        out.append(rec)
    return out


def _suppressed_group(row: dict[str, Any]) -> dict[str, Any]:
    n = int(row.get("n") or 0)
    if n >= SUPPRESSION_N:
        return row
    keep = {k: v for k, v in row.items() if k not in {"n", "avg_overall", "bad_pct", "p0"}}
    return {**keep, "n": None, "n_bucket": f"<{SUPPRESSION_N}", "suppressed": True}


def _organization_groups(
    records: list[dict[str, Any]],
    field: str,
    states: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        label = str(rec.get(field) or "").strip()
        if label:
            groups.setdefault(label, []).append(rec)
    output = []
    for label, rows in groups.items():
        scores = [float(r["overall_pct"]) for r in rows if isinstance(r.get("overall_pct"), (int, float))]
        item = {
            field: label,
            "n": len(rows),
            "avg_overall": round(sum(scores) / len(scores), 1) if scores else None,
            "bad_pct": round(100 * sum(score < 75 for score in scores) / len(scores), 1) if scores else None,
            "p0": sum(int(r.get("p0") or 0) for r in rows),
        }
        if field == "doctor_fio":
            item["specialization"] = next((str(r.get("specialization") or "") for r in rows if r.get("specialization")), "")
            item["open_cases"] = sum(
                1
                for r in rows
                if _needs_review(r)
                and (states or {}).get(r["case_id"], {}).get("status", "new")
                not in {"false_positive", "resolved", "closed"}
            )
        output.append(item)
    output.sort(key=lambda row: int(row.get("n") or 0), reverse=True)
    return [_suppressed_group(row) for row in output]


def _public_row(rec: dict[str, Any]) -> dict[str, Any]:
    blocked = {"patient_id", "source_path", "_month"}
    return {k: v for k, v in rec.items() if k not in blocked and not k.startswith("_")}


def _crm_states(case_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not case_ids:
        return {}
    with closing(_connect()) as conn:
        out: dict[str, dict[str, Any]] = {}
        for offset in range(0, len(case_ids), 500):
            batch = case_ids[offset : offset + 500]
            marks = ",".join("?" for _ in batch)
            for row in conn.execute(f"SELECT * FROM crm_case_state WHERE case_id IN ({marks})", batch):
                item = dict(row)
                item["tags"] = json.loads(item.pop("tags_json") or "[]")
                item["finding_decisions"] = json.loads(item.pop("finding_decisions_json") or "{}")
                out[item["case_id"]] = item
        return out


def build_cases(params: dict[str, Any]) -> dict[str, Any]:
    all_records = _records(params)
    filtered = _filter_records(all_records, params)
    states = _crm_states([r["case_id"] for r in filtered])
    crm_statuses = set(_values(params.get("crm_statuses")))
    assignees = set(_values(params.get("assignees")))
    if crm_statuses:
        filtered = [r for r in filtered if states.get(r["case_id"], {}).get("status", "new") in crm_statuses]
    if assignees:
        filtered = [r for r in filtered if states.get(r["case_id"], {}).get("assignee") in assignees]
    if str(params.get("queue_only") or "").lower() in {"1", "true", "yes"}:
        filtered = [
            r
            for r in filtered
            if states.get(r["case_id"], {}).get("status", "new")
            not in {"false_positive", "resolved", "closed"}
        ]
    sort_map = {
        "date": "date",
        "overall": "overall_pct",
        "priority": "p0",
        "updated_at": "updated_at",
    }
    sort_field = sort_map.get(str(params.get("sort_by") or ""), "date")
    reverse = str(params.get("sort_dir") or "desc").lower() == "desc"
    filtered.sort(key=lambda r: (r.get(sort_field) is None, r.get(sort_field) or ""), reverse=reverse)
    page = max(1, int(params.get("page") or 1))
    page_size = max(1, min(200, int(params.get("page_size") or 50)))
    start = (page - 1) * page_size
    rows = []
    for rec in filtered[start : start + page_size]:
        crm = states.get(rec["case_id"]) or {"status": "new", "tags": [], "finding_decisions": {}}
        rows.append({**_public_row(rec), "crm": crm})
    agg = _filtered_agg(filtered)
    agg["by_specialty"] = [_suppressed_group(r) for r in agg.get("by_specialty") or []]
    agg["by_chapter"] = [_suppressed_group(r) for r in agg.get("by_chapter") or []]
    return {
        "ok": True,
        "namespace": "mo",
        "total": len(filtered),
        "page": page,
        "page_size": page_size,
        "rows": rows,
        "aggregate": agg,
        "suppression_n": SUPPRESSION_N,
        "applied_filters": {k: v for k, v in params.items() if v not in (None, "", [], False)},
    }


def build_facets(params: dict[str, Any]) -> dict[str, Any]:
    filtered = _filter_records(_records(params), params)
    facets = _facets(filtered)
    doctor_counts = Counter(str(r.get("doctor_fio") or "") for r in filtered if r.get("doctor_fio"))
    facets["doctors"] = [
        {
            "value": key,
            "n": count if count >= SUPPRESSION_N else None,
            "n_bucket": None if count >= SUPPRESSION_N else f"<{SUPPRESSION_N}",
            "suppressed": count < SUPPRESSION_N,
        }
        for key, count in doctor_counts.most_common(200)
    ]
    for values in facets.values():
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict) and isinstance(item.get("n"), int) and item["n"] < SUPPRESSION_N:
                item["n"] = None
                item["n_bucket"] = f"<{SUPPRESSION_N}"
                item["suppressed"] = True
    states = _crm_states([r["case_id"] for r in filtered])
    crm = Counter((states.get(r["case_id"]) or {}).get("status", "new") for r in filtered)
    facets["crm_statuses"] = [
        {"value": key, "n": n if n >= SUPPRESSION_N else None, "n_bucket": None if n >= SUPPRESSION_N else f"<{SUPPRESSION_N}"}
        for key, n in crm.most_common()
    ]
    return {"ok": True, "facets": facets, "n_filtered": len(filtered), "suppression_n": SUPPRESSION_N}


def build_overview(params: dict[str, Any]) -> dict[str, Any]:
    filtered = _filter_records(_records(params), params)
    states = _crm_states([r["case_id"] for r in filtered])
    agg = _filtered_agg(filtered)
    kinds = Counter(r.get("document_kind") or "unknown" for r in filtered)
    eligible = sum(kinds.get(k, 0) for k in ("medical_exam", "consultation"))
    small_slice = len(filtered) < SUPPRESSION_N
    return {
        "ok": True,
        "namespace": "mo",
        "period": {"date_from": params.get("date_from"), "date_to": params.get("date_to")},
        "kpi": {
            "source_records": None if small_slice else len(filtered),
            "n_bucket": f"<{SUPPRESSION_N}" if small_slice else None,
            "eligible": None if small_slice else eligible,
            "evaluated": None if small_slice else sum(1 for r in filtered if isinstance(r.get("overall_pct"), (int, float))),
            "avg_score": None if small_slice else agg.get("avg_overall"),
            "needs_attention": None if small_slice else agg.get("n_bad"),
            "needs_attention_pct": None if small_slice else agg.get("pct_bad"),
            "critical": None if small_slice else sum(1 for r in filtered if int(r.get("p0") or 0) > 0),
            "suppressed": small_slice,
        },
        "document_kind_distribution": {} if small_slice else dict(kinds),
        "data_through": max((str(r.get("date") or "") for r in filtered), default=""),
        "axis_means": None if small_slice else agg.get("axis_means"),
        "severity_totals": None if small_slice else agg.get("severity_totals"),
        "status_distribution": None if small_slice else agg.get("status_distribution"),
        "by_specialty": [_suppressed_group(r) for r in agg.get("by_specialty") or []],
        "by_chapter": [_suppressed_group(r) for r in agg.get("by_chapter") or []],
        "by_doctor": _organization_groups(filtered, "doctor_fio", states),
        "by_branch": _organization_groups(filtered, "filial"),
        "suppression_n": SUPPRESSION_N,
    }


def build_daily_report(report_date: str) -> dict[str, Any]:
    try:
        chosen = date.fromisoformat(report_date)
    except ValueError:
        return {"ok": False, "error": "invalid_date"}
    configured_root = (os.environ.get("MO_DATA_ROOT") or "").strip()
    roots = [
        Path(configured_root).expanduser() if configured_root else Path("/var/data/medical_exams"),
        ROOT / "data" / "medical_exams",
    ]
    for root in roots:
        path = root / "reports" / f"{chosen:%Y}" / f"{chosen:%m}" / f"{chosen:%d}" / "report.json"
        if not path.is_file():
            continue
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = stored.get("summary") or {}
        overview = {
            "n": summary.get("source_rows"),
            "n_evaluated": summary.get("scored"),
            "avg_overall": summary.get("avg_score"),
            "n_bad": summary.get("needs_attention"),
            "severity_totals": {"P0": summary.get("critical")},
            "avg_coverage": (stored.get("month_to_date") or {}).get("avg_coverage"),
        }
        return {
            "ok": True,
            "date": stored.get("date") or chosen.isoformat(),
            "revision": stored.get("revision"),
            "generated_at": stored.get("generated_at"),
            "quality_status": "ok" if (stored.get("quality") or {}).get("passed") else "blocked",
            "executive_summary": {
                "ok": True,
                "kpi": {
                    "n": summary.get("source_rows"),
                    "eligible": summary.get("eligible_rows"),
                    "scored": summary.get("scored"),
                    "avg_score": summary.get("avg_score"),
                    "needs_attention": summary.get("needs_attention"),
                    "critical": summary.get("critical"),
                },
                "axes": stored.get("axes") or {},
                "organizations": stored.get("organizations") or {},
            },
            "overview": overview,
            "comparison": stored.get("comparisons") or {},
            "month_to_date": stored.get("month_to_date") or {},
            "action_queue": stored.get("action_queue") or [],
            "data_quality": stored.get("quality") or {},
        }
    params = {"date_from": chosen.isoformat(), "date_to": chosen.isoformat()}
    overview = build_overview(params)
    cases = build_cases({**params, "page": 1, "page_size": 20, "crm_statuses": "new,assigned,in_review"})
    previous = build_overview(
        {"date_from": (chosen - timedelta(days=1)).isoformat(), "date_to": (chosen - timedelta(days=1)).isoformat()}
    )
    current_avg = overview["kpi"].get("avg_score")
    previous_avg = previous["kpi"].get("avg_score")
    delta = (
        round(float(current_avg) - float(previous_avg), 1)
        if isinstance(current_avg, (int, float)) and isinstance(previous_avg, (int, float))
        else None
    )
    return {
        "ok": True,
        "date": chosen.isoformat(),
        "revision": 1,
        "generated_at": _utc(),
        "quality_status": "no_data" if overview["kpi"].get("n_bucket") == "<5" and not cases["total"] else "ok",
        "executive_summary": overview,
        "comparison": {"previous_date": (chosen - timedelta(days=1)).isoformat(), "avg_score_delta": delta},
        "action_queue": cases["rows"],
        "data_quality": build_data_quality(params),
    }


def build_trends(params: dict[str, Any]) -> dict[str, Any]:
    records = _filter_records(_records(params), params)
    by_day: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        if rec.get("date"):
            by_day.setdefault(rec["date"], []).append(rec)
    daily = []
    for day, rows in sorted(by_day.items()):
        agg = _filtered_agg(rows)
        small = len(rows) < SUPPRESSION_N
        daily.append(
            {
                "date": day,
                "n": None if small else len(rows),
                "n_bucket": f"<{SUPPRESSION_N}" if small else None,
                "avg_score": None if small else agg.get("avg_overall"),
                "critical": None if small else sum(r["p0"] for r in rows),
                "suppressed": small,
            }
        )
    legacy = build_kz_dynamics(months=_selected_months(params))
    return {"ok": True, "daily": daily, "monthly": legacy.get("series") or []}


def build_data_quality(params: dict[str, Any]) -> dict[str, Any]:
    rows = _filter_records(_records(params), params)
    n = len(rows)
    count = lambda pred: sum(1 for row in rows if pred(row))
    parse_fail = count(lambda r: str(r.get("parse_ok") or "").lower() not in {"1", "true"})
    date_mismatch = count(lambda r: str(r.get("date_mismatch") or "").lower() in {"1", "true"})
    small = n < SUPPRESSION_N
    return {
        "ok": True,
        "n": None if small else n,
        "n_bucket": f"<{SUPPRESSION_N}" if small else None,
        "suppressed": small,
        "parse_rate": None if small else round(100 * (n - parse_fail) / n, 2),
        "date_mismatch_rate": None if small else round(100 * date_mismatch / n, 2),
        "missing_doctor": None if small else count(lambda r: not str(r.get("doctor_fio") or "").strip(" -")),
        "missing_specialty": None if small else count(lambda r: not str(r.get("specialization") or "").strip(" -")),
        "missing_branch": None if small else count(lambda r: not str(r.get("filial") or "").strip(" -")),
        "missing_mkb": None if small else count(lambda r: not r.get("mkb_code_main")),
        "duplicate_case_ids": None if small else n - len({r["case_id"] for r in rows}),
        "unknown_document_kind": None if small else count(lambda r: r.get("document_kind") == "unknown"),
    }


def build_case_detail(case_id: str, month: str | None = None) -> dict[str, Any]:
    selected_month = month or _month_for_date("")
    detail: dict[str, Any] = {"ok": False}
    for rec in _pipeline_records_for_month(selected_month):
        if rec["case_id"] == case_id:
            detail = {
                "ok": True,
                "record": _public_row(dict(rec)),
                "axes": {
                    "documentation": rec.get("axis_documentation"),
                    "clinical_concordance": rec.get("axis_concordance"),
                    "safety": rec.get("axis_safety"),
                    "regulatory": rec.get("axis_regulatory"),
                },
                "findings": rec.get("_findings") or [],
                "source": "daily_pipeline",
            }
            break
    if not detail.get("ok"):
        detail = build_kz_case_detail(month=selected_month, visit_id=case_id)
    if not detail.get("ok") and not month:
        for candidate in reversed(build_kz_dynamics().get("months") or []):
            detail = build_kz_case_detail(month=candidate, visit_id=case_id)
            if detail.get("ok"):
                break
    if not detail.get("ok"):
        return detail
    record = dict(detail.get("record") or {})
    record.pop("patient_id", None)
    detail["record"] = record
    detail["case_id"] = case_id
    with closing(_connect()) as conn:
        state = conn.execute("SELECT * FROM crm_case_state WHERE case_id=?", (case_id,)).fetchone()
        events = conn.execute(
            "SELECT event_id,event_type,actor,payload_json,created_at FROM crm_case_event "
            "WHERE case_id=? ORDER BY created_at DESC LIMIT 200",
            (case_id,),
        ).fetchall()
    detail["crm"] = dict(state) if state else {"case_id": case_id, "status": "new", "tags_json": "[]"}
    detail["events"] = [
        {**dict(row), "payload": json.loads(row["payload_json"] or "{}")} for row in events
    ]
    for event in detail["events"]:
        event.pop("payload_json", None)
    return detail


def save_view(*, actor: str, payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()[:120]
    if not name:
        raise ValueError("name_required")
    view_id = str(payload.get("view_id") or uuid.uuid4())
    scope = str(payload.get("scope") or "private").lower()
    if scope not in {"private", "team"}:
        raise ValueError("invalid_scope")
    now = _utc()
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    with closing(_connect()) as conn:
        existing = conn.execute("SELECT owner,created_at FROM saved_view WHERE view_id=?", (view_id,)).fetchone()
        if existing and existing["owner"] != actor:
            raise PermissionError("view_owner_mismatch")
        conn.execute(
            "INSERT OR REPLACE INTO saved_view(view_id,owner,scope,name,filters_json,config_json,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (view_id, actor, scope, name, json.dumps(filters, ensure_ascii=False), json.dumps(config, ensure_ascii=False), existing["created_at"] if existing else now, now),
        )
        conn.commit()
    return {"ok": True, "view_id": view_id, "updated_at": now}


def list_views(actor: str) -> dict[str, Any]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM saved_view WHERE owner=? OR scope='team' ORDER BY updated_at DESC", (actor,)
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["filters"] = json.loads(item.pop("filters_json") or "{}")
        item["config"] = json.loads(item.pop("config_json") or "{}")
        items.append(item)
    return {"ok": True, "items": items}


def delete_view(*, actor: str, view_id: str) -> dict[str, Any]:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT owner FROM saved_view WHERE view_id=?", (view_id,)).fetchone()
        if not row:
            return {"ok": False, "error": "view_not_found"}
        if row["owner"] != actor:
            raise PermissionError("view_owner_mismatch")
        conn.execute("DELETE FROM saved_view WHERE view_id=?", (view_id,))
        conn.commit()
    return {"ok": True, "view_id": view_id}


def apply_bulk_action(*, actor: str, role: str, payload: dict[str, Any]) -> dict[str, Any]:
    if role not in CRM_ROLES:
        raise PermissionError("mutation_requires_methodist_role")
    case_ids = list(dict.fromkeys(str(v).strip() for v in payload.get("case_ids") or [] if str(v).strip()))
    if not case_ids or len(case_ids) > 500:
        raise ValueError("case_ids_required_or_too_many")
    changes = payload.get("changes") if isinstance(payload.get("changes"), dict) else {}
    status = changes.get("status")
    if status is not None and status not in CRM_STATUSES:
        raise ValueError("invalid_status")
    assignee = str(changes.get("assignee") or "").strip()[:120] or None if "assignee" in changes else ...
    due_date = str(changes.get("due_date") or "").strip()[:10] or None if "due_date" in changes else ...
    tags = [str(v).strip()[:50] for v in changes.get("tags") or [] if str(v).strip()][:30] if "tags" in changes else ...
    finding_decisions = changes.get("finding_decisions", ...)
    if finding_decisions is not ... and not isinstance(finding_decisions, dict):
        raise ValueError("finding_decisions_must_be_object")
    comment = str(payload.get("comment") or "").strip()[:2000]
    now = _utc()
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        for case_id in case_ids:
            current = conn.execute("SELECT * FROM crm_case_state WHERE case_id=?", (case_id,)).fetchone()
            state = dict(current) if current else {
                "status": "new", "assignee": None, "tags_json": "[]", "due_date": None, "finding_decisions_json": "{}"
            }
            if status is not None:
                state["status"] = status
            if assignee is not ...:
                state["assignee"] = assignee
            if due_date is not ...:
                state["due_date"] = due_date
            if tags is not ...:
                state["tags_json"] = json.dumps(tags, ensure_ascii=False)
            if finding_decisions is not ...:
                current_decisions = json.loads(state["finding_decisions_json"] or "{}")
                for finding_code, decision in finding_decisions.items():
                    if str(decision) not in {"confirmed", "false_positive", "needs_more_data", "unreviewed"}:
                        raise ValueError("invalid_finding_decision")
                    current_decisions[str(finding_code)[:120]] = str(decision)
                state["finding_decisions_json"] = json.dumps(current_decisions, ensure_ascii=False)
            conn.execute(
                "INSERT OR REPLACE INTO crm_case_state(case_id,status,assignee,tags_json,due_date,finding_decisions_json,updated_at,updated_by) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (case_id, state["status"], state["assignee"], state["tags_json"], state["due_date"], state["finding_decisions_json"], now, actor),
            )
            event_payload = {"changes": changes}
            if comment:
                event_payload["comment"] = comment
            conn.execute(
                "INSERT INTO crm_case_event(event_id,case_id,event_type,actor,payload_json,created_at) VALUES(?,?,?,?,?,?)",
                (str(uuid.uuid4()), case_id, "bulk_action" if len(case_ids) > 1 else "case_action", actor, json.dumps(event_payload, ensure_ascii=False), now),
            )
        conn.commit()
    return {"ok": True, "updated": len(case_ids), "case_ids": case_ids, "updated_at": now}


def build_reports() -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    roots = [Path("/var/data/medical_exams/reports"), ROOT / "data" / "medical_exams" / "reports"]
    for base in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*/*/*/report.json"), reverse=True)[:120]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            reports.append(
                {
                    "date": data.get("date") or path.parent.name,
                    "revision": data.get("revision"),
                    "generated_at": data.get("generated_at"),
                    "quality_status": data.get("quality_status")
                    or ("ok" if (data.get("quality") or {}).get("passed") else "blocked"),
                }
            )
    if not reports:
        reports = [{"month": month, "kind": "legacy_monthly"} for month in reversed(build_kz_dynamics().get("months") or [])]
    return {"ok": True, "items": reports}


def build_entity(kind: str, entity_id: str, params: dict[str, Any]) -> dict[str, Any]:
    field = {"doctors": "doctor_fio", "specialties": "specialization", "branches": "filial"}.get(kind)
    if not field:
        return {"ok": False, "error": "unknown_entity_kind"}
    rows = [r for r in _filter_records(_records(params), params) if str(r.get(field) or "") == entity_id]
    if len(rows) < SUPPRESSION_N:
        return {"ok": True, "suppressed": True, "n_bucket": f"<{SUPPRESSION_N}", "entity_id": entity_id}
    agg = _filtered_agg(rows)
    return {"ok": True, "entity_id": entity_id, "entity_kind": kind, "n": len(rows), "aggregate": agg}


def create_export(*, actor: str, payload: dict[str, Any]) -> dict[str, Any]:
    kind = str(payload.get("kind") or "aggregates")
    if kind not in {"aggregates", "cases"}:
        raise ValueError("invalid_export_kind")
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    data = build_overview(filters) if kind == "aggregates" else build_cases({**filters, "page": 1, "page_size": 200})
    job_id = str(uuid.uuid4())
    export_dir = _db_path().parent / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / f"{job_id}.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(hours=24)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO export_job(job_id,owner,status,kind,filters_json,result_path,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?)",
            (job_id, actor, "ready", kind, json.dumps(filters, ensure_ascii=False), str(target), _utc(), expires),
        )
        conn.commit()
    return {
        "ok": True,
        "job_id": job_id,
        "status": "ready",
        "expires_at": expires,
        "download_url": f"/api/methodist/mo/exports/{job_id}",
    }


def get_export(*, actor: str, job_id: str) -> Path:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT owner,status,result_path,expires_at FROM export_job WHERE job_id=?",
            (job_id,),
        ).fetchone()
    if not row:
        raise FileNotFoundError("export_not_found")
    if row["owner"] != actor:
        raise PermissionError("export_owner_mismatch")
    expires = datetime.fromisoformat(str(row["expires_at"]).replace("Z", "+00:00"))
    if expires < datetime.now(timezone.utc):
        raise FileNotFoundError("export_expired")
    path = Path(row["result_path"])
    if row["status"] != "ready" or not path.is_file():
        raise FileNotFoundError("export_not_ready")
    return path


def compatibility_metadata() -> dict[str, Any]:
    return {
        "deprecated": True,
        "replacement": "/api/methodist/mo",
        "sunset_after_releases": 2,
        "legacy_fields_preserved": ["kz_kind", "evaluation_v3"],
    }
