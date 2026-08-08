"""Совместимый backend аналитики МО и локального CRM для методиста."""
from __future__ import annotations

import csv
import calendar
import json
import os
import re
import sqlite3
import statistics
import uuid
from collections import Counter
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from .mo_daily import CRM_SCHEMA_SQL, initialize_warehouse, migrate_crm, sanitize_mo_org_label
from .mo_metrics import (
    METRICS,
    SCHEMA_VERSION,
    DateRange,
    mean_confidence_interval,
    metric_catalog,
    resolve_periods,
    suppress_values,
)
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
    {
        "clinical_visit",
        "procedure_session",
        "medical_exam",
        "consultation",
        "certificate",
        "diagnostic",
        "non_clinical",
        "empty",
        "unknown",
    }
)
DOCUMENT_KIND_LABELS = {
    "clinical_visit": "Клинический приём",
    "procedure_session": "Манипуляция / процедура",
    "medical_exam": "Профосмотр / медосмотр",
    "consultation": "Клинический приём (legacy)",
    "certificate": "Справка",
    "diagnostic": "Диагностическое исследование",
    "non_clinical": "Неклинический документ",
    "empty": "Пустой документ",
    "unknown": "Не определён",
}
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
CRM_ROLES = frozenset({"methodist", "lead", "admin", "expert"})
_CRM_MIGRATED = False
_HEX_ID_RX = re.compile(r"^[a-f0-9]{32,64}$", re.IGNORECASE)
_ICD_CODE_RX = re.compile(r"^[A-Za-zА-Яа-я]\d{2}(?:\.\d{1,2})?$")


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _looks_like_opaque_id(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and bool(_HEX_ID_RX.match(text))


def _is_valid_icd_code(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and bool(_ICD_CODE_RX.match(text))


def _safe_diagnosis_text(*values: Any) -> str:
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        if _looks_like_opaque_id(text):
            continue
        return text
    return ""


def _db_path() -> Path:
    """Единый файл витрины: аналитика pipeline и CRM методиста живут вместе."""
    configured = (os.environ.get("MO_ANALYTICS_DB") or "").strip()
    if configured:
        return Path(configured)
    for root in _medical_exam_roots():
        warehouse = root / "warehouse"
        if warehouse.is_dir() and os.access(warehouse, os.W_OK):
            return warehouse / "mo_analytics.sqlite"
        # Диск Render смонтирован, но каталога ещё нет: создаём, иначе CRM осядет
        # в контейнере и исчезнет при следующем деплое.
        if root.parent.is_dir() and os.access(root.parent, os.W_OK):
            return warehouse / "mo_analytics.sqlite"
    return ROOT / "data" / "medical_exams" / "warehouse" / "mo_analytics.sqlite"


def _legacy_crm_paths() -> list[Path]:
    """Старые файлы CRM: читаются один раз для миграции и больше не пишутся."""
    return [
        *(root / "warehouse" / "mo_methodist.sqlite" for root in _medical_exam_roots()),
        ROOT / "data" / "ml" / "secure" / "mo_methodist.sqlite",
    ]


def _migrate_legacy_crm(target: Path) -> dict[str, int]:
    """Перенести CRM из старого файла один раз за процесс.

    При явном `MO_ANALYTICS_DB` (тесты, разбор инцидента) не трогаем чужие данные:
    файлом управляет тот, кто его указал.
    """
    global _CRM_MIGRATED
    if _CRM_MIGRATED or (os.environ.get("MO_ANALYTICS_DB") or "").strip():
        return {}
    _CRM_MIGRATED = True
    moved: dict[str, int] = {}
    for legacy in _legacy_crm_paths():
        if not legacy.is_file() or legacy.resolve() == target.resolve():
            continue
        for table, count in migrate_crm(legacy, target).items():
            if count:
                moved[table] = moved.get(table, 0) + count
    return moved


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    initialize_warehouse(path)
    if fresh or not _CRM_MIGRATED:
        _migrate_legacy_crm(path)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(CRM_SCHEMA_SQL)
    conn.commit()
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return conn


def _warehouse_available() -> bool:
    path = _db_path()
    if not path.is_file():
        return False
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2) as conn:
            return conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fact_mo_case'"
            ).fetchone() is not None
    except sqlite3.Error:
        return False


def _backend_source() -> str:
    configured = (os.environ.get("MO_BACKEND_SOURCE") or "auto").strip().lower()
    if configured not in {"auto", "warehouse", "jsonl"}:
        raise ValueError("MO_BACKEND_SOURCE должен быть auto, warehouse или jsonl")
    if configured == "jsonl":
        return "jsonl_fallback"
    if configured == "warehouse":
        if not _warehouse_available():
            raise RuntimeError("Витрина МО недоступна при MO_BACKEND_SOURCE=warehouse")
        return "warehouse"
    return "warehouse" if _warehouse_available() else "jsonl_fallback"


def _read_connection() -> sqlite3.Connection:
    if _backend_source() != "warehouse":
        raise RuntimeError("SQL-витрина МО недоступна")
    conn = sqlite3.connect(f"file:{_db_path()}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _warehouse_has_column(db_path: str, table: str, column: str) -> bool:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        try:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            return column in cols
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def _finding_shadow_select(alias: str = "f") -> str:
    """SELECT-фрагмент is_shadow: на старых витринах колонки ещё нет."""
    path = str(_db_path())
    if _warehouse_has_column(path, "fact_mo_finding", "is_shadow"):
        return f"COALESCE({alias}.is_shadow, 0) AS is_shadow"
    return "0 AS is_shadow"


def _finding_link_select(alias: str = "f") -> str:
    path = str(_db_path())
    linked = (
        f"{alias}.linked_fields_json"
        if _warehouse_has_column(path, "fact_mo_finding", "linked_fields_json")
        else "NULL AS linked_fields_json"
    )
    hint = (
        f"{alias}.link_hint_ru"
        if _warehouse_has_column(path, "fact_mo_finding", "link_hint_ru")
        else "'' AS link_hint_ru"
    )
    return f"{linked}, {hint}"


def _source_for_period(period: DateRange) -> str:
    """В auto используем SQL только если в выбранном периоде есть факты."""
    source = _backend_source()
    if source != "warehouse" or (os.environ.get("MO_BACKEND_SOURCE") or "auto").strip().lower() != "auto":
        return source
    try:
        with closing(_read_connection()) as conn:
            exists = conn.execute(
                "SELECT 1 FROM fact_mo_case WHERE visit_date BETWEEN ? AND ? LIMIT 1",
                (period.date_from.isoformat(), period.date_to.isoformat()),
            ).fetchone()
    except sqlite3.Error:
        return "jsonl_fallback"
    return "warehouse" if exists else "jsonl_fallback"


def _sql_case_filter(
    period: DateRange,
    params: dict[str, Any],
    *,
    alias: str = "c",
) -> tuple[str, list[Any]]:
    clauses = [f"{alias}.visit_date BETWEEN ? AND ?"]
    values: list[Any] = [period.date_from.isoformat(), period.date_to.isoformat()]
    mappings = {
        "specializations": "specialty",
        "filials": "filial",
        "document_kinds": "document_kind",
        "statuses": "status",
    }
    for param, column in mappings.items():
        selected = _values(params.get(param))
        if not selected:
            continue
        clauses.append(f"{alias}.{column} IN ({','.join('?' for _ in selected)})")
        values.extend(selected)
    doctors = _values(params.get("doctors"))
    if doctors:
        clauses.append(
            f"{alias}.doctor_key IN (SELECT doctor_key FROM dim_doctor "
            f"WHERE doctor_fio IN ({','.join('?' for _ in doctors)}))"
        )
        values.extend(doctors)
    return " AND ".join(clauses), values


def _month_for_date(value: str) -> str:
    value = (value or "").strip()
    return value[:7] if len(value) >= 7 else datetime.now(ZoneInfo("Europe/Minsk")).strftime("%Y-%m")


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
    """Разбор multi-select.

    UI шлёт фильтры через `|` (адреса филиалов содержат запятые).
    Legacy CSV (metrics/коды без пробелов) по-прежнему через `,`.
    Строка с запятой и пробелами в сегментах - одно значение (филиал).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        # «overall,volume» - коды; «ул. Захарова, 50Д» - один адрес (есть пробелы).
        if parts and all(" " not in part for part in parts):
            return parts
    return [text]


def _medical_exam_roots() -> list[Path]:
    configured = (os.environ.get("MO_DATA_ROOT") or "").strip()
    candidates = (
        [Path(configured).expanduser()]
        if configured
        else [Path("/var/data/medical_exams"), ROOT / "data" / "medical_exams"]
    )
    return list(dict.fromkeys(path.resolve() for path in candidates))


def _latest_pipeline_report() -> tuple[dict[str, Any] | None, Path | None]:
    latest: tuple[str, dict[str, Any], Path] | None = None
    for root in _medical_exam_roots():
        base = root / "reports"
        if not base.is_dir():
            continue
        for path in base.glob("*/*/*/report.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            report_date = str(payload.get("date") or "")
            if not report_date:
                continue
            if latest is None or report_date > latest[0]:
                latest = (report_date, payload, path)
    if latest is None:
        return None, None
    return latest[1], latest[2]


def _pipeline_state_snapshot() -> dict[str, Any]:
    for root in _medical_exam_roots():
        state_path = root / "state" / "pipeline.json"
        if not state_path.is_file():
            continue
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "invalid", "path": str(state_path)}
        dates = payload.get("dates") if isinstance(payload.get("dates"), dict) else {}
        last_date = max(dates.keys(), default="")
        last_entry = dates.get(last_date) if last_date else {}
        return {
            "status": "present",
            "path": str(state_path),
            "last_date": last_date or None,
            "last_stage": (last_entry or {}).get("status"),
            "last_heartbeat": (last_entry or {}).get("heartbeat"),
            "runs_total": len(payload.get("runs") or []),
        }
    return {"status": "missing"}


def _describe_empty_state(*, total_records: int, filtered_records: int, params: dict[str, Any]) -> dict[str, Any]:
    if total_records == 0:
        return {
            "reason_code": "no_source_data",
            "title": "Нет загруженных данных за выбранный период",
            "hint": "Проверьте ежедневный запуск и свежесть последнего отчёта.",
        }
    if filtered_records == 0:
        applied = [k for k, v in params.items() if v not in (None, "", [], False)]
        return {
            "reason_code": "filters_excluded_all",
            "title": "Фильтры исключили все записи",
            "hint": "Сбросьте часть фильтров или расширьте диапазон дат.",
            "applied_keys": applied,
        }
    return {"reason_code": "ok", "title": "", "hint": ""}


def build_freshness(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    records = _records(params)
    filtered = _filter_records(records, params)
    now_minsk = datetime.now(ZoneInfo("Europe/Minsk"))
    report_payload, report_path = _latest_pipeline_report()
    state_info = _pipeline_state_snapshot()
    latest_case_date = max((str(r.get("date") or "") for r in filtered), default="")
    latest_any_case_date = max((str(r.get("date") or "") for r in records), default="")
    report_date = str((report_payload or {}).get("date") or "")
    data_through = max([d for d in (report_date, latest_case_date, latest_any_case_date) if d], default="")
    lag_days: int | None = None
    if data_through:
        try:
            lag_days = (now_minsk.date() - date.fromisoformat(data_through[:10])).days
        except ValueError:
            lag_days = None
    status = "unknown"
    if lag_days is None:
        status = "missing"
    elif lag_days <= 1:
        status = "fresh"
    elif lag_days <= 3:
        status = "stale"
    else:
        status = "critical"
    empty_state = _describe_empty_state(
        total_records=len(records),
        filtered_records=len(filtered),
        params=params,
    )
    return {
        "ok": True,
        "source": _backend_source(),
        "status": status,
        "lag_days": lag_days,
        "data_through": data_through or None,
        "latest_report": {
            "date": report_date or None,
            "generated_at": (report_payload or {}).get("generated_at"),
            "revision": (report_payload or {}).get("revision"),
            "path": str(report_path) if report_path else None,
        },
        "state": state_info,
        "roots": [
            {
                "path": str(root),
                "exists": root.is_dir(),
                "has_reports": (root / "reports").is_dir(),
                "has_secure_cases": (root / "secure_cases").is_dir(),
            }
            for root in _medical_exam_roots()
        ],
        "filtered_records": len(filtered),
        "total_records": len(records),
        "empty_state": empty_state,
        "checked_at": now_minsk.replace(microsecond=0).isoformat(),
    }


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


def _jsonl_records(params: dict[str, Any]) -> list[dict[str, Any]]:
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
    # Витрину заполняет ежедневный pipeline (upsert_warehouse); API здесь ничего не дублирует.
    return list(records_by_key.values())


def _warehouse_records(params: dict[str, Any]) -> list[dict[str, Any]]:
    where: list[str] = []
    values: list[Any] = []
    if params.get("date_from"):
        where.append("c.visit_date >= ?")
        values.append(str(params["date_from"])[:10])
    if params.get("date_to"):
        where.append("c.visit_date <= ?")
        values.append(str(params["date_to"])[:10])
    months = _selected_months(params)
    if not params.get("date_from") and not params.get("date_to") and months:
        marks = ",".join("?" for _ in months)
        where.append(f"substr(c.visit_date, 1, 7) IN ({marks})")
        values.extend(months)
    sql = """
        SELECT c.*, d.doctor_fio,
               COALESCE(d.specialty, c.specialty) AS doctor_specialty,
               COALESCE(d.filial, c.filial) AS doctor_filial,
               COALESCE(NULLIF(dx.diagnosis_label, ''), '') AS diagnosis_label,
               COALESCE(f.p0, 0) AS p0, COALESCE(f.p1, 0) AS p1,
               COALESCE(f.p2, 0) AS p2, COALESCE(f.p3, 0) AS p3,
               COALESCE(f.finding_codes, '') AS finding_codes,
               ax_reg.score AS reg55_pct
        FROM fact_mo_case c
        LEFT JOIN dim_doctor d ON d.doctor_key = c.doctor_key
        LEFT JOIN dim_diagnosis dx ON dx.diagnosis_code = c.diagnosis_code
        LEFT JOIN fact_mo_score_axis ax_reg
               ON ax_reg.mis_id = c.mis_id AND ax_reg.axis = 'regulatory'
        LEFT JOIN (
          SELECT mis_id,
                 SUM(severity='P0') p0, SUM(severity='P1') p1,
                 SUM(severity='P2') p2, SUM(severity='P3') p3,
                 GROUP_CONCAT(DISTINCT finding_code) finding_codes
          FROM fact_mo_finding GROUP BY mis_id
        ) f ON f.mis_id = c.mis_id
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    with closing(_read_connection()) as conn:
        rows = conn.execute(sql, values).fetchall()
    output = []
    for row in rows:
        item = dict(row)
        score = item.get("overall_pct")
        diagnosis_code = str(item.get("diagnosis_code") or "").strip()
        diagnosis_label = _safe_diagnosis_text(item.get("diagnosis_label"))
        diagnosis_short = _safe_diagnosis_text(
            diagnosis_label,
            diagnosis_code if _is_valid_icd_code(diagnosis_code) else "",
        ) or "Не указан"
        diagnosis_code_public = diagnosis_code if _is_valid_icd_code(diagnosis_code) else ""
        scorer_version = str(item.get("scorer_version") or "")
        schema_version = str(item.get("score_schema_version") or "")
        specialization = sanitize_mo_org_label(
            item.get("doctor_specialty") or item.get("specialty"),
            scorer_version=scorer_version,
            schema_version=schema_version,
        )
        filial = sanitize_mo_org_label(
            item.get("doctor_filial") or item.get("filial"),
            scorer_version=scorer_version,
            schema_version=schema_version,
        )
        output.append(
            {
                "case_id": str(item.get("visit_id") or item["mis_id"]),
                "mis_id": str(item["mis_id"]),
                "visit_id": str(item.get("visit_id") or ""),
                "date": item["visit_date"],
                "doctor_fio": item.get("doctor_fio") or "",
                "specialization": specialization,
                "filial": filial,
                "document_kind": item.get("document_kind") or "unknown",
                "document_kind_label": DOCUMENT_KIND_LABELS.get(
                    str(item.get("document_kind") or "unknown"),
                    str(item.get("document_kind") or ""),
                ),
                "diagnosis_code": diagnosis_code_public,
                "diagnosis_short": diagnosis_short,
                "mkb_code_main": diagnosis_code_public,
                "mkb_code_main_source": str(item.get("mkb_code_main_source") or ""),
                "mkb_code_main_slot": str(item.get("mkb_code_main_slot") or ""),
                "icd_chapter": item.get("icd_chapter") or "",
                "history_prior_n": int(item.get("history_prior_n") or 0),
                "history_tier": str(item.get("history_tier") or ""),
                "zone1_pct": (
                    float(item["zone1_pct"])
                    if isinstance(item.get("zone1_pct"), (int, float))
                    else None
                ),
                "zone2a_pct": (
                    float(item["zone2a_pct"])
                    if isinstance(item.get("zone2a_pct"), (int, float))
                    else None
                ),
                "zone2b_pct": (
                    float(item["zone2b_pct"])
                    if isinstance(item.get("zone2b_pct"), (int, float))
                    else None
                ),
                "zone1_band": str(item.get("zone1_band") or "") or None,
                "zone2a_band": str(item.get("zone2a_band") or "") or None,
                "zone2b_band": str(item.get("zone2b_band") or "") or None,
                "zone2b_kp_status": str(item.get("zone2b_kp_status") or "") or None,
                "attention_primary": str(item.get("attention_primary") or "") or None,
                "attention_reason_ru": str(item.get("attention_reason_ru") or "") or None,
                "overall_pct": score,
                "reg55_pct": (
                    float(item["reg55_pct"])
                    if isinstance(item.get("reg55_pct"), (int, float))
                    else None
                ),
                "score_reason": (
                    None
                    if isinstance(score, (int, float))
                    else (
                        "Не оценивается: не клинический приём (процедура / диагностика / профосмотр / стоматология)"
                        if str(item.get("document_kind") or "")
                        not in {"clinical_visit", "consultation"}
                        else "Оценка ещё не рассчитана"
                    )
                ),
                "status": item.get("status") or "",
                "score_band": (
                    "90-100" if isinstance(score, (int, float)) and score >= 90
                    else "75-90" if isinstance(score, (int, float)) and score >= 75
                    else "0-75" if isinstance(score, (int, float))
                    else "unscored"
                ),
                "p0": int(item.get("p0") or 0),
                "p1": int(item.get("p1") or 0),
                "p2": int(item.get("p2") or 0),
                "p3": int(item.get("p3") or 0),
                "finding_codes": [
                    code for code in str(item.get("finding_codes") or "").split(",") if code
                ],
                "document_url": f"/api/methodist/mo/cases/{item.get('visit_id') or item['mis_id']}/document",
                "pdf_url": f"/api/methodist/mo/cases/{item.get('visit_id') or item['mis_id']}/pdf",
                "parse_ok": "1",
                "date_mismatch": "0",
                "_source": "warehouse",
            }
        )
    return output


def _records(params: dict[str, Any]) -> list[dict[str, Any]]:
    return _warehouse_records(params) if _backend_source() == "warehouse" else _jsonl_records(params)


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
    finding_codes = set(_values(params.get("finding_codes")))
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
        record_findings = {
            str(value)
            for value in (rec.get("finding_codes") or [])
            if str(value)
        }
        if not record_findings:
            record_findings = {
                str(item.get("code") or item.get("finding_code") or "")
                for item in (rec.get("_findings") or [])
                if isinstance(item, dict)
            }
        if finding_codes and not (finding_codes & record_findings):
            continue
        if str(params.get("queue_only") or "").lower() in {"1", "true", "yes"}:
            if not _needs_review(rec):
                continue
        zone = str(params.get("zone") or "").strip().lower()
        zone_band = str(params.get("zone_band") or "").strip().lower()
        if zone in {"zone1", "documentation"}:
            if zone_band and str(rec.get("zone1_band") or "").lower() != zone_band:
                continue
        elif zone in {"zone2a", "diagnosis"}:
            if zone_band and str(rec.get("zone2a_band") or "").lower() != zone_band:
                continue
        elif zone in {"zone2b", "plan"}:
            if zone_band and str(rec.get("zone2b_band") or "").lower() != zone_band:
                continue
        elif zone in {"safety"}:
            if str(rec.get("attention_primary") or "") != "safety":
                continue
        elif zone_band:
            # band без zone - любое из трёх
            bands = {
                str(rec.get("zone1_band") or "").lower(),
                str(rec.get("zone2a_band") or "").lower(),
                str(rec.get("zone2b_band") or "").lower(),
            }
            if zone_band not in bands:
                continue
        if str(params.get("attention_only") or "").lower() in {"1", "true", "yes"}:
            primary = str(rec.get("attention_primary") or "none")
            if primary in {"", "none"}:
                continue
        kp_status = str(params.get("kp_status") or "").strip().lower()
        if kp_status and str(rec.get("zone2b_kp_status") or "").lower() != kp_status:
            continue
        history_tier = str(params.get("history_tier") or "").strip()
        if history_tier and str(rec.get("history_tier") or "") != history_tier:
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
            z_scored = [r for r in rows if r.get("zone1_band") or r.get("zone2a_band") or r.get("zone2b_band")]
            n_z = len(z_scored)
            if n_z:
                z1 = sum(1 for r in z_scored if str(r.get("zone1_band") or "") == "bad")
                z2a = sum(1 for r in z_scored if str(r.get("zone2a_band") or "") == "bad")
                z2b = sum(1 for r in z_scored if str(r.get("zone2b_band") or "") == "bad")
                item["zone1_bad_pct"] = round(100.0 * z1 / n_z, 1)
                item["zone2a_bad_pct"] = round(100.0 * z2a / n_z, 1)
                item["zone2b_bad_pct"] = round(100.0 * z2b / n_z, 1)
                item["attention_n"] = sum(
                    1
                    for r in z_scored
                    if str(r.get("attention_primary") or "none") not in {"", "none"}
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


def _apply_score_eligible_default(params: dict[str, Any]) -> dict[str, Any]:
    """Таблица случаев: clinical_visit + legacy consultation (жёстко).

    Процедуры / профосмотры / диагностика / стоматология не попадают в строки.
    Opt-out через score_eligible_only=0 и чужие document_kinds игнорируются.
    """
    out = dict(params or {})
    requested = [str(v).strip() for v in _values(out.get("document_kinds")) if str(v).strip()]
    allowed = [v for v in requested if v in {"clinical_visit", "consultation"}]
    # В фильтре API канон - clinical_visit; consultation подтягивается SQL scored_kind.
    out["document_kinds"] = "clinical_visit|consultation" if not allowed else "|".join(dict.fromkeys(allowed))
    out["score_eligible_only"] = "1"
    return out


def is_case_score_eligible(
    record: Mapping[str, Any] | None = None,
    *,
    document_kind: str | None = None,
    document_kinds: list[str] | None = None,
) -> bool:
    """Гейт оценки: clinical_visit или legacy consultation.

    Можно передать несколько kind (document + record): достаточно одного eligible.
    """
    try:
        from .mo_daily import is_scored_document_kind
    except Exception:  # noqa: BLE001
        def is_scored_document_kind(kind: str | None) -> bool:  # type: ignore[misc]
            return str(kind or "").strip() in {"clinical_visit", "consultation"}

    kinds: list[str] = []
    if document_kind is not None:
        kinds.append(str(document_kind or "").strip())
    if document_kinds:
        kinds.extend(str(k or "").strip() for k in document_kinds)
    if not kinds:
        kinds.append(str((record or {}).get("document_kind") or "").strip())
    return any(is_scored_document_kind(k) for k in kinds if k)


def build_cases(params: dict[str, Any]) -> dict[str, Any]:
    params = _apply_score_eligible_default(params)
    all_records = _records(params)
    filtered = _filter_records(all_records, params)
    states = _crm_states([r["case_id"] for r in filtered])
    crm_statuses = set(_values(params.get("crm_statuses")))
    assignees = set(_values(params.get("assignees")))
    include_patient_id = str(params.get("include_patient_id") or "").lower() in {
        "1",
        "true",
        "yes",
    }
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
        "reg55": "reg55_pct",
        "priority": "p0",
        "updated_at": "updated_at",
        "doctor": "doctor_fio",
        "specialty": "specialty",
        "filial": "filial",
        "status": "status",
        "visit_id": "visit_id",
        "patient_id": "patient_id",
        "zone1": "zone1_pct",
        "zone2a": "zone2a_pct",
        "zone2b": "zone2b_pct",
        "attention": "attention_primary",
    }
    sort_field = sort_map.get(str(params.get("sort_by") or ""), "date")
    reverse = str(params.get("sort_dir") or "desc").lower() == "desc"
    filtered.sort(key=lambda r: (r.get(sort_field) is None, r.get(sort_field) or ""), reverse=reverse)
    page = max(1, int(params.get("page") or 1))
    # Дневной срез: по умолчанию больше строк, чтобы таблица дня была «полной».
    default_page = 100 if (
        str(params.get("date_from") or "")
        and str(params.get("date_from") or "") == str(params.get("date_to") or "")
    ) else 50
    page_size = max(1, min(500, int(params.get("page_size") or default_page)))
    start = (page - 1) * page_size
    rows = []
    for rec in filtered[start : start + page_size]:
        crm = states.get(rec["case_id"]) or {"status": "new", "tags": [], "finding_decisions": {}}
        public = _public_row(rec)
        if include_patient_id:
            public["patient_id"] = str(rec.get("patient_id") or "")
            public["visit_id"] = str(rec.get("visit_id") or rec.get("case_id") or "")
        try:
            from .mo_icd_visit_status import chip_label_ru, chip_title_ru, status_from_finding_codes

            icd_status = status_from_finding_codes(rec.get("finding_codes"))
            public["icd_visit_status"] = icd_status
            public["icd_visit_status_label_ru"] = chip_label_ru(icd_status)
            public["icd_visit_status_title_ru"] = chip_title_ru(icd_status)
        except Exception:  # noqa: BLE001
            pass
        rows.append({**public, "crm": crm})
    if include_patient_id:
        try:
            from .mo_review_pack import enrich_rows_with_patient_id

            enrich_rows_with_patient_id(rows)
        except Exception:  # noqa: BLE001
            pass
    agg = _filtered_agg(filtered)
    agg["by_specialty"] = [_suppressed_group(r) for r in agg.get("by_specialty") or []]
    agg["by_chapter"] = [_suppressed_group(r) for r in agg.get("by_chapter") or []]
    return {
        "ok": True,
        "namespace": "mo",
        "source": _backend_source(),
        "total": len(filtered),
        "page": page,
        "page_size": page_size,
        "rows": rows,
        "aggregate": agg,
        "suppression_n": SUPPRESSION_N,
        "applied_filters": {
            k: v
            for k, v in params.items()
            if k != "include_patient_id" and v not in (None, "", [], False)
        },
        "empty_state": _describe_empty_state(
            total_records=len(all_records),
            filtered_records=len(filtered),
            params=params,
        ),
    }


def build_facets(params: dict[str, Any]) -> dict[str, Any]:
    all_records = _records(params)
    filtered = _filter_records(all_records, params)
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
    # Типы документов - по всему срезу периода (без фильтра document_kinds),
    # чтобы при default «только clinical_visit» в меню остались процедуры/профосмотры.
    params_without_kinds = {
        key: value for key, value in params.items() if key not in {"document_kinds", "score_eligible_only"}
    }
    kind_base = _filter_records(all_records, params_without_kinds)
    kind_counts = Counter(str(r.get("document_kind") or "unknown") for r in kind_base)
    facets["document_kinds"] = [
        {
            "value": key,
            "label": DOCUMENT_KIND_LABELS.get(key, key),
            "n": count if count >= SUPPRESSION_N else None,
            "n_bucket": None if count >= SUPPRESSION_N else f"<{SUPPRESSION_N}",
            "suppressed": count < SUPPRESSION_N,
            "score_eligible": key in {"clinical_visit", "consultation"},
        }
        for key, count in kind_counts.most_common()
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
    return {
        "ok": True,
        "source": _backend_source(),
        "facets": facets,
        "n_filtered": len(filtered),
        "suppression_n": SUPPRESSION_N,
        "default_document_kinds": ["clinical_visit"],
    }


def _overview_attention_from_warehouse(params: dict[str, Any]) -> dict[str, Any] | None:
    """Агрегаты зон для overview (без UI-флага; клиент может игнорировать)."""
    date_from = str(params.get("date_from") or "")[:10]
    date_to = str(params.get("date_to") or "")[:10]
    if not date_from or not date_to:
        return None
    try:
        with closing(_read_connection()) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(fact_mo_case)")}
            if "zone1_band" not in cols:
                return None
            row = conn.execute(
                """SELECT
                     COUNT(*) AS n_evaluated,
                     SUM(CASE WHEN zone1_band='bad' THEN 1 ELSE 0 END) AS zone1_bad,
                     SUM(CASE WHEN zone2a_band='bad' THEN 1 ELSE 0 END) AS zone2a_bad,
                     SUM(CASE WHEN zone2b_band='bad' THEN 1 ELSE 0 END) AS zone2b_bad,
                     SUM(CASE WHEN attention_primary='safety' THEN 1 ELSE 0 END) AS safety_critical,
                     AVG(zone1_pct) AS zone1_avg,
                     AVG(zone2a_pct) AS zone2a_avg,
                     AVG(zone2b_pct) AS zone2b_avg
                   FROM fact_mo_case
                   WHERE visit_date BETWEEN ? AND ?
                     AND document_kind IN ('clinical_visit', 'consultation')
                     AND layer_engine IS NOT NULL""",
                (date_from, date_to),
            ).fetchone()
            trend = conn.execute(
                """SELECT visit_date AS date,
                          AVG(zone1_pct) AS zone1_avg,
                          AVG(zone2a_pct) AS zone2a_avg,
                          AVG(zone2b_pct) AS zone2b_avg,
                          SUM(CASE WHEN attention_primary='safety' THEN 1 ELSE 0 END) AS safety_critical
                   FROM fact_mo_case
                   WHERE visit_date BETWEEN ? AND ?
                     AND document_kind IN ('clinical_visit', 'consultation')
                     AND layer_engine IS NOT NULL
                   GROUP BY visit_date
                   ORDER BY visit_date""",
                (date_from, date_to),
            ).fetchall()
    except Exception:  # noqa: BLE001
        return None
    if not row:
        return None
    n = int(row["n_evaluated"] or 0)
    if n <= 0:
        return {
            "n_evaluated": 0,
            "zone1_bad": 0,
            "zone1_bad_pct": 0.0,
            "zone2a_bad": 0,
            "zone2a_bad_pct": 0.0,
            "zone2b_bad": 0,
            "zone2b_bad_pct": 0.0,
            "safety_critical": 0,
            "queue_critical": None,
            "queue_important": None,
            "zone_avgs": {"zone1": None, "zone2a": None, "zone2b": None},
            "zone_trends": [],
        }

    def _pct(bad: int) -> float:
        return round(100.0 * bad / n, 1)

    z1_bad = int(row["zone1_bad"] or 0)
    z2a_bad = int(row["zone2a_bad"] or 0)
    z2b_bad = int(row["zone2b_bad"] or 0)
    return {
        "n_evaluated": n,
        "zone1_bad": z1_bad,
        "zone1_bad_pct": _pct(z1_bad),
        "zone2a_bad": z2a_bad,
        "zone2a_bad_pct": _pct(z2a_bad),
        "zone2b_bad": z2b_bad,
        "zone2b_bad_pct": _pct(z2b_bad),
        "safety_critical": int(row["safety_critical"] or 0),
        "queue_critical": None,
        "queue_important": None,
        "zone_avgs": {
            "zone1": round(float(row["zone1_avg"]), 1) if row["zone1_avg"] is not None else None,
            "zone2a": round(float(row["zone2a_avg"]), 1) if row["zone2a_avg"] is not None else None,
            "zone2b": round(float(row["zone2b_avg"]), 1) if row["zone2b_avg"] is not None else None,
        },
        "zone_trends": [
            {
                "date": str(t["date"]),
                "zone1_avg": round(float(t["zone1_avg"]), 1) if t["zone1_avg"] is not None else None,
                "zone2a_avg": round(float(t["zone2a_avg"]), 1) if t["zone2a_avg"] is not None else None,
                "zone2b_avg": round(float(t["zone2b_avg"]), 1) if t["zone2b_avg"] is not None else None,
                "safety_critical": int(t["safety_critical"] or 0),
            }
            for t in trend
        ],
    }


def build_overview(params: dict[str, Any]) -> dict[str, Any]:
    all_records = _records(params)
    filtered = _filter_records(all_records, params)
    states = _crm_states([r["case_id"] for r in filtered])
    agg = _filtered_agg(filtered)
    kinds = Counter(r.get("document_kind") or "unknown" for r in filtered)
    eligible = sum(kinds.get(k, 0) for k in ("clinical_visit", "consultation"))
    small_slice = len(filtered) < SUPPRESSION_N
    attention = None if small_slice else _overview_attention_from_warehouse(params)
    return {
        "ok": True,
        "namespace": "mo",
        "source": _backend_source(),
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
        "attention": attention,
        "zone_trends": (attention or {}).get("zone_trends") if attention else None,
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
        "data_freshness": build_freshness(params),
        "empty_state": _describe_empty_state(
            total_records=len(all_records),
            filtered_records=len(filtered),
            params=params,
        ),
    }


def _pipeline_report_for_date(chosen: date) -> dict[str, Any] | None:
    for root in _medical_exam_roots():
        path = root / "reports" / f"{chosen:%Y}" / f"{chosen:%m}" / f"{chosen:%d}" / "report.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _unavailable(reason: str, **values: Any) -> dict[str, Any]:
    return {"available": False, "reason": reason, **values}


_DAILY_AXES = {
    "documentation": ("Оформление", "avg_documentation"),
    "clinical_concordance": ("Клиническая согласованность", "avg_clinical_concordance"),
    "safety": ("Безопасность", "avg_safety"),
    "regulatory": ("Регуляторика", "avg_regulatory"),
}


def _daily_warehouse_contract(chosen: date, stored: dict[str, Any] | None) -> dict[str, Any]:
    """Bounded SQL contract for the authenticated detailed day report."""
    day = chosen.isoformat()
    previous_day = (chosen - timedelta(days=1)).isoformat()
    history_start = (chosen - timedelta(days=56)).isoformat()
    quality = (stored or {}).get("quality") or {}
    quality_metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
    stored_summary = (stored or {}).get("summary") or {}
    completeness = (stored or {}).get("completeness") or {}

    with closing(_read_connection()) as conn:
        daily_rows = {
            str(row["visit_date"]): dict(row)
            for row in conn.execute(
                """SELECT * FROM fact_mo_daily
                   WHERE visit_date BETWEEN ? AND ? ORDER BY visit_date""",
                (history_start, day),
            ).fetchall()
        }
        current = daily_rows.get(day)
        warehouse_rows = int(
            conn.execute(
                "SELECT COUNT(*) FROM fact_mo_case WHERE visit_date = ?", (day,)
            ).fetchone()[0]
        )
        kind_rows = conn.execute(
            """SELECT c.document_kind, COALESCE(k.label, c.document_kind) AS label,
                      COUNT(*) AS source,
                      SUM(c.document_kind IN ('clinical_visit', 'consultation')) AS eligible,
                      SUM(c.document_kind IN ('clinical_visit', 'consultation')
                          AND c.overall_pct IS NOT NULL) AS evaluated
               FROM fact_mo_case c
               LEFT JOIN dim_document_kind k ON k.document_kind=c.document_kind
               WHERE c.visit_date=?
               GROUP BY c.document_kind, label
               ORDER BY source DESC LIMIT 20""",
            (day,),
        ).fetchall()
        finding_rows = conn.execute(
            """SELECT f.finding_code, f.severity, COUNT(DISTINCT f.mis_id) AS cases,
                      COALESCE(
                        NULLIF(MAX(f.title_ru), ''),
                        NULLIF(MAX(df.title_ru), ''),
                        f.finding_code
                      ) AS title_ru
               FROM fact_mo_finding f
               JOIN fact_mo_case c ON c.mis_id=f.mis_id
               LEFT JOIN dim_finding df ON df.finding_code=f.finding_code
               WHERE c.visit_date=?
                 AND c.document_kind IN ('clinical_visit', 'consultation')
                 AND f.severity IN ('P0','P1','P2','P3')
                 AND COALESCE(f.passed, 0) = 0
               GROUP BY f.finding_code, f.severity
               ORDER BY CASE f.severity WHEN 'P0' THEN 0 WHEN 'P1' THEN 1
                         WHEN 'P2' THEN 2 ELSE 3 END, cases DESC
               LIMIT 30""",
            (day,),
        ).fetchall()
        sample_rows = conn.execute(
            """SELECT f.finding_code, f.severity,
                      COALESCE(NULLIF(c.visit_id,''), c.mis_id) AS case_id,
                      COALESCE(NULLIF(d.doctor_fio,''), 'Врач не указан') AS doctor,
                      COALESCE(c.specialty, '') AS specialty
               FROM fact_mo_finding f
               JOIN fact_mo_case c ON c.mis_id=f.mis_id
               LEFT JOIN dim_doctor d ON d.doctor_key=c.doctor_key
               WHERE c.visit_date=?
                 AND c.document_kind IN ('clinical_visit', 'consultation')
                 AND f.severity IN ('P0','P1','P2','P3')
                 AND COALESCE(f.passed, 0) = 0
               ORDER BY CASE f.severity WHEN 'P0' THEN 0 WHEN 'P1' THEN 1
                         WHEN 'P2' THEN 2 ELSE 3 END, c.mis_id
               LIMIT 400""",
            (day,),
        ).fetchall()
        from .mo_action_queue_select import sql_finding_code_in_clause

        detail_select = (
            "COALESCE(NULLIF(f.detail_ru,''), '') AS detail_ru"
            if _warehouse_has_column(str(_db_path()), "fact_mo_finding", "detail_ru")
            else "'' AS detail_ru"
        )
        _zone_select = (
            "c.zone1_band, c.zone2a_band, c.zone2b_band, c.zone2b_kp_status, "
            "c.attention_primary, c.attention_reason_ru,"
            if _warehouse_has_column(str(_db_path()), "fact_mo_case", "zone1_band")
            else "NULL AS zone1_band, NULL AS zone2a_band, NULL AS zone2b_band, "
            "NULL AS zone2b_kp_status, NULL AS attention_primary, NULL AS attention_reason_ru,"
        )
        action_raw = conn.execute(
            """SELECT c.mis_id, c.visit_id, c.visit_date, c.filial, c.specialty, c.diagnosis_code,
                      c.overall_pct, c.document_kind,
                      c.scorer_version, c.score_schema_version,
                      """
            + _zone_select
            + """
                      COALESCE(NULLIF(dx.diagnosis_label,''), c.diagnosis_code) AS diagnosis,
                      COALESCE(NULLIF(d.doctor_fio,''), 'Врач не указан') AS doctor,
                      COALESCE(NULLIF(d.specialty,''), c.specialty, '') AS doctor_specialty,
                      COALESCE(NULLIF(d.filial,''), c.filial, '') AS doctor_filial,
                      f.finding_code, f.severity,
                      """
            + _finding_shadow_select("f")
            + """,
                      COALESCE(NULLIF(f.title_ru,''), NULLIF(df.title_ru,''), f.finding_code) AS finding_title,
                      COALESCE(NULLIF(f.evidence,''), '') AS evidence,
                      """
            + detail_select
            + """,
                      COALESCE(s.status, 'new') AS crm_status,
                      ax_doc.score AS axis_documentation,
                      ax_clin.score AS axis_clinical_concordance,
                      ax_safe.score AS axis_safety,
                      ax_reg.score AS axis_regulatory
               FROM fact_mo_case c
               JOIN fact_mo_finding f ON f.mis_id=c.mis_id
               LEFT JOIN dim_doctor d ON d.doctor_key=c.doctor_key
               LEFT JOIN dim_diagnosis dx ON dx.diagnosis_code=c.diagnosis_code
               LEFT JOIN dim_finding df ON df.finding_code=f.finding_code
               LEFT JOIN crm_case_state s
                 ON s.case_id=COALESCE(NULLIF(c.visit_id,''), c.mis_id)
               LEFT JOIN fact_mo_score_axis ax_doc
                 ON ax_doc.mis_id=c.mis_id AND ax_doc.axis='documentation'
               LEFT JOIN fact_mo_score_axis ax_clin
                 ON ax_clin.mis_id=c.mis_id AND ax_clin.axis='clinical_concordance'
               LEFT JOIN fact_mo_score_axis ax_safe
                 ON ax_safe.mis_id=c.mis_id AND ax_safe.axis='safety'
               LEFT JOIN fact_mo_score_axis ax_reg
                 ON ax_reg.mis_id=c.mis_id AND ax_reg.axis='regulatory'
               WHERE c.visit_date=?
                 AND c.document_kind IN ('clinical_visit', 'consultation')
                 AND """
            + sql_finding_code_in_clause("f")
            + """
                 AND COALESCE(f.passed, 0) = 0
               ORDER BY c.overall_pct ASC, c.mis_id
               LIMIT 800""",
            (day,),
        ).fetchall()
        flow_rows: dict[str, list[sqlite3.Row]] = {}
        for dimension, column in (
            ("specialty", "specialty"),
            ("branch", "filial"),
            ("document_kind", "document_kind"),
        ):
            flow_rows[dimension] = conn.execute(
                f"""SELECT {column} AS key, visit_date, COUNT(*) AS n
                    FROM fact_mo_case
                    WHERE visit_date IN (?, ?)
                    GROUP BY {column}, visit_date
                    ORDER BY n DESC LIMIT 60""",
                (day, previous_day),
            ).fetchall()

    weekday_history = [
        row
        for row in daily_rows.values()
        if row["visit_date"] < day
        and date.fromisoformat(str(row["visit_date"])).weekday() == chosen.weekday()
    ][-8:]
    baseline_counts = [
        int(row["source_rows"]) for row in weekday_history if row.get("source_rows") is not None
    ]
    expected = statistics.median(baseline_counts) if baseline_counts else None
    actual = stored_summary.get("source_rows")
    if actual is None and current:
        actual = current.get("source_rows")
    actual = int(actual) if actual is not None else warehouse_rows
    lag_days = (datetime.now(ZoneInfo("Europe/Minsk")).date() - chosen).days
    flags = [
        {
            "code": str(item.get("code") or ""),
            "level": str(item.get("severity") or level),
            "message": str(item.get("message") or ""),
        }
        for level in ("blocking", "warnings")
        for item in (quality.get(level) or [])
        if isinstance(item, dict)
    ]
    expected_payload = (
        {
            "available": True,
            "value": expected,
            "samples": len(baseline_counts),
            "method": "Медиана того же дня недели за предыдущие 8 недель",
        }
        if expected is not None
        else _unavailable("Нет истории того же дня недели", value=None, samples=0)
    )
    data_completeness = {
        "available": bool(current or stored),
        "actual_rows": actual,
        "warehouse_rows": warehouse_rows,
        "expected_rows": expected_payload,
        "actual_vs_expected_pct": (
            round(100 * actual / expected, 1) if expected not in (None, 0) else None
        ),
        "lag_days": lag_days,
        "flags": flags,
        "revision": (stored or {}).get("revision") if stored else (current or {}).get("revision"),
        "partial": bool((stored or {}).get("partial") or (current or {}).get("partial")),
        "partial_reasons": list(
            (completeness.get("reasons") if isinstance(completeness, Mapping) else None)
            or (stored or {}).get("partial_reasons")
            or (current or {}).get("partial_reasons")
            or []
        ),
        "advisory_reasons": list(
            (completeness.get("advisory_reasons") if isinstance(completeness, Mapping) else None)
            or (stored or {}).get("advisory_reasons")
            or (current or {}).get("advisory_reasons")
            or []
        ),
        "llm_queue_pending": (
            completeness.get("llm_queue_pending")
            if isinstance(completeness, Mapping) and completeness.get("llm_queue_pending") is not None
            else (stored or {}).get("llm_queue_pending")
            if stored
            else (current or {}).get("llm_queue_pending")
        ),
        "quality_status": (
            "blocked" if stored and not quality.get("passed") else (current or {}).get("quality_status")
        ),
        "coverage_pct": completeness.get("coverage_pct")
        if completeness.get("coverage_pct") is not None
        else (current or {}).get("coverage_pct"),
    }
    if not data_completeness["available"]:
        data_completeness["reason"] = "Нет отчёта и дневного агрегата витрины"

    document_kinds = []
    for row in kind_rows:
        n = int(row["source"])
        item = {
            "key": str(row["document_kind"]),
            "label": str(row["label"]),
            "source": n,
            "eligible": int(row["eligible"] or 0),
            "evaluated": int(row["evaluated"] or 0),
            "excluded": n - int(row["eligible"] or 0),
        }
        document_kinds.append(
            suppress_values(
                item,
                n=n,
                threshold=SUPPRESSION_N,
                protected={"key", "label"},
            )
        )
    eligible = int(stored_summary.get("eligible_rows") or (current or {}).get("eligible_rows") or 0)
    evaluated = int(stored_summary.get("scored") or (current or {}).get("scored_rows") or 0)
    funnel = {
        "available": bool(current or warehouse_rows or stored),
        "source": actual,
        "eligible": eligible,
        "evaluated": evaluated,
        "excluded": max(0, actual - eligible),
        "evaluation_errors": stored_summary.get("scoring_errors"),
        "document_kinds": document_kinds,
    }
    if not funnel["available"]:
        funnel["reason"] = "Нет строк за выбранный день"

    previous = daily_rows.get(previous_day) or {}
    indices = []
    stored_axes = (stored or {}).get("axes") or {}
    for key, (label, column) in _DAILY_AXES.items():
        value = stored_axes.get(key)
        if value is None and current:
            value = current.get(column)
        weekday_values = [
            float(row[column])
            for row in weekday_history
            if row.get(column) is not None
        ]
        weekday_mean = round(statistics.fmean(weekday_values), 2) if weekday_values else None
        previous_value = previous.get(column)
        available = value is not None
        indices.append(
            {
                "key": key,
                "label": label,
                "available": available,
                "reason": None if available else "Ось не записана в отчёте или витрине",
                "value": value,
                "previous_day": previous_value,
                "delta_previous_day": round(float(value) - float(previous_value), 2)
                if value is not None and previous_value is not None
                else None,
                "weekday_mean_8w": weekday_mean,
                "weekday_samples": len(weekday_values),
                "delta_weekday_mean": round(float(value) - weekday_mean, 2)
                if value is not None and weekday_mean is not None
                else None,
            }
        )

    from .mo_action_queue_select import (
        BAND_LABEL_RU,
        BAND_TO_INTERNAL,
        finding_eligible_for_action_queue,
        pick_primary_queue_finding,
        queue_reason_ru,
        signal_band_for_finding,
        strip_pn_tokens,
    )
    from .mo_finding_labels_ru import (
        demote_stale_reg55_p0,
        finding_label_ru,
        severity_tone_css,
    )

    samples_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in sample_rows:
        key = (str(row["finding_code"]), str(row["severity"]))
        bucket = samples_by_key.setdefault(key, [])
        if len(bucket) >= 5:
            continue
        case_id = str(row["case_id"] or "").strip()
        if not case_id or any(item["case_id"] == case_id for item in bucket):
            continue
        bucket.append(
            {
                "case_id": case_id,
                "doctor": str(row["doctor"] or "Врач не указан"),
                "specialty": str(row["specialty"] or ""),
            }
        )

    top_findings = []
    for row in finding_rows:
        if int(row["cases"]) < SUPPRESSION_N:
            continue
        code = str(row["finding_code"])
        severity = str(row["severity"])
        label = finding_label_ru(code, str(row["title_ru"] or ""))
        top_findings.append(
            {
                "finding_code": code,
                "label": label,
                "severity": severity,
                "cases": int(row["cases"]),
                "suppressed": False,
                "sample_cases": samples_by_key.get((code, severity), [])[:5],
            }
        )
    findings_contract = {
        "available": bool(top_findings),
        "items": top_findings,
        "suppression_n": SUPPRESSION_N,
        "day": day,
    }
    if not top_findings:
        findings_contract["reason"] = "Нет замечаний выше порога публикации"

    action_cases = []
    seen_cases: set[str] = set()
    findings_by_case: dict[str, list[dict[str, Any]]] = {}
    try:
        from .mo_review_pack import visit_identity_map_for_day

        identity_map = visit_identity_map_for_day(day)
    except Exception:  # noqa: BLE001
        identity_map = {}
    for row in action_raw:
        case_id = str(row["visit_id"] or row["mis_id"])
        is_shadow = bool(int(row["is_shadow"] or 0)) if "is_shadow" in row.keys() else False
        finding_row = {
            "finding_code": str(row["finding_code"] or ""),
            "severity": str(row["severity"] or ""),
            "finding_title": str(row["finding_title"] or ""),
            "evidence": str(row["evidence"] or "") if "evidence" in row.keys() else "",
            "detail_ru": str(row["detail_ru"] or "") if "detail_ru" in row.keys() else "",
            "is_shadow": is_shadow,
            "_case_row": row,
        }
        if not finding_eligible_for_action_queue(finding_row):
            continue
        findings_by_case.setdefault(case_id, []).append(finding_row)

    for case_id, findings in findings_by_case.items():
        primary = pick_primary_queue_finding(findings)
        if not primary:
            continue
        row = primary.get("_case_row") or findings[0].get("_case_row")
        if row is None:
            continue
        score = row["overall_pct"]
        code = str(primary.get("finding_code") or "")
        demoted = demote_stale_reg55_p0(
            code=code,
            severity=str(primary.get("severity") or ""),
            title_ru=str(primary.get("finding_title") or ""),
        )
        finding_title = strip_pn_tokens(str(demoted["title_ru"]))
        band = signal_band_for_finding(primary) or "important"
        finding_sev = BAND_TO_INTERNAL.get(band, "P1")
        is_shadow = bool(primary.get("is_shadow"))
        axes = {
            "documentation": row["axis_documentation"]
            if "axis_documentation" in row.keys()
            else None,
            "clinical_concordance": row["axis_clinical_concordance"]
            if "axis_clinical_concordance" in row.keys()
            else None,
            "safety": row["axis_safety"] if "axis_safety" in row.keys() else None,
            "regulatory": row["axis_regulatory"] if "axis_regulatory" in row.keys() else None,
        }
        display_score = float(score) if isinstance(score, (int, float)) else None
        reason = strip_pn_tokens(
            queue_reason_ru(band=band, finding_title=finding_title, finding_code=code)
        )
        scorer_version = str(row["scorer_version"] or "") if "scorer_version" in row.keys() else ""
        schema_version = (
            str(row["score_schema_version"] or "") if "score_schema_version" in row.keys() else ""
        )
        specialty_raw = (
            row["doctor_specialty"] if "doctor_specialty" in row.keys() else row["specialty"]
        )
        filial_raw = row["doctor_filial"] if "doctor_filial" in row.keys() else row["filial"]
        specialty = sanitize_mo_org_label(
            specialty_raw, scorer_version=scorer_version, schema_version=schema_version
        )
        branch = sanitize_mo_org_label(
            filial_raw, scorer_version=scorer_version, schema_version=schema_version
        )
        visit_id = str(row["visit_id"] or case_id)
        mis_id = str(row["mis_id"])
        visit_date = str(row["visit_date"] if "visit_date" in row.keys() else day)[:10]
        identity = identity_map.get(visit_id) or identity_map.get(case_id) or identity_map.get(mis_id) or {}
        patient_id = str(identity.get("patient_id") or "")
        doctor_id = str(identity.get("doctor_id") or "")
        doctor_name = str(row["doctor"] or "").strip()
        if (not doctor_name or doctor_name == "Врач не указан") and identity.get("doctor_fio"):
            doctor_name = identity["doctor_fio"]
        if not specialty and identity.get("specialty"):
            specialty = sanitize_mo_org_label(
                identity.get("specialty"),
                scorer_version=scorer_version,
                schema_version=schema_version,
            )
        if not branch and identity.get("filial"):
            branch = sanitize_mo_org_label(
                identity.get("filial"),
                scorer_version=scorer_version,
                schema_version=schema_version,
            )
        if (not doctor_name or doctor_name == "Врач не указан") and doctor_id:
            doctor_name = f"ID врача: {doctor_id}"
        seen_cases.add(case_id)
        attention_primary = (
            str(row["attention_primary"] or "")
            if "attention_primary" in row.keys()
            else ""
        )
        layer_ru = {
            "safety": "Риск",
            "zone1": "Оформление",
            "zone2a": "Диагноз",
            "zone2b": "План по протоколу",
        }.get(attention_primary, "")
        action_cases.append(
            {
                "case_id": case_id,
                "visit_id": visit_id,
                "mis_id": mis_id,
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "visit_date": visit_date,
                "severity": finding_sev,
                "severity_label_ru": BAND_LABEL_RU.get(band, "Важно"),
                "severity_tone": severity_tone_css(finding_sev),
                "queue_band": band,
                "finding_severity": finding_sev,
                "finding_severity_label_ru": BAND_LABEL_RU.get(band, "Важно"),
                "attention_primary": attention_primary or None,
                "attention_reason_ru": (
                    str(row["attention_reason_ru"] or "")
                    if "attention_reason_ru" in row.keys()
                    else ""
                )
                or None,
                "layer_ru": layer_ru or None,
                "zone1_band": (
                    str(row["zone1_band"] or "") if "zone1_band" in row.keys() else ""
                )
                or None,
                "zone2a_band": (
                    str(row["zone2a_band"] or "") if "zone2a_band" in row.keys() else ""
                )
                or None,
                "zone2b_band": (
                    str(row["zone2b_band"] or "") if "zone2b_band" in row.keys() else ""
                )
                or None,
                "zone2b_kp_status": (
                    str(row["zone2b_kp_status"] or "")
                    if "zone2b_kp_status" in row.keys()
                    else ""
                )
                or None,
                "doctor": doctor_name or "Врач не указан",
                "doctor_fio": doctor_name or "Врач не указан",
                "specialty": specialty or "Специальность не указана",
                "branch": branch or "Филиал не указан",
                "filial": branch or "Филиал не указан",
                "diagnosis": str(row["diagnosis"] or "Диагноз не указан"),
                "diagnosis_code": str(row["diagnosis_code"] or ""),
                "overall_pct": display_score,
                "overall_pct_stored": float(score) if isinstance(score, (int, float)) else None,
                "reg55_pct": (
                    float(axes["regulatory"])
                    if isinstance(axes.get("regulatory"), (int, float))
                    else None
                ),
                "finding_code": code,
                "finding_title": finding_title,
                "is_shadow": is_shadow,
                "demoted_stale_reg55_p0": False,
                "reason": reason,
                "crm_status": str(row["crm_status"]),
                "document_url": f"/api/methodist/mo/cases/{case_id}/document",
                "pdf_url": f"/api/methodist/mo/cases/{case_id}/pdf",
                "formula_note_ru": (
                    f"справка: формула {round(display_score)}%"
                    if isinstance(display_score, (int, float))
                    else ""
                ),
            }
        )
        if len(action_cases) >= 100:
            break
    # Сортировка: Критично → Важно → ниже формула.
    _rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    action_cases.sort(
        key=lambda item: (
            _rank.get(str(item.get("severity") or ""), 9),
            float(item["overall_pct"])
            if isinstance(item.get("overall_pct"), (int, float))
            else 999.0,
            str(item.get("case_id") or ""),
        )
    )
    try:
        from .mo_llm_action_judge import load_llm_action_judge_index

        judge_index = load_llm_action_judge_index(day)
    except Exception:  # noqa: BLE001
        judge_index = {}
    if judge_index:
        for item in action_cases:
            judge = judge_index.get(str(item.get("case_id") or "")) or judge_index.get(
                str(item.get("mis_id") or "")
            )
            if judge:
                item["llm_action_judge"] = judge
    action_contract = {
        "available": bool(action_cases),
        "items": action_cases,
        "limit": 100,
    }
    if not action_cases:
        action_contract["reason"] = "Случаев для разбора за день не найдено"

    doctor_rows = _doctor_breakdown(
        DateRange(chosen, chosen), max(5, SUPPRESSION_N), {}
    )
    # За один день R² case-mix почти всегда < 0.30 - это не повод прятать график.
    # Порог: enough_data (n) + delta < -10; надёжность case-mix - мягкий флаг в карточке.
    outliers = [
        {**row, "statistically_distinct": (row.get("delta_ci95") or {}).get("high") is not None
         and float(row["delta_ci95"]["high"]) < 0}
        for row in doctor_rows
        if row.get("enough_data")
        and row.get("delta") is not None
        and float(row["delta"]) < -10
    ][:50]
    doctor_contract = {
        "available": bool(outliers),
        "items": outliers,
        "sample_gate": max(5, SUPPRESSION_N),
        "rule": "Ожидаемое по специальности, дельта ниже -10 п.п., за день n не меньше 5",
        "case_mix_soft": True,
    }
    if outliers and not any(row.get("case_mix_reliable") for row in outliers):
        doctor_contract["note"] = (
            "Модель case-mix за день слабая (R² < 0.30) - ожидаемое упрощено до средней "
            "по специальности; ранжирование ориентировочное"
        )
    if not outliers:
        doctor_contract["reason"] = "Нет врачей, прошедших порог отклонения и размера выборки"

    flow_dimensions: dict[str, list[dict[str, Any]]] = {}
    for dimension, rows in flow_rows.items():
        grouped: dict[str, dict[str, int]] = {}
        for row in rows:
            key = str(row["key"] or "Не указано")
            grouped.setdefault(key, {})[str(row["visit_date"])] = int(row["n"])
        current_total = sum(values.get(day, 0) for values in grouped.values())
        previous_total = sum(values.get(previous_day, 0) for values in grouped.values())
        items = []
        for key, values in grouped.items():
            current_n = values.get(day, 0)
            previous_n = values.get(previous_day, 0)
            if current_n < SUPPRESSION_N or (previous_n and previous_n < SUPPRESSION_N):
                items.append(
                    _unavailable(
                        f"Группа меньше {SUPPRESSION_N}",
                        key=key,
                        suppressed=True,
                    )
                )
                continue
            current_share = round(100 * current_n / current_total, 2) if current_total else None
            previous_share = round(100 * previous_n / previous_total, 2) if previous_total else None
            items.append(
                {
                    "key": key,
                    "available": True,
                    "n": current_n,
                    "previous_n": previous_n,
                    "share_pct": current_share,
                    "previous_share_pct": previous_share,
                    "share_delta_pp": round(current_share - previous_share, 2)
                    if current_share is not None and previous_share is not None
                    else None,
                }
            )
        flow_dimensions[dimension] = sorted(
            items,
            key=lambda item: (item.get("n") is None, -(item.get("n") or 0)),
        )[:20]

    source_metric_specs = (
        ("parse_ok_pct", "Распознано"),
        ("doctor_fio_filled_pct", "Врач заполнен"),
        ("doctor_specialization_filled_pct", "Специальность заполнена"),
        ("filial_filled_pct", "Филиал заполнен"),
        ("mkb_code_main_filled_pct", "МКБ заполнен"),
        ("date_mismatch_pct", "Расхождение дат"),
    )
    source_items = []
    for key, label in source_metric_specs:
        if key in quality_metrics:
            source_items.append({"key": key, "label": label, "available": True, "value": quality_metrics[key]})
        else:
            source_items.append(_unavailable("Метрика отсутствует в сохранённом отчёте", key=key, label=label, value=None))
    source_quality = {
        "available": any(item["available"] for item in source_items),
        "items": source_items,
        "flags": flags,
    }
    if not source_quality["available"]:
        source_quality["reason"] = "Сохранённый отчёт не содержит метрик качества источника"

    indices_available = any(item["available"] for item in indices)
    flow_available = any(
        item.get("available")
        for items in flow_dimensions.values()
        for item in items
    )
    attention = _overview_attention_from_warehouse(
        {"date_from": day, "date_to": day}
    )
    return {
        "data_completeness": data_completeness,
        "funnel": funnel,
        "indices": {
            "available": indices_available,
            "reason": None if indices_available else "Четыре оси не записаны за выбранный день",
            "items": indices,
        },
        "attention": attention,
        "top_findings": findings_contract,
        "action_cases": action_contract,
        "doctor_outliers": doctor_contract,
        "flow_changes": {
            "available": flow_available,
            "reason": None if flow_available else "Нет публикуемых групп потока за выбранный день",
            "comparison_date": previous_day,
            "dimensions": flow_dimensions,
            "suppression_n": SUPPRESSION_N,
        },
        "source_quality": source_quality,
    }


def build_daily_report(report_date: str) -> dict[str, Any]:
    try:
        chosen = date.fromisoformat(report_date)
    except ValueError:
        return {"ok": False, "error": "invalid_date"}
    stored = _pipeline_report_for_date(chosen)
    summary = (stored or {}).get("summary") or {}
    overview = {
        "n": summary.get("source_rows"),
        "n_evaluated": summary.get("scored"),
        "avg_overall": summary.get("avg_score"),
        "n_bad": summary.get("needs_attention"),
        "severity_totals": {"P0": summary.get("critical")},
        "avg_coverage": ((stored or {}).get("month_to_date") or {}).get("avg_coverage"),
    }
    base = {
        "ok": True,
        "date": (stored or {}).get("date") or chosen.isoformat(),
        "revision": (stored or {}).get("revision"),
        "generated_at": (stored or {}).get("generated_at") or _utc(),
        "quality_status": (
            "ok" if stored and ((stored.get("quality") or {}).get("passed")) else
            "blocked" if stored else "no_data"
        ),
        "partial": bool((stored or {}).get("partial")),
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
            "axes": (stored or {}).get("axes") or {},
            "organizations": (stored or {}).get("organizations") or {},
        },
        "overview": overview,
        "comparison": (stored or {}).get("comparisons") or {},
        "month_to_date": (stored or {}).get("month_to_date") or {},
        "action_queue": (stored or {}).get("action_queue") or [],
        "data_quality": (stored or {}).get("quality") or {},
        "source": _backend_source(),
        "schema_version": SCHEMA_VERSION,
        "suppression_n": SUPPRESSION_N,
    }
    if _source_for_period(DateRange(chosen, chosen)) == "warehouse":
        expanded = _daily_warehouse_contract(chosen, stored)
        if not stored:
            funnel = expanded["funnel"]
            completeness = expanded["data_completeness"]
            axis_values = {
                item["key"]: item["value"]
                for item in expanded["indices"]["items"]
                if item.get("available")
            }
            legacy_kpi = {
                "n": funnel.get("source"),
                "eligible": funnel.get("eligible"),
                "scored": funnel.get("evaluated"),
                "avg_score": None,
                "needs_attention": None,
                "critical": sum(
                    1
                    for item in expanded["action_cases"]["items"]
                    if item.get("severity") == "P0"
                ),
            }
            base.update(
                {
                    "revision": completeness.get("revision"),
                    "quality_status": completeness.get("quality_status") or "no_data",
                    "partial": completeness.get("partial", False),
                    "executive_summary": {
                        "ok": True,
                        "kpi": legacy_kpi,
                        "axes": axis_values,
                        "organizations": {},
                    },
                    "overview": {
                        "n": funnel.get("source"),
                        "n_evaluated": funnel.get("evaluated"),
                        "avg_overall": None,
                        "n_bad": None,
                        "severity_totals": {"P0": legacy_kpi["critical"]},
                        "avg_coverage": completeness.get("coverage_pct"),
                    },
                    "action_queue": expanded["action_cases"]["items"],
                }
            )
        return {**base, **expanded}
    reason = "SQL-витрина не содержит данных за выбранный день"
    return {
        **base,
        "data_completeness": _unavailable(reason),
        "funnel": _unavailable(reason, document_kinds=[]),
        "indices": _unavailable(reason, items=[]),
        "top_findings": _unavailable(reason, items=[]),
        "action_cases": _unavailable(reason, items=[]),
        "doctor_outliers": _unavailable(reason, items=[]),
        "flow_changes": _unavailable(reason, dimensions={}),
        "source_quality": _unavailable(
            "Сохранённый отчёт недоступен", items=[]
        ),
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
    return {
        "ok": True,
        "source": _backend_source(),
        "daily": daily,
        "monthly": legacy.get("series") or [],
    }


def _resolve_request_period(params: dict[str, Any]):
    period = str(params.get("period") or "").strip().lower()
    if not period:
        period = "custom" if params.get("date_from") or params.get("date_to") else "month"
    compare = str(params.get("compare_period") or params.get("compare") or "none")
    return resolve_periods(
        period=period,
        month=str(params.get("month") or "") or None,
        compare=compare,
        date_from=str(params.get("date_from") or "") or None,
        date_to=str(params.get("date_to") or "") or None,
    )


def _sql_summary(period: DateRange, params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    where, values = _sql_case_filter(period, params)
    score_expr = (
        "COALESCE(c.overall_pct_v3,c.overall_pct)"
        if str(params.get("methodology") or "").lower() == "v3"
        else "c.overall_pct"
    )
    with closing(_read_connection()) as conn:
        row = conn.execute(
            f"""SELECT COUNT(*) AS source_records,
                      SUM(document_kind IN ('clinical_visit', 'consultation')) AS eligible,
                      SUM(document_kind IN ('clinical_visit', 'consultation')
                          AND {score_expr} IS NOT NULL) AS evaluated,
                      AVG(CASE WHEN document_kind IN ('clinical_visit', 'consultation')
                          THEN {score_expr} END) AS avg_score
               FROM fact_mo_case c
               WHERE {where}""",
            values,
        ).fetchone()
        attention = conn.execute(
            """SELECT COUNT(DISTINCT c.mis_id)
               FROM fact_mo_case c JOIN fact_mo_finding f ON f.mis_id=c.mis_id
               WHERE """ + where + """
                 AND c.document_kind IN ('clinical_visit', 'consultation')
                 AND f.passed=0 AND f.severity IN ('P0','P1')""",
            values,
        ).fetchone()[0]
        critical = conn.execute(
            """SELECT COUNT(DISTINCT c.mis_id)
               FROM fact_mo_case c JOIN fact_mo_finding f ON f.mis_id=c.mis_id
               WHERE """ + where + """
                 AND c.document_kind IN ('clinical_visit', 'consultation')
                 AND f.passed=0 AND f.severity='P0'""",
            values,
        ).fetchone()[0]
        axes = {
            str(axis): round(float(value), 2)
            for axis, value in conn.execute(
                """SELECT a.axis, AVG(a.score)
                   FROM fact_mo_score_axis a JOIN fact_mo_case c ON c.mis_id=a.mis_id
                   WHERE """ + where + """
                     AND c.document_kind IN ('clinical_visit', 'consultation')
                   GROUP BY a.axis""",
                values,
            )
            if value is not None
        }
    result = dict(row)
    evaluated = int(result.get("evaluated") or 0)
    eligible = int(result.get("eligible") or 0)
    attention = int(attention or 0)
    return {
        "source_records": int(result.get("source_records") or 0),
        "eligible": eligible,
        "evaluated": evaluated,
        "avg_score": round(float(result["avg_score"]), 2) if result.get("avg_score") is not None else None,
        "needs_attention": attention,
        "needs_attention_pct": round(100 * attention / evaluated, 2) if evaluated else None,
        "critical": int(critical or 0),
        "coverage_pct": round(100 * evaluated / eligible, 2) if eligible else None,
        "axes": axes,
    }


def _fallback_summary(period: DateRange, params: dict[str, Any] | None = None) -> dict[str, Any]:
    filters = {
        **(params or {}),
        "date_from": period.date_from.isoformat(),
        "date_to": period.date_to.isoformat(),
    }
    rows = _filter_records(_jsonl_records(filters), filters)
    eligible_rows = [
        row for row in rows if row.get("document_kind") in {"clinical_visit", "consultation"}
    ]
    scores = [
        float(row["overall_pct"])
        for row in eligible_rows
        if isinstance(row.get("overall_pct"), (int, float))
    ]
    eligible = len(eligible_rows)
    attention = sum(
        1
        for row in eligible_rows
        if int(row.get("p0") or 0) > 0 or int(row.get("p1") or 0) > 0
    )
    return {
        "source_records": len(rows),
        "eligible": eligible,
        "evaluated": len(scores),
        "avg_score": round(statistics.fmean(scores), 2) if scores else None,
        "needs_attention": attention,
        "needs_attention_pct": round(100 * attention / len(scores), 2) if scores else None,
        "critical": sum(int(row.get("p0") or 0) > 0 for row in rows),
        "coverage_pct": round(100 * len(scores) / eligible, 2) if eligible else None,
        "axes": {},
    }


def build_summary(params: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve_request_period(params)
    source = _source_for_period(resolved.current)
    aggregate = _sql_summary if source == "warehouse" else _fallback_summary
    current = aggregate(resolved.current, params)
    comparison = aggregate(resolved.comparison, params) if resolved.comparison else None
    deltas: dict[str, float | int | None] = {}
    if comparison is not None:
        for key in ("source_records", "eligible", "evaluated", "avg_score", "needs_attention_pct", "critical", "coverage_pct"):
            left, right = current.get(key), comparison.get(key)
            deltas[key] = round(float(left) - float(right), 2) if left is not None and right is not None else None
    return {
        "ok": True,
        "source": source,
        "schema_version": SCHEMA_VERSION,
        "periods": resolved.to_dict(),
        "kpi": current,
        "comparison": comparison,
        "deltas": deltas,
        "suppression_n": SUPPRESSION_N,
    }


def _shift_year(value: date, years: int) -> date:
    """Shift a date while keeping leap-day comparisons valid."""
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def _previous_month_period(period: DateRange) -> DateRange:
    previous_end = period.date_from - timedelta(days=1)
    previous_start = previous_end.replace(day=1)
    wanted_end = previous_start + timedelta(days=period.days - 1)
    return DateRange(previous_start, min(previous_end, wanted_end))


def _comparison_payload(
    current: dict[str, Any],
    period: DateRange,
    params: dict[str, Any],
) -> dict[str, Any]:
    metrics = (
        "source_records",
        "eligible",
        "evaluated",
        "avg_score",
        "needs_attention_pct",
        "critical",
        "coverage_pct",
    )

    def one(label: str, comparison_period: DateRange) -> dict[str, Any]:
        values = _sql_summary(comparison_period, params)
        if values["source_records"] == 0:
            return _unavailable(
                f"Нет данных для сравнения: {label}",
                period=comparison_period.to_dict(),
                kpi=None,
                deltas={key: None for key in metrics},
            )
        return {
            "available": True,
            "label": label,
            "period": comparison_period.to_dict(),
            "kpi": values,
            "deltas": {
                key: (
                    round(float(current[key]) - float(values[key]), 2)
                    if current.get(key) is not None and values.get(key) is not None
                    else None
                )
                for key in metrics
            },
        }

    previous = _previous_month_period(period)
    year_period = DateRange(_shift_year(period.date_from, -1), _shift_year(period.date_to, -1))
    return {
        "previous_month_equal_length": one("Равный период прошлого месяца", previous),
        "previous_year_same_period": one("Тот же период прошлого года", year_period),
    }


def build_month_report(params: dict[str, Any]) -> dict[str, Any]:
    """Build the bounded, privacy-safe Month/MTD BI contract from warehouse SQL."""
    requested = _resolve_request_period(params)
    if _source_for_period(requested.current) != "warehouse":
        raise RuntimeError("Экран месяца требует SQL-витрину МО")
    requested_end = requested.current.date_to
    with closing(_read_connection()) as conn:
        latest_raw = conn.execute(
            """SELECT MAX(visit_date) FROM fact_mo_case
               WHERE visit_date BETWEEN ? AND ?""",
            (requested.current.date_from.isoformat(), requested_end.isoformat()),
        ).fetchone()[0]
    if not latest_raw:
        period = requested.current
        return {
            "ok": True,
            "source": "warehouse",
            "schema_version": SCHEMA_VERSION,
            "timezone": "Europe/Minsk",
            "period_mode": requested.period,
            "period": period.to_dict(),
            "data_through": None,
            "days_elapsed": 0,
            "days_in_month": calendar.monthrange(period.date_from.year, period.date_from.month)[1],
            "suppression_n": SUPPRESSION_N,
            "available": False,
            "reason": "Нет данных в витрине за выбранный период",
        }

    data_through = min(date.fromisoformat(str(latest_raw)), requested_end)
    period = DateRange(requested.current.date_from, data_through)
    bounded_params = {
        **params,
        "period": "custom",
        "date_from": period.date_from.isoformat(),
        "date_to": period.date_to.isoformat(),
        "compare": "none",
        "compare_period": "none",
    }
    current = _sql_summary(period, bounded_params)
    timeseries = build_timeseries(
        {
            **bounded_params,
            "metrics": "overall,documentation,clinical_concordance,safety,regulatory,volume",
            "granularity": "day",
        }
    )
    daily = timeseries["series"]
    volumes = [int(row.get("volume") or 0) for row in daily]
    volume_median = statistics.median(volumes) if volumes else 0
    for row in daily:
        volume = int(row.get("volume") or 0)
        deviation = (
            round(100 * (volume - volume_median) / volume_median, 1)
            if volume_median
            else None
        )
        row["anomaly"] = bool(deviation is not None and abs(deviation) >= 40)
        row["anomaly_reason"] = (
            f"Объём отличается от медианы периода на {deviation:+.1f}%"
            if row["anomaly"]
            else None
        )

    days_elapsed = period.days
    days_in_month = calendar.monthrange(period.date_from.year, period.date_from.month)[1]
    is_month = requested.period == "month"
    pace = days_in_month / days_elapsed if is_month and days_elapsed else None
    forecast = {
        "available": bool(is_month and days_elapsed and current["source_records"]),
        "reason": None if is_month else "Прогноз до конца месяца доступен только для периода Месяц",
        "method": "Линейный темп по календарным дням" if is_month else None,
        "assumptions": [
            "Средний суточный объём MTD сохранится до конца месяца",
            "Прогноз не корректируется на выходные, праздники и изменения расписания",
            "Средняя оценка и покрытие считаются неизменными",
        ],
        "projected_source": round(current["source_records"] * pace) if pace else None,
        "projected_evaluated": round(current["evaluated"] * pace) if pace else None,
        "projected_findings_cases": None,
        "projected_avg_score": current["avg_score"],
    }
    comparison = _comparison_payload(current, period, bounded_params)
    heatmap = build_heatmap({**bounded_params, "rows": "specialty", "cols": "icd_chapter"})
    doctors = _doctor_breakdown(period, max(20, SUPPRESSION_N), bounded_params)

    where, values = _sql_case_filter(period, bounded_params)
    with closing(_read_connection()) as conn:
        finding_rows = conn.execute(
            """SELECT f.finding_code, f.severity, COUNT(DISTINCT f.mis_id) cases,
                      COALESCE(
                        NULLIF(MAX(f.title_ru), ''),
                        NULLIF(MAX(df.title_ru), ''),
                        f.finding_code
                      ) AS title_ru
               FROM fact_mo_finding f
               JOIN fact_mo_case c ON c.mis_id=f.mis_id
               LEFT JOIN dim_finding df ON df.finding_code=f.finding_code
               WHERE """ + where + """
                 AND c.document_kind IN ('clinical_visit', 'consultation')
                 AND COALESCE(f.passed, 0) = 0
               GROUP BY f.finding_code, f.severity
               ORDER BY cases DESC, f.finding_code
               LIMIT 200""",
            values,
        ).fetchall()
        finding_cases = int(
            conn.execute(
                """SELECT COUNT(DISTINCT c.mis_id)
                   FROM fact_mo_case c JOIN fact_mo_finding f ON f.mis_id=c.mis_id
                   WHERE """ + where + """
                     AND c.document_kind IN ('clinical_visit', 'consultation')
                     AND COALESCE(f.passed, 0) = 0""",
                values,
            ).fetchone()[0]
        )
        crm_rows = conn.execute(
            """SELECT COALESCE(s.status, 'new') status, COUNT(*) cases
               FROM fact_mo_case c
               LEFT JOIN crm_case_state s
                 ON s.case_id=COALESCE(NULLIF(c.visit_id,''), c.mis_id)
               WHERE """ + where + """
                 AND c.document_kind IN ('clinical_visit', 'consultation')
                 AND c.overall_pct IS NOT NULL
               GROUP BY COALESCE(s.status, 'new')""",
            values,
        ).fetchall()

    from .mo_finding_labels_ru import finding_label_ru

    published_findings = [
        {
            "finding_code": str(row["finding_code"]),
            "label": finding_label_ru(str(row["finding_code"]), str(row["title_ru"] or "")),
            "severity": str(row["severity"] or ""),
            "cases": int(row["cases"]),
        }
        for row in finding_rows
        if int(row["cases"]) >= SUPPRESSION_N
    ]
    finding_total = sum(item["cases"] for item in published_findings)
    running = 0
    for item in published_findings:
        running += item["cases"]
        item["share_pct"] = round(100 * item["cases"] / finding_total, 2) if finding_total else None
        item["cumulative_share_pct"] = round(100 * running / finding_total, 2) if finding_total else None
    statuses = {str(row["status"]): int(row["cases"]) for row in crm_rows}
    in_work_statuses = {
        "assigned", "in_review", "confirmed_issue", "needs_more_data", "sent_to_doctor"
    }
    closed_statuses = {"resolved", "closed", "false_positive"}
    in_work = sum(statuses.get(status, 0) for status in in_work_statuses)
    closed = sum(statuses.get(status, 0) for status in closed_statuses)

    daily_source = sum(int(row.get("volume") or 0) for row in daily)
    daily_evaluated = sum(int(row.get("evaluated") or 0) for row in daily)
    reconciliation = {
        "status": (
            "ok"
            if daily_source == current["source_records"] and daily_evaluated == current["evaluated"]
            else "diverged"
        ),
        "daily_source_sum": daily_source,
        "mtd_source": current["source_records"],
        "source_delta": daily_source - current["source_records"],
        "daily_evaluated_sum": daily_evaluated,
        "mtd_evaluated": current["evaluated"],
        "evaluated_delta": daily_evaluated - current["evaluated"],
    }
    return {
        "ok": True,
        "available": True,
        "source": "warehouse",
        "schema_version": SCHEMA_VERSION,
        "timezone": "Europe/Minsk",
        "period_mode": requested.period,
        "period": period.to_dict(),
        "period_label": "MTD" if requested.period == "month" else "Выбранный период",
        "data_through": data_through.isoformat(),
        "days_elapsed": days_elapsed,
        "days_in_month": days_in_month,
        "suppression_n": SUPPRESSION_N,
        "kpi": current,
        "forecast": forecast,
        "comparison": comparison,
        "timeseries": {
            "items": daily,
            "metrics": metric_catalog(),
            "anomaly_rule": "Отклонение объёма на 40% и более от медианы выбранного периода",
        },
        "reconciliation": reconciliation,
        "heatmap": heatmap,
        "doctor_case_mix": {
            "available": any(item.get("enough_data") and not item.get("suppressed") for item in doctors),
            "items": doctors[:100],
            "sample_gate": max(20, SUPPRESSION_N),
            "rule": "Дельта к средней оценке специальности; 95% ДИ; рейтинг только при n не меньше 20",
        },
        "pareto": {
            "available": bool(published_findings),
            "items": published_findings,
            "suppression_n": SUPPRESSION_N,
            "reason": None if published_findings else "Нет замечаний выше порога публикации",
        },
        "funnel": {
            "source": current["source_records"],
            "eligible": current["eligible"],
            "evaluated": current["evaluated"],
            "with_findings": finding_cases,
            "in_crm_work": in_work,
            "closed": closed,
        },
        "crm_progress": {
            "available": bool(statuses),
            "statuses": statuses,
            "in_work": in_work,
            "closed": closed,
        },
        "reg55": _unavailable(
            "В текущей витрине нет отдельной проверенной метрики соответствия постановлению №55"
        ),
    }


def build_timeseries(params: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve_request_period(params)
    requested = set(_values(params.get("metrics"))) or {
        "overall", "documentation", "clinical_concordance", "safety", "regulatory", "volume", "coverage", "critical"
    }
    unknown = requested - set(METRICS)
    if unknown:
        raise ValueError(f"Неизвестные metrics: {', '.join(sorted(unknown))}")
    granularity = str(params.get("granularity") or "day").lower()
    if granularity not in {"day", "week"}:
        raise ValueError("granularity должен быть day или week")
    source = _source_for_period(resolved.current)
    if source != "warehouse":
        legacy = build_trends(resolved.current.to_dict())
        return {
            "ok": True,
            "source": source,
            "schema_version": SCHEMA_VERSION,
            "periods": resolved.to_dict(),
            "granularity": "day",
            "series": legacy["daily"],
        }
    case_filters = any(
        _values(params.get(key))
        for key in ("specializations", "filials", "doctors", "document_kinds", "statuses")
    ) or str(params.get("methodology") or "").lower() == "v3"
    score_expr = (
        "COALESCE(c.overall_pct_v3,c.overall_pct)"
        if str(params.get("methodology") or "").lower() == "v3"
        else "c.overall_pct"
    )
    bucket = "c.visit_date" if granularity == "day" else "strftime('%Y-W%W', c.visit_date)"
    with closing(_read_connection()) as conn:
        if case_filters:
            where, values = _sql_case_filter(resolved.current, params)
            rows = [
                dict(row)
                for row in conn.execute(
                    f"""SELECT {bucket} AS date, COUNT(*) AS volume,
                               SUM(c.document_kind IN ('clinical_visit', 'consultation')
                                   AND {score_expr} IS NOT NULL) AS evaluated,
                               ROUND(AVG(CASE WHEN c.document_kind IN ('clinical_visit', 'consultation')
                                   THEN {score_expr} END), 2) AS overall,
                               ROUND(SUM(c.document_kind IN ('clinical_visit', 'consultation')
                                   AND {score_expr} IS NOT NULL) * 100.0 /
                                   NULLIF(SUM(c.document_kind IN ('clinical_visit', 'consultation')), 0), 2)
                                   AS coverage,
                               0 AS critical
                        FROM fact_mo_case c WHERE {where}
                        GROUP BY {bucket} ORDER BY date""",
                    values,
                )
            ]
            by_date = {str(row["date"]): row for row in rows}
            for row in conn.execute(
                f"""SELECT {bucket} AS date, a.axis, ROUND(AVG(a.score), 2) AS score
                    FROM fact_mo_score_axis a
                    JOIN fact_mo_case c ON c.mis_id=a.mis_id
                    WHERE {where}
                      AND c.document_kind IN ('clinical_visit', 'consultation')
                    GROUP BY {bucket}, a.axis""",
                values,
            ):
                target = by_date.get(str(row["date"]))
                if target is not None:
                    target[str(row["axis"])] = row["score"]
            for row in conn.execute(
                f"""SELECT {bucket} AS date, COUNT(DISTINCT c.mis_id) AS critical
                    FROM fact_mo_case c
                    JOIN fact_mo_finding f ON f.mis_id=c.mis_id
                    WHERE {where}
                      AND c.document_kind IN ('clinical_visit', 'consultation')
                      AND f.severity='P0'
                    GROUP BY {bucket}""",
                values,
            ):
                target = by_date.get(str(row["date"]))
                if target is not None:
                    target["critical"] = int(row["critical"])
        else:
            daily_bucket = (
                "visit_date" if granularity == "day" else "strftime('%Y-W%W', visit_date)"
            )
            rows = [
                dict(row)
                for row in conn.execute(
                    f"""SELECT {daily_bucket} AS date, SUM(source_rows) AS volume,
                               SUM(scored_rows) AS evaluated,
                               ROUND(SUM(avg_score * scored_rows) /
                                   NULLIF(SUM(scored_rows), 0), 2) AS overall,
                               ROUND(AVG(avg_documentation), 2) AS documentation,
                               ROUND(AVG(avg_clinical_concordance), 2) AS clinical_concordance,
                               ROUND(AVG(avg_safety), 2) AS safety,
                               ROUND(AVG(avg_regulatory), 2) AS regulatory,
                               ROUND(AVG(coverage_pct), 2) AS coverage,
                               SUM(critical) AS critical
                        FROM fact_mo_daily WHERE visit_date BETWEEN ? AND ?
                        GROUP BY {daily_bucket} ORDER BY date""",
                    (resolved.current.date_from.isoformat(), resolved.current.date_to.isoformat()),
                )
            ]
    series = [
        {
            key: value
            for key, value in row.items()
            if key in {"date", "evaluated"} or key in requested
        }
        for row in rows
    ]
    return {
        "ok": True,
        "source": source,
        "schema_version": SCHEMA_VERSION,
        "periods": resolved.to_dict(),
        "granularity": granularity,
        "series": series,
    }


_BREAKDOWN_DIMENSIONS = {
    "doctor": ("c.doctor_key", "d.doctor_fio"),
    "specialty": ("c.specialty", "c.specialty"),
    "branch": ("c.filial", "c.filial"),
    "document_kind": ("c.document_kind", "COALESCE(k.label, c.document_kind)"),
}


def _doctor_breakdown(
    period: DateRange,
    sample_threshold: int,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    where, values = _sql_case_filter(period, params)
    with closing(_read_connection()) as conn:
        rows = conn.execute(
            """SELECT c.doctor_key, COALESCE(d.doctor_fio, c.doctor_key) doctor,
                      c.specialty, c.icd_chapter, c.overall_pct
               FROM fact_mo_case c LEFT JOIN dim_doctor d ON d.doctor_key=c.doctor_key
               WHERE """ + where + """ AND c.doctor_key <> ''
                 AND c.document_kind IN ('clinical_visit', 'consultation')
                 AND c.overall_pct IS NOT NULL""",
            values,
        ).fetchall()
    specialty_scores: dict[str, list[float]] = {}
    case_mix_scores: dict[tuple[str, str], list[float]] = {}
    doctors: dict[tuple[str, str, str], list[tuple[float, str]]] = {}
    for row in rows:
        specialty = str(row["specialty"] or "Не указано")
        score = float(row["overall_pct"])
        specialty_scores.setdefault(specialty, []).append(score)
        chapter = str(row["icd_chapter"] or "Не указано")
        case_mix_scores.setdefault((specialty, chapter), []).append(score)
        doctors.setdefault(
            (str(row["doctor_key"]), str(row["doctor"]), specialty), []
        ).append((score, chapter))
    overall_mean = statistics.fmean(float(row["overall_pct"]) for row in rows) if rows else 0.0
    predictions = []
    actual = []
    for row in rows:
        specialty = str(row["specialty"] or "Не указано")
        chapter = str(row["icd_chapter"] or "Не указано")
        group = case_mix_scores[(specialty, chapter)]
        predicted = (
            statistics.fmean(group)
            if len(group) >= 20
            else statistics.fmean(specialty_scores[specialty])
        )
        predictions.append(predicted)
        actual.append(float(row["overall_pct"]))
    denominator = sum((value - overall_mean) ** 2 for value in actual)
    model_r_squared = (
        1.0 - sum((value - predicted) ** 2 for value, predicted in zip(actual, predictions)) / denominator
        if denominator
        else 0.0
    )
    # На коротком MTD (начало месяца) R² case-mix часто < 0.30 - график всё равно
    # нужен методисту. Порог n оставляет enough_data; valid отдельным флагом.
    model_valid = model_r_squared >= 0.30
    output = []
    for (doctor_key, doctor, specialty), observations in doctors.items():
        scores = [score for score, _chapter in observations]
        n = len(scores)
        expected = statistics.fmean(
            statistics.fmean(case_mix_scores[(specialty, chapter)])
            if len(case_mix_scores[(specialty, chapter)]) >= 20
            else statistics.fmean(specialty_scores[specialty])
            for _score, chapter in observations
        )
        interval = mean_confidence_interval(scores)
        row = {
            "key": doctor_key,
            "label": doctor,
            "specialty": specialty,
            "n": n,
            "avg_score": round(statistics.fmean(scores), 2),
            "expected_score": round(expected, 2),
            "delta": round(statistics.fmean(scores) - expected, 2),
            "ci95": {"low": interval["low"], "high": interval["high"]},
            "delta_ci95": {
                "low": round(float(interval["low"]) - expected, 2) if interval["low"] is not None else None,
                "high": round(float(interval["high"]) - expected, 2) if interval["high"] is not None else None,
            },
            "enough_data": n >= sample_threshold,
            "case_mix_reliable": model_valid,
            "case_mix_model": {
                "features": ["specialty", "icd_chapter"],
                "r_squared": round(model_r_squared, 4),
                "valid": model_valid,
                "minimum_r_squared": 0.30,
            },
        }
        output.append(
            suppress_values(
                row,
                n=n,
                threshold=SUPPRESSION_N,
                protected={"key", "label", "specialty", "enough_data"},
            )
        )
    output.sort(key=lambda item: (item.get("delta") is None, item.get("delta") or 0))
    return output


def build_breakdown(params: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve_request_period(params)
    dimension = str(params.get("dimension") or "specialty").lower()
    if dimension not in _BREAKDOWN_DIMENSIONS:
        raise ValueError(
            f"Неизвестный dimension={dimension!r}; допустимо: {', '.join(sorted(_BREAKDOWN_DIMENSIONS))}"
        )
    sample_threshold = max(SUPPRESSION_N, int(params.get("sample_threshold") or (20 if dimension == "doctor" else SUPPRESSION_N)))
    source = _source_for_period(resolved.current)
    if source != "warehouse":
        rows = _filter_records(_jsonl_records(resolved.current.to_dict()), resolved.current.to_dict())
        field = {"doctor": "doctor_fio", "specialty": "specialization", "branch": "filial", "document_kind": "document_kind"}[dimension]
        items = _organization_groups(rows, field)
    elif dimension == "doctor":
        items = _doctor_breakdown(resolved.current, sample_threshold, params)
    else:
        key_sql, label_sql = _BREAKDOWN_DIMENSIONS[dimension]
        joins = (
            "LEFT JOIN dim_doctor d ON d.doctor_key=c.doctor_key "
            "LEFT JOIN dim_document_kind k ON k.document_kind=c.document_kind"
        )
        where, values = _sql_case_filter(resolved.current, params)
        with closing(_read_connection()) as conn:
            raw = conn.execute(
                f"""SELECT {key_sql} AS key, {label_sql} AS label, COUNT(*) AS n,
                           AVG(c.overall_pct) AS avg_score,
                           SUM(EXISTS(
                             SELECT 1 FROM fact_mo_finding af
                             WHERE af.mis_id=c.mis_id AND af.passed=0
                               AND af.severity IN ('P0','P1')
                           )) AS needs_attention
                    FROM fact_mo_case c {joins}
                    WHERE {where}
                      AND c.document_kind IN ('clinical_visit', 'consultation')
                    GROUP BY {key_sql}, {label_sql} ORDER BY n DESC""",
                values,
            ).fetchall()
        items = [
            suppress_values(
                {
                    "key": str(row["key"] or ""),
                    "label": str(row["label"] or "Не указано"),
                    "avg_score": round(float(row["avg_score"]), 2) if row["avg_score"] is not None else None,
                    "needs_attention": int(row["needs_attention"] or 0),
                },
                n=int(row["n"]),
                threshold=SUPPRESSION_N,
                protected={"key", "label"},
            )
            for row in raw
        ]
    return {
        "ok": True,
        "source": source,
        "schema_version": SCHEMA_VERSION,
        "periods": resolved.to_dict(),
        "dimension": dimension,
        "sample_threshold": sample_threshold,
        "items": items,
    }


def build_heatmap(params: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve_request_period(params)
    rows = str(params.get("rows") or "specialty")
    cols = str(params.get("cols") or "icd_chapter")
    if rows != "specialty" or cols != "icd_chapter":
        raise ValueError("Доступная heatmap: rows=specialty&cols=icd_chapter")
    source = _source_for_period(resolved.current)
    cells: list[dict[str, Any]] = []
    if source == "warehouse":
        where, values = _sql_case_filter(resolved.current, params)
        with closing(_read_connection()) as conn:
            raw = conn.execute(
                f"""SELECT c.specialty AS row_key, c.icd_chapter AS col_key,
                           COUNT(*) AS n, AVG(c.overall_pct) AS avg_score
                    FROM fact_mo_case c
                    WHERE {where}
                      AND c.document_kind IN ('clinical_visit', 'consultation')
                      AND c.icd_chapter <> ''
                    GROUP BY c.specialty, c.icd_chapter
                    ORDER BY n DESC""",
                values,
            ).fetchall()
        cells = [
            {
                "row": str(item["row_key"] or "Не указано"),
                "col": str(item["col_key"]),
                "n": int(item["n"]),
                "avg_score": round(float(item["avg_score"]), 2)
                if item["avg_score"] is not None
                else None,
            }
            for item in raw
            if int(item["n"]) >= SUPPRESSION_N
        ]
    else:
        filters = {
            **params,
            "date_from": resolved.current.date_from.isoformat(),
            "date_to": resolved.current.date_to.isoformat(),
        }
        records = _filter_records(_jsonl_records(filters), filters)
        grouped: dict[tuple[str, str], list[float]] = {}
        for record in records:
            chapter = str(record.get("icd_chapter") or "")
            if not chapter:
                continue
            key = (str(record.get("specialization") or "Не указано"), chapter)
            score = record.get("overall_pct")
            if isinstance(score, (int, float)):
                grouped.setdefault(key, []).append(float(score))
        cells = [
            {
                "row": row_key,
                "col": col_key,
                "n": len(scores),
                "avg_score": round(statistics.fmean(scores), 2),
            }
            for (row_key, col_key), scores in grouped.items()
            if len(scores) >= SUPPRESSION_N
        ]
    return {
        "ok": True,
        "source": source,
        "schema_version": SCHEMA_VERSION,
        "periods": resolved.to_dict(),
        "status": "ok" if cells else "not_available",
        "reason": None
        if cells
        else "В выбранном срезе нет групп специальность × глава МКБ выше порога публикации.",
        "rows": rows,
        "cols": cols,
        "cells": cells,
    }


def build_findings(params: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve_request_period(params)
    source = _source_for_period(resolved.current)
    items: list[dict[str, Any]] = []
    if source == "warehouse":
        where, values = _sql_case_filter(resolved.current, params)
        with closing(_read_connection()) as conn:
            rows = conn.execute(
                """SELECT f.finding_code, f.severity, COUNT(DISTINCT f.mis_id) AS cases
                   FROM fact_mo_finding f JOIN fact_mo_case c ON c.mis_id=f.mis_id
                   WHERE """ + where + """
                     AND c.document_kind IN ('clinical_visit', 'consultation')
                   GROUP BY f.finding_code, f.severity ORDER BY cases DESC""",
                values,
            ).fetchall()
        items = [
            {
                "finding_code": str(row["finding_code"]),
                "label": f"Замечание: {row['finding_code']}",
                "severity": row["severity"],
                "cases": int(row["cases"]),
            }
            for row in rows
            if int(row["cases"]) >= SUPPRESSION_N
        ]
    return {
        "ok": True,
        "source": source,
        "schema_version": SCHEMA_VERSION,
        "periods": resolved.to_dict(),
        "items": items,
        "suppression_n": SUPPRESSION_N,
    }


def build_meta() -> dict[str, Any]:
    return {
        "ok": True,
        "source": _backend_source(),
        "schema_version": SCHEMA_VERSION,
        "timezone": "Europe/Minsk",
        "periods": sorted({"yesterday", "7d", "month", "custom"}),
        "compare": sorted({"previous", "weekday", "none"}),
        "dimensions": sorted(_BREAKDOWN_DIMENSIONS),
        "metrics": metric_catalog(),
        "suppression_n": SUPPRESSION_N,
    }


def build_data_quality(params: dict[str, Any]) -> dict[str, Any]:
    all_records = _records(params)
    rows = _filter_records(all_records, params)
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
        "empty_state": _describe_empty_state(
            total_records=len(all_records),
            filtered_records=n,
            params=params,
        ),
    }


def _normalize_finding_row(finding_row: Mapping[str, Any]) -> dict[str, Any]:
    """Нормализовать строку fact_mo_finding для case detail / API."""
    from .mo_finding_labels_ru import (
        demote_stale_reg55_p0,
        enrich_finding_detail_ru,
        severity_tone_css,
        source_ref_display_ru,
    )

    finding = dict(finding_row)
    code = str(finding.get("code") or finding.get("finding_code") or "")
    finding["code"] = code
    demoted = demote_stale_reg55_p0(
        code=code,
        severity=str(finding.get("severity") or ""),
        title_ru=str(finding.get("title_ru") or ""),
    )
    finding["title_ru"] = str(demoted["title_ru"])
    source_ref = str(finding.get("source_ref") or "")
    finding["source_ref"] = source_ref
    finding["source_ref_ru"] = source_ref_display_ru(source_ref)
    severity = str(demoted["severity"] or "").strip()
    finding["severity"] = severity
    finding["severity_label_ru"] = str(demoted["severity_label_ru"])
    finding["severity_hint_ru"] = str(demoted["severity_hint_ru"])
    finding["severity_tone"] = str(demoted.get("severity_tone") or severity_tone_css(severity))
    finding["demoted_stale_reg55_p0"] = bool(demoted.get("demoted_stale_reg55_p0"))
    finding["detail_ru"] = enrich_finding_detail_ru(
        code=code,
        detail=str(finding.get("detail_ru") or finding.get("detail") or ""),
        source_ref=source_ref,
        title_ru=str(finding.get("title_ru") or ""),
    )
    is_shadow = bool(int(finding.get("is_shadow") or 0))
    finding["is_shadow"] = is_shadow
    finding["shadow"] = is_shadow
    linked_raw = finding.pop("linked_fields_json", None)
    linked: list[str] = []
    if isinstance(linked_raw, str) and linked_raw.strip():
        try:
            parsed = json.loads(linked_raw)
            if isinstance(parsed, list):
                linked = [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            linked = []
    if not linked and finding.get("linked_fields"):
        linked = [str(item) for item in finding.get("linked_fields") or [] if item]
    finding["linked_fields"] = linked
    finding["link_hint_ru"] = str(finding.get("link_hint_ru") or "")
    return finding


def build_case_detail(case_id: str, month: str | None = None) -> dict[str, Any]:
    selected_month = month or _month_for_date("")
    detail: dict[str, Any] = {"ok": False}
    if _backend_source() == "warehouse":
        with closing(_read_connection()) as conn:
            row = conn.execute(
                """SELECT c.*, d.doctor_fio,
                          COALESCE(d.specialty, c.specialty) AS doctor_specialty,
                          COALESCE(d.filial, c.filial) AS doctor_filial
                   FROM fact_mo_case c
                   LEFT JOIN dim_doctor d ON d.doctor_key = c.doctor_key
                   WHERE c.visit_id = ? OR c.mis_id = ?
                   ORDER BY CASE WHEN c.document_kind IN ('clinical_visit', 'consultation') THEN 0 ELSE 1 END
                   LIMIT 1""",
                (case_id, case_id),
            ).fetchone()
            if row:
                item = dict(row)
                mis_id = str(item["mis_id"])
                axes = {
                    str(axis_row["axis"]): axis_row["score"]
                    for axis_row in conn.execute(
                        "SELECT axis, score FROM fact_mo_score_axis WHERE mis_id = ?", (mis_id,)
                    )
                }
                findings = [
                    _normalize_finding_row(finding_row)
                    for finding_row in conn.execute(
                        """SELECT f.finding_code AS code, f.finding_code, f.severity,
                                  f.evidence, f.source_ref, f.axis,
                                  """
                        + _finding_shadow_select("f")
                        + ", "
                        + _finding_link_select("f")
                        + """,
                                  COALESCE(NULLIF(f.title_ru,''), NULLIF(df.title_ru,''), f.finding_code) AS title_ru,
                                  COALESCE(NULLIF(f.detail_ru,''), df.why_important_ru, '') AS detail_ru
                           FROM fact_mo_finding f
                           LEFT JOIN dim_finding df ON df.finding_code = f.finding_code
                           WHERE f.mis_id = ? AND COALESCE(f.passed, 0) = 0
                           ORDER BY CASE f.severity WHEN 'P0' THEN 0 WHEN 'P1' THEN 1
                                    WHEN 'P2' THEN 2 ELSE 3 END""",
                        (mis_id,),
                    )
                ]
                document_kind = str(item.get("document_kind") or "unknown")
                score = item.get("overall_pct")
                # Снять ложный hard-cap 40%: если единственный «P0» был stale №55
                if any(f.get("demoted_stale_reg55_p0") for f in findings):
                    still_p0 = any(
                        str(f.get("severity") or "").upper() == "P0" for f in findings
                    )
                    if not still_p0:
                        from .mo_finding_labels_ru import recompute_overall_from_axes

                        axis_score = recompute_overall_from_axes(axes)
                        if axis_score is not None:
                            score = axis_score
                scorer_version = str(item.get("scorer_version") or "")
                schema_version = str(item.get("score_schema_version") or "")
                specialization = sanitize_mo_org_label(
                    item.get("doctor_specialty") or item.get("specialty"),
                    scorer_version=scorer_version,
                    schema_version=schema_version,
                )
                filial = sanitize_mo_org_label(
                    item.get("doctor_filial") or item.get("filial"),
                    scorer_version=scorer_version,
                    schema_version=schema_version,
                )
                record = {
                    "case_id": str(item.get("visit_id") or mis_id),
                    "mis_id": mis_id,
                    "visit_id": str(item.get("visit_id") or ""),
                    "date": item.get("visit_date"),
                    "doctor_fio": item.get("doctor_fio") or "",
                    "specialization": specialization,
                    "filial": filial,
                    "document_kind": document_kind,
                    "document_kind_label": DOCUMENT_KIND_LABELS.get(document_kind, document_kind),
                    "diagnosis_code": item.get("diagnosis_code") or "",
                    "diagnosis_short": item.get("diagnosis_code") or "",
                    "mkb_code_main": item.get("diagnosis_code") or "",
                    "icd_chapter": item.get("icd_chapter") or "",
                    "overall_pct": score,
                    "score_reason": (
                        None
                        if isinstance(score, (int, float))
                        else (
                            "Не оценивается: не клинический приём (процедура / диагностика / профосмотр / стоматология)"
                            if document_kind not in {"clinical_visit", "consultation"}
                            else "Оценка ещё не рассчитана"
                        )
                    ),
                    "status": item.get("status") or "",
                }
                if isinstance(axes.get("regulatory"), (int, float)):
                    record["reg55_pct"] = float(axes["regulatory"])
                detail = {
                    "ok": True,
                    "record": record,
                    "deep_overall_pct": score,
                    "deep_status": item.get("status") or "",
                    "axes": axes,
                    "findings": findings,
                    "source": "warehouse",
                }
    if not detail.get("ok"):
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
        with closing(_read_connection()) as conn:
            row = conn.execute(
                """SELECT c.*, d.doctor_fio,
                          COALESCE(d.specialty, c.specialty) AS doctor_specialty,
                          COALESCE(d.filial, c.filial) AS doctor_filial,
                          COALESCE(NULLIF(dx.diagnosis_label, ''), '') AS diagnosis_label
                   FROM fact_mo_case c
                   LEFT JOIN dim_doctor d ON d.doctor_key = c.doctor_key
                   LEFT JOIN dim_diagnosis dx ON dx.diagnosis_code = c.diagnosis_code
                   WHERE c.visit_id = ? OR c.mis_id = ?
                   ORDER BY CASE
                            WHEN c.document_kind IN ('clinical_visit', 'consultation') THEN 0
                            WHEN c.document_kind='procedure_session' THEN 1
                            WHEN c.document_kind='medical_exam' THEN 2
                            ELSE 3
                          END
                   LIMIT 1""",
                (case_id, case_id),
            ).fetchone()
            if row:
                item = dict(row)
                findings = [
                    _normalize_finding_row(finding_row)
                    for finding_row in conn.execute(
                        """SELECT f.finding_code AS code, f.severity, f.source_ref, f.axis,
                                  """
                        + _finding_shadow_select("f")
                        + ", "
                        + _finding_link_select("f")
                        + """,
                                  COALESCE(NULLIF(f.title_ru,''), NULLIF(df.title_ru,''), f.finding_code) AS title_ru,
                                  COALESCE(f.detail_ru, f.evidence, '') AS detail_ru,
                                  f.evidence
                           FROM fact_mo_finding f
                           LEFT JOIN dim_finding df ON df.finding_code = f.finding_code
                           WHERE f.mis_id = ?
                             AND COALESCE(f.passed, 0) = 0
                           ORDER BY CASE f.severity WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 ELSE 3 END""",
                        (item.get("mis_id"),),
                    )
                ]
                axis_rows = conn.execute(
                    "SELECT axis, score FROM fact_mo_score_axis WHERE mis_id = ?",
                    (item.get("mis_id"),),
                ).fetchall()
                axes = {str(r["axis"]): r["score"] for r in axis_rows}
                diagnosis_code = str(item.get("diagnosis_code") or "").strip()
                diagnosis_label = _safe_diagnosis_text(item.get("diagnosis_label"))
                diagnosis_short = _safe_diagnosis_text(
                    diagnosis_label,
                    diagnosis_code if _is_valid_icd_code(diagnosis_code) else "",
                ) or "Не указан"
                score = item.get("overall_pct")
                scorer_version = str(item.get("scorer_version") or "")
                schema_version = str(item.get("score_schema_version") or "")
                specialization = sanitize_mo_org_label(
                    item.get("doctor_specialty") or item.get("specialty"),
                    scorer_version=scorer_version,
                    schema_version=schema_version,
                )
                filial = sanitize_mo_org_label(
                    item.get("doctor_filial") or item.get("filial"),
                    scorer_version=scorer_version,
                    schema_version=schema_version,
                )
                detail = {
                    "ok": True,
                    "record": {
                        "case_id": str(item.get("visit_id") or item.get("mis_id") or case_id),
                        "visit_id": str(item.get("visit_id") or ""),
                        "mis_id": str(item.get("mis_id") or ""),
                        "date": str(item.get("visit_date") or ""),
                        "doctor_fio": item.get("doctor_fio") or "",
                        "specialization": specialization,
                        "filial": filial,
                        "document_kind": item.get("document_kind") or "unknown",
                        "document_kind_label": {
                            "clinical_visit": "Клинический приём",
                            "procedure_session": "Манипуляция / процедура",
                            "medical_exam": "Профосмотр / медосмотр",
                            "consultation": "Клинический приём (legacy)",
                            "certificate": "Справка",
                            "diagnostic": "Диагностическое исследование",
                            "non_clinical": "Неклинический документ",
                            "empty": "Пустой документ",
                            "unknown": "Не определён",
                        }.get(str(item.get("document_kind") or "unknown"), str(item.get("document_kind") or "")),
                        "diagnosis_code": diagnosis_code if _is_valid_icd_code(diagnosis_code) else "",
                        "mkb_code_main": diagnosis_code if _is_valid_icd_code(diagnosis_code) else "",
                        "mkb_code_main_source": str(item.get("mkb_code_main_source") or ""),
                        "mkb_code_main_slot": str(item.get("mkb_code_main_slot") or ""),
                        "diagnosis_short": diagnosis_short,
                        "diagnosis_text": str(item.get("diagnosis_text") or "")[:200],
                        "patient_key": str(item.get("patient_key") or ""),
                        "doctor_id": str(item.get("doctor_id") or ""),
                        "doctor_key": str(item.get("doctor_key") or ""),
                        "history_prior_n": int(item.get("history_prior_n") or 0),
                        "history_tier": str(item.get("history_tier") or ""),
                        "overall_pct": score,
                        "status": item.get("status") or "",
                        "score_reason": (
                            None
                            if isinstance(score, (int, float))
                            else (
                                "Не оценивается: не клинический приём (процедура / диагностика / профосмотр / стоматология)"
                                if str(item.get("document_kind") or "") not in {"clinical_visit", "consultation"}
                                else "Оценка ещё не рассчитана"
                            )
                        ),
                        "parse_ok": "1",
                        "date_mismatch": "0",
                        "_source": "warehouse_case_detail",
                    },
                    "axes": {
                        "documentation": axes.get("documentation"),
                        "clinical_concordance": axes.get("clinical_concordance"),
                        "safety": axes.get("safety"),
                        "regulatory": axes.get("regulatory"),
                    },
                    "findings": [dict(row) for row in findings],
                    "source": "warehouse",
                }
            else:
                return detail
    record = dict(detail.get("record") or {})
    diagnosis_code = str(
        record.get("diagnosis_code")
        or record.get("mkb_code_main")
        or ""
    ).strip()
    diagnosis_short = _safe_diagnosis_text(
        record.get("diagnosis_short"),
        record.get("diagnosis"),
        diagnosis_code if _is_valid_icd_code(diagnosis_code) else "",
    )
    record["diagnosis_short"] = diagnosis_short or "Не указан"
    record["diagnosis_code"] = diagnosis_code if _is_valid_icd_code(diagnosis_code) else ""
    record["mkb_code_main"] = record["diagnosis_code"]
    axes = detail.get("axes") if isinstance(detail.get("axes"), dict) else {}
    if detail.get("coverage_pct") is None:
        if isinstance(record.get("coverage_pct"), (int, float)):
            detail["coverage_pct"] = float(record["coverage_pct"])
        else:
            populated_axes = sum(1 for key in ("documentation", "clinical_concordance", "safety", "regulatory") if axes.get(key) is not None)
            if populated_axes:
                detail["coverage_pct"] = round(100.0 * populated_axes / 4.0, 2)
            elif str(record.get("parse_ok") or "").strip() == "1":
                detail["coverage_pct"] = 100.0
    if detail.get("confidence_pct") is None:
        if isinstance(record.get("confidence_pct"), (int, float)):
            detail["confidence_pct"] = float(record["confidence_pct"])
        else:
            parse_ok = str(record.get("parse_ok") or "").strip() == "1"
            mismatch = str(record.get("date_mismatch") or "0").strip() == "1"
            if parse_ok and mismatch:
                detail["confidence_pct"] = 75.0
            elif parse_ok:
                detail["confidence_pct"] = 90.0
            elif axes:
                detail["confidence_pct"] = 55.0
    patient_id_raw = str(record.pop("patient_id", None) or "").strip()
    detail["record"] = record
    detail["case_id"] = case_id
    with closing(_connect()) as conn:
        try:
            from .mo_review_pack import ensure_review_pack_schema

            ensure_review_pack_schema(conn)
        except Exception:  # noqa: BLE001
            pass
        state = conn.execute("SELECT * FROM crm_case_state WHERE case_id=?", (case_id,)).fetchone()
        events = conn.execute(
            "SELECT event_id,event_type,actor,payload_json,created_at FROM crm_case_event "
            "WHERE case_id=? ORDER BY created_at DESC LIMIT 200",
            (case_id,),
        ).fetchall()
        review_packs = []
        try:
            review_packs = conn.execute(
                """SELECT pack_id, created_at, actor, training_use, decision_json, supersedes_pack_id
                   FROM crm_review_pack WHERE case_id=?
                   ORDER BY created_at DESC LIMIT 20""",
                (case_id,),
            ).fetchall()
        except sqlite3.Error:
            review_packs = []
    crm = dict(state) if state else {
        "case_id": case_id,
        "status": "new",
        "tags_json": "[]",
        "finding_decisions_json": "{}",
    }
    crm["tags"] = json.loads(crm.pop("tags_json") or "[]")
    crm["finding_decisions"] = json.loads(crm.pop("finding_decisions_json") or "{}")
    detail["crm"] = crm
    detail["events"] = [
        {**dict(row), "payload": json.loads(row["payload_json"] or "{}")} for row in events
    ]
    for event in detail["events"]:
        event.pop("payload_json", None)
    pack_items = []
    for row in review_packs:
        item = dict(row)
        decision = {}
        try:
            decision = json.loads(item.pop("decision_json", None) or "{}")
        except json.JSONDecodeError:
            item.pop("decision_json", None)
        pack_items.append(
            {
                "pack_id": item.get("pack_id"),
                "created_at": item.get("created_at"),
                "actor": item.get("actor") or "",
                "training_use": bool(int(item.get("training_use") or 0)),
                "supersedes_pack_id": item.get("supersedes_pack_id"),
                "decision_summary": {
                    "status": decision.get("status"),
                    "verdict_completeness": decision.get("verdict_completeness"),
                    "verdict_diagnosis": decision.get("verdict_diagnosis"),
                    "verdict_recommendations": decision.get("verdict_recommendations"),
                    "summary_ru": (decision.get("summary_ru") or "")[:240],
                },
            }
        )
    detail["review_packs"] = pack_items
    # patient_id только по явному флагу в API (methodist+); по умолчанию скрыт.
    if patient_id_raw:
        detail["_patient_id_hint"] = patient_id_raw
    # Черновой бандл; case detail API пересоберёт после identity lookup.
    try:
        from clinical_knowledge.mo_patient_history_bundle import (
            attach_bundle_to_case,
            public_bundle_for_ui,
        )

        hist_case = {
            "patient_id": patient_id_raw or "",
            "patient_key": str(record.get("patient_key") or ""),
            "visit_date": str(record.get("date") or record.get("visit_date") or "")[:10],
            "doctor_id": str(record.get("doctor_id") or ""),
            "doctor_key": str(record.get("doctor_key") or ""),
            "doctor_fio": str(record.get("doctor_fio") or ""),
            "specialty": str(record.get("specialization") or record.get("specialty") or ""),
            "diagnosis_code": str(record.get("diagnosis_code") or record.get("mkb_code_main") or ""),
            "mis_id": str(record.get("mis_id") or case_id),
            "visit_id": str(record.get("visit_id") or ""),
        }
        detail["patient_history"] = public_bundle_for_ui(attach_bundle_to_case(hist_case))
    except Exception:  # noqa: BLE001
        pass
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


def build_reports(*, min_date: str | None = None) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    min_day = str(min_date or "").strip()[:10]
    roots = [root / "reports" for root in _medical_exam_roots()]
    for base in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*/*/*/report.json"), reverse=True)[:120]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            day = str(data.get("date") or path.parent.name)
            if min_day and day < min_day:
                continue
            if day in seen_dates:
                continue
            seen_dates.add(day)
            summary = data.get("summary") or {}
            reports.append(
                {
                    "date": day,
                    "revision": data.get("revision"),
                    "generated_at": data.get("generated_at"),
                    "quality_status": data.get("quality_status")
                    or ("partial" if data.get("partial") else ("ok" if (data.get("quality") or {}).get("passed") else "blocked")),
                    "source_rows": summary.get("source_rows"),
                    "evaluated": summary.get("scored") or summary.get("eligible_rows"),
                    "avg_score": summary.get("avg_score"),
                    "needs_attention": summary.get("needs_attention"),
                    "critical": summary.get("critical"),
                    "has_report_file": True,
                }
            )
    # Дополняем днями из витрины, если HTML/JSON отчёт ещё не опубликован, но данные есть.
    if _warehouse_available():
        with closing(_read_connection()) as conn:
            try:
                rows = conn.execute(
                    """SELECT visit_date, source_rows, scored_rows, avg_score, needs_attention, critical, quality_status
                       FROM fact_mo_daily ORDER BY visit_date DESC LIMIT 120"""
                ).fetchall()
            except sqlite3.Error:
                rows = []
        for row in rows:
            day = str(row["visit_date"])
            if min_day and day < min_day:
                continue
            if day in seen_dates:
                # обогащаем уже найденный отчёт, если KPI пустые
                for item in reports:
                    if item["date"] == day:
                        item.setdefault("source_rows", row["source_rows"])
                        item.setdefault("evaluated", row["scored_rows"])
                        item.setdefault("avg_score", row["avg_score"])
                        item.setdefault("needs_attention", row["needs_attention"])
                        item.setdefault("critical", row["critical"])
                        break
                continue
            seen_dates.add(day)
            reports.append(
                {
                    "date": day,
                    "revision": None,
                    "generated_at": None,
                    "quality_status": str(row["quality_status"] or "warehouse_only"),
                    "source_rows": row["source_rows"],
                    "evaluated": row["scored_rows"],
                    "avg_score": row["avg_score"],
                    "needs_attention": row["needs_attention"],
                    "critical": row["critical"],
                    "has_report_file": False,
                    "empty_reason": "Есть витрина, файл дневного отчёта ещё не сформирован",
                }
            )
    reports.sort(key=lambda item: str(item.get("date") or ""), reverse=True)
    if not reports and not min_day:
        reports = [{"month": month, "kind": "legacy_monthly"} for month in reversed(build_kz_dynamics().get("months") or [])]
    return {
        "ok": True,
        "items": reports,
        "min_date": min_day or None,
        "freshness": build_freshness({}),
    }


def build_mo_health() -> dict[str, Any]:
    """Сводка здоровья модуля МО (фаза 7): витрина, лаг, сверка, LLM."""
    freshness = build_freshness({})
    db_path = _db_path()
    schema_version = None
    warehouse_cases = None
    daily_days = None
    reconcile: dict[str, Any] = {"available": False}
    if _warehouse_available():
        with closing(_read_connection()) as conn:
            try:
                schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
            except sqlite3.Error:
                schema_version = None
            try:
                warehouse_cases = int(conn.execute("SELECT COUNT(*) FROM fact_mo_case").fetchone()[0])
                daily_days = int(conn.execute("SELECT COUNT(*) FROM fact_mo_daily").fetchone()[0])
            except sqlite3.Error:
                pass
            try:
                # Сверка: дни витрины против дней отчётов на диске.
                days = [
                    str(row[0])
                    for row in conn.execute(
                        "SELECT visit_date FROM fact_mo_daily ORDER BY visit_date DESC LIMIT 14"
                    ).fetchall()
                ]
                missing_reports = []
                for day in days:
                    found = False
                    for root in _medical_exam_roots():
                        path = root / "reports" / day[0:4] / day[5:7] / day[8:10] / "report.json"
                        if path.is_file():
                            found = True
                            break
                    if not found:
                        missing_reports.append(day)
                reconcile = {
                    "available": True,
                    "checked_days": len(days),
                    "missing_report_files": missing_reports,
                    "mismatch_pct": round(100 * len(missing_reports) / len(days), 2) if days else 0.0,
                    "alert": bool(days) and (100 * len(missing_reports) / len(days) > 0.5),
                }
            except sqlite3.Error as exc:
                reconcile = {"available": False, "reason": str(exc)}
    data_through = str(freshness.get("data_through") or "")[:10]
    source_formats: list[str] = []
    checked_roots = 0
    if data_through:
        year, month = data_through[:4], data_through[5:7]
        for root in _medical_exam_roots():
            checked_roots += 1
            raw = root / "raw" / year / month / f"mo_{data_through}.parquet"
            secure = root / "secure_cases" / year / month / f"mo_{data_through}.csv"
            quarantine = root / "quarantine" / year / month
            if raw.is_file() and "raw_parquet" not in source_formats:
                source_formats.append("raw_parquet")
            if secure.is_file() and "secure_csv" not in source_formats:
                source_formats.append("secure_csv")
            if quarantine.is_dir() and any(quarantine.glob(f"*/mo_{data_through}.parquet")):
                if "quarantine_parquet" not in source_formats:
                    source_formats.append("quarantine_parquet")
    document_source = {
        "status": "ready" if source_formats else ("missing" if data_through else "unknown"),
        "data_through": data_through or None,
        "formats": source_formats,
        "checked_roots": checked_roots,
        "case_document_available": bool(source_formats),
    }
    status = "ok"
    reason_codes: list[str] = []
    if freshness.get("status") in {"critical", "stale"}:
        status = str(freshness.get("status"))
        reason_codes.append(f"freshness_{freshness.get('status')}")
    if reconcile.get("alert"):
        status = "degraded"
        reason_codes.append("report_reconciliation_mismatch")
    if document_source["status"] == "missing":
        status = "degraded" if status == "ok" else status
        reason_codes.append("case_document_source_missing")
    if not _warehouse_available():
        reason_codes.append("warehouse_unavailable")

    yesterday = (datetime.now(ZoneInfo("Europe/Minsk")).date() - timedelta(days=1))
    yesterday_report = _pipeline_report_for_date(yesterday) or {}
    yesterday_completeness = (
        yesterday_report.get("completeness")
        if isinstance(yesterday_report.get("completeness"), dict)
        else {}
    )
    state_info = freshness.get("state") if isinstance(freshness.get("state"), dict) else {}
    state_dates = {}
    if state_info.get("path"):
        try:
            state_payload = json.loads(Path(str(state_info["path"])).read_text(encoding="utf-8"))
            state_dates = state_payload.get("dates") if isinstance(state_payload.get("dates"), dict) else {}
        except (OSError, json.JSONDecodeError, TypeError):
            state_dates = {}
    yesterday_state = state_dates.get(yesterday.isoformat()) if isinstance(state_dates, dict) else None
    if isinstance(yesterday_state, dict) and isinstance(yesterday_state.get("completeness"), dict):
        # Состояние пайплайна может быть свежее отчёта до публикации.
        yesterday_completeness = {**yesterday_completeness, **yesterday_state["completeness"]}
    yesterday_partial = bool(
        yesterday_report.get("partial")
        or yesterday_completeness.get("partial")
        or (isinstance(yesterday_state, dict) and yesterday_state.get("status") == "partial")
    )
    yesterday_reasons = list(yesterday_completeness.get("reasons") or [])
    yesterday_advisory = list(yesterday_completeness.get("advisory_reasons") or [])
    if yesterday_partial:
        status = "degraded" if status == "ok" else status
        reason_codes.append("yesterday_partial")
        for code in yesterday_reasons:
            reason_codes.append(f"yesterday_{code}")

    components = {
        "warehouse": {
            "status": "ready" if _warehouse_available() else "missing",
            "cases": warehouse_cases,
            "days": daily_days,
            "schema_version": schema_version,
        },
        "freshness": {
            "status": freshness.get("status") or "unknown",
            "data_through": freshness.get("data_through"),
            "lag_days": freshness.get("lag_days"),
        },
        "reports": {
            "status": "degraded" if reconcile.get("alert") else ("ready" if reconcile.get("available") else "unknown"),
            "missing_days": len(reconcile.get("missing_report_files") or []),
        },
        "case_document_source": document_source,
        "pipeline": {
            "status": ((freshness.get("state") or {}).get("last_stage") or "unknown"),
            "heartbeat": (freshness.get("state") or {}).get("last_heartbeat"),
        },
        "yesterday": {
            "date": yesterday.isoformat(),
            "partial": yesterday_partial,
            "reasons": yesterday_reasons,
            "advisory_reasons": yesterday_advisory,
            "llm_queue_pending": yesterday_completeness.get("llm_queue_pending"),
            "coverage_pct": yesterday_completeness.get("coverage_pct"),
            "pipeline_status": (yesterday_state or {}).get("status") if isinstance(yesterday_state, dict) else None,
        },
    }
    return {
        "ok": True,
        "status": status,
        "reason_codes": reason_codes,
        "schema_version": schema_version,
        "warehouse_path": str(db_path),
        "warehouse_cases": warehouse_cases,
        "daily_days": daily_days,
        "freshness": freshness,
        "reconcile": reconcile,
        "yesterday": components["yesterday"],
        "components": components,
        "features": {
            "case_document": True,
            "case_pdf": True,
            "methodology_toggle": True,
            "llm_costs": True,
            "v4_primary": False,
        },
        "checked_at": _utc(),
    }


def build_mo_capabilities(role: str = "methodist") -> dict[str, Any]:
    """Явный контракт возможностей, чтобы UI не угадывал доступность функций."""
    normalized_role = (
        role
        if role in {"viewer", "doctor", "methodist", "lead", "admin", "expert"}
        else "viewer"
    )
    is_expert = normalized_role == "expert"
    can_work_cases = normalized_role in {"methodist", "lead", "admin", "expert"}
    can_view_population = normalized_role in {"viewer", "methodist", "lead", "admin"}
    from .mo_expert_auth import reports_min_date

    expert_min = reports_min_date() if is_expert else None
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "role": normalized_role,
        "reports_min_date": expert_min,
        "pages": {
            "overview": can_view_population and not is_expert,
            "yesterday": can_view_population or is_expert,
            "queue": can_work_cases and not is_expert,
            "documents": can_view_population and not is_expert,
            "doctors": can_view_population and not is_expert,
            "specialties": can_view_population and not is_expert,
            "diagnoses": can_view_population and not is_expert,
            "safety": can_view_population and not is_expert,
            "doctor_cabinet": normalized_role in {"doctor", "methodist", "lead", "admin"},
            "access_log": normalized_role == "admin",
            "data_quality": can_view_population and not is_expert,
            "reports": can_view_population or is_expert,
            "settings": normalized_role in {"methodist", "lead", "admin"},
        },
        "actions": {
            "case_document": normalized_role in {"doctor", "methodist", "lead", "admin", "expert"},
            "case_pdf": normalized_role in {"doctor", "methodist", "lead", "admin", "expert"},
            "rubric_mz": can_view_population or is_expert,
            "case_decision": can_work_cases,
            "bulk_action": can_work_cases and not is_expert,
            "export_aggregates": can_view_population and not is_expert,
            "export_clinical": normalized_role in {"methodist", "lead", "admin"},
            "manage_saved_views": normalized_role in {"methodist", "lead", "admin"},
            "review_pack": can_work_cases,
        },
        "metric_states": ["available", "missing", "not_applicable", "scoring_error", "suppressed"],
        "checked_at": _utc(),
    }


def build_entity(kind: str, entity_id: str, params: dict[str, Any]) -> dict[str, Any]:
    field = {"doctors": "doctor_fio", "specialties": "specialization", "branches": "filial"}.get(kind)
    if not field:
        return {"ok": False, "error": "unknown_entity_kind"}
    rows = [r for r in _filter_records(_records(params), params) if str(r.get(field) or "") == entity_id]
    if len(rows) < SUPPRESSION_N:
        return {"ok": True, "suppressed": True, "n_bucket": f"<{SUPPRESSION_N}", "entity_id": entity_id}
    agg = _filtered_agg(rows)
    return {"ok": True, "entity_id": entity_id, "entity_kind": kind, "n": len(rows), "aggregate": agg}


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 2)


def build_dimension(dimension: str, params: dict[str, Any]) -> dict[str, Any]:
    """Ограниченные SQL-контракты четырёх интерактивных экранов."""
    if dimension not in {"doctors", "specialties", "diagnoses", "safety"}:
        raise ValueError("unknown_dimension")
    resolved = _resolve_request_period(params)
    if _source_for_period(resolved.current) != "warehouse":
        raise RuntimeError("Интерактивные разрезы требуют SQL-витрину МО")
    where, values = _sql_case_filter(resolved.current, params)
    result: dict[str, Any] = {
        "ok": True,
        "source": "warehouse",
        "dimension": dimension,
        "periods": resolved.to_dict(),
        "suppression_n": SUPPRESSION_N,
    }
    if dimension == "doctors":
        items = _doctor_breakdown(resolved.current, max(20, SUPPRESSION_N), params)
        with closing(_read_connection()) as conn:
            p0_rows = conn.execute(
                """SELECT c.doctor_key, COUNT(DISTINCT c.mis_id) p0_cases
                   FROM fact_mo_case c JOIN fact_mo_finding f ON f.mis_id=c.mis_id
                   WHERE """ + where + """ AND f.severity='P0'
                   GROUP BY c.doctor_key""",
                values,
            ).fetchall()
            zone_by_key: dict[str, dict[str, Any]] = {}
            if _warehouse_has_column(str(_db_path()), "fact_mo_case", "zone1_band"):
                zone_rows = conn.execute(
                    """SELECT c.doctor_key,
                              COUNT(*) AS n_scored,
                              SUM(CASE WHEN c.zone1_band='bad' THEN 1 ELSE 0 END) AS zone1_bad,
                              SUM(CASE WHEN c.zone2a_band='bad' THEN 1 ELSE 0 END) AS zone2a_bad,
                              SUM(CASE WHEN c.zone2b_band='bad' THEN 1 ELSE 0 END) AS zone2b_bad,
                              SUM(CASE WHEN COALESCE(c.attention_primary, 'none') NOT IN ('', 'none')
                                       THEN 1 ELSE 0 END) AS attention_n
                       FROM fact_mo_case c
                       WHERE """ + where + """
                         AND c.doctor_key <> ''
                         AND c.document_kind IN ('clinical_visit', 'consultation')
                         AND c.layer_engine IS NOT NULL
                       GROUP BY c.doctor_key""",
                    values,
                ).fetchall()
                for row in zone_rows:
                    n_scored = int(row["n_scored"] or 0)
                    z1 = int(row["zone1_bad"] or 0)
                    z2a = int(row["zone2a_bad"] or 0)
                    z2b = int(row["zone2b_bad"] or 0)

                    def _bad_pct(bad: int) -> float | None:
                        if n_scored <= 0:
                            return None
                        return round(100.0 * bad / n_scored, 1)

                    zone_by_key[str(row["doctor_key"])] = {
                        "n_zone_scored": n_scored,
                        "zone1_bad": z1,
                        "zone1_bad_pct": _bad_pct(z1),
                        "zone2a_bad": z2a,
                        "zone2a_bad_pct": _bad_pct(z2a),
                        "zone2b_bad": z2b,
                        "zone2b_bad_pct": _bad_pct(z2b),
                        "attention_n": int(row["attention_n"] or 0),
                    }
        p0 = {str(row["doctor_key"]): int(row["p0_cases"]) for row in p0_rows}
        for item in items:
            key = str(item["key"])
            item["p0_cases"] = p0.get(key, 0) if not item.get("suppressed") else None
            item["drilldown"] = {"level": "doctor", "id": item["key"]}
            zones = zone_by_key.get(key) or {}
            if item.get("suppressed"):
                item.update(
                    {
                        "zone1_bad": None,
                        "zone1_bad_pct": None,
                        "zone2a_bad": None,
                        "zone2a_bad_pct": None,
                        "zone2b_bad": None,
                        "zone2b_bad_pct": None,
                        "attention_n": None,
                        "n_zone_scored": None,
                    }
                )
            else:
                item.update(zones)
        ranked = [
            item
            for item in items
            if item.get("enough_data")
            and item.get("case_mix_reliable")
            and not item.get("suppressed")
        ]
        ranked.sort(key=lambda item: float(item["delta"]))
        # ui-target: сортировка по доле плохого выбранного раздела (по умолчанию оформление)
        zone_ranked = [
            item
            for item in items
            if not item.get("suppressed") and item.get("zone1_bad_pct") is not None
        ]
        zone_ranked.sort(
            key=lambda item: (
                -(float(item.get("zone1_bad_pct") or 0)),
                -(int(item.get("n") or 0)),
            )
        )
        result.update(
            {
                "items": items[:250],
                "ranking": ranked,
                "zone_ranking": zone_ranked[:50],
                "ranking_metric": "expected_delta",
                "sample_gate": max(20, SUPPRESSION_N),
                "no_raw_score_ranking": True,
                "zone_metrics": True,
            }
        )
        return result
    with closing(_read_connection()) as conn:
        if dimension == "specialties":
            rows = conn.execute(
                """SELECT c.specialty, c.overall_pct
                   FROM fact_mo_case c WHERE """ + where + """
                     AND c.document_kind IN ('clinical_visit', 'consultation')
                     AND c.overall_pct IS NOT NULL
                   ORDER BY c.specialty LIMIT 20000""",
                values,
            ).fetchall()
            grouped: dict[str, list[float]] = {}
            for row in rows:
                grouped.setdefault(str(row["specialty"] or "Не указано"), []).append(float(row["overall_pct"]))
            items = []
            for specialty, scores in grouped.items():
                if len(scores) < SUPPRESSION_N:
                    continue
                items.append(
                    {
                        "key": specialty,
                        "label": specialty,
                        "n": len(scores),
                        "boxplot": [
                            _percentile(scores, 0),
                            _percentile(scores, 0.25),
                            _percentile(scores, 0.5),
                            _percentile(scores, 0.75),
                            _percentile(scores, 1),
                        ],
                        "drilldown": {"level": "specialty", "id": specialty},
                    }
                )
            items.sort(key=lambda item: (-item["n"], item["label"]))
            result["items"] = items[:100]
        elif dimension == "diagnoses":
            rows = conn.execute(
                """SELECT c.icd_chapter, c.diagnosis_code, COUNT(*) n,
                          ROUND(AVG(c.overall_pct),2) avg_score
                   FROM fact_mo_case c WHERE """ + where + """
                     AND c.document_kind IN ('clinical_visit', 'consultation')
                     AND c.icd_chapter <> '' AND c.overall_pct IS NOT NULL
                   GROUP BY c.icd_chapter,c.diagnosis_code
                   ORDER BY n DESC LIMIT 500""",
                values,
            ).fetchall()
            chapters: dict[str, dict[str, Any]] = {}
            for row in rows:
                n = int(row["n"])
                if n < SUPPRESSION_N:
                    continue
                chapter = str(row["icd_chapter"])
                target = chapters.setdefault(
                    chapter, {"name": chapter, "value": 0, "_weighted": 0.0, "children": []}
                )
                target["value"] += n
                target["_weighted"] += n * float(row["avg_score"])
                target["children"].append(
                    {
                        "name": str(row["diagnosis_code"] or "Без кода"),
                        "value": n,
                        "score": float(row["avg_score"]),
                        "drilldown": {
                            "level": "diagnosis",
                            "id": str(row["diagnosis_code"] or ""),
                            "parent": chapter,
                        },
                    }
                )
            items = []
            for chapter in chapters.values():
                if chapter["value"] < SUPPRESSION_N:
                    continue
                chapter["score"] = round(chapter.pop("_weighted") / chapter["value"], 2)
                chapter["drilldown"] = {"level": "icd_chapter", "id": chapter["name"]}
                items.append(chapter)
            result["items"] = sorted(items, key=lambda item: -item["value"])[:50]
            result["encoding"] = {"size": "volume", "color": "avg_score"}
        else:
            rows = conn.execute(
                """SELECT c.visit_date, f.severity, COUNT(DISTINCT c.mis_id) cases
                   FROM fact_mo_case c JOIN fact_mo_finding f ON f.mis_id=c.mis_id
                   WHERE """ + where + """
                     AND f.severity IN ('P0','P1','P2','P3')
                   GROUP BY c.visit_date,f.severity ORDER BY c.visit_date""",
                values,
            ).fetchall()
            by_day: dict[str, dict[str, Any]] = {}
            for row in rows:
                day = str(row["visit_date"])
                target = by_day.setdefault(day, {"date": day, "P0": 0, "P1": 0, "P2": 0, "P3": 0})
                target[str(row["severity"])] = int(row["cases"])
            incidents = [
                {
                    "date": str(row["visit_date"]),
                    "case_id": str(row["mis_id"]),
                    "finding_code": str(row["finding_code"]),
                    "source_ref": str(row["source_ref"] or ""),
                }
                for row in conn.execute(
                    """SELECT c.visit_date,c.mis_id,f.finding_code,f.source_ref
                       FROM fact_mo_case c JOIN fact_mo_finding f ON f.mis_id=c.mis_id
                       WHERE """ + where + """ AND f.severity='P0'
                       ORDER BY c.visit_date DESC LIMIT 200""",
                    values,
                ).fetchall()
            ]
            result.update({"items": list(by_day.values()), "incidents": incidents})
    return result


def build_drilldown(level: str, entity_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Путь специальность -> врач -> случай -> замечание -> источник."""
    resolved = _resolve_request_period(params)
    if _source_for_period(resolved.current) != "warehouse":
        raise RuntimeError("Drill-down требует SQL-витрину МО")
    where, values = _sql_case_filter(resolved.current, params)
    with closing(_read_connection()) as conn:
        if level == "specialty":
            rows = conn.execute(
                """SELECT c.doctor_key,COALESCE(d.doctor_fio,c.doctor_key) label,COUNT(*) n
                   FROM fact_mo_case c LEFT JOIN dim_doctor d ON d.doctor_key=c.doctor_key
                   WHERE """ + where + """ AND c.specialty=? GROUP BY c.doctor_key,label
                   HAVING COUNT(*)>=? ORDER BY n DESC LIMIT 200""",
                (*values, entity_id, SUPPRESSION_N),
            ).fetchall()
            items = [
                {"level": "doctor", "id": str(row["doctor_key"]), "label": str(row["label"]), "n": int(row["n"])}
                for row in rows
            ]
        elif level == "doctor":
            rows = conn.execute(
                """SELECT c.mis_id,c.visit_id,c.visit_date,c.overall_pct,c.diagnosis_code
                   FROM fact_mo_case c WHERE """ + where + """
                     AND c.doctor_key=? ORDER BY c.visit_date DESC LIMIT 500""",
                (*values, entity_id),
            ).fetchall()
            items = [
                {
                    "level": "case",
                    "id": str(row["visit_id"] or row["mis_id"]),
                    "mis_id": str(row["mis_id"]),
                    "date": str(row["visit_date"]),
                    "score": row["overall_pct"],
                    "diagnosis_code": str(row["diagnosis_code"] or ""),
                }
                for row in rows
            ]
        elif level == "case":
            rows = conn.execute(
                """SELECT f.finding_code,f.severity,f.source_ref
                   FROM fact_mo_finding f JOIN fact_mo_case c ON c.mis_id=f.mis_id
                   WHERE (c.mis_id=? OR c.visit_id=?) ORDER BY f.severity,f.finding_code LIMIT 200""",
                (entity_id, entity_id),
            ).fetchall()
            items = [
                {
                    "level": "finding",
                    "id": str(row["finding_code"]),
                    "severity": str(row["severity"] or ""),
                    "source_ref": str(row["source_ref"] or ""),
                }
                for row in rows
            ]
        else:
            raise ValueError("unknown_drilldown_level")
    return {"ok": True, "level": level, "entity_id": entity_id, "items": items}


def record_access(
    *,
    actor: str,
    role: str,
    action: str,
    doctor_key: str | None = None,
    case_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    access_id = str(uuid.uuid4())
    safe_metadata = {
        str(key)[:80]: value
        for key, value in (metadata or {}).items()
        if key not in {"token", "text", "patient_text", "evidence"}
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    with closing(_connect()) as conn:
        conn.execute(
            """INSERT INTO access_log
               (access_id,actor,role,action,doctor_key,case_id,metadata_json,created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                access_id,
                actor[:120],
                role[:30],
                action[:80],
                doctor_key,
                case_id,
                json.dumps(safe_metadata, ensure_ascii=False),
                _utc(),
            ),
        )
        conn.commit()
    return access_id


def list_access_log(*, role: str, limit: int = 200) -> dict[str, Any]:
    if role != "admin":
        raise PermissionError("admin_role_required")
    with closing(_connect()) as conn:
        rows = conn.execute(
            """SELECT access_id,actor,role,action,doctor_key,case_id,metadata_json,created_at
               FROM access_log ORDER BY created_at DESC LIMIT ?""",
            (min(max(limit, 1), 500),),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        items.append(item)
    return {"ok": True, "items": items}


def build_doctor_cabinet(*, doctor_key: str, actor: str, role: str, include_unscored: bool = False) -> dict[str, Any]:
    if not doctor_key:
        raise PermissionError("trusted_doctor_identity_required")
    record_access(actor=actor, role=role, action="doctor_cabinet_open", doctor_key=doctor_key)
    with closing(_read_connection()) as conn:
        doctor = conn.execute(
            "SELECT doctor_key,doctor_fio,specialty,filial FROM dim_doctor WHERE doctor_key=?",
            (doctor_key,),
        ).fetchone()
        if not doctor:
            return {"ok": False, "error": "doctor_not_found"}
        cases = conn.execute(
            """SELECT mis_id,visit_id,visit_date,overall_pct,status,diagnosis_code,
                      document_kind,specialty,filial,scorer_version
               FROM fact_mo_case WHERE doctor_key=?
               ORDER BY visit_date DESC, CASE WHEN document_kind IN ('clinical_visit', 'consultation') THEN 0 ELSE 1 END
               LIMIT 500""",
            (doctor_key,),
        ).fetchall()
        findings = conn.execute(
            """SELECT f.mis_id,f.finding_code,f.severity,f.source_ref,
                      COALESCE(df.title_ru, f.finding_code) AS title_ru
               FROM fact_mo_finding f JOIN fact_mo_case c ON c.mis_id=f.mis_id
               LEFT JOIN dim_finding df ON df.finding_code=f.finding_code
               WHERE c.doctor_key=? ORDER BY c.visit_date DESC,f.severity LIMIT 1000""",
            (doctor_key,),
        ).fetchall()
        pairs = conn.execute(
            """SELECT pair_id,case_id_a,case_id_b,similarity,algorithm,threshold,
                      provenance_json,detected_at
               FROM fact_mo_template_pair WHERE doctor_key=?
               ORDER BY similarity DESC LIMIT 200""",
            (doctor_key,),
        ).fetchall()
        actions = conn.execute(
            """SELECT e.event_id,e.case_id,e.event_type,e.actor,e.payload_json,e.created_at
               FROM crm_case_event e
               WHERE e.case_id IN (
                 SELECT COALESCE(NULLIF(visit_id,''),mis_id)
                 FROM fact_mo_case WHERE doctor_key=?
               )
               ORDER BY e.created_at DESC LIMIT 500""",
            (doctor_key,),
        ).fetchall()
        dispute_rows = conn.execute(
            """SELECT COUNT(*) total,
                      SUM(status='submitted') submitted,
                      SUM(status='resolved_false_positive') false_positive
               FROM crm_dispute_state WHERE case_id IN (
                   SELECT COALESCE(NULLIF(visit_id,''),mis_id)
                   FROM fact_mo_case WHERE doctor_key=?
                 )""",
            (doctor_key,),
        ).fetchone()
    kind_labels = dict(DOCUMENT_KIND_LABELS)
    scored_kinds = {"clinical_visit"}
    case_items = []
    for row in cases:
        item = dict(row)
        kind = str(item.get("document_kind") or "unknown")
        if not include_unscored and kind not in scored_kinds:
            continue
        score = item.get("overall_pct")
        case_id = str(item.get("visit_id") or item.get("mis_id"))
        diagnosis_code = str(item.get("diagnosis_code") or "").strip()
        safe_diagnosis_code = diagnosis_code if _is_valid_icd_code(diagnosis_code) else ""
        item.update(
            {
                "case_id": case_id,
                "diagnosis_code": safe_diagnosis_code,
                "document_kind_label": kind_labels.get(kind, kind),
                "score_reason": (
                    None
                    if isinstance(score, (int, float))
                    else (
                        f"Не оценивается: {kind_labels.get(kind, kind)}"
                        if kind not in scored_kinds
                        else "Оценка ещё не рассчитана"
                    )
                ),
                "title": " · ".join(
                    part
                    for part in (
                        str(item.get("visit_date") or ""),
                        str(safe_diagnosis_code or "Без кода МКБ"),
                        kind_labels.get(kind, kind),
                    )
                    if part
                ),
                "document_url": f"/api/methodist/mo/cases/{case_id}/document",
                "pdf_url": f"/api/methodist/mo/cases/{case_id}/pdf",
            }
        )
        # Никогда не отдаём content_hash и сырые opaque id как заголовок.
        item.pop("content_hash", None)
        case_items.append(item)
    pair_items = []
    for row in pairs:
        item = dict(row)
        item["provenance"] = json.loads(item.pop("provenance_json") or "{}")
        pair_items.append(item)
    action_items = []
    for row in actions:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        action_items.append(item)
    finding_items = []
    scored_mis_ids = {str(item["mis_id"]) for item in case_items}
    for row in findings:
        item = dict(row)
        if scored_mis_ids and str(item.get("mis_id")) not in scored_mis_ids and not include_unscored:
            continue
        item["citation"] = item.get("source_ref") or None
        finding_items.append(item)
    return {
        "ok": True,
        "doctor": dict(doctor),
        "cases": case_items,
        "findings": finding_items,
        "actions": action_items,
        "template_pairs": pair_items,
        "dispute_stats": {
            "total": int(dispute_rows["total"] or 0),
            "submitted": int(dispute_rows["submitted"] or 0),
            "false_positive": int(dispute_rows["false_positive"] or 0),
        },
        "what_to_fix": sorted({str(row["finding_code"]) for row in finding_items})[:50],
        "include_unscored": include_unscored,
        "hidden_unscored": max(0, len(cases) - len(case_items)),
    }


def create_dispute(
    *,
    actor: str,
    role: str,
    doctor_key: str,
    case_id: str,
    finding_code: str,
    reason: str,
) -> dict[str, Any]:
    reason = reason.strip()
    if not reason:
        raise ValueError("dispute_reason_required")
    with closing(_connect()) as conn:
        owned = conn.execute(
            """SELECT 1 FROM fact_mo_case
               WHERE doctor_key=? AND (mis_id=? OR visit_id=?) LIMIT 1""",
            (doctor_key, case_id, case_id),
        ).fetchone()
        if not owned:
            raise PermissionError("case_not_owned_by_doctor")
        if finding_code:
            finding_exists = conn.execute(
                """SELECT 1 FROM fact_mo_finding f
                   JOIN fact_mo_case c ON c.mis_id=f.mis_id
                   WHERE c.doctor_key=? AND (c.mis_id=? OR c.visit_id=?)
                     AND f.finding_code=? LIMIT 1""",
                (doctor_key, case_id, case_id, finding_code),
            ).fetchone()
            if not finding_exists:
                raise ValueError("finding_not_found_for_case")
        now = _utc()
        event_id = str(uuid.uuid4())
        dispute_id = str(uuid.uuid4())
        payload = {
            "finding_code": finding_code[:120],
            "reason": reason[:2000],
            "status": "submitted",
        }
        conn.execute(
            """INSERT INTO crm_case_event
               (event_id,case_id,event_type,actor,payload_json,created_at)
               VALUES(?,?,?,?,?,?)""",
            (event_id, case_id, "doctor_dispute", actor[:120], json.dumps(payload, ensure_ascii=False), now),
        )
        conn.execute(
            """INSERT INTO crm_dispute_state
               (dispute_id,event_id,case_id,finding_code,status,reason,actor,
                created_at,updated_at,resolved_by)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                dispute_id,
                event_id,
                case_id,
                finding_code[:120] or None,
                "submitted",
                reason[:2000],
                actor[:120],
                now,
                now,
                None,
            ),
        )
        conn.commit()
    return {
        "ok": True,
        "dispute_id": dispute_id,
        "event_id": event_id,
        "status": "submitted",
        "created_at": now,
    }


def create_doctor_export(*, doctor_key: str, actor: str, role: str) -> dict[str, Any]:
    cabinet = build_doctor_cabinet(doctor_key=doctor_key, actor=actor, role=role)
    if not cabinet.get("ok"):
        return cabinet
    job_id = str(uuid.uuid4())
    export_dir = _db_path().parent / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / f"{job_id}.json"
    target.write_text(json.dumps(cabinet, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(hours=24)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with closing(_connect()) as conn:
        conn.execute(
            """INSERT INTO export_job
               (job_id,owner,status,kind,filters_json,result_path,created_at,expires_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                job_id,
                actor,
                "ready",
                "doctor_cabinet",
                json.dumps({"doctor_key": doctor_key}),
                str(target),
                _utc(),
                expires,
            ),
        )
        conn.commit()
    record_access(
        actor=actor,
        role=role,
        action="doctor_personal_export",
        doctor_key=doctor_key,
        metadata={"job_id": job_id},
    )
    return {
        "ok": True,
        "job_id": job_id,
        "status": "ready",
        "expires_at": expires,
        "download_url": f"/api/methodist/mo/exports/{job_id}",
    }


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


def _llm_night_coverage(date_from: date, date_to: date) -> list[dict[str, Any]]:
    """Покрытие night-queue и action-judge по дням (из артефактов на диске)."""
    rows: list[dict[str, Any]] = []
    day = date_from
    while day <= date_to:
        y, m, d = day.isoformat()[:4], day.isoformat()[5:7], day.isoformat()[8:10]
        queue_n = grades_ok = grades_err = judges = 0
        for root in _medical_exam_roots():
            secure = root / "secure_cases" / y / m
            queue_path = secure / f"kz_l1_{day.isoformat()}_llm_queue.json"
            grades_path = secure / f"kz_l1_{day.isoformat()}_llm_grades.jsonl"
            judge_path = root / "llm_action_judge" / y / m / d / "judges.jsonl"
            if queue_path.is_file():
                try:
                    payload = json.loads(queue_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = {}
                if isinstance(payload, dict):
                    ids = payload.get("visit_ids") or payload.get("queue") or []
                    queue_n = max(queue_n, len(ids) if isinstance(ids, list) else int(payload.get("n") or 0))
                elif isinstance(payload, list):
                    queue_n = max(queue_n, len(payload))
            if grades_path.is_file():
                ok = err = 0
                try:
                    for line in grades_path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            err += 1
                            continue
                        if isinstance(row, dict) and (row.get("_error") or row.get("error")):
                            err += 1
                        else:
                            ok += 1
                except OSError:
                    ok = err = 0
                grades_ok = max(grades_ok, ok)
                grades_err = max(grades_err, err)
            if judge_path.is_file():
                try:
                    judges = max(
                        judges,
                        sum(1 for line in judge_path.read_text(encoding="utf-8").splitlines() if line.strip()),
                    )
                except OSError:
                    pass
        pending = max(0, queue_n - grades_ok)
        rows.append(
            {
                "date": day.isoformat(),
                "queue": queue_n,
                "grades_ok": grades_ok,
                "grades_error": grades_err,
                "pending": pending,
                "action_judges": judges,
                "night_complete": queue_n > 0 and pending == 0,
            }
        )
        day += timedelta(days=1)
    return rows


def build_llm_costs(params: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve_request_period(params)
    coverage = _llm_night_coverage(resolved.current.date_from, resolved.current.date_to)
    with closing(_read_connection()) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fact_llm_usage'"
        ).fetchone()
        if not table_exists:
            return {
                "ok": True,
                "available": False,
                "reason": "Учёт расходов появится после первого LLM-прогона v4",
                "periods": resolved.to_dict(),
                "total_usd": 0.0,
                "calls": 0,
                "items": [],
                "coverage": coverage,
            }
        date_values = (
            resolved.current.date_from.isoformat(),
            resolved.current.date_to.isoformat(),
        )
        rows = conn.execute(
            """SELECT usage_date,tier,model,COUNT(*) calls,
                      SUM(prompt_tokens) prompt_tokens,
                      SUM(completion_tokens) completion_tokens,
                      ROUND(SUM(cost_usd),6) cost_usd,
                      ROUND(AVG(latency_ms),1) avg_latency_ms,
                      SUM(status NOT LIKE 'ok%') failed
               FROM fact_llm_usage
               WHERE usage_date BETWEEN ? AND ?
               GROUP BY usage_date,tier,model
               ORDER BY usage_date,tier,model""",
            date_values,
        ).fetchall()
        totals = conn.execute(
            """SELECT COUNT(*) calls,ROUND(COALESCE(SUM(cost_usd),0),6) total_usd,
                      COUNT(DISTINCT case_id) cases
               FROM fact_llm_usage WHERE usage_date BETWEEN ? AND ?""",
            date_values,
        ).fetchone()
    cases = int(totals["cases"] or 0)
    total = float(totals["total_usd"] or 0)
    return {
        "ok": True,
        "available": True,
        "periods": resolved.to_dict(),
        "currency": "USD",
        "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
        "calls": int(totals["calls"] or 0),
        "cases": cases,
        "total_usd": total,
        "cost_per_case_usd": round(total / cases, 6) if cases else None,
        "items": [dict(row) for row in rows],
        "coverage": coverage,
        "coverage_summary": {
            "days": len(coverage),
            "night_complete_days": sum(1 for row in coverage if row.get("night_complete")),
            "grades_ok": sum(int(row.get("grades_ok") or 0) for row in coverage),
            "pending": sum(int(row.get("pending") or 0) for row in coverage),
            "action_judges": sum(int(row.get("action_judges") or 0) for row in coverage),
        },
    }


def build_methodology_status(params: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve_request_period(params)
    where, values = _sql_case_filter(resolved.current, params)
    with closing(_read_connection()) as conn:
        versions = [
            {"scorer_version": str(row[0] or "legacy"), "n": int(row[1])}
            for row in conn.execute(
                "SELECT scorer_version,COUNT(*) FROM fact_mo_case c WHERE "
                + where
                + " GROUP BY scorer_version ORDER BY COUNT(*) DESC",
                values,
            )
        ]
        visits = conn.execute(
            """SELECT COUNT(*) visits,COALESCE(SUM(records),0) records,
                      COALESCE(SUM(scored_records),0) scored_records
               FROM fact_mo_visit WHERE visit_date BETWEEN ? AND ?""",
            (resolved.current.date_from.isoformat(), resolved.current.date_to.isoformat()),
        ).fetchone()
        explanation = conn.execute(
            """SELECT COUNT(DISTINCT f.finding_code) shown,
                      COUNT(DISTINCT d.finding_code) explained
               FROM fact_mo_finding f
               LEFT JOIN dim_finding d ON d.finding_code=f.finding_code
               JOIN fact_mo_case c ON c.mis_id=f.mis_id
               WHERE """
            + where,
            values,
        ).fetchone()
    shown = int(explanation["shown"] or 0)
    explained = int(explanation["explained"] or 0)
    return {
        "ok": True,
        "periods": resolved.to_dict(),
        "primary_scorer": "v4.0.0",
        "weights": {
            "documentation": 0.25,
            "clinical_concordance": 0.35,
            "safety": 0.30,
            "regulatory": 0.10,
        },
        "versions": versions,
        "visit_denominators": dict(visits),
        "finding_explanations": {
            "shown": shown,
            "explained": explained,
            "coverage_pct": round(100 * explained / shown, 2) if shown else 100.0,
        },
        "attention_rule": "P0/P1 или низкая доказательность; порог общего балла не используется",
        "trust_rule": "Штраф и risk-cap применяются только к правилам trust A/B",
    }


def compatibility_metadata() -> dict[str, Any]:
    return {
        "deprecated": True,
        "replacement": "/api/methodist/mo",
        "sunset_after_releases": 2,
        "legacy_fields_preserved": ["kz_kind", "evaluation_v3"],
    }
