"""Каталог визитов МИС для рубрики «Поиск МИС» (волна 4).

Источник строк - `mis_data` (дата, врач, короткий diagnos), не `mis_protocol.result`.
Поиск и покрытие читаются из SQLite склада. Живой SELECT - только точный visit_id.
PHI: в каталоге patient_key, не patient_id; в лог только visit_id и статус job.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from clinical_knowledge.mo_daily import patient_key_for
from clinical_knowledge.mo_lab_bundle import default_lab_path
from clinical_knowledge.mo_patient_history_bundle import default_warehouse_path

ENGINE = "mo_mis_catalog_v1"
DX_SHORT_MAX = 80
SEARCH_LIMIT = 50
MAX_QUEUED_JOBS = 20
_ID_RE = re.compile(r"^\d{4,12}$")
_JOB_LOCK = threading.Lock()
_log = logging.getLogger("protocol.mo_mis_catalog")

CATALOG_DDL = """
CREATE TABLE IF NOT EXISTS fact_mis_catalog (
  visit_id TEXT PRIMARY KEY,
  visit_date TEXT NOT NULL,
  specialist_id TEXT,
  doctor_fio TEXT,
  specialization TEXT,
  filial TEXT,
  dx_short TEXT,
  patient_key TEXT,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mis_catalog_date ON fact_mis_catalog(visit_date);
CREATE INDEX IF NOT EXISTS idx_mis_catalog_dx ON fact_mis_catalog(dx_short);
CREATE INDEX IF NOT EXISTS idx_mis_catalog_doctor ON fact_mis_catalog(doctor_fio);
CREATE INDEX IF NOT EXISTS idx_mis_catalog_spec ON fact_mis_catalog(specialization);
CREATE INDEX IF NOT EXISTS idx_mis_catalog_patient ON fact_mis_catalog(patient_key, visit_date);
CREATE TABLE IF NOT EXISTS mo_ingest_job (
  job_id TEXT PRIMARY KEY,
  visit_id TEXT NOT NULL,
  status TEXT NOT NULL,
  actor TEXT,
  case_id TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ingest_job_status ON mo_ingest_job(status, created_at);
CREATE TABLE IF NOT EXISTS fact_mis_catalog_meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""

LAB_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_lab_indicator ON fact_mo_lab(indicator_name);
CREATE INDEX IF NOT EXISTS idx_lab_type ON fact_mo_lab(type_name);
CREATE INDEX IF NOT EXISTS idx_lab_date ON fact_mo_lab(test_date);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clip(value: Any, n: int) -> str:
    return str(value or "").strip()[:n]


def warehouse_path(path: Path | str | None = None) -> Path | None:
    if path:
        resolved = Path(path)
        return resolved if resolved.is_file() or resolved.parent.is_dir() else None
    return default_warehouse_path()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(CATALOG_DDL)
    conn.commit()


def ensure_lab_indexes(conn: sqlite3.Connection) -> None:
    try:
        conn.executescript(LAB_INDEX_DDL)
        conn.commit()
    except sqlite3.Error:
        pass


def seed_catalog_from_cases(conn: sqlite3.Connection) -> int:
    """Визиты склада сразу видны как «В аналитике», без ночного МИС-прогона."""
    ensure_schema(conn)
    if conn.execute(
        "SELECT 1 FROM fact_mis_catalog_meta WHERE key = 'seeded_from_cases'"
    ).fetchone():
        return 0
    tables = _table_names(conn)
    if "fact_mo_case" not in tables:
        return 0
    cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(fact_mo_case)").fetchall()}
    if "visit_id" not in cols:
        return 0
    doctor_join = (
        "LEFT JOIN dim_doctor d ON d.doctor_key = c.doctor_key"
        if "dim_doctor" in tables
        else ""
    )
    doctor_fio = "COALESCE(d.doctor_fio, '')" if doctor_join else "''"
    dx_select = (
        "substr(COALESCE(c.diagnosis_text, ''), 1, ?)"
        if "diagnosis_text" in cols
        else "''"
    )
    before = conn.execute("SELECT COUNT(*) FROM fact_mis_catalog").fetchone()[0]
    params: tuple[Any, ...] = (
        (DX_SHORT_MAX, _utc_now()) if "diagnosis_text" in cols else (_utc_now(),)
    )
    conn.execute(
        f"""
        INSERT OR IGNORE INTO fact_mis_catalog(
          visit_id, visit_date, specialist_id, doctor_fio, specialization,
          filial, dx_short, patient_key, updated_at
        )
        SELECT c.visit_id, c.visit_date, '', {doctor_fio},
               COALESCE(c.specialty, ''), COALESCE(c.filial, ''),
               {dx_select},
               COALESCE(c.patient_key, ''), ?
        FROM fact_mo_case c
        {doctor_join}
        WHERE c.visit_id IS NOT NULL AND trim(c.visit_id) != ''
        """,
        params,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM fact_mis_catalog").fetchone()[0]
    conn.execute(
        "INSERT OR REPLACE INTO fact_mis_catalog_meta(key, value) VALUES (?, ?)",
        ("seeded_from_cases", _utc_now()),
    )
    conn.commit()
    return int(after) - int(before)


def upsert_catalog_rows(conn: sqlite3.Connection, rows: list[Mapping[str, Any]]) -> int:
    ensure_schema(conn)
    now = _utc_now()
    payload = []
    for row in rows:
        visit_id = str(row.get("visit_id") or "").strip()
        visit_date = str(row.get("visit_date") or "")[:10]
        if not visit_id or not visit_date:
            continue
        pid = str(row.get("patient_id") or "").strip()
        payload.append(
            (
                visit_id,
                visit_date,
                _clip(row.get("specialist_id"), 32),
                _clip(row.get("doctor_fio"), 160),
                _clip(row.get("specialization"), 120),
                _clip(row.get("filial"), 80),
                _clip(row.get("dx_short") or row.get("diagnos"), DX_SHORT_MAX),
                str(row.get("patient_key") or "").strip() or patient_key_for(pid),
                now,
            )
        )
    if not payload:
        return 0
    conn.executemany(
        """
        INSERT INTO fact_mis_catalog(
          visit_id, visit_date, specialist_id, doctor_fio, specialization,
          filial, dx_short, patient_key, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(visit_id) DO UPDATE SET
          visit_date=excluded.visit_date,
          specialist_id=excluded.specialist_id,
          doctor_fio=excluded.doctor_fio,
          specialization=excluded.specialization,
          filial=excluded.filial,
          dx_short=excluded.dx_short,
          patient_key=excluded.patient_key,
          updated_at=excluded.updated_at
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _in_analytics_sql(conn: sqlite3.Connection) -> str:
    if "fact_mo_case" not in _table_names(conn):
        return "0"
    return (
        "EXISTS (SELECT 1 FROM fact_mo_case c "
        "WHERE c.visit_id = fact_mis_catalog.visit_id)"
    )


def _like_term(q: str) -> str:
    escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _open_warehouse(path: Path | str | None) -> sqlite3.Connection | None:
    resolved = warehouse_path(path)
    if resolved is None:
        return None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def coverage_for_window(
    conn: sqlite3.Connection, *, date_from: str, date_to: str
) -> dict[str, Any]:
    ensure_schema(conn)
    found = conn.execute(
        """
        SELECT COUNT(*) FROM fact_mis_catalog
        WHERE visit_date >= ? AND visit_date <= ?
        """,
        (date_from, date_to),
    ).fetchone()[0]
    scored = conn.execute(
        f"""
        SELECT COUNT(*) FROM fact_mis_catalog
        WHERE visit_date >= ? AND visit_date <= ?
          AND {_in_analytics_sql(conn)}
        """,
        (date_from, date_to),
    ).fetchone()[0]
    found_n = int(found or 0)
    scored_n = int(scored or 0)
    return {
        "engine": ENGINE,
        "label_ru": "покрытие МИС, это не KPI склада",
        "found": found_n,
        "in_analytics": scored_n,
        "not_scored": max(found_n - scored_n, 0),
        "date_from": date_from,
        "date_to": date_to,
    }


def _public_visit(row: Mapping[str, Any], *, in_analytics: bool, source: str) -> dict[str, Any]:
    data = dict(row) if isinstance(row, sqlite3.Row) else dict(row)
    visit_id = str(data.get("visit_id") or "")
    return {
        "visit_id": visit_id,
        "visit_date": str(data.get("visit_date") or "")[:10],
        "doctor_fio": _clip(data.get("doctor_fio"), 160),
        "specialization": _clip(data.get("specialization"), 120),
        "filial": _clip(data.get("filial"), 80),
        "dx_short": _clip(data.get("dx_short"), DX_SHORT_MAX),
        "in_analytics": bool(in_analytics),
        "badge": "В аналитике" if in_analytics else "Не разобрано",
        "source": source,
        "case_id": str(data.get("case_id") or "") if in_analytics else "",
        "patient_spark": [],
    }


def _case_id_for(conn: sqlite3.Connection, visit_id: str) -> str:
    if "fact_mo_case" not in _table_names(conn):
        return ""
    row = conn.execute(
        "SELECT mis_id FROM fact_mo_case WHERE visit_id = ? LIMIT 1",
        (visit_id,),
    ).fetchone()
    if not row:
        return ""
    return str(row["mis_id"] if isinstance(row, sqlite3.Row) else row[0] or "")


def _analytics_flag(conn: sqlite3.Connection, visit_id: str) -> bool:
    if "fact_mo_case" not in _table_names(conn):
        return False
    row = conn.execute(
        "SELECT 1 FROM fact_mo_case WHERE visit_id = ? LIMIT 1",
        (visit_id,),
    ).fetchone()
    return bool(row)


def patient_visit_spark(
    conn: sqlite3.Connection, *, patient_key: str, limit: int = 12
) -> list[dict[str, Any]]:
    if not patient_key:
        return []
    rows = conn.execute(
        f"""
        SELECT visit_id, visit_date,
               CASE WHEN {_in_analytics_sql(conn)} THEN 1 ELSE 0 END AS in_analytics
        FROM fact_mis_catalog
        WHERE patient_key = ?
        ORDER BY visit_date DESC
        LIMIT ?
        """,
        (patient_key, limit),
    ).fetchall()
    return [
        {
            "visit_id": str(row["visit_id"]),
            "visit_date": str(row["visit_date"])[:10],
            "in_analytics": bool(row["in_analytics"]),
        }
        for row in rows
    ]


def search_visits(
    *,
    q: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = SEARCH_LIMIT,
    warehouse: Path | str | None = None,
    live_lookup: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    needle = str(q or "").strip()
    date_from = str(date_from or "")[:10]
    date_to = str(date_to or "")[:10]
    limit = max(1, min(int(limit or SEARCH_LIMIT), SEARCH_LIMIT))
    conn = _open_warehouse(warehouse)
    if conn is None:
        return {
            "ok": False,
            "engine": ENGINE,
            "items": [],
            "empty_reason": "склад недоступен",
            "coverage": {},
        }
    try:
        seed_catalog_from_cases(conn)
        clauses = ["1=1"]
        params: list[Any] = []
        if date_from:
            clauses.append("visit_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("visit_date <= ?")
            params.append(date_to)
        exact_id = bool(_ID_RE.match(needle))
        if exact_id:
            clauses.append("visit_id = ?")
            params.append(needle)
        elif needle:
            clauses.append(
                "(dx_short LIKE ? ESCAPE '\\' "
                "OR doctor_fio LIKE ? ESCAPE '\\' "
                "OR specialization LIKE ? ESCAPE '\\' "
                "OR filial LIKE ? ESCAPE '\\')"
            )
            like = _like_term(needle)
            params.extend([like, like, like, like])
        sql = f"""
            SELECT visit_id, visit_date, doctor_fio, specialization, filial, dx_short,
                   patient_key, {_in_analytics_sql(conn)} AS in_analytics
            FROM fact_mis_catalog
            WHERE {' AND '.join(clauses)}
            ORDER BY visit_date DESC, visit_id DESC
            LIMIT ?
        """
        rows = conn.execute(sql, [*params, limit]).fetchall()
        items = []
        for row in rows:
            public = _public_visit(row, in_analytics=bool(row["in_analytics"]), source="catalog")
            if public["in_analytics"]:
                public["case_id"] = _case_id_for(conn, public["visit_id"])
            public["patient_spark"] = patient_visit_spark(
                conn, patient_key=str(row["patient_key"] or "")
            )
            items.append(public)
        empty_reason = ""
        if not items and exact_id:
            live_row = None
            lookup = live_lookup if live_lookup is not None else lookup_mis_visit_meta
            try:
                live_row = lookup(needle)
            except Exception:
                live_row = None
            in_wh = _analytics_flag(conn, needle)
            if live_row:
                if not in_wh:
                    upsert_catalog_rows(conn, [live_row])
                public = _public_visit(
                    live_row, in_analytics=in_wh, source="live_mis"
                )
                if in_wh:
                    public["case_id"] = _case_id_for(conn, needle)
                items = [public]
            else:
                public = _public_visit(
                    {"visit_id": needle, "visit_date": date_from or date_to},
                    in_analytics=in_wh,
                    source="typed",
                )
                if in_wh:
                    public["case_id"] = _case_id_for(conn, needle)
                    items = [public]
                else:
                    empty_reason = "в каталоге нет, попробуйте visit_id"
                    items = [public]
        elif not items:
            empty_reason = (
                "в каталоге нет, попробуйте visit_id"
                if needle
                else "в каталоге нет визитов за даты фильтра"
            )
        coverage = coverage_for_window(
            conn, date_from=date_from or "0000-01-01", date_to=date_to or "9999-12-31"
        )
        return {
            "ok": True,
            "engine": ENGINE,
            "items": items,
            "empty_reason": empty_reason,
            "coverage": coverage,
            "q": needle,
        }
    finally:
        conn.close()


def coverage_payload(
    *,
    date_from: str = "",
    date_to: str = "",
    warehouse: Path | str | None = None,
) -> dict[str, Any]:
    date_from = str(date_from or "0000-01-01")[:10]
    date_to = str(date_to or "9999-12-31")[:10]
    conn = _open_warehouse(warehouse)
    if conn is None:
        return {
            "ok": False,
            "engine": ENGINE,
            "found": 0,
            "in_analytics": 0,
            "not_scored": 0,
            "label_ru": "покрытие МИС, это не KPI склада",
            "empty_reason": "склад недоступен",
        }
    try:
        out = coverage_for_window(conn, date_from=date_from, date_to=date_to)
        out["ok"] = True
        return out
    finally:
        conn.close()


def mis_dsn_available() -> bool:
    return bool((os.environ.get("KRAVIRA_DB_PASSWORD") or "").strip())


def lookup_mis_visit_meta(visit_id: str) -> dict[str, Any] | None:
    """Точный SELECT mis_data LIMIT 1. Не трогает result. None, если нет DSN."""
    if not _ID_RE.match(str(visit_id or "").strip()) or not mis_dsn_available():
        return None
    try:
        import pymysql
    except ImportError:
        return None
    pw = (os.environ.get("KRAVIRA_DB_PASSWORD") or "").strip()
    con = pymysql.connect(
        host=os.environ.get("KRAVIRA_DB_HOST") or "178.163.240.131",
        port=int(os.environ.get("KRAVIRA_DB_PORT") or 6330),
        user=os.environ.get("KRAVIRA_DB_USER") or "kravira_mc_user",
        password=pw,
        database=os.environ.get("KRAVIRA_DB_NAME") or "kravira_mc",
        charset="utf8mb4",
        connect_timeout=int(os.environ.get("MIS_DB_CONNECT_TIMEOUT") or 30),
        read_timeout=int(os.environ.get("MIS_DB_READ_TIMEOUT") or 60),
    )
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT visit_id, vdate, specialist_id, specialist_name, specialization,
                   diagnos, filial, patient_id
            FROM mis_data WHERE visit_id=%s LIMIT 1
            """,
            (int(visit_id),),
        )
        meta = cur.fetchone()
    finally:
        con.close()
    if not meta:
        return None
    return {
        "visit_id": str(meta[0]),
        "visit_date": str(meta[1] or "")[:10],
        "specialist_id": str(meta[2] or ""),
        "doctor_fio": str(meta[3] or ""),
        "specialization": str(meta[4] or ""),
        "dx_short": _clip(meta[5], DX_SHORT_MAX),
        "filial": str(meta[6] or ""),
        "patient_key": patient_key_for(meta[7]),
    }


def search_labs(
    *,
    q: str = "",
    date_from: str = "",
    date_to: str = "",
    visit_id: str = "",
    limit: int = SEARCH_LIMIT,
    lab_db: Path | str | sqlite3.Connection | None = None,
    warehouse: Path | str | None = None,
) -> dict[str, Any]:
    needle = str(q or "").strip()
    visit_id = str(visit_id or "").strip()
    if _ID_RE.match(needle) and not visit_id:
        visit_id = needle
    date_from = str(date_from or "")[:10]
    date_to = str(date_to or "")[:10]
    limit = max(1, min(int(limit or SEARCH_LIMIT), SEARCH_LIMIT))
    own = False
    conn: sqlite3.Connection | None = None
    if isinstance(lab_db, sqlite3.Connection):
        conn = lab_db
    else:
        path = Path(lab_db) if lab_db else default_lab_path()
        if path is None or not path.is_file():
            return {
                "ok": True,
                "engine": ENGINE,
                "items": [],
                "empty_reason": "тест не найден",
                "timeline": [],
            }
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        own = True
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(fact_mo_lab)").fetchall()}
        if "indicator_name" not in columns:
            return {
                "ok": True,
                "engine": ENGINE,
                "items": [],
                "empty_reason": "тест не найден",
                "timeline": [],
            }
        patient_key = ""
        if visit_id:
            wh = _open_warehouse(warehouse)
            if wh is not None:
                try:
                    row = wh.execute(
                        "SELECT patient_key FROM fact_mis_catalog WHERE visit_id = ? "
                        "UNION SELECT patient_key FROM fact_mo_case WHERE visit_id = ? LIMIT 1",
                        (visit_id, visit_id),
                    ).fetchone()
                    if row:
                        patient_key = str(row[0] or "")
                finally:
                    wh.close()
        clauses = ["1=1"]
        params: list[Any] = []
        if date_from:
            clauses.append("test_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("test_date <= ?")
            params.append(date_to)
        if patient_key:
            clauses.append("patient_key = ?")
            params.append(patient_key)
        if needle and not _ID_RE.match(needle):
            clauses.append(
                "(indicator_name LIKE ? ESCAPE '\\' OR type_name LIKE ? ESCAPE '\\')"
            )
            like = _like_term(needle)
            params.extend([like, like])
        elif needle and _ID_RE.match(needle) and not patient_key:
            return {
                "ok": True,
                "engine": ENGINE,
                "items": [],
                "empty_reason": "тест не найден",
                "timeline": [],
            }
        sql = f"""
            SELECT indicator_name, type_name, test_date, value, unit, COUNT(*) AS n
            FROM fact_mo_lab
            WHERE {' AND '.join(clauses)}
            GROUP BY indicator_name, type_name
            ORDER BY n DESC, indicator_name
            LIMIT ?
        """
        rows = conn.execute(sql, [*params, limit]).fetchall()
        items = [
            {
                "indicator_name": _clip(row["indicator_name"], 160),
                "type_name": _clip(row["type_name"], 160),
                "n": int(row["n"] or 0),
                "last_date": str(row["test_date"] or "")[:10],
                "last_value": _clip(row["value"], 40),
                "unit": _clip(row["unit"], 24),
            }
            for row in rows
        ]
        timeline: list[dict[str, Any]] = []
        if items:
            top = items[0]["indicator_name"]
            timeline = lab_timeline(
                indicator_name=top,
                date_from=date_from,
                date_to=date_to,
                patient_key=patient_key,
                conn=conn,
            )
        return {
            "ok": True,
            "engine": ENGINE,
            "items": items,
            "empty_reason": "" if items else "тест не найден",
            "timeline": timeline,
            "reference_available": False,
            "reference_note_ru": "референс в складе нет - показаны значения по датам",
        }
    except sqlite3.Error:
        return {
            "ok": True,
            "engine": ENGINE,
            "items": [],
            "empty_reason": "тест не найден",
            "timeline": [],
        }
    finally:
        if own and conn is not None:
            conn.close()


def lab_timeline(
    *,
    indicator_name: str,
    date_from: str = "",
    date_to: str = "",
    patient_key: str = "",
    conn: sqlite3.Connection,
    limit: int = 60,
) -> list[dict[str, Any]]:
    name = str(indicator_name or "").strip()
    if not name:
        return []
    clauses = ["indicator_name = ?"]
    params: list[Any] = [name]
    if date_from:
        clauses.append("test_date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("test_date <= ?")
        params.append(date_to)
    if patient_key:
        clauses.append("patient_key = ?")
        params.append(patient_key)
    rows = conn.execute(
        f"""
        SELECT test_date, value, unit
        FROM fact_mo_lab
        WHERE {' AND '.join(clauses)}
        ORDER BY test_date ASC
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    points = []
    for row in rows:
        raw = str(row["value"] or "").replace(",", ".").strip()
        try:
            num = float(raw)
        except ValueError:
            num = None
        points.append(
            {
                "test_date": str(row["test_date"] or "")[:10],
                "value": _clip(row["value"], 40),
                "numeric": num,
                "unit": _clip(row["unit"], 24),
                "ref_low": None,
                "ref_high": None,
            }
        )
    return points


_JOB_ERROR_RU: dict[str, str] = {
    "visit_not_found_in_mis_protocol": (
        "В МИС по этому визиту нет консультативного заключения: это процедура, анализ "
        "или пустой документ, оценивать нечего."
    ),
    "docker_score_failed": "Оценка внутри protocol-web не выполнилась; см. лог очереди на GCE.",
    "mis_unavailable": "МИС недоступна с GCE; повторите позже.",
}


def job_error_ru(error: str) -> str:
    """Человеческое объяснение кода ошибки очереди (код:детали → текст)."""
    code = str(error or "").split(":", 1)[0].strip()
    if not code:
        return ""
    for key, text in _JOB_ERROR_RU.items():
        if code == key or code.startswith(key):
            return text
    return "Разбор визита не удался: " + str(error)[:160]


def job_status_ru(status: str, *, age_sec: float | None = None) -> str:
    if status == "queued":
        if age_sec is not None and age_sec > 120:
            return "В очереди дольше 2 минут: воркер очереди на GCE не работает или МИС недоступна."
        return "В очереди: воркер заберёт визит в течение нескольких секунд."
    if status == "running":
        return "Выгружаем из МИС и оцениваем."
    if status == "done":
        return "Визит в аналитике."
    if status == "error":
        return "Ошибка."
    return ""


def _job_public(row: Mapping[str, Any] | sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    status = str(data.get("status") or "")
    created = str(data.get("created_at") or "")
    age_sec: float | None = None
    if created:
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_sec = max(0.0, (datetime.now(timezone.utc) - created_dt).total_seconds())
        except ValueError:
            age_sec = None
    error = str(data.get("error") or "")
    return {
        "job_id": str(data.get("job_id") or ""),
        "visit_id": str(data.get("visit_id") or ""),
        "status": status,
        "status_ru": job_status_ru(status, age_sec=age_sec),
        "case_id": str(data.get("case_id") or ""),
        "error": error,
        "error_ru": job_error_ru(error) if status == "error" else "",
        "age_sec": int(age_sec) if age_sec is not None else None,
        "created_at": created,
        "updated_at": str(data.get("updated_at") or ""),
        "engine": ENGINE,
    }


def enqueue_ingest_job(
    *,
    visit_id: str,
    actor: str = "",
    warehouse: Path | str | None = None,
) -> dict[str, Any]:
    vid = str(visit_id or "").strip()
    if not _ID_RE.match(vid):
        raise ValueError("Нужен точный visit_id.")
    conn = _open_warehouse(warehouse)
    if conn is None:
        raise RuntimeError("склад недоступен")
    try:
        if _analytics_flag(conn, vid):
            case_id = _case_id_for(conn, vid)
            return {
                "job_id": "",
                "visit_id": vid,
                "status": "done",
                "case_id": case_id,
                "error": "",
                "already_in_analytics": True,
                "engine": ENGINE,
            }
        queued_n = conn.execute(
            "SELECT COUNT(*) FROM mo_ingest_job WHERE status IN ('queued', 'running')"
        ).fetchone()[0]
        if int(queued_n or 0) >= MAX_QUEUED_JOBS:
            raise RuntimeError("Очередь ingest полна. Дождитесь текущих заданий.")
        running = conn.execute(
            "SELECT job_id FROM mo_ingest_job WHERE status = 'running' LIMIT 1"
        ).fetchone()
        job_id = uuid.uuid4().hex[:12]
        now = _utc_now()
        status = "queued"
        conn.execute(
            """
            INSERT INTO mo_ingest_job(
              job_id, visit_id, status, actor, case_id, error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, '', '', ?, ?)
            """,
            (job_id, vid, status, (actor or "")[:80], now, now),
        )
        conn.commit()
        _log.info("mo_ingest_enqueued visit_id=%s job_id=%s running=%s", vid, job_id, bool(running))
        return get_ingest_job(job_id, warehouse=warehouse) or {
            "job_id": job_id,
            "visit_id": vid,
            "status": status,
            "engine": ENGINE,
        }
    finally:
        conn.close()


def get_ingest_job(job_id: str, *, warehouse: Path | str | None = None) -> dict[str, Any] | None:
    conn = _open_warehouse(warehouse)
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT * FROM mo_ingest_job WHERE job_id = ? LIMIT 1",
            (str(job_id),),
        ).fetchone()
        return _job_public(row) if row else None
    finally:
        conn.close()


def update_ingest_job(
    job_id: str,
    *,
    status: str,
    error: str = "",
    case_id: str = "",
    warehouse: Path | str | None = None,
) -> None:
    conn = _open_warehouse(warehouse)
    if conn is None:
        return
    try:
        conn.execute(
            """
            UPDATE mo_ingest_job
            SET status = ?, error = ?, case_id = COALESCE(NULLIF(?, ''), case_id),
                updated_at = ?
            WHERE job_id = ?
            """,
            (status, (error or "")[:300], case_id, _utc_now(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def next_queued_job(*, warehouse: Path | str | None = None) -> dict[str, Any] | None:
    with _JOB_LOCK:
        conn = _open_warehouse(warehouse)
        if conn is None:
            return None
        try:
            running = conn.execute(
                "SELECT job_id FROM mo_ingest_job WHERE status = 'running' LIMIT 1"
            ).fetchone()
            if running:
                return None
            row = conn.execute(
                """
                SELECT * FROM mo_ingest_job
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            job_id = str(row["job_id"])
            conn.execute(
                "UPDATE mo_ingest_job SET status = 'running', updated_at = ? WHERE job_id = ?",
                (_utc_now(), job_id),
            )
            conn.commit()
            return _job_public(conn.execute(
                "SELECT * FROM mo_ingest_job WHERE job_id = ?", (job_id,)
            ).fetchone())
        finally:
            conn.close()


def mark_job_done_from_warehouse(visit_id: str, *, warehouse: Path | str | None = None) -> str:
    conn = _open_warehouse(warehouse)
    if conn is None:
        return ""
    try:
        return _case_id_for(conn, visit_id)
    finally:
        conn.close()


def process_next_ingest_job(
    *,
    warehouse: Path | str | None = None,
    ingest_fn: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    """Снять одно queued-задание. ingest_fn подставляют тесты; на GCE - CLI/host."""
    job = next_queued_job(warehouse=warehouse)
    if not job:
        return None
    job_id = job["job_id"]
    visit_id = job["visit_id"]
    try:
        fn = ingest_fn
        if fn is None:
            from scripts.ingest_mo_visit_from_mis import ingest_visit as fn
        root = Path(os.environ.get("MO_DATA_ROOT") or "/var/data/medical_exams")
        fn(visit_id, data_root=root)
        case_id = mark_job_done_from_warehouse(visit_id, warehouse=warehouse)
        update_ingest_job(job_id, status="done", case_id=case_id, warehouse=warehouse)
        _log.info("mo_ingest_done visit_id=%s job_id=%s", visit_id, job_id)
        return {
            "job_id": job_id,
            "visit_id": visit_id,
            "status": "done",
            "case_id": case_id,
        }
    except LookupError as exc:
        update_ingest_job(job_id, status="error", error=str(exc)[:280], warehouse=warehouse)
        _log.info("mo_ingest_error visit_id=%s job_id=%s", visit_id, job_id)
        return {"job_id": job_id, "visit_id": visit_id, "status": "error"}
    except Exception as exc:
        update_ingest_job(job_id, status="error", error=str(exc)[:280], warehouse=warehouse)
        _log.info("mo_ingest_error visit_id=%s job_id=%s", visit_id, job_id)
        return {"job_id": job_id, "visit_id": visit_id, "status": "error"}
