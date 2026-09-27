"""Логический паспорт клиента на patient_key (не файл на человека).

Канон: docs/plans/2026-09-27-mo-client-passport-score-ui-v2.md волна P0.
Тексты КЗ и ФИО не кладём. Дубль режет пара (patient_key, visit_id).
Оценку визита этот модуль не запускает.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ENGINE = "mo_patient_passport_v1"
DX_LABEL_MAX = 80

KIND_RANK = {
    "clinical_visit": 0,
    "consultation": 1,
}


def passport_enabled() -> bool:
    raw = (os.environ.get("MO_PASSPORT") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def default_warehouse_path() -> Path | None:
    from clinical_knowledge.mo_patient_history_bundle import default_warehouse_path as _wh

    return _wh()


def default_lab_path(warehouse: Path | None = None) -> Path | None:
    roots: list[Path] = []
    env = (os.environ.get("MO_DATA_ROOT") or "").strip()
    if env:
        roots.append(Path(env))
    if warehouse is not None:
        roots.append(warehouse.parent.parent)
    roots.append(Path("/var/data/medical_exams"))
    for root in roots:
        path = root / "warehouse" / "mo_lab.sqlite"
        if path.is_file():
            return path
    return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _dx_label(value: Any) -> str:
    text = " ".join(_norm(value).split())
    if len(text) <= DX_LABEL_MAX:
        return text
    return text[: DX_LABEL_MAX - 1].rstrip() + "..."


def _kind_rank(kind: str) -> int:
    return KIND_RANK.get(kind, 9)


def ensure_passport_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS fact_mo_visit_index (
          patient_key TEXT NOT NULL,
          visit_id TEXT NOT NULL,
          visit_date TEXT,
          specialty TEXT,
          diagnosis_code TEXT,
          dx_label TEXT,
          overall_grade TEXT,
          document_kind TEXT,
          mis_id TEXT,
          doctor_key TEXT,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (patient_key, visit_id)
        );
        CREATE INDEX IF NOT EXISTS idx_visit_index_date
          ON fact_mo_visit_index(patient_key, visit_date);
        CREATE TABLE IF NOT EXISTS fact_mo_patient_specialty (
          patient_key TEXT NOT NULL,
          specialty TEXT NOT NULL,
          n_visits INTEGER NOT NULL,
          last_date TEXT,
          last_icd TEXT,
          last_grade TEXT,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (patient_key, specialty)
        );
        CREATE TABLE IF NOT EXISTS fact_mo_patient_passport (
          patient_key TEXT PRIMARY KEY,
          n_visits INTEGER NOT NULL,
          n_specialties INTEGER NOT NULL,
          first_date TEXT,
          last_date TEXT,
          n_lab_rows INTEGER NOT NULL DEFAULT 0,
          n_lab_dates INTEGER NOT NULL DEFAULT 0,
          rebuilt_at TEXT NOT NULL,
          engine TEXT NOT NULL
        );
        """
    )


def _table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in db.execute(f"PRAGMA table_info({table})")}


def _pick_visit_card(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def _key(row: Mapping[str, Any]) -> tuple[int, str, str]:
        return (
            -_kind_rank(_norm(row.get("document_kind"))),
            _norm(row.get("visit_date")),
            _norm(row.get("mis_id")),
        )

    winner = max(rows, key=_key)
    return {
        "patient_key": _norm(winner.get("patient_key")),
        "visit_id": _norm(winner.get("visit_id")),
        "visit_date": _norm(winner.get("visit_date"))[:10],
        "specialty": _norm(winner.get("specialty")),
        "diagnosis_code": _norm(winner.get("diagnosis_code")),
        "dx_label": _dx_label(winner.get("diagnosis_text") or winner.get("dx_label")),
        "overall_grade": _norm(winner.get("overall_grade")),
        "document_kind": _norm(winner.get("document_kind")),
        "mis_id": _norm(winner.get("mis_id")),
        "doctor_key": _norm(winner.get("doctor_key")),
    }


def _iter_case_rows(
    db: sqlite3.Connection,
    keys: Sequence[str] | None,
) -> list[dict[str, Any]]:
    cols = _table_columns(db, "fact_mo_case")
    if "patient_key" not in cols or "visit_id" not in cols:
        return []
    select = [
        "patient_key",
        "visit_id",
        "visit_date",
        "specialty",
        "document_kind",
        "mis_id",
        "doctor_key",
    ]
    select.append("diagnosis_code" if "diagnosis_code" in cols else "NULL AS diagnosis_code")
    select.append("diagnosis_text" if "diagnosis_text" in cols else "NULL AS diagnosis_text")
    select.append("overall_grade" if "overall_grade" in cols else "NULL AS overall_grade")
    sql = (
        "SELECT "
        + ", ".join(select)
        + " FROM fact_mo_case WHERE TRIM(COALESCE(patient_key,'')) != ''"
        " AND TRIM(COALESCE(visit_id,'')) != ''"
    )
    params: list[Any] = []
    if keys:
        placeholders = ",".join("?" for _ in keys)
        sql += f" AND patient_key IN ({placeholders})"
        params.extend(keys)
    db.row_factory = sqlite3.Row
    return [dict(row) for row in db.execute(sql, params)]


def _lab_counts(
    lab_path: Path | None,
    keys: Sequence[str] | None,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if lab_path is None or not lab_path.is_file():
        return out
    with sqlite3.connect(str(lab_path)) as db:
        tables = {str(r[0]) for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "fact_mo_lab" not in tables:
            return out
        sql = (
            "SELECT patient_key, COUNT(*) AS n_rows, COUNT(DISTINCT test_date) AS n_dates "
            "FROM fact_mo_lab WHERE TRIM(COALESCE(patient_key,'')) != ''"
        )
        params: list[Any] = []
        if keys:
            placeholders = ",".join("?" for _ in keys)
            sql += f" AND patient_key IN ({placeholders})"
            params.extend(keys)
        sql += " GROUP BY patient_key"
        for row in db.execute(sql, params):
            out[str(row[0])] = {"n_lab_rows": int(row[1] or 0), "n_lab_dates": int(row[2] or 0)}
    return out


def rebuild_passports(
    warehouse: Path,
    *,
    lab_path: Path | None = None,
    patient_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Собрать индекс визитов и паспорта со склада. Без PHI в результате."""
    keys = sorted({_norm(k) for k in (patient_keys or []) if _norm(k)}) or None
    now = _now()
    cards_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with sqlite3.connect(str(warehouse)) as db:
        ensure_passport_schema(db)
        if keys:
            placeholders = ",".join("?" for _ in keys)
            for table in (
                "fact_mo_visit_index",
                "fact_mo_patient_specialty",
                "fact_mo_patient_passport",
            ):
                db.execute(f"DELETE FROM {table} WHERE patient_key IN ({placeholders})", keys)
        else:
            db.execute("DELETE FROM fact_mo_visit_index")
            db.execute("DELETE FROM fact_mo_patient_specialty")
            db.execute("DELETE FROM fact_mo_patient_passport")
        for row in _iter_case_rows(db, keys):
            pair = (_norm(row.get("patient_key")), _norm(row.get("visit_id")))
            if not pair[0] or not pair[1]:
                continue
            cards_by_pair.setdefault(pair, []).append(row)
        cards = [_pick_visit_card(group) for group in cards_by_pair.values()]
        db.executemany(
            """
            INSERT OR REPLACE INTO fact_mo_visit_index(
              patient_key, visit_id, visit_date, specialty, diagnosis_code, dx_label,
              overall_grade, document_kind, mis_id, doctor_key, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    c["patient_key"],
                    c["visit_id"],
                    c["visit_date"] or None,
                    c["specialty"] or None,
                    c["diagnosis_code"] or None,
                    c["dx_label"] or None,
                    c["overall_grade"] or None,
                    c["document_kind"] or None,
                    c["mis_id"] or None,
                    c["doctor_key"] or None,
                    now,
                )
                for c in cards
            ],
        )
        by_patient: dict[str, list[dict[str, Any]]] = {}
        for card in cards:
            by_patient.setdefault(card["patient_key"], []).append(card)
        labs = _lab_counts(lab_path, keys)
        specialty_rows: list[tuple[Any, ...]] = []
        passport_rows: list[tuple[Any, ...]] = []
        for patient_key, visits in by_patient.items():
            visits_sorted = sorted(visits, key=lambda c: (c["visit_date"], c["visit_id"]))
            by_spec: dict[str, list[dict[str, Any]]] = {}
            for card in visits_sorted:
                spec = card["specialty"] or "не указана"
                by_spec.setdefault(spec, []).append(card)
            for spec, spec_visits in by_spec.items():
                last = spec_visits[-1]
                specialty_rows.append(
                    (
                        patient_key,
                        spec,
                        len(spec_visits),
                        last["visit_date"] or None,
                        last["diagnosis_code"] or None,
                        last["overall_grade"] or None,
                        now,
                    )
                )
            lab = labs.get(patient_key) or {"n_lab_rows": 0, "n_lab_dates": 0}
            passport_rows.append(
                (
                    patient_key,
                    len(visits_sorted),
                    len(by_spec),
                    visits_sorted[0]["visit_date"] or None,
                    visits_sorted[-1]["visit_date"] or None,
                    lab["n_lab_rows"],
                    lab["n_lab_dates"],
                    now,
                    ENGINE,
                )
            )
        # Клиенты только с анализами, без визитов склада.
        if labs:
            known = set(by_patient)
            for patient_key, lab in labs.items():
                if patient_key in known:
                    continue
                passport_rows.append(
                    (
                        patient_key,
                        0,
                        0,
                        None,
                        None,
                        lab["n_lab_rows"],
                        lab["n_lab_dates"],
                        now,
                        ENGINE,
                    )
                )
        db.executemany(
            """
            INSERT OR REPLACE INTO fact_mo_patient_specialty(
              patient_key, specialty, n_visits, last_date, last_icd, last_grade, updated_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            specialty_rows,
        )
        db.executemany(
            """
            INSERT OR REPLACE INTO fact_mo_patient_passport(
              patient_key, n_visits, n_specialties, first_date, last_date,
              n_lab_rows, n_lab_dates, rebuilt_at, engine
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            passport_rows,
        )
        db.commit()
        n_passports = db.execute("SELECT COUNT(*) FROM fact_mo_patient_passport").fetchone()[0]
        n_visits = db.execute("SELECT COUNT(*) FROM fact_mo_visit_index").fetchone()[0]
        n_specs = db.execute("SELECT COUNT(*) FROM fact_mo_patient_specialty").fetchone()[0]
    return {
        "ok": True,
        "engine": ENGINE,
        "passports": int(n_passports),
        "visit_cards": int(n_visits),
        "specialty_rows": int(n_specs),
        "keys": len(keys) if keys else None,
        "lab_attached": bool(lab_path and Path(lab_path).is_file()),
        "rebuilt_at": now,
    }


def public_passport(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Сводка без patient_id и ФИО."""
    rec = dict(row or {})
    return {
        "engine": ENGINE,
        "n_visits": int(rec.get("n_visits") or 0),
        "n_specialties": int(rec.get("n_specialties") or 0),
        "first_date": rec.get("first_date") or None,
        "last_date": rec.get("last_date") or None,
        "n_lab_rows": int(rec.get("n_lab_rows") or 0),
        "n_lab_dates": int(rec.get("n_lab_dates") or 0),
        "rebuilt_at": rec.get("rebuilt_at") or None,
    }
