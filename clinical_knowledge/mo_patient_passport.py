"""Логический паспорт клиента на patient_key (не файл на человека).

Канон: docs/plans/2026-09-27-mo-client-passport-score-ui-v2.md волны P0/P1.
Тексты КЗ и ФИО не кладём. Дубль режет пара (patient_key, visit_id).
Оценку визита этот модуль не запускает.
Карты source=mis переживают пересборку склада.
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
        CREATE TABLE IF NOT EXISTS fact_mo_lab_coverage (
          patient_key TEXT PRIMARY KEY,
          n_lab_rows INTEGER NOT NULL DEFAULT 0,
          n_lab_dates INTEGER NOT NULL DEFAULT 0,
          first_date TEXT,
          last_date TEXT,
          updated_at TEXT NOT NULL
        );
        """
    )
    cols = _table_columns(db, "fact_mo_visit_index")
    if "source" not in cols:
        db.execute("ALTER TABLE fact_mo_visit_index ADD COLUMN source TEXT")


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


def _coverage_counts(
    db: sqlite3.Connection,
    keys: Sequence[str] | None,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    tables = {str(r[0]) for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "fact_mo_lab_coverage" not in tables:
        return out
    sql = (
        "SELECT patient_key, n_lab_rows, n_lab_dates "
        "FROM fact_mo_lab_coverage WHERE TRIM(COALESCE(patient_key,'')) != ''"
    )
    params: list[Any] = []
    if keys:
        placeholders = ",".join("?" for _ in keys)
        sql += f" AND patient_key IN ({placeholders})"
        params.extend(keys)
    for row in db.execute(sql, params):
        out[str(row[0])] = {
            "n_lab_rows": int(row[1] or 0),
            "n_lab_dates": int(row[2] or 0),
        }
    return out


def _merge_lab_maps(
    *maps: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for mapping in maps:
        for key, rec in mapping.items():
            cur = out.setdefault(key, {"n_lab_rows": 0, "n_lab_dates": 0})
            cur["n_lab_rows"] = max(cur["n_lab_rows"], int(rec.get("n_lab_rows") or 0))
            cur["n_lab_dates"] = max(cur["n_lab_dates"], int(rec.get("n_lab_dates") or 0))
    return out


def _load_index_cards(
    db: sqlite3.Connection,
    keys: Sequence[str] | None,
) -> list[dict[str, Any]]:
    sql = (
        "SELECT patient_key, visit_id, visit_date, specialty, diagnosis_code, "
        "dx_label, overall_grade, document_kind, mis_id, doctor_key "
        "FROM fact_mo_visit_index WHERE TRIM(COALESCE(patient_key,'')) != ''"
    )
    params: list[Any] = []
    if keys:
        placeholders = ",".join("?" for _ in keys)
        sql += f" AND patient_key IN ({placeholders})"
        params.extend(keys)
    db.row_factory = sqlite3.Row
    return [dict(row) for row in db.execute(sql, params)]


def _refresh_aggregates_on(
    db: sqlite3.Connection,
    *,
    keys: Sequence[str] | None,
    lab_path: Path | None,
    now: str,
) -> None:
    cards = _load_index_cards(db, keys)
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        pk = _norm(card.get("patient_key"))
        if not pk:
            continue
        card["visit_date"] = _norm(card.get("visit_date"))[:10]
        card["specialty"] = _norm(card.get("specialty"))
        card["diagnosis_code"] = _norm(card.get("diagnosis_code"))
        card["overall_grade"] = _norm(card.get("overall_grade"))
        card["visit_id"] = _norm(card.get("visit_id"))
        by_patient.setdefault(pk, []).append(card)
    labs = _merge_lab_maps(_lab_counts(lab_path, keys), _coverage_counts(db, keys))
    if keys:
        placeholders = ",".join("?" for _ in keys)
        db.execute(
            f"DELETE FROM fact_mo_patient_specialty WHERE patient_key IN ({placeholders})",
            keys,
        )
    else:
        db.execute("DELETE FROM fact_mo_patient_specialty")
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
    known = set(by_patient)
    for patient_key, lab in labs.items():
        if patient_key in known:
            continue
        if keys is not None and patient_key not in set(keys):
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
    if keys is None:
        keep = sorted(set(by_patient) | set(labs))
        if keep:
            placeholders = ",".join("?" for _ in keep)
            db.execute(
                f"DELETE FROM fact_mo_patient_passport WHERE patient_key NOT IN ({placeholders})",
                keep,
            )
        else:
            db.execute("DELETE FROM fact_mo_patient_passport")


def _dedup_mis_cards(cards: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in cards:
        pk = _norm(raw.get("patient_key"))
        vid = _norm(raw.get("visit_id"))
        if not pk or not vid:
            continue
        card = {
            "patient_key": pk,
            "visit_id": vid,
            "visit_date": _norm(raw.get("visit_date"))[:10],
            "specialty": _norm(raw.get("specialty")),
            "diagnosis_code": _norm(raw.get("diagnosis_code")),
            "dx_label": _dx_label(raw.get("dx_label") or raw.get("diagnosis_text")),
            "document_kind": _norm(raw.get("document_kind")) or "mis_index",
            "mis_id": _norm(raw.get("mis_id")),
        }
        prev = by_pair.get((pk, vid))
        if prev is None:
            by_pair[(pk, vid)] = card
            continue
        score = (
            1 if card["specialty"] else 0,
            1 if card["diagnosis_code"] else 0,
            1 if card["dx_label"] else 0,
            card["visit_date"],
            card["mis_id"],
        )
        prev_score = (
            1 if prev["specialty"] else 0,
            1 if prev["diagnosis_code"] else 0,
            1 if prev["dx_label"] else 0,
            prev["visit_date"],
            prev["mis_id"],
        )
        if score > prev_score:
            by_pair[(pk, vid)] = card
    return list(by_pair.values())


def merge_mis_visit_cards(
    warehouse: Path,
    cards: Sequence[Mapping[str, Any]],
    *,
    lab_path: Path | None = None,
    refresh: bool = True,
) -> dict[str, Any]:
    """Добавить карты МИС. Оценку и складские поля не затирает."""
    now = _now()
    prepared = _dedup_mis_cards(cards)
    with sqlite3.connect(str(warehouse)) as db:
        ensure_passport_schema(db)
        db.execute("DROP TABLE IF EXISTS tmp_mis_idx")
        db.execute(
            """
            CREATE TEMP TABLE tmp_mis_idx (
              patient_key TEXT NOT NULL,
              visit_id TEXT NOT NULL,
              visit_date TEXT,
              specialty TEXT,
              diagnosis_code TEXT,
              dx_label TEXT,
              document_kind TEXT,
              mis_id TEXT,
              PRIMARY KEY (patient_key, visit_id)
            )
            """
        )
        db.executemany(
            """
            INSERT OR REPLACE INTO tmp_mis_idx
            VALUES (?,?,?,?,?,?,?,?)
            """,
            [
                (
                    c["patient_key"],
                    c["visit_id"],
                    c["visit_date"] or None,
                    c["specialty"] or None,
                    c["diagnosis_code"] or None,
                    c["dx_label"] or None,
                    c["document_kind"] or None,
                    c["mis_id"] or None,
                )
                for c in prepared
            ],
        )
        before = int(db.execute("SELECT COUNT(*) FROM fact_mo_visit_index").fetchone()[0])
        db.execute(
            """
            INSERT OR IGNORE INTO fact_mo_visit_index(
              patient_key, visit_id, visit_date, specialty, diagnosis_code, dx_label,
              overall_grade, document_kind, mis_id, doctor_key, updated_at, source
            )
            SELECT patient_key, visit_id, visit_date, specialty, diagnosis_code, dx_label,
                   NULL, document_kind, mis_id, NULL, ?, 'mis'
              FROM tmp_mis_idx
            """,
            (now,),
        )
        db.execute(
            """
            UPDATE fact_mo_visit_index
               SET visit_date = COALESCE(NULLIF(visit_date, ''), (
                        SELECT visit_date FROM tmp_mis_idx t
                         WHERE t.patient_key = fact_mo_visit_index.patient_key
                           AND t.visit_id = fact_mo_visit_index.visit_id
                   )),
                   specialty = COALESCE(NULLIF(specialty, ''), (
                        SELECT specialty FROM tmp_mis_idx t
                         WHERE t.patient_key = fact_mo_visit_index.patient_key
                           AND t.visit_id = fact_mo_visit_index.visit_id
                   )),
                   diagnosis_code = COALESCE(NULLIF(diagnosis_code, ''), (
                        SELECT diagnosis_code FROM tmp_mis_idx t
                         WHERE t.patient_key = fact_mo_visit_index.patient_key
                           AND t.visit_id = fact_mo_visit_index.visit_id
                   )),
                   dx_label = COALESCE(NULLIF(dx_label, ''), (
                        SELECT dx_label FROM tmp_mis_idx t
                         WHERE t.patient_key = fact_mo_visit_index.patient_key
                           AND t.visit_id = fact_mo_visit_index.visit_id
                   )),
                   updated_at = ?
             WHERE EXISTS (
                SELECT 1 FROM tmp_mis_idx t
                 WHERE t.patient_key = fact_mo_visit_index.patient_key
                   AND t.visit_id = fact_mo_visit_index.visit_id
             )
            """,
            (now,),
        )
        after = int(db.execute("SELECT COUNT(*) FROM fact_mo_visit_index").fetchone()[0])
        touched = sorted({c["patient_key"] for c in prepared})
        if refresh:
            _refresh_aggregates_on(db, keys=touched or None, lab_path=lab_path, now=now)
        db.commit()
        n_passports = db.execute("SELECT COUNT(*) FROM fact_mo_patient_passport").fetchone()[0]
    return {
        "ok": True,
        "engine": ENGINE,
        "inserted_cards": after - before,
        "merged_cards": len(prepared),
        "touched_keys": len(touched),
        "passports": int(n_passports),
        "visit_cards": after,
        "rebuilt_at": now,
    }


def merge_mis_lab_coverage(
    warehouse: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    lab_path: Path | None = None,
    refresh: bool = True,
) -> dict[str, Any]:
    """Добор покрытия анализов без значений и без PHI."""
    now = _now()
    prepared: list[tuple[Any, ...]] = []
    keys: list[str] = []
    for raw in rows:
        pk = _norm(raw.get("patient_key"))
        if not pk:
            continue
        keys.append(pk)
        prepared.append(
            (
                pk,
                int(raw.get("n_lab_rows") or 0),
                int(raw.get("n_lab_dates") or 0),
                _norm(raw.get("first_date"))[:10] or None,
                _norm(raw.get("last_date"))[:10] or None,
                now,
            )
        )
    with sqlite3.connect(str(warehouse)) as db:
        ensure_passport_schema(db)
        db.executemany(
            """
            INSERT INTO fact_mo_lab_coverage(
              patient_key, n_lab_rows, n_lab_dates, first_date, last_date, updated_at
            ) VALUES (?,?,?,?,?,?)
            ON CONFLICT(patient_key) DO UPDATE SET
              n_lab_rows = CASE
                WHEN excluded.n_lab_rows > n_lab_rows THEN excluded.n_lab_rows
                ELSE n_lab_rows
              END,
              n_lab_dates = CASE
                WHEN excluded.n_lab_dates > n_lab_dates THEN excluded.n_lab_dates
                ELSE n_lab_dates
              END,
              first_date = CASE
                WHEN excluded.first_date IS NULL THEN first_date
                WHEN first_date IS NULL THEN excluded.first_date
                WHEN excluded.first_date < first_date THEN excluded.first_date
                ELSE first_date
              END,
              last_date = CASE
                WHEN excluded.last_date IS NULL THEN last_date
                WHEN last_date IS NULL THEN excluded.last_date
                WHEN excluded.last_date > last_date THEN excluded.last_date
                ELSE last_date
              END,
              updated_at = excluded.updated_at
            """,
            prepared,
        )
        if refresh:
            _refresh_aggregates_on(db, keys=sorted(set(keys)) or None, lab_path=lab_path, now=now)
        db.commit()
        n_cov = db.execute("SELECT COUNT(*) FROM fact_mo_lab_coverage").fetchone()[0]
        n_passports = db.execute("SELECT COUNT(*) FROM fact_mo_patient_passport").fetchone()[0]
        n_lab_pass = db.execute(
            "SELECT COUNT(*) FROM fact_mo_patient_passport WHERE n_lab_rows > 0 OR n_lab_dates > 0"
        ).fetchone()[0]
    return {
        "ok": True,
        "engine": ENGINE,
        "coverage_rows": int(n_cov),
        "lab_passports": int(n_lab_pass),
        "passports": int(n_passports),
        "merged": len(prepared),
        "rebuilt_at": now,
    }


def refresh_passport_aggregates(
    warehouse: Path,
    *,
    lab_path: Path | None = None,
    patient_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Пересчитать сводки, не трогая индекс визитов."""
    keys = sorted({_norm(k) for k in (patient_keys or []) if _norm(k)}) or None
    now = _now()
    with sqlite3.connect(str(warehouse)) as db:
        ensure_passport_schema(db)
        _refresh_aggregates_on(db, keys=keys, lab_path=lab_path, now=now)
        db.commit()
        n_passports = db.execute("SELECT COUNT(*) FROM fact_mo_patient_passport").fetchone()[0]
        n_visits = db.execute("SELECT COUNT(*) FROM fact_mo_visit_index").fetchone()[0]
        n_specs = db.execute("SELECT COUNT(*) FROM fact_mo_patient_specialty").fetchone()[0]
        n_lab = db.execute(
            "SELECT COUNT(*) FROM fact_mo_patient_passport WHERE n_lab_rows > 0 OR n_lab_dates > 0"
        ).fetchone()[0]
    return {
        "ok": True,
        "engine": ENGINE,
        "passports": int(n_passports),
        "visit_cards": int(n_visits),
        "specialty_rows": int(n_specs),
        "lab_passports": int(n_lab),
        "rebuilt_at": now,
    }


def rebuild_passports(
    warehouse: Path,
    *,
    lab_path: Path | None = None,
    patient_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Собрать индекс визитов со склада. Карты source=mis не удаляет."""
    keys = sorted({_norm(k) for k in (patient_keys or []) if _norm(k)}) or None
    now = _now()
    cards_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with sqlite3.connect(str(warehouse)) as db:
        ensure_passport_schema(db)
        if keys:
            placeholders = ",".join("?" for _ in keys)
            db.execute(
                f"""
                DELETE FROM fact_mo_visit_index
                 WHERE patient_key IN ({placeholders})
                   AND COALESCE(source, 'warehouse') != 'mis'
                """,
                keys,
            )
        else:
            db.execute(
                "DELETE FROM fact_mo_visit_index WHERE COALESCE(source, 'warehouse') != 'mis'"
            )
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
              overall_grade, document_kind, mis_id, doctor_key, updated_at, source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
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
                    "warehouse",
                )
                for c in cards
            ],
        )
        _refresh_aggregates_on(db, keys=keys, lab_path=lab_path, now=now)
        db.commit()
        n_passports = db.execute("SELECT COUNT(*) FROM fact_mo_patient_passport").fetchone()[0]
        n_visits = db.execute("SELECT COUNT(*) FROM fact_mo_visit_index").fetchone()[0]
        n_specs = db.execute("SELECT COUNT(*) FROM fact_mo_patient_specialty").fetchone()[0]
        n_mis = db.execute(
            "SELECT COUNT(*) FROM fact_mo_visit_index WHERE source = 'mis'"
        ).fetchone()[0]
    return {
        "ok": True,
        "engine": ENGINE,
        "passports": int(n_passports),
        "visit_cards": int(n_visits),
        "specialty_rows": int(n_specs),
        "mis_cards_kept": int(n_mis),
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
