"""Login and dashboard must wait on a busy warehouse, not 500 after 5 s."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from clinical_knowledge import mo_app_accounts, mo_backend
from clinical_knowledge.mo_daily import (
    WAREHOUSE_BUSY_TIMEOUT_MS,
    connect_warehouse,
    initialize_warehouse,
    upsert_warehouse,
)
from tests.test_mo_zone_scores import _rich_clinical


def test_connect_warehouse_sets_busy_timeout(tmp_path: Path) -> None:
    path = tmp_path / "mo.sqlite"
    initialize_warehouse(path)
    conn = connect_warehouse(path)
    try:
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        assert int(row[0]) >= WAREHOUSE_BUSY_TIMEOUT_MS
    finally:
        conn.close()


def test_backend_connect_skips_ddl_on_same_path(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    initialize_warehouse(db)
    mo_backend._WAREHOUSE_SCHEMA_READY_PATH = None
    first = mo_backend._connect()
    first.close()
    calls = {"n": 0}
    orig = mo_backend.initialize_warehouse

    def wrapped(path):
        calls["n"] += 1
        return orig(path)

    monkeypatch.setattr(mo_backend, "initialize_warehouse", wrapped)
    second = mo_backend._connect()
    second.close()
    assert calls["n"] == 0


def test_login_waits_out_short_exclusive_lock(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    initialize_warehouse(db)
    mo_backend._WAREHOUSE_SCHEMA_READY_PATH = None
    mo_app_accounts._APP_SCHEMA_READY_PATH = None
    mo_app_accounts.upsert_user(
        login="lockwait",
        password="secret-pass-1",
        display_name="Lock Wait",
        role="methodist",
        mo_access="full",
    )

    holder = sqlite3.connect(str(db), timeout=60, check_same_thread=False)
    holder.execute("PRAGMA busy_timeout=60000")
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("SELECT 1").fetchone()

    def unlock() -> None:
        time.sleep(0.5)
        holder.commit()
        holder.close()

    threading.Thread(target=unlock, daemon=True).start()
    started = time.monotonic()
    session = mo_app_accounts.login_user(login="lockwait", password="secret-pass-1")
    elapsed = time.monotonic() - started
    assert session["ok"] is True
    assert elapsed >= 0.4
    assert elapsed < 15


def test_upsert_writes_history_cache_after_suggest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MO_WAREHOUSE_PROTOCOL_SUGGEST", "1")
    order: list[str] = []

    def fake_suggest(**kwargs):
        order.append("suggest")
        return {
            "ok": True,
            "items": [
                {
                    "title": "Острый тонзиллит",
                    "match_kind": "clinical",
                    "score": 77,
                    "protocol_id": "live_t1",
                }
            ],
        }

    import clinical_knowledge.mo_patient_history_bundle as history_mod

    orig_cache = history_mod.upsert_history_cache

    def wrapped_cache(*args, **kwargs):
        order.append("cache")
        return orig_cache(*args, **kwargs)

    monkeypatch.setattr(
        "clinical_knowledge.case_protocol_suggest.suggest_protocols_for_mo_case",
        fake_suggest,
    )
    monkeypatch.setattr(history_mod, "upsert_history_cache", wrapped_cache)
    path = tmp_path / "wh.sqlite"
    initialize_warehouse(path)
    raw_rows = [
        {
            "id": "1",
            "visit_id": "v1",
            "visit_date": "2026-08-02",
            "document_kind": "clinical_visit",
            "doctor_fio": "Иванов И.И.",
            "doctor_specialization": "Терапия",
            "filial": "1",
            "patient_id": "p1",
            "visit_time": "10:15",
            **_rich_clinical(),
            "mkb_code_main": "J03.9",
        }
    ]
    cases = [
        {
            "mis_id": "1",
            "visit_id": "v1",
            "overall_pct": 80,
            "status": "good",
            "block_scores": {"exams": 70, "treatment": 65},
            "evaluation_v4": {"scorer_version": "test", "schema_version": "1", "findings": []},
            **_rich_clinical(),
        }
    ]
    report = {"date": "2026-08-02", "quality": {"passed": True}, "partial": False}
    upsert_warehouse(path, raw_rows, cases, report)
    assert "suggest" in order
    assert "cache" in order
    assert order.index("suggest") < order.index("cache")
