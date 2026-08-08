from __future__ import annotations

import json
import plistlib
import subprocess
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from clinical_knowledge.mo_daily import (
    add_document_taxonomy,
    assess_completeness,
    atomic_write_text,
    build_daily_report,
    case_overall_pct,
    case_status,
    catch_up_dates,
    day_status,
    doctor_key_for,
    icd_chapter,
    initialize_warehouse,
    install_daily_partition,
    merge_daily_partitions,
    migrate_crm,
    previous_week_dates,
    resolve_run_date,
    this_week_dates,
    upsert_warehouse,
    validate_export,
    write_daily_report,
)
from clinical_knowledge.mo_orchestrator import (
    MoDailyPipeline,
    PipelineState,
    PipelinePaths,
    VpnController,
    build_digest,
    read_sql_epam_health,
    run_with_retry,
)
from scripts.run_mis_protocol_l1_batch import split_kz_rows


def valid_frame(day: str = "2026-07-27") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "visit_id": 11,
                "patient_id": 111,
                "visit_date": day,
                "parse_ok": "1",
                "date_mismatch": "0",
                "doctor_fio": "Иванов И.И.",
                "doctor_specialization": "Терапевт",
                "filial": "A",
                "patient_age_years": 40,
                "mkb_code_main": "I10",
                "complaints": "головная боль",
                "clinical_diagnosis": "I10",
                "kz_kind": "kz",
                "result_raw": "secure",
            }
        ]
    )


def test_yesterday_is_calculated_in_minsk() -> None:
    now = datetime.fromisoformat("2026-07-28T00:30:00+03:00")
    assert resolve_run_date("yesterday", now=now) == date(2026, 7, 27)
    with pytest.raises(ValueError):
        resolve_run_date("2026-07-28", now=now)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"service_names": "Периодический медицинский осмотр"}, "medical_exam"),
        ({"kz_kind": "kz"}, "consultation"),
        ({"service_names": "Справка в бассейн"}, "certificate"),
        ({"doctor_specialization": "Врач УЗИ"}, "diagnostic"),
        ({"doctor_specialization": "Медицинская сестра"}, "non_clinical"),
        ({"complaints": "", "clinical_diagnosis": ""}, "empty"),
        ({"kz_kind": "", "doctor_specialization": "Неизвестно"}, "unknown"),
    ],
)
def test_document_taxonomy(changes: dict, expected: str) -> None:
    row = valid_frame().iloc[0].to_dict()
    row.update(changes)
    if expected == "empty":
        for field in (
            "complaints",
            "anamnesis_doctor",
            "anamnesis_auto",
            "objective_status",
            "exam_data",
            "clinical_diagnosis",
            "diagnosis_list",
            "exam_recommendations",
            "treatment_recommendations",
        ):
            row[field] = ""
    classified = add_document_taxonomy(pd.DataFrame([row]))
    assert classified.iloc[0]["document_kind"] == expected


def test_batch_prefers_explicit_mo_eligibility_over_legacy_kind() -> None:
    rows = [
        {
            "visit_id": "1",
            "kz_kind": "certificate",
            "document_kind": "medical_exam",
            "mo_score_eligible": "true",
            "parse_ok": "1",
        },
        {
            "visit_id": "2",
            "kz_kind": "certificate",
            "document_kind": "certificate",
            "mo_score_eligible": "false",
            "parse_ok": "1",
        },
    ]
    scored, excluded = split_kz_rows(rows)
    assert [row["visit_id"] for row in scored] == ["1"]
    assert excluded["n_excluded"] == 1
    assert excluded["excluded_top_specialties"]["certificate"][" - "] == 1


def test_validation_blocks_duplicates_and_accepts_verified_empty_day() -> None:
    frame = valid_frame()
    duplicate = pd.concat([frame, frame], ignore_index=True)
    failed = validate_export(duplicate, day=date(2026, 7, 27), source_rows=2)
    assert not failed.passed
    assert {issue.code for issue in failed.blocking} == {"duplicates"}

    empty = frame.iloc[0:0]
    result = validate_export(empty, day=date(2026, 7, 27), source_rows=0)
    assert result.passed
    assert result.metrics["parse_ok_pct"] == 100.0


def wide_frame(rows: int, *, day: str = "2026-07-27", **overrides: object) -> pd.DataFrame:
    base = valid_frame(day).iloc[0].to_dict()
    records = []
    for index in range(rows):
        row = dict(base)
        row.update({"id": index + 1, "visit_id": 1000 + index, "patient_id": 5000 + index})
        row.update(overrides)
        records.append(row)
    return pd.DataFrame(records)


def test_volume_collapse_blocks_incomplete_night_export(monkeypatch: pytest.MonkeyPatch) -> None:
    thin = wide_frame(2)
    history = [580, 601, 575, 590]
    blocked = validate_export(thin, day=date(2026, 7, 27), source_rows=2, historical_same_weekday_counts=history)
    assert not blocked.passed
    assert {issue.code for issue in blocked.blocking} == {"volume_collapse"}
    assert blocked.metrics["volume_ratio_pct"] < 1

    # Осознанный перезабор праздничного дня: гейт снимается переменной окружения.
    monkeypatch.setenv("MO_VOLUME_COLLAPSE_RATIO", "0")
    override = validate_export(thin, day=date(2026, 7, 27), source_rows=2, historical_same_weekday_counts=history)
    assert override.passed
    assert {issue.code for issue in override.warnings} == {"volume_anomaly"}


def test_volume_gate_needs_enough_history_and_allows_normal_day() -> None:
    thin = wide_frame(2)
    short_history = validate_export(
        thin, day=date(2026, 7, 27), source_rows=2, historical_same_weekday_counts=[580, 590]
    )
    assert short_history.passed

    normal = wide_frame(560)
    healthy = validate_export(
        normal, day=date(2026, 7, 27), source_rows=560, historical_same_weekday_counts=[580, 601, 575, 590]
    )
    assert healthy.passed
    assert not healthy.warnings


def test_broken_doctor_join_blocks_the_day() -> None:
    frame = wide_frame(40, doctor_fio="")
    result = validate_export(frame, day=date(2026, 7, 27), source_rows=40)
    assert not result.passed
    assert {issue.code for issue in result.blocking} == {"doctor_join_broken"}

    # Маленький день с редкими пропусками остаётся предупреждением, а не блокировкой.
    small = wide_frame(10, doctor_fio="")
    warned = validate_export(small, day=date(2026, 7, 27), source_rows=10)
    assert warned.passed
    assert {issue.code for issue in warned.warnings} == {"doctor_missing"}


def test_blocked_day_goes_to_quarantine_not_raw(tmp_path: Path) -> None:
    frame = wide_frame(2)
    quality = validate_export(
        frame, day=date(2026, 7, 27), source_rows=2, historical_same_weekday_counts=[580, 601, 575, 590]
    )
    partition, meta_path = install_daily_partition(
        frame,
        day=date(2026, 7, 27),
        root=tmp_path,
        quality=quality,
        run_id="run-1",
        source_meta={"rows": 2},
    )
    assert "quarantine" in partition.parts
    assert not (tmp_path / "raw" / "2026" / "07" / "mo_2026-07-27.parquet").exists()
    assert json.loads(meta_path.read_text())["quality"]["blocking"][0]["code"] == "volume_collapse"


def test_merge_is_idempotent_and_late_row_wins(tmp_path: Path) -> None:
    first = valid_frame()
    second = valid_frame()
    first["visit_time"] = pd.to_timedelta(["08:00:00"])
    second["visit_time"] = pd.to_timedelta(["08:30:00"])
    second.loc[0, "complaints"] = "уточнённая жалоба"
    first_path = tmp_path / "mo_2026-07-26.parquet"
    second_path = tmp_path / "mo_2026-07-27.parquet"
    first.to_parquet(first_path, index=False)
    second.to_parquet(second_path, index=False)

    out = tmp_path / "month"
    parquet, _, info = merge_daily_partitions([first_path, second_path], month="2026-07", out_dir=out)
    assert info["rows"] == 1
    assert info["upserted_duplicates"] == 1
    assert pd.read_parquet(parquet).iloc[0]["complaints"] == "уточнённая жалоба"
    first_hash = info["sha256"]
    _, _, rerun = merge_daily_partitions([first_path, second_path], month="2026-07", out_dir=out)
    assert rerun["sha256"] == first_hash


def test_atomic_write_replaces_content(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_text(path, "old")
    atomic_write_text(path, "new")
    assert path.read_text() == "new"
    assert not list(tmp_path.glob("*.tmp"))


class FakeRunner:
    def __init__(self, status: str = "Connected", fail_body: bool = False):
        self.current = status
        self.calls: list[tuple[str, ...]] = []
        self.fail_body = fail_body

    def __call__(self, command):
        command = tuple(command)
        self.calls.append(command)
        action = command[-1]
        if action == "status":
            return subprocess.CompletedProcess(command, 0, stdout=self.current)
        if action == "ensure-off":
            self.current = "Disconnected"
        if action == "ensure-on":
            self.current = "Connected"
        return subprocess.CompletedProcess(command, 0, stdout="")


def test_vpn_restored_in_finally() -> None:
    runner = FakeRunner("Connected")
    vpn = VpnController(runner, wait_seconds=0)
    with pytest.raises(RuntimeError):
        with vpn.sql_window():
            assert runner.current == "Disconnected"
            raise RuntimeError("SQL failed")
    assert runner.current == "Connected"
    assert any(call[-1] == "ensure-on" for call in runner.calls)


def test_vpn_stays_disconnected_if_initially_disconnected() -> None:
    runner = FakeRunner("Disconnected")
    vpn = VpnController(runner, wait_seconds=0)
    with vpn.sql_window():
        pass
    assert runner.current == "Disconnected"
    assert not any(call[-1] == "ensure-on" for call in runner.calls)


def test_vpn_reports_primary_and_restore_failures() -> None:
    runner = FakeRunner("Connected")
    original = runner.__call__

    def fail_restore(command):
        if command[-1] == "ensure-on":
            raise OSError("restore unavailable")
        return original(command)

    vpn = VpnController(fail_restore, wait_seconds=0)
    with pytest.raises(ExceptionGroup) as caught:
        with vpn.sql_window():
            raise RuntimeError("SQL unavailable")
    messages = [str(error) for error in caught.value.exceptions]
    assert any("SQL unavailable" in message for message in messages)
    assert any("restore unavailable" in message for message in messages)


def test_retry_eventually_succeeds_without_leaking_command() -> None:
    attempts = 0

    def runner(command):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    result = run_with_retry(runner, ("safe-command",), attempts=3, base_delay_seconds=0, sleep=lambda _: None)
    assert result.stdout == "ok"
    assert attempts == 3


def test_catch_up_is_bounded_and_skips_successes() -> None:
    dates = catch_up_dates(
        successful_dates=["2026-07-24", "2026-07-26"],
        first_date=date(2026, 7, 24),
        yesterday=date(2026, 7, 27),
        limit=2,
    )
    assert dates == [date(2026, 7, 25), date(2026, 7, 27)]


def test_previous_and_this_week_windows_in_minsk() -> None:
    # Среда 2026-07-29: this week = Пн 27 - Вт 28; previous = Пн 20 - Вс 26.
    now = datetime.fromisoformat("2026-07-29T07:00:00+03:00")
    assert this_week_dates(now=now) == [date(2026, 7, 27), date(2026, 7, 28)]
    assert previous_week_dates(now=now) == [
        date(2026, 7, 20),
        date(2026, 7, 21),
        date(2026, 7, 22),
        date(2026, 7, 23),
        date(2026, 7, 24),
        date(2026, 7, 25),
        date(2026, 7, 26),
    ]


def test_stale_run_is_failed_for_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "pipeline.json"
    state_path.write_text(
        json.dumps(
            {
                "dates": {
                    "2026-07-27": {
                        "status": "scoring",
                        "heartbeat": "2020-01-01T00:00:00Z",
                    }
                },
                "runs": [],
            }
        )
    )
    state = PipelineState(state_path)
    assert state.mark_stale_runs() == ["2026-07-27"]
    assert state.data["dates"]["2026-07-27"]["status"] == "failed"


def test_sql_epam_health_signal_is_read_only(tmp_path: Path) -> None:
    status = tmp_path / "sync-status.json"
    status.write_text(json.dumps({"status": "success", "finished_at": "2026-07-28T06:45:00+03:00"}))
    signal = read_sql_epam_health(status)
    assert signal["status"] == "present"
    assert signal["upstream_status"] == "success"
    assert read_sql_epam_health(None) == {"status": "not_configured"}


def test_report_has_secure_queue_but_public_has_no_pii() -> None:
    raw = add_document_taxonomy(valid_frame()).to_dict(orient="records")
    cases = [
        {
            "mis_id": 1,
            "visit_id": 11,
            "patient_id": 111,
            "doctor_fio": "Иванов И.И.",
            "doctor_specialization": "Терапевт",
            "filial": "A",
            "overall_pct": 42,
            "status": "non_compliant",
        }
    ]
    secure, public = build_daily_report(
        raw,
        cases,
        day=date(2026, 7, 27),
        run_id="test",
        revision=1,
        quality={"passed": True},
    )
    assert secure["action_queue"][0]["visit_id"] == 11
    assert secure["summary"]["eligible_rows"] == 1
    assert secure["summary"]["eligible_visits"] == 1
    assert secure["summary"]["critical"] == 1
    assert secure["summary"]["critical_low_score"] == 1
    assert secure["summary"]["eligible_rows"] == 1
    assert secure["summary"]["eligible_visits"] == 1
    assert secure["summary"]["critical"] == 1
    assert secure["summary"]["critical_low_score"] == 1
    encoded = json.dumps(public, ensure_ascii=False)
    assert "patient_id" not in encoded
    assert "visit_id" not in encoded
    assert "Иванов" not in encoded


def test_daily_html_report_has_kpi_axes_and_action_queue(tmp_path: Path) -> None:
    raw = add_document_taxonomy(valid_frame()).to_dict(orient="records")
    cases = [{
        "mis_id": 1,
        "visit_id": 11,
        "doctor_fio": "Иванов И.И.",
        "doctor_specialization": "Терапевт",
        "filial": "A",
        "overall_pct": 42,
        "deep": {"axes": {"documentation": 75}, "n_by_severity": {"P1": 1}},
    }]
    secure, public = build_daily_report(
        raw,
        cases,
        day=date(2026, 7, 27),
        run_id="test",
        revision=1,
        quality={"passed": True, "metrics": {"parse_ok_pct": 100}},
    )
    assert secure["summary"]["critical_clinical"] == 1
    assert secure["summary"]["critical_clinical"] == 1
    paths = write_daily_report(secure, public, day=date(2026, 7, 27), root=tmp_path)
    report_html = (paths["report_dir"] / "report.html").read_text()
    assert "Отчёт МО за 27.07.2026" in report_html
    assert "Оценка по направлениям" in report_html
    assert "Очередь разбора" in report_html
    assert "font-family" not in report_html or "sans-serif" in report_html


def test_partial_day_is_marked_in_report_html_and_warehouse(tmp_path: Path) -> None:
    import sqlite3

    raw = add_document_taxonomy(wide_frame(3)).to_dict(orient="records")
    scored_one = [{"mis_id": 1, "visit_id": 1000, "overall_pct": 88, "status": "compliant"}]
    completeness = assess_completeness(raw, scored_one)
    assert completeness["partial"]
    assert completeness["reasons"] == ["scoring_coverage"]
    assert completeness["coverage_pct"] == pytest.approx(33.33, abs=0.01)

    secure, public = build_daily_report(
        raw,
        scored_one,
        day=date(2026, 7, 27),
        run_id="test",
        revision=1,
        quality={"passed": True, "metrics": {"parse_ok_pct": 100}},
        completeness=completeness,
    )
    assert secure["partial"] is True
    assert public["partial"] is True
    assert day_status(secure) == "partial"

    paths = write_daily_report(secure, public, day=date(2026, 7, 27), root=tmp_path)
    assert "День доделывается" in (paths["report_dir"] / "report.html").read_text()

    warehouse = tmp_path / "warehouse.sqlite"
    upsert_warehouse(warehouse, raw, scored_one, secure)
    with sqlite3.connect(warehouse) as db:
        status = db.execute("SELECT quality_status FROM fact_mo_daily WHERE visit_date = ?", ("2026-07-27",)).fetchone()
    assert status[0] == "partial"


def test_complete_day_is_not_partial_and_blocked_day_wins() -> None:
    raw = add_document_taxonomy(wide_frame(2)).to_dict(orient="records")
    cases = [
        {"mis_id": 1, "visit_id": 1000, "overall_pct": 88},
        {"mis_id": 2, "visit_id": 1001, "overall_pct": 91},
    ]
    completeness = assess_completeness(raw, cases)
    assert not completeness["partial"]
    assert completeness["coverage_pct"] == 100.0
    assert day_status({"quality": {"passed": True}, "partial": False}) == "passed"
    assert day_status({"quality": {"passed": False}, "partial": True}) == "blocked"

    failed_case = assess_completeness(raw, [cases[0], {"mis_id": 2, "error": "llm timeout"}])
    assert failed_case["partial"]
    assert set(failed_case["reasons"]) == {"scoring_coverage", "scoring_errors"}

    queued = assess_completeness(raw, cases, llm_queue_pending=4)
    assert queued["reasons"] == []
    assert queued["advisory_reasons"] == ["llm_queue_pending"]
    assert not queued["partial"]
    assert queued["llm_queue_pending"] == 4

    queued_with_gap = assess_completeness(raw, [cases[0]], llm_queue_pending=2)
    assert set(queued_with_gap["reasons"]) == {"scoring_coverage", "llm_queue_pending"}
    assert queued_with_gap["partial"]


def test_partial_day_is_retried_by_catch_up_until_attempts_run_out(tmp_path: Path) -> None:
    state_path = tmp_path / "pipeline.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dates": {
                    "2026-07-25": {"status": "success", "attempts": 1},
                    "2026-07-26": {"status": "partial", "attempts": 2},
                    "2026-07-27": {"status": "partial", "attempts": 9},
                },
                "runs": [],
            }
        )
    )
    state = PipelineState(state_path)
    assert state.settled_dates() == ["2026-07-25", "2026-07-27"]
    assert catch_up_dates(
        successful_dates=state.settled_dates(),
        yesterday=date(2026, 7, 27),
        first_date=date(2026, 7, 25),
    ) == [date(2026, 7, 26)]


def test_catch_up_picks_partial_day_behind_a_successful_one(tmp_path: Path) -> None:
    state_path = tmp_path / "data" / "state" / "pipeline.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dates": {
                    "2026-07-25": {"status": "partial", "attempts": 1},
                    "2026-07-26": {"status": "success", "attempts": 1},
                    "2026-07-27": {"status": "success", "attempts": 1},
                },
                "runs": [],
            }
        )
    )
    pipeline = MoDailyPipeline(PipelinePaths(tmp_path, tmp_path / "data"), runner=lambda _c: None)
    state = PipelineState(state_path)
    now = datetime.fromisoformat("2026-07-28T10:00:00+03:00")

    # retry / hourly: только --catch-up, но недоделанный 25-е обязан вернуться в работу.
    assert pipeline.select_days(catch_up=True, state=state, now=now) == [date(2026, 7, 25)]

    # Основной приём: вчера плюс окно сверки, без дублей дат.
    assert pipeline.select_days(catch_up=True, reconcile_days=3, state=state, now=now) == [
        date(2026, 7, 25),
        date(2026, 7, 26),
        date(2026, 7, 27),
    ]


def test_partial_day_survives_stale_sweep(tmp_path: Path) -> None:
    state_path = tmp_path / "pipeline.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dates": {"2026-07-26": {"status": "partial", "heartbeat": "2020-01-01T00:00:00+03:00"}},
                "runs": [],
            }
        )
    )
    state = PipelineState(state_path)
    assert state.mark_stale_runs() == []
    assert state.data["dates"]["2026-07-26"]["status"] == "partial"


def test_warehouse_schema_contains_star_and_crm_tables(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "warehouse.sqlite"
    initialize_warehouse(path)
    with sqlite3.connect(path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"fact_mo_case", "fact_mo_daily", "dim_doctor", "crm_case_state", "crm_case_event"} <= tables


def warehouse_day_rows(day: str = "2026-07-27") -> tuple[list[dict], list[dict]]:
    frame = wide_frame(3, day=day)
    frame.loc[0, "mkb_codes"] = "I10|E11.9"
    frame.loc[1, "mkb_codes"] = "J06.9"
    frame.loc[2, "mkb_codes"] = ""
    frame["service_codes"] = "10.100.1. | 10.64."
    frame["service_names"] = "Консультация врача-терапевта, категория | Перчатки нитриловые"
    frame.loc[1, "doctor_fio"] = "Петров П.П."
    frame.loc[1, "doctor_specialization"] = "Кардиолог"
    frame.loc[1, "filial"] = "B"
    raw = add_document_taxonomy(frame).to_dict(orient="records")
    cases = [
        {
            "mis_id": 1,
            "visit_id": 1000,
            "doctor_fio": "Иванов И.И.",
            "doctor_specialization": "Терапевт",
            "filial": "A",
            "overall_pct": 58,
            "status": "partially_compliant",
            "deep": {
                "axes": {"documentation": 60, "clinical_concordance": 55, "safety": 40, "regulatory": 90},
                "n_by_severity": {"P0": 0, "P1": 1, "P2": 2},
                "findings": [
                    {"code": "no_bp", "axis": "documentation", "severity": "P1", "passed": False, "evidence": "нет АД"},
                    {"code": "no_plan", "axis": "safety", "severity": "P2", "passed": False, "evidence": "нет плана"},
                ],
            },
        },
        {
            "mis_id": 2,
            "visit_id": 1001,
            "doctor_fio": "Петров П.П.",
            "doctor_specialization": "Кардиолог",
            "filial": "B",
            "overall_pct": 92,
            "status": "compliant",
            "deep": {"axes": {"documentation": 95, "safety": 90}, "n_by_severity": {}, "findings": []},
        },
        {
            "mis_id": 3,
            "visit_id": 1002,
            "doctor_fio": "Иванов И.И.",
            "doctor_specialization": "Терапевт",
            "filial": "A",
            "overall_pct": 81,
            "status": "mostly_compliant",
            "deep": {"axes": {"documentation": 80}, "findings": []},
        },
    ]
    return raw, cases


def test_one_day_fills_every_warehouse_table(tmp_path: Path) -> None:
    import sqlite3

    raw, cases = warehouse_day_rows()
    secure, _public = build_daily_report(
        raw, cases, day=date(2026, 7, 27), run_id="wh", revision=2, quality={"passed": True}
    )
    path = tmp_path / "warehouse.sqlite"
    written = upsert_warehouse(path, raw, cases, secure)

    assert written["fact_mo_case"] == 3
    assert written["fact_mo_score_axis"] == 7
    assert written["fact_mo_finding"] == 2
    assert written["fact_mo_doctor_daily"] == 2

    with sqlite3.connect(path) as db:
        counts = {
            table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "fact_mo_case",
                "fact_mo_finding",
                "fact_mo_score_axis",
                "fact_mo_daily",
                "fact_mo_doctor_daily",
                "dim_date",
                "dim_doctor",
                "dim_specialty",
                "dim_branch",
                "dim_diagnosis",
                "dim_service",
                "dim_document_kind",
                "crm_case_state",
                "crm_case_event",
            )
        }
        empty = sorted(name for name, count in counts.items() if count == 0)
        assert empty == [], f"пустые таблицы витрины: {empty}"
        # saved_view и export_job наполняет кабинет методиста, а не pipeline.
        assert db.execute("SELECT COUNT(*) FROM saved_view").fetchone()[0] == 0

        day_row = dict(
            zip(
                [c[0] for c in db.execute("SELECT * FROM fact_mo_daily LIMIT 0").description],
                db.execute("SELECT * FROM fact_mo_daily WHERE visit_date = ?", ("2026-07-27",)).fetchone(),
            )
        )
        assert day_row["eligible_rows"] == 3
        assert day_row["avg_safety"] == pytest.approx(65.0)
        assert day_row["needs_attention"] == 1
        assert day_row["quality_status"] == "passed"

        # Врач с двумя случаями агрегирован отдельно от второго врача.
        ivanov = db.execute(
            "SELECT cases, scored, avg_score, needs_attention, critical FROM fact_mo_doctor_daily"
            " WHERE doctor_key = ?",
            (doctor_key_for("Иванов И.И."),),
        ).fetchone()
        assert ivanov == (2, 2, 69.5, 1, 1)
        assert db.execute("SELECT chapter FROM dim_diagnosis WHERE diagnosis_code = 'I10'").fetchone()[0] == (
            "Болезни системы кровообращения"
        )


def test_warehouse_soft_fills_mkb_from_full_doc_without_touching_agreement(tmp_path: Path) -> None:
    import sqlite3

    raw, cases = warehouse_day_rows()
    # mis_id=3: слот пуст, код только в статусе → soft_fill; agreement остаётся как в raw
    for row in raw:
        if str(row.get("id") or row.get("mis_id") or "") in {"3", "3.0"} or int(float(row.get("id") or 0)) == 3:
            row["mkb_code_main"] = ""
            row["mkb_codes"] = ""
            row["mkb_code_agreement"] = "unknown"
            row["objective_status"] = "Локально спокойно. N47.1 после операции."
            row["clinical_diagnosis"] = "Состояние после циркумцизио"
            break
    secure, _ = build_daily_report(
        raw, cases, day=date(2026, 7, 27), run_id="wh-soft", revision=1, quality={"passed": True}
    )
    path = tmp_path / "warehouse.sqlite"
    upsert_warehouse(path, raw, cases, secure)
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT diagnosis_code, mkb_code_main_source, mkb_code_main_slot "
            "FROM fact_mo_case WHERE mis_id = '3'"
        ).fetchone()
        assert row is not None
        assert row[0] == "N47.1"
        assert row[1] == "soft_fill_full_doc"
        assert row[2] == ""
        # слот с кодом не перетирается soft-fill
        slot_row = db.execute(
            "SELECT diagnosis_code, mkb_code_main_source, mkb_code_main_slot "
            "FROM fact_mo_case WHERE mis_id = '1'"
        ).fetchone()
        assert slot_row[0] == "I10"
        assert slot_row[1] == "slot"
        assert slot_row[2] == "I10"
    # raw agreement не мутировали
    raw3 = next(r for r in raw if int(float(r.get("id") or 0)) == 3)
    assert raw3.get("mkb_code_agreement") == "unknown"
    assert not (raw3.get("mkb_code_main") or "").strip()


def test_warehouse_upsert_is_idempotent_and_drops_vanished_rows(tmp_path: Path) -> None:
    import sqlite3

    raw, cases = warehouse_day_rows()
    secure, _ = build_daily_report(
        raw, cases, day=date(2026, 7, 27), run_id="wh", revision=1, quality={"passed": True}
    )
    path = tmp_path / "warehouse.sqlite"
    upsert_warehouse(path, raw, cases, secure)
    upsert_warehouse(path, raw, cases, secure)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM fact_mo_case").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM fact_mo_finding").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM fact_mo_doctor_daily").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM crm_case_event").fetchone()[0] == 1

    # Запись удалили в МИС: повторный день не должен оставлять её в витрине.
    shrunk_raw, shrunk_cases = raw[:2], cases[:2]
    secure_second, _ = build_daily_report(
        shrunk_raw, shrunk_cases, day=date(2026, 7, 27), run_id="wh2", revision=2, quality={"passed": True}
    )
    written = upsert_warehouse(path, shrunk_raw, shrunk_cases, secure_second)
    assert written["deleted_stale_cases"] == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM fact_mo_case").fetchone()[0] == 2
        assert db.execute("SELECT revision FROM fact_mo_daily").fetchone()[0] == 2


def legacy_crm_db(path: Path) -> None:
    """Старый файл кабинета: своя схема CRM без звёздных таблиц pipeline."""
    import sqlite3

    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE crm_case_state (
              case_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'new', assignee TEXT,
              tags_json TEXT NOT NULL DEFAULT '[]', due_date TEXT,
              finding_decisions_json TEXT NOT NULL DEFAULT '{}',
              updated_at TEXT NOT NULL, updated_by TEXT NOT NULL
            );
            CREATE TABLE crm_case_event (
              event_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, event_type TEXT NOT NULL,
              actor TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE saved_view (
              view_id TEXT PRIMARY KEY, owner TEXT NOT NULL, scope TEXT NOT NULL, name TEXT NOT NULL,
              filters_json TEXT NOT NULL, config_json TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE export_job (
              job_id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL, kind TEXT NOT NULL,
              filters_json TEXT NOT NULL, result_path TEXT, created_at TEXT NOT NULL, expires_at TEXT NOT NULL
            );
            """
        )
        db.execute(
            "INSERT INTO crm_case_state VALUES ('1001','in_review','ИП','[\"P1\"]','2026-07-28','{}','2026-07-27T10:00:00Z','ИП')"
        )
        db.execute(
            "INSERT INTO crm_case_event VALUES ('ev1','1001','status_changed','ИП','{}','2026-07-27T10:00:00Z')"
        )
        db.execute(
            "INSERT INTO saved_view VALUES ('v1','ИП','private','Моя очередь','{}','{}','2026-07-27T10:00:00Z','2026-07-27T10:00:00Z')"
        )


def test_crm_migration_preserves_methodist_work_and_is_idempotent(tmp_path: Path) -> None:
    import sqlite3

    legacy = tmp_path / "mo_methodist.sqlite"
    legacy_crm_db(legacy)
    warehouse = tmp_path / "mo_analytics.sqlite"

    moved = migrate_crm(legacy, warehouse)
    assert moved == {"crm_case_state": 1, "crm_case_event": 1, "saved_view": 1}
    assert migrate_crm(legacy, warehouse) == {"crm_case_state": 0, "crm_case_event": 0, "saved_view": 0}

    with sqlite3.connect(warehouse) as db:
        db.row_factory = sqlite3.Row
        state = db.execute("SELECT * FROM crm_case_state WHERE case_id = '1001'").fetchone()
        assert state["status"] == "in_review"
        assert state["assignee"] == "ИП"
        assert db.execute("SELECT COUNT(*) FROM crm_case_event").fetchone()[0] == 1

    # Источник остаётся нетронутым: это резервная копия.
    with sqlite3.connect(f"file:{legacy}?mode=ro", uri=True) as origin:
        assert origin.execute("SELECT COUNT(*) FROM crm_case_state").fetchone()[0] == 1


def test_old_crm_tables_are_upgraded_without_losing_rows(tmp_path: Path) -> None:
    import sqlite3

    warehouse = tmp_path / "mo_analytics.sqlite"
    with sqlite3.connect(warehouse) as db:
        db.executescript(
            """
            CREATE TABLE crm_case_state (
              mis_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'new', assignee TEXT,
              due_date TEXT, updated_at TEXT NOT NULL
            );
            CREATE TABLE crm_case_event (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT, mis_id TEXT NOT NULL,
              event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE saved_view (
              view_id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
              filters_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            """
        )
        db.execute("INSERT INTO crm_case_state VALUES ('7','in_review','ИП','2026-07-28','2026-07-27T10:00:00Z')")

    initialize_warehouse(warehouse)
    with sqlite3.connect(warehouse) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        # Непустая таблица сохранена под другим именем, пустые просто пересозданы.
        assert "crm_case_state_legacy" in tables
        assert db.execute("SELECT status FROM crm_case_state_legacy WHERE mis_id = '7'").fetchone()[0] == "in_review"
        assert "crm_case_event_legacy" not in tables
        columns = {row[1] for row in db.execute("PRAGMA table_info(crm_case_state)")}
        assert {"case_id", "tags_json", "finding_decisions_json", "updated_by"} <= columns
        assert "scope" in {row[1] for row in db.execute("PRAGMA table_info(saved_view)")}


def test_pipeline_queue_does_not_overwrite_methodist_status(tmp_path: Path) -> None:
    import sqlite3

    warehouse = tmp_path / "mo_analytics.sqlite"
    initialize_warehouse(warehouse)
    with sqlite3.connect(warehouse) as db:
        db.execute(
            "INSERT INTO crm_case_state VALUES ('1000','confirmed_issue','ИП','[]',NULL,'{}','2026-07-27T09:00:00Z','ИП')"
        )

    raw, cases = warehouse_day_rows()
    cases[0]["overall_pct"] = 41
    secure, _ = build_daily_report(
        raw, cases, day=date(2026, 7, 27), run_id="wh", revision=1, quality={"passed": True}
    )
    assert secure["action_queue"], "низкая оценка обязана попасть в очередь разбора"
    upsert_warehouse(warehouse, raw, cases, secure)

    with sqlite3.connect(warehouse) as db:
        assert db.execute("SELECT status FROM crm_case_state WHERE case_id = '1000'").fetchone()[0] == (
            "confirmed_issue"
        )
        assert db.execute("SELECT updated_by FROM crm_case_state WHERE case_id = '1000'").fetchone()[0] == "ИП"


def test_deep_only_cases_keep_their_scores_in_warehouse(tmp_path: Path) -> None:
    import sqlite3

    # Так выглядят месячные прогоны --deep-only: верхний overall_pct пуст, оценка в deep.
    raw = add_document_taxonomy(wide_frame(2)).to_dict(orient="records")
    cases = [
        {
            "mis_id": 1,
            "visit_id": 1000,
            "overall_pct": None,
            "status": None,
            "doctor_fio": "Иванов И.И.",
            "deep": {
                "overall_pct": 93.0,
                "status": "good",
                "axes": {"documentation": 100.0, "safety": 88.0},
                "findings": [{"code": "minor", "severity": "P3", "passed": False}],
            },
        },
        {
            "mis_id": 2,
            "visit_id": 1001,
            "core_overall_pct": 64.0,
            "doctor_fio": "Иванов И.И.",
        },
    ]
    assert case_overall_pct(cases[0]) == 93.0
    assert case_status(cases[0]) == "good"
    assert case_overall_pct(cases[1]) == 64.0
    assert case_overall_pct({"overall_pct": None}) is None
    # Витрина живёт на deep-шкале: верхний L1-балл дневного прогона её не подменяет,
    # иначе тренд рвётся на стыке истории и свежих дней.
    mixed = {"overall_pct": 71.5, "status": "review", "deep": {"overall_pct": 86.2, "status": "good"}}
    assert case_overall_pct(mixed) == 86.2
    assert case_status(mixed) == "good"

    completeness = assess_completeness(raw, cases)
    assert completeness["coverage_pct"] == 100.0
    assert not completeness["partial"]

    secure, _ = build_daily_report(
        raw, cases, day=date(2026, 7, 20), run_id="deep-only", revision=1, quality={"passed": True}
    )
    assert secure["summary"]["avg_score"] == 78.5
    path = tmp_path / "warehouse.sqlite"
    upsert_warehouse(path, raw, cases, secure)
    with sqlite3.connect(path) as db:
        scores = [row[0] for row in db.execute("SELECT overall_pct FROM fact_mo_case ORDER BY mis_id")]
        assert scores == [93.0, 64.0]
        assert db.execute("SELECT status FROM fact_mo_case WHERE mis_id = '1'").fetchone()[0] == "good"
        assert db.execute("SELECT avg_score FROM fact_mo_daily").fetchone()[0] == 78.5
        assert db.execute("SELECT COUNT(*) FROM fact_mo_case WHERE overall_pct IS NULL").fetchone()[0] == 0


def test_icd_chapter_maps_codes_and_ignores_garbage() -> None:
    assert icd_chapter("I10") == "Болезни системы кровообращения"
    assert icd_chapter("s72.0") == "Травмы и отравления"
    assert icd_chapter("Z00.0") == "Факторы, влияющие на здоровье"
    assert icd_chapter("") == ""
    assert icd_chapter("нет") == ""


def test_dry_run_does_not_call_runner_or_vpn(tmp_path: Path) -> None:
    def forbidden(_command):
        raise AssertionError("subprocess must not run")

    pipeline = MoDailyPipeline(
        PipelinePaths(tmp_path, tmp_path / "data"),
        runner=forbidden,
        dry_run=True,
    )
    result = pipeline.run(date_value="2026-07-27", now=datetime.fromisoformat("2026-07-28T08:00:00+03:00"))
    assert result[0]["dry_run"] is True
    assert "export_mis_protocol_month.py" in " ".join(result[0]["commands"][0])
    assert not (tmp_path / "data").exists()


def test_orchestrator_runs_with_injected_exporter_and_batch(tmp_path: Path) -> None:
    class FakeVpn:
        entered = False

        @contextmanager
        def sql_window(self):
            self.entered = True
            yield

    def runner(command):
        command = list(command)
        script = Path(command[1]).name
        if script == "export_mis_protocol_month.py":
            out_dir = Path(command[command.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = "2026-07-27_2026-07-28"
            valid_frame().to_parquet(out_dir / f"mis_protocol_{tag}.parquet", index=False)
            (out_dir / f"mis_protocol_{tag}.meta.json").write_text(
                json.dumps({"source_rows": 1, "rows": 1})
            )
        elif script == "run_mis_protocol_l1_batch.py":
            out_dir = Path(command[command.index("--out-dir") + 1])
            tag = command[command.index("--month") + 1]
            case = {
                "mis_id": 1,
                "visit_id": 11,
                "date": "2026-07-27",
                "doctor_fio": "Иванов И.И.",
                "doctor_specialization": "Терапевт",
                "filial": "A",
                "overall_pct": 80,
                "status": "mostly_compliant",
            }
            (out_dir / f"kz_l1_{tag}_cases.jsonl").write_text(json.dumps(case) + "\n")
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    vpn = FakeVpn()
    paths = PipelinePaths(tmp_path, tmp_path / "data")
    pipeline = MoDailyPipeline(paths, runner=runner, vpn=vpn, notify=lambda _: True)
    result = pipeline.run(
        date_value="2026-07-27",
        now=datetime.fromisoformat("2026-07-28T08:00:00+03:00"),
    )
    assert result[0]["status"] == "success"
    assert vpn.entered
    assert (paths.data_root / "reports" / "2026" / "07" / "27" / "report.json").is_file()
    public = json.loads((paths.data_root / "public" / "latest.json").read_text())
    assert public["summary"]["source_rows"] == 1
    assert "Иванов" not in json.dumps(public, ensure_ascii=False)


def test_digest_leads_with_problems_and_groups_successes() -> None:
    text = build_digest(
        [
            {"day": "2026-07-25", "level": "ok", "rows": 500, "scored": 480, "detail": "строк 500"},
            {"day": "2026-07-26", "level": "partial", "detail": "покрытие оценки 62.5% (scoring_coverage)"},
            {"day": "2026-07-27", "level": "blocked", "detail": "data-quality gate blocked: ['volume_collapse']"},
            {"day": "2026-07-28", "level": "ok", "rows": 610, "scored": 600, "detail": "строк 610"},
        ],
        now=datetime.fromisoformat("2026-07-30T06:12:00+03:00"),
    )
    lines = text.splitlines()
    assert lines[0] == "МО, приём 30.07 06:12 Europe/Minsk"
    assert lines[1].startswith("Не принято: 27.07")
    assert lines[2].startswith("Доделывается: 26.07")
    assert lines[3] == "Готово (2): 25.07, 28.07; строк 1110, оценено 1080"


def test_run_sends_one_digest_and_survives_a_failing_day(tmp_path: Path) -> None:
    sent: list[str] = []
    pipeline = MoDailyPipeline(
        PipelinePaths(tmp_path, tmp_path / "data"),
        runner=lambda _c: None,
        vpn=None,
        notify=lambda message: sent.append(message) or True,
    )
    calls: list[date] = []

    def fake_run_day(day: date, *, force: bool = False) -> dict:
        calls.append(day)
        if day == date(2026, 7, 26):
            pipeline._emit(day, "blocked", detail="data-quality gate blocked: ['volume_collapse']")
            raise RuntimeError("data-quality gate blocked: ['volume_collapse']")
        pipeline._emit(day, "ok", rows=10, scored=9, detail="строк 10")
        return {"date": day.isoformat(), "status": "success"}

    pipeline.run_day = fake_run_day  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="2026-07-26"):
        pipeline.run(reconcile_days=3, now=datetime.fromisoformat("2026-07-28T06:05:00+03:00"))

    # Плохой день не обрывает прогон: 27-е тоже обработано.
    assert calls == [date(2026, 7, 25), date(2026, 7, 26), date(2026, 7, 27)]
    assert len(sent) == 1
    assert "Не принято: 26.07" in sent[0]
    assert "Готово (2)" in sent[0]


def test_weekly_reconciliation_reuses_scores_unless_force_is_explicit(tmp_path: Path) -> None:
    pipeline = MoDailyPipeline(
        PipelinePaths(tmp_path, tmp_path / "data"),
        runner=lambda _c: None,
        notify=lambda _message: True,
    )
    calls: list[tuple[date, bool]] = []

    def fake_run_day(day: date, *, force: bool = False) -> dict:
        calls.append((day, force))
        return {"date": day.isoformat(), "status": "success"}

    pipeline.run_day = fake_run_day  # type: ignore[method-assign]
    now = datetime.fromisoformat("2026-08-03T06:00:00+03:00")
    pipeline.run(previous_week=True, now=now)
    assert len(calls) == 7
    assert all(force is False for _, force in calls)

    calls.clear()
    pipeline.run(previous_week=True, force=True, now=now)
    assert len(calls) == 7
    assert all(force is True for _, force in calls)


def test_run_stays_silent_when_there_is_nothing_to_do(tmp_path: Path) -> None:
    sent: list[str] = []
    state_path = tmp_path / "data" / "state" / "pipeline.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({"schema_version": 1, "dates": {"2026-07-27": {"status": "success", "attempts": 1}}, "runs": []})
    )
    pipeline = MoDailyPipeline(
        PipelinePaths(tmp_path, tmp_path / "data"),
        runner=lambda _c: None,
        notify=lambda message: sent.append(message) or True,
    )
    results = pipeline.run(catch_up=True, now=datetime.fromisoformat("2026-07-28T11:00:00+03:00"))
    assert results == []
    assert sent == []


def test_launchd_templates_are_valid_plists() -> None:
    root = Path(__file__).resolve().parents[1]
    by_label = {}
    for path in (root / "deploy" / "launchd").glob("*.plist.in"):
        rendered = (
            path.read_text()
            .replace("__ROOT__", str(root))
            .replace("__WRAPPER__", str(root / "scripts" / "run_mo_daily_launchd.sh"))
            .replace("__LOG_DIR__", "/tmp")
            .replace("__PYTHON__", "python3")
        )
        payload = plistlib.loads(rendered.encode())
        assert payload["Label"].startswith("by.protocol.mo-daily")
        by_label[payload["Label"]] = payload
    assert by_label["by.protocol.mo-daily"]["StartCalendarInterval"]["Hour"] == 6
    weekly = by_label["by.protocol.mo-daily-weekly"]["StartCalendarInterval"]
    assert weekly["Weekday"] == 1
    assert weekly["Hour"] == 11
    wrapper = (root / "scripts" / "run_mo_daily_launchd.sh").read_text(encoding="utf-8")
    assert "--previous-week" in wrapper
    assert 'weekly)' in wrapper