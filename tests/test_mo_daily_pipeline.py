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
    atomic_write_text,
    build_daily_report,
    catch_up_dates,
    initialize_warehouse,
    merge_daily_partitions,
    previous_week_dates,
    resolve_run_date,
    this_week_dates,
    validate_export,
    write_daily_report,
)
from clinical_knowledge.mo_orchestrator import (
    MoDailyPipeline,
    PipelineState,
    PipelinePaths,
    VpnController,
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


def test_warehouse_schema_contains_star_and_crm_tables(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "warehouse.sqlite"
    initialize_warehouse(path)
    with sqlite3.connect(path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"fact_mo_case", "fact_mo_daily", "dim_doctor", "crm_case_state", "crm_case_event"} <= tables


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