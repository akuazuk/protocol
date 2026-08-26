from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from clinical_knowledge.mo_lab_bundle import lab_primary_enabled
from clinical_knowledge.mo_lab_rollout import (
    build_rollout_report,
    ensure_shadow_state,
    lab_primary_guard,
    rollout_report_path,
)

ROOT = Path(__file__).resolve().parents[1]


def _seed_databases(root: Path) -> tuple[Path, Path]:
    analytics = root / "warehouse" / "mo_analytics.sqlite"
    lab = root / "warehouse" / "mo_lab.sqlite"
    analytics.parent.mkdir(parents=True)
    with sqlite3.connect(analytics) as conn:
        conn.executescript(
            """
            CREATE TABLE fact_mo_case (
              mis_id TEXT PRIMARY KEY,
              visit_date TEXT,
              patient_key TEXT
            );
            CREATE TABLE fact_mo_finding (
              mis_id TEXT,
              finding_code TEXT
            );
            CREATE TABLE crm_review_pack (
              visit_date TEXT,
              decision_json TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO fact_mo_case VALUES (?,?,?)",
            ("case-secret", "2026-08-26", "patient-key-secret"),
        )
        conn.execute(
            "INSERT INTO fact_mo_finding VALUES (?,?)",
            ("case-secret", "B_lab_present_not_in_mo"),
        )
        conn.executemany(
            "INSERT INTO crm_review_pack VALUES (?,?)",
            [
                (
                    "2026-08-26",
                    json.dumps(
                        {
                            "finding_decisions": {
                                "B_lab_present_not_in_mo": "confirmed",
                                "B_dx_absent": "false_positive",
                            }
                        }
                    ),
                )
                for _ in range(5)
            ],
        )
    with sqlite3.connect(lab) as conn:
        conn.executescript(
            """
            CREATE TABLE fact_mo_lab (
              patient_key TEXT,
              test_date TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO fact_mo_lab VALUES (?,?)",
            ("patient-key-secret", "2026-08-26"),
        )
    return analytics, lab


def test_guard_requires_shadow_days_and_successful_nights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_LAB_IN_PRIMARY", "1")
    ensure_shadow_state(
        data_root=tmp_path,
        now=datetime(2026, 8, 20, tzinfo=timezone.utc),
        git_commit_sha="a" * 40,
    )
    blocked = lab_primary_guard(data_root=tmp_path, today=date(2026, 8, 26))
    assert blocked["allowed"] is False
    assert "successful_nights_incomplete" in blocked["block_reasons"]
    assert lab_primary_enabled() is False


def test_rollout_runner_works_without_site_packages(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts" / "run_mo_lab_rollout_metrics.py"),
            "--data-root",
            str(tmp_path),
            "--init-shadow-state-only",
            "--git-sha",
            "c" * 40,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


def test_report_is_aggregate_only_and_unlocks_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    analytics, lab = _seed_databases(tmp_path)
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_LAB_IN_PRIMARY", "1")
    ensure_shadow_state(
        data_root=tmp_path,
        now=datetime(2026, 8, 20, tzinfo=timezone.utc),
        git_commit_sha="b" * 40,
    )
    state_dir = tmp_path / "state"
    for offset in range(7):
        day = date(2026, 8, 26) - timedelta(days=offset)
        (state_dir / f"gce_lab_{day.isoformat()}.json").write_text(
            json.dumps({"status": "success"}),
            encoding="utf-8",
        )
    report = build_rollout_report(
        analytics_db=analytics,
        lab_db=lab,
        data_root=tmp_path,
        end_date=date(2026, 8, 26),
    )
    assert report["metrics"]["finding_n"] == 1
    assert report["metrics"]["same_day_lab_case_n"] == 1
    assert report["review_pack"]["finding_decisions"] == {"confirmed": 5}
    assert report["guard_inputs"]["successful_lab_nights"] == 7
    blob = json.dumps(report)
    assert "case-secret" not in blob
    assert "patient-key-secret" not in blob
    assert report["phi_check"] == {
        "contains_row_identifiers": False,
        "contains_clinical_text": False,
        "contains_lab_values": False,
    }
    guard = lab_primary_guard(data_root=tmp_path, today=date(2026, 8, 26))
    assert guard["allowed"] is True
    assert guard["effective"] is True
    assert lab_primary_enabled() is True
    report["review_pack"]["finding_decisions"] = {
        "confirmed": 3,
        "false_positive": 2,
    }
    rollout_report_path(tmp_path).write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    blocked = lab_primary_guard(data_root=tmp_path, today=date(2026, 8, 26))
    assert blocked["allowed"] is False
    assert blocked["false_positive_pct"] == 40.0
    assert "false_positive_rate_high" in blocked["block_reasons"]
