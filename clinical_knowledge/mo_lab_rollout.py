"""PHI-safe rollout telemetry and guard for MO laboratory findings."""
from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ENGINE = "mo_lab_rollout_v1"
LAB_CODES = (
    "B_lab_ordered_already_done",
    "B_lab_present_not_in_mo",
)
DEFAULT_SHADOW_DAYS = 7
DEFAULT_MIN_REVIEWED = 5
DEFAULT_MAX_FALSE_POSITIVE_PCT = 20.0


def _data_root(data_root: Path | str | None = None) -> Path:
    if data_root is not None:
        return Path(data_root)
    return Path(os.environ.get("MO_DATA_ROOT") or "/var/data/medical_exams")


def shadow_state_path(data_root: Path | str | None = None) -> Path:
    return _data_root(data_root) / "state" / "mo_lab_shadow_since.json"


def rollout_report_path(data_root: Path | str | None = None) -> Path:
    return _data_root(data_root) / "reports" / "mo_lab_rollout_latest.json"


def minimum_shadow_days() -> int:
    try:
        value = int(os.environ.get("MO_LAB_MIN_SHADOW_DAYS") or DEFAULT_SHADOW_DAYS)
    except ValueError:
        return DEFAULT_SHADOW_DAYS
    return max(1, value)


def ensure_shadow_state(
    *,
    git_commit_sha: str = "",
    data_root: Path | str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    path = shadow_state_path(data_root)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    current = now or datetime.now(timezone.utc)
    payload = {
        "engine": ENGINE,
        "shadow_since": current.date().isoformat(),
        "first_deploy_at": current.replace(microsecond=0).isoformat(),
        "git_commit_sha": str(git_commit_sha or "")[:40],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return payload


def read_rollout_report(data_root: Path | str | None = None) -> dict[str, Any]:
    path = rollout_report_path(data_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def lab_primary_guard(
    *,
    data_root: Path | str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    requested = (os.environ.get("MO_LAB_IN_PRIMARY") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    min_days = minimum_shadow_days()
    current = today or datetime.now(timezone.utc).date()
    state: dict[str, Any] = {}
    try:
        state = json.loads(shadow_state_path(data_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    shadow_since = str(state.get("shadow_since") or "")[:10]
    try:
        days_in_shadow = max(0, (current - date.fromisoformat(shadow_since)).days + 1)
    except ValueError:
        days_in_shadow = 0
    report = read_rollout_report(data_root)
    successful_nights = int(
        (report.get("guard_inputs") or {}).get("successful_lab_nights") or 0
    )
    report_date = str(report.get("generated_date") or "")[:10]
    try:
        report_age_days = max(0, (current - date.fromisoformat(report_date)).days)
    except ValueError:
        report_age_days = 999
    decisions = (report.get("review_pack") or {}).get("finding_decisions") or {}
    reviewed_n = sum(
        int(decisions.get(key) or 0)
        for key in ("confirmed", "false_positive", "needs_more_data")
    )
    false_positive_n = int(decisions.get("false_positive") or 0)
    false_positive_pct = (
        round(100.0 * false_positive_n / reviewed_n, 1) if reviewed_n else None
    )
    try:
        min_reviewed = max(
            1, int(os.environ.get("MO_LAB_MIN_REVIEWED") or DEFAULT_MIN_REVIEWED)
        )
    except ValueError:
        min_reviewed = DEFAULT_MIN_REVIEWED
    try:
        max_false_positive_pct = float(
            os.environ.get("MO_LAB_MAX_FALSE_POSITIVE_PCT")
            or DEFAULT_MAX_FALSE_POSITIVE_PCT
        )
    except ValueError:
        max_false_positive_pct = DEFAULT_MAX_FALSE_POSITIVE_PCT
    reasons: list[str] = []
    if not shadow_since:
        reasons.append("shadow_state_missing")
    if days_in_shadow < min_days:
        reasons.append("shadow_period_incomplete")
    if successful_nights < min_days:
        reasons.append("successful_nights_incomplete")
    if report_age_days > 1:
        reasons.append("rollout_report_stale")
    if reviewed_n < min_reviewed:
        reasons.append("review_sample_incomplete")
    if (
        reviewed_n >= min_reviewed
        and false_positive_pct is not None
        and false_positive_pct > max_false_positive_pct
    ):
        reasons.append("false_positive_rate_high")
    allowed = not reasons
    return {
        "engine": ENGINE,
        "requested": requested,
        "allowed": allowed,
        "effective": requested and allowed,
        "shadow_since": shadow_since or None,
        "days_in_shadow": days_in_shadow,
        "minimum_shadow_days": min_days,
        "successful_lab_nights": successful_nights,
        "report_age_days": report_age_days if report_age_days < 999 else None,
        "reviewed_n": reviewed_n,
        "false_positive_pct": false_positive_pct,
        "max_false_positive_pct": max_false_positive_pct,
        "block_reasons": reasons,
    }


def _successful_lab_nights(root: Path, *, end: date, days: int) -> int:
    success = 0
    for offset in range(days):
        day = end - timedelta(days=offset)
        path = root / "state" / f"gce_lab_{day.isoformat()}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if payload.get("status") == "success":
            success += 1
    return success


def build_rollout_report(
    *,
    analytics_db: Path | str,
    lab_db: Path | str,
    data_root: Path | str | None = None,
    end_date: date | None = None,
    days: int = DEFAULT_SHADOW_DAYS,
) -> dict[str, Any]:
    root = _data_root(data_root)
    end = end_date or datetime.now(timezone.utc).date()
    start = end - timedelta(days=max(1, days) - 1)
    metrics: dict[str, Any] = {
        "case_n": 0,
        "finding_n": 0,
        "cases_with_lab_finding_n": 0,
        "finding_by_code": {code: 0 for code in LAB_CODES},
        "same_day_lab_case_n": 0,
    }
    review_counts: Counter[str] = Counter()
    pack_with_lab_n = 0
    conn = sqlite3.connect(f"file:{Path(analytics_db)}?mode=ro", uri=True)
    try:
        metrics["case_n"] = int(
            conn.execute(
                "SELECT COUNT(*) FROM fact_mo_case WHERE visit_date BETWEEN ? AND ?",
                (start.isoformat(), end.isoformat()),
            ).fetchone()[0]
        )
        rows = conn.execute(
            """
            SELECT f.finding_code, COUNT(*) AS n
            FROM fact_mo_finding f
            JOIN fact_mo_case c ON c.mis_id=f.mis_id
            WHERE c.visit_date BETWEEN ? AND ?
              AND f.finding_code IN (?, ?)
            GROUP BY f.finding_code
            """,
            (start.isoformat(), end.isoformat(), *LAB_CODES),
        ).fetchall()
        for code, count in rows:
            metrics["finding_by_code"][str(code)] = int(count or 0)
            metrics["finding_n"] += int(count or 0)
        metrics["cases_with_lab_finding_n"] = int(
            conn.execute(
                """
                SELECT COUNT(DISTINCT f.mis_id)
                FROM fact_mo_finding f
                JOIN fact_mo_case c ON c.mis_id=f.mis_id
                WHERE c.visit_date BETWEEN ? AND ?
                  AND f.finding_code IN (?, ?)
                """,
                (start.isoformat(), end.isoformat(), *LAB_CODES),
            ).fetchone()[0]
        )
        conn.execute("ATTACH DATABASE ? AS labdb", (f"file:{Path(lab_db)}?mode=ro",))
        metrics["same_day_lab_case_n"] = int(
            conn.execute(
                """
                SELECT COUNT(DISTINCT c.mis_id)
                FROM fact_mo_case c
                WHERE c.visit_date BETWEEN ? AND ?
                  AND EXISTS (
                    SELECT 1 FROM labdb.fact_mo_lab l
                    WHERE l.patient_key=c.patient_key AND l.test_date=c.visit_date
                  )
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchone()[0]
        )
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='crm_review_pack'"
        ).fetchone()
        if table_exists:
            for (raw,) in conn.execute(
                "SELECT decision_json FROM crm_review_pack WHERE visit_date BETWEEN ? AND ?",
                (start.isoformat(), end.isoformat()),
            ):
                try:
                    decision = json.loads(raw or "{}")
                except (json.JSONDecodeError, TypeError):
                    continue
                lab_decisions = {
                    code: value
                    for code, value in (decision.get("finding_decisions") or {}).items()
                    if str(code) in LAB_CODES
                }
                if lab_decisions:
                    pack_with_lab_n += 1
                    review_counts.update(str(value) for value in lab_decisions.values())
    finally:
        conn.close()
    successful_nights = _successful_lab_nights(root, end=end, days=days)
    generated = datetime.now(timezone.utc).replace(microsecond=0)
    report = {
        "engine": ENGINE,
        "generated_at": generated.isoformat(),
        "generated_date": end.isoformat(),
        "window": {"from": start.isoformat(), "to": end.isoformat(), "days": days},
        "metrics": metrics,
        "review_pack": {
            "pack_with_lab_n": pack_with_lab_n,
            "finding_decisions": dict(sorted(review_counts.items())),
        },
        "guard_inputs": {
            "successful_lab_nights": successful_nights,
            "required_nights": minimum_shadow_days(),
        },
        "phi_check": {
            "contains_row_identifiers": False,
            "contains_clinical_text": False,
            "contains_lab_values": False,
        },
    }
    path = rollout_report_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return report
