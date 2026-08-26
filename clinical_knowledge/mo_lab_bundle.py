"""Лаборатория клиента для разбора МО из warehouse/mo_lab.sqlite.

План: docs/plans/2026-08-26-mo-lab-from-mis-tests-v1.md
Канон: клеим к patient_key + окну дат, не к visit_id.
Пустой блок лучше, чем чужие результаты. В exam_data не пишем.
Флаги: MO_LAB_BUNDLE (default 1), MO_LAB_IN_PRIMARY (default 0).
"""
from __future__ import annotations

import os
import sqlite3
from collections import OrderedDict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

ENGINE = "mo_lab_v1"
DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_LOOKAHEAD_DAYS = 1
ROW_CAP = 400
USAGE_FOR_SCORES_RU = (
    "Лаборатория - контекст для методиста из склада mis_tests. "
    "По умолчанию не меняет итоговую оценку (MO_LAB_IN_PRIMARY=0). "
    "Флаг включает только замечание «анализы есть, в МО не указаны» (P3). "
    "Не подменяет графу «Данные обследований» и не ставит «плохой анализ» без референса."
)


def lab_bundle_enabled() -> bool:
    raw = (os.environ.get("MO_LAB_BUNDLE") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def lab_primary_enabled() -> bool:
    """Документационный gap в primary. «Плохой анализ» из value - нет (волна 4)."""
    raw = (os.environ.get("MO_LAB_IN_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def lookback_days() -> int:
    raw = (os.environ.get("MO_LAB_LOOKBACK_DAYS") or "").strip()
    if not raw:
        return DEFAULT_LOOKBACK_DAYS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_LOOKBACK_DAYS
    return value if value > 0 else DEFAULT_LOOKBACK_DAYS


def lookahead_days() -> int:
    raw = (os.environ.get("MO_LAB_LOOKAHEAD_DAYS") or "").strip()
    if not raw:
        return DEFAULT_LOOKAHEAD_DAYS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_LOOKAHEAD_DAYS
    return value if value >= 0 else DEFAULT_LOOKAHEAD_DAYS


def default_lab_path() -> Path | None:
    env = (os.environ.get("MO_LAB_DB") or "").strip()
    if env:
        path = Path(env)
        return path if path.is_file() else None
    roots: list[Path] = []
    data_root = (os.environ.get("MO_DATA_ROOT") or "").strip()
    if data_root:
        roots.append(Path(data_root))
    roots.append(Path("/var/data/medical_exams"))
    for root in roots:
        path = root / "warehouse" / "mo_lab.sqlite"
        if path.is_file():
            return path
    return None


def empty_bundle(*, reason: str) -> dict[str, Any]:
    return {
        "engine": ENGINE,
        "enabled": lab_bundle_enabled(),
        "window": {},
        "summary": {"n_rows": 0, "n_dates": 0, "n_types": 0, "same_day_rows": 0},
        "days": [],
        "reason": reason,
        "usage_for_scores_ru": USAGE_FOR_SCORES_RU,
    }


def _clip(value: Any, n: int) -> str:
    return str(value or "")[:n]


def _window_for(visit_date: str) -> tuple[str, str, dict[str, Any]] | None:
    day = str(visit_date or "")[:10]
    try:
        visit = date.fromisoformat(day)
    except ValueError:
        return None
    back = lookback_days()
    ahead = lookahead_days()
    start = (visit - timedelta(days=back)).isoformat()
    end = (visit + timedelta(days=ahead)).isoformat()
    return start, end, {
        "from": start,
        "to": end,
        "visit_date": visit.isoformat(),
        "lookback_days": back,
        "lookahead_days": ahead,
    }


def _group_rows(rows: list[tuple], *, visit_date: str) -> list[dict[str, Any]]:
    by_date: OrderedDict[str, OrderedDict[str, dict[str, Any]]] = OrderedDict()
    for row in rows:
        test_date = str(row[0] or "")[:10]
        type_id = int(row[2] or 0)
        type_name = _clip(row[3], 160) or "анализ"
        type_key = f"{type_id}:{type_name}"
        if test_date not in by_date:
            by_date[test_date] = OrderedDict()
        types = by_date[test_date]
        if type_key not in types:
            types[type_key] = {
                "type_id": type_id,
                "type_name": type_name,
                "indicators": [],
            }
        types[type_key]["indicators"].append(
            {
                "name": _clip(row[5], 120) or "показатель",
                "value": _clip(row[6], 80),
                "unit": _clip(row[7], 24),
            }
        )
    days: list[dict[str, Any]] = []
    for test_date, types in by_date.items():
        days.append(
            {
                "test_date": test_date,
                "same_day": test_date == visit_date,
                "types": list(types.values()),
            }
        )
    return days


def build_lab_bundle(
    *,
    patient_key: str = "",
    visit_date: str = "",
    lab_db: Path | str | sqlite3.Connection | None = None,
) -> dict[str, Any]:
    if not lab_bundle_enabled():
        return empty_bundle(reason="disabled")
    key = str(patient_key or "").strip()
    if not key:
        return empty_bundle(reason="missing_key")
    window = _window_for(visit_date)
    if window is None:
        return empty_bundle(reason="missing_date")
    start, end, window_meta = window
    own_conn = False
    conn: sqlite3.Connection | None = None
    path: Path | None = None
    if isinstance(lab_db, sqlite3.Connection):
        conn = lab_db
    elif lab_db is not None:
        path = Path(lab_db)
    else:
        path = default_lab_path()
    if conn is None:
        if path is None or not path.is_file():
            return empty_bundle(reason="db_missing")
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            own_conn = True
        except sqlite3.Error:
            return empty_bundle(reason="db_missing")
    try:
        rows = conn.execute(
            """
            SELECT test_date, test_id, type_id, type_name,
                   indicator_id, indicator_name, value, unit
            FROM fact_mo_lab
            WHERE patient_key = ?
              AND test_date >= ?
              AND test_date <= ?
            ORDER BY test_date DESC, type_name, indicator_name
            LIMIT ?
            """,
            (key, start, end, ROW_CAP),
        ).fetchall()
    except sqlite3.Error:
        if own_conn and conn is not None:
            conn.close()
        return empty_bundle(reason="db_missing")
    if own_conn and conn is not None:
        conn.close()
    if not rows:
        out = empty_bundle(reason="empty")
        out["window"] = window_meta
        return out
    days = _group_rows(rows, visit_date=window_meta["visit_date"])
    type_names = {
        item["type_name"]
        for day in days
        for item in day.get("types") or []
    }
    same_day_rows = sum(
        len(item.get("indicators") or [])
        for day in days
        if day.get("same_day")
        for item in day.get("types") or []
    )
    return {
        "engine": ENGINE,
        "enabled": True,
        "window": window_meta,
        "summary": {
            "n_rows": len(rows),
            "n_dates": len(days),
            "n_types": len(type_names),
            "same_day_rows": same_day_rows,
            "truncated": len(rows) >= ROW_CAP,
        },
        "days": days,
        "reason": "",
        "usage_for_scores_ru": USAGE_FOR_SCORES_RU,
    }


def build_lab_reconcile_bundle(
    *,
    patient_key: str = "",
    visit_date: str = "",
    lab_db: Path | str | sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Полный panel/indicator index окна без values и UI ROW_CAP.

    Reconcile нельзя считать по обрезанному display payload: у пациентов с
    большим числом строк это даёт ложные пропуски панелей.
    """
    if not lab_bundle_enabled():
        return empty_bundle(reason="disabled")
    key = str(patient_key or "").strip()
    if not key:
        return empty_bundle(reason="missing_key")
    window = _window_for(visit_date)
    if window is None:
        return empty_bundle(reason="missing_date")
    start, end, window_meta = window
    own_conn = False
    conn: sqlite3.Connection | None = None
    path: Path | None = None
    if isinstance(lab_db, sqlite3.Connection):
        conn = lab_db
    elif lab_db is not None:
        path = Path(lab_db)
    else:
        path = default_lab_path()
    if conn is None:
        if path is None or not path.is_file():
            return empty_bundle(reason="db_missing")
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            own_conn = True
        except sqlite3.Error:
            return empty_bundle(reason="db_missing")
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT test_date, 0, type_id, type_name,
                            indicator_id, indicator_name, '', ''
            FROM fact_mo_lab
            WHERE patient_key = ?
              AND test_date >= ?
              AND test_date <= ?
            ORDER BY test_date DESC, type_name, indicator_name
            """,
            (key, start, end),
        ).fetchall()
    except sqlite3.Error:
        if own_conn and conn is not None:
            conn.close()
        return empty_bundle(reason="db_missing")
    if own_conn and conn is not None:
        conn.close()
    if not rows:
        out = empty_bundle(reason="empty")
        out["window"] = window_meta
        return out
    days = _group_rows(rows, visit_date=window_meta["visit_date"])
    return {
        "engine": ENGINE,
        "enabled": True,
        "window": window_meta,
        "summary": {
            "n_rows": len(rows),
            "n_dates": len(days),
            "n_types": len(
                {
                    item["type_name"]
                    for day in days
                    for item in day.get("types") or []
                }
            ),
            "same_day_rows": sum(
                len(item.get("indicators") or [])
                for day in days
                if day.get("same_day")
                for item in day.get("types") or []
            ),
            "truncated": False,
        },
        "days": days,
        "reason": "",
        "usage_for_scores_ru": USAGE_FOR_SCORES_RU,
    }


def public_lab_for_ui(bundle: Mapping[str, Any] | None) -> dict[str, Any]:
    """Публичный объект для API/UI без patient_key / patient_id."""
    if not isinstance(bundle, Mapping):
        return empty_bundle(reason="empty")
    days = []
    for day in list(bundle.get("days") or [])[:12]:
        if not isinstance(day, Mapping):
            continue
        types = []
        for item in list(day.get("types") or [])[:24]:
            if not isinstance(item, Mapping):
                continue
            indicators = [
                {
                    "name": _clip(ind.get("name"), 120),
                    "value": _clip(ind.get("value"), 80),
                    "unit": _clip(ind.get("unit"), 24),
                }
                for ind in list(item.get("indicators") or [])[:40]
                if isinstance(ind, Mapping)
            ]
            types.append(
                {
                    "type_id": int(item.get("type_id") or 0),
                    "type_name": _clip(item.get("type_name"), 160),
                    "indicators": indicators,
                }
            )
        days.append(
            {
                "test_date": str(day.get("test_date") or "")[:10],
                "same_day": bool(day.get("same_day")),
                "types": types,
            }
        )
    summary = dict(bundle.get("summary") or {})
    window = dict(bundle.get("window") or {})
    return {
        "engine": ENGINE,
        "enabled": bool(bundle.get("enabled", True)),
        "window": window,
        "summary": {
            "n_rows": int(summary.get("n_rows") or 0),
            "n_dates": int(summary.get("n_dates") or 0),
            "n_types": int(summary.get("n_types") or 0),
            "same_day_rows": int(summary.get("same_day_rows") or 0),
            "truncated": bool(summary.get("truncated")),
        },
        "days": days,
        "reason": str(bundle.get("reason") or ""),
        "usage_for_scores_ru": USAGE_FOR_SCORES_RU,
    }


def attach_lab_to_case(
    case: dict[str, Any],
    *,
    lab_db: Path | str | sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Положить бандл в case['_lab'] для разбора; patient_id в бандл не кладём."""
    if not isinstance(case, dict):
        bundle = empty_bundle(reason="bad_case")
        return bundle
    from clinical_knowledge.mo_daily import patient_key_for

    patient_id = str(
        case.get("patient_id")
        or case.get("patientId")
        or (case.get("raw") or {}).get("patient_id")
        or ""
    ).strip()
    patient_key = str(case.get("patient_key") or "").strip() or patient_key_for(patient_id)
    visit = str(case.get("visit_date") or case.get("date") or "")[:10]
    bundle = build_lab_bundle(
        patient_key=patient_key,
        visit_date=visit,
        lab_db=lab_db,
    )
    case["_lab"] = bundle
    return bundle


def lab_payload_for_case(
    case: dict[str, Any],
    *,
    lab_db: Path | str | sqlite3.Connection | None = None,
) -> dict[str, Any]:
    return public_lab_for_ui(attach_lab_to_case(case, lab_db=lab_db))


def lab_reconcile_payload_for_case(
    case: Mapping[str, Any],
    *,
    lab_db: Path | str | sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Полный безопасный payload для reconcile, без identity и values."""
    from clinical_knowledge.mo_daily import patient_key_for

    patient_id = str(
        case.get("patient_id")
        or case.get("patientId")
        or (case.get("raw") or {}).get("patient_id")
        or ""
    ).strip()
    patient_key = str(case.get("patient_key") or "").strip() or patient_key_for(patient_id)
    visit = str(case.get("visit_date") or case.get("date") or "")[:10]
    return build_lab_reconcile_bundle(
        patient_key=patient_key,
        visit_date=visit,
        lab_db=lab_db,
    )
