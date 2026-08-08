"""История пациента у врача / специальности → бандл + одно shadow МО.

План: docs/plans/2026-08-08-mo-patient-history-bundle-v2.md
Флаги: MO_PATIENT_HISTORY_BUNDLE (default 1), MO_PATIENT_HISTORY_IN_PRIMARY (default 0).
patient_id наружу / в логи не отдаём - только patient_key внутри склада.
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ENGINE = "mo_patient_history_v1"
_SOURCE = "mo_patient_history_v1"
FINDING_CODE = "B_patient_history_context"

TIER_KNOWN_DOCTOR = "known_to_doctor"
TIER_KNOWN_SPECIALTY = "known_in_specialty_only"
TIER_NEW_PROFILE = "new_for_profile"
TIER_FIRST_CONTACT = "first_contact"
TIER_INSUFFICIENT = "insufficient"


def patient_history_enabled() -> bool:
    raw = (os.environ.get("MO_PATIENT_HISTORY_BUNDLE") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def patient_history_primary_enabled() -> bool:
    raw = (os.environ.get("MO_PATIENT_HISTORY_IN_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def history_lookback_days() -> int | None:
    raw = (os.environ.get("MO_PATIENT_HISTORY_LOOKBACK_DAYS") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def patient_key_for(patient_id: Any) -> str:
    """Hash patient_id для склада (как doctor_key_for для ФИО)."""
    from clinical_knowledge.mo_daily import patient_key_for as _pk

    return _pk(patient_id)


def default_warehouse_path() -> Path | None:
    roots: list[Path] = []
    env = (os.environ.get("MO_DATA_ROOT") or "").strip()
    if env:
        roots.append(Path(env))
    roots.append(Path("/var/data/medical_exams"))
    for root in roots:
        path = root / "warehouse" / "mo_analytics.sqlite"
        if path.is_file():
            return path
    return None


def _norm_specialty(value: Any) -> str:
    return str(value or "").strip().lower()


def _visit_row_public(row: Mapping[str, Any]) -> dict[str, Any]:
    pct = row.get("overall_pct")
    try:
        pct_f = float(pct) if pct is not None else None
    except (TypeError, ValueError):
        pct_f = None
    return {
        "mis_id": str(row.get("mis_id") or ""),
        "visit_id": str(row.get("visit_id") or ""),
        "visit_date": str(row.get("visit_date") or "")[:10],
        "doctor_key": str(row.get("doctor_key") or ""),
        "doctor_id": str(row.get("doctor_id") or ""),
        "specialty": str(row.get("specialty") or ""),
        "diagnosis_code": str(row.get("diagnosis_code") or ""),
        "diagnosis_text": str(row.get("diagnosis_text") or "")[:200],
        "overall_pct": pct_f,
        "document_kind": str(row.get("document_kind") or ""),
    }


def _same_doctor(row: Mapping[str, Any], *, doctor_id: str, doctor_key: str) -> bool:
    row_doc_id = str(row.get("doctor_id") or "").strip()
    row_key = str(row.get("doctor_key") or "").strip()
    if doctor_id and row_doc_id and row_doc_id == doctor_id:
        return True
    if doctor_key and row_key and row_key == doctor_key:
        return True
    return False


def _aggregate_summary(
    same_doctor: list[dict[str, Any]],
    same_specialty: list[dict[str, Any]],
    other: list[dict[str, Any]],
    *,
    current_code: str,
) -> dict[str, Any]:
    code = (current_code or "").strip().upper()

    def _codes(rows: list[dict[str, Any]]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for row in rows:
            c = str(row.get("diagnosis_code") or "").strip().upper()
            if c:
                counter[c] += 1
        return dict(counter.most_common(40))

    codes_doc = _codes(same_doctor)
    codes_spec = _codes(same_specialty)
    seen_doc = bool(code and codes_doc.get(code))
    seen_spec = bool(code and (codes_doc.get(code) or codes_spec.get(code)))
    return {
        "n_same_doctor": len(same_doctor),
        "n_same_specialty": len(same_specialty),
        "n_other": len(other),
        "n_visits": len(same_doctor) + len(same_specialty) + len(other),
        "codes_same_doctor": codes_doc,
        "codes_same_specialty": codes_spec,
        "last_same_doctor_date": same_doctor[-1]["visit_date"] if same_doctor else "",
        "last_same_specialty_date": same_specialty[-1]["visit_date"] if same_specialty else "",
        "current_code": code,
        "current_code_seen_by_doctor": seen_doc,
        "current_code_seen_in_specialty": seen_spec,
    }


def _tier_from_summary(summary: Mapping[str, Any]) -> str:
    if int(summary.get("n_visits") or 0) <= 0 and not summary.get("current_code"):
        # ещё может быть first_contact с n=0
        pass
    if int(summary.get("n_visits") or 0) < 0:
        return TIER_INSUFFICIENT
    if summary.get("current_code_seen_by_doctor"):
        return TIER_KNOWN_DOCTOR
    if int(summary.get("n_same_doctor") or 0) == 0:
        return TIER_FIRST_CONTACT
    if summary.get("current_code_seen_in_specialty"):
        return TIER_KNOWN_SPECIALTY
    return TIER_NEW_PROFILE


def empty_bundle(*, reason: str = "insufficient") -> dict[str, Any]:
    summary = {
        "n_same_doctor": 0,
        "n_same_specialty": 0,
        "n_other": 0,
        "n_visits": 0,
        "codes_same_doctor": {},
        "codes_same_specialty": {},
        "last_same_doctor_date": "",
        "last_same_specialty_date": "",
        "current_code": "",
        "current_code_seen_by_doctor": False,
        "current_code_seen_in_specialty": False,
    }
    return {
        "engine": ENGINE,
        "same_doctor": [],
        "same_specialty": [],
        "other": [],
        "summary": summary,
        "coverage": {"first_date": "", "last_date": "", "n_visits": 0},
        "tier": TIER_INSUFFICIENT,
        "reason": reason,
    }


def build_patient_history_bundle(
    *,
    patient_id: str = "",
    patient_key: str = "",
    as_of_date: str,
    doctor_id: str = "",
    doctor_key: str = "",
    specialty: str = "",
    current_code: str = "",
    exclude_ids: set[str] | None = None,
    lookback_days: int | None = None,
    warehouse: Path | str | sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Собрать бандл истории as-of даты случая. Без patient_id в результате."""
    key = (patient_key or "").strip() or patient_key_for(patient_id)
    day = (as_of_date or "")[:10]
    if not key or len(day) < 10:
        return empty_bundle(reason="missing_patient_or_date")

    excluded = {str(x).strip() for x in (exclude_ids or set()) if str(x or "").strip()}
    lookback = history_lookback_days() if lookback_days is None else lookback_days
    own_conn = False
    db: sqlite3.Connection | None
    if isinstance(warehouse, sqlite3.Connection):
        db = warehouse
    else:
        path = Path(warehouse) if warehouse else default_warehouse_path()
        if path is None or not Path(path).is_file():
            return empty_bundle(reason="no_warehouse")
        db = sqlite3.connect(str(path))
        db.row_factory = sqlite3.Row
        own_conn = True

    try:
        cols = {row[1] for row in db.execute("PRAGMA table_info(fact_mo_case)")}
        if "patient_key" not in cols:
            return empty_bundle(reason="schema_no_patient_key")
        select_cols = [
            "mis_id",
            "visit_id",
            "visit_date",
            "doctor_key",
            "specialty",
            "diagnosis_code",
            "overall_pct",
            "document_kind",
        ]
        for optional in ("doctor_id", "diagnosis_text"):
            if optional in cols:
                select_cols.append(optional)
        sql = (
            f'SELECT {", ".join(select_cols)} FROM fact_mo_case '
            "WHERE patient_key = ? AND visit_date < ?"
        )
        params: list[Any] = [key, day]
        if lookback is not None:
            # опциональный потолок: visit_date >= as_of - lookback
            sql += " AND visit_date >= date(?, ?)"
            params.extend([day, f"-{int(lookback)} days"])
        sql += " ORDER BY visit_date ASC, mis_id ASC"
        cursor = db.execute(sql, params)
        col_names = [d[0] for d in cursor.description]
        rows = []
        for raw_row in cursor.fetchall():
            if isinstance(raw_row, sqlite3.Row):
                rows.append(dict(raw_row))
            elif isinstance(raw_row, Mapping):
                rows.append(dict(raw_row))
            else:
                rows.append({col_names[i]: raw_row[i] for i in range(len(col_names))})
    finally:
        if own_conn and db is not None:
            db.close()

    same_doctor: list[dict[str, Any]] = []
    same_specialty: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    spec_norm = _norm_specialty(specialty)
    for raw in rows:
        public = _visit_row_public(raw)
        ids = {public["mis_id"], public["visit_id"]} - {""}
        if ids & excluded:
            continue
        if _same_doctor(public, doctor_id=str(doctor_id or "").strip(), doctor_key=str(doctor_key or "").strip()):
            same_doctor.append(public)
        elif spec_norm and _norm_specialty(public.get("specialty")) == spec_norm:
            same_specialty.append(public)
        else:
            other.append(public)

    summary = _aggregate_summary(
        same_doctor, same_specialty, other, current_code=current_code
    )
    all_dates = [
        r["visit_date"]
        for r in (*same_doctor, *same_specialty, *other)
        if r.get("visit_date")
    ]
    tier = _tier_from_summary(summary)
    if summary["n_visits"] == 0 and not key:
        tier = TIER_INSUFFICIENT
    return {
        "engine": ENGINE,
        "same_doctor": same_doctor,
        "same_specialty": same_specialty,
        "other": other,
        "summary": summary,
        "coverage": {
            "first_date": min(all_dates) if all_dates else "",
            "last_date": max(all_dates) if all_dates else "",
            "n_visits": summary["n_visits"],
            "lookback_days": lookback,
        },
        "tier": tier,
        "reason": "",
    }


def history_summary_for_analyzers(bundle: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(bundle, Mapping):
        return {}
    summary = dict(bundle.get("summary") or {})
    summary["tier"] = str(bundle.get("tier") or "")
    summary["coverage"] = dict(bundle.get("coverage") or {})
    return summary


def public_bundle_for_ui(bundle: Mapping[str, Any] | None) -> dict[str, Any]:
    """Публичный объект для API/UI без patient_id."""
    if not isinstance(bundle, Mapping):
        return empty_bundle(reason="empty")
    return {
        "engine": ENGINE,
        "tier": str(bundle.get("tier") or TIER_INSUFFICIENT),
        "summary": dict(bundle.get("summary") or {}),
        "coverage": dict(bundle.get("coverage") or {}),
        "same_doctor": list(bundle.get("same_doctor") or [])[:40],
        "same_specialty": list(bundle.get("same_specialty") or [])[:40],
        "other": list(bundle.get("other") or [])[:20],
        "reason": str(bundle.get("reason") or ""),
    }


def _history_detail(bundle: Mapping[str, Any]) -> str:
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), Mapping) else {}
    coverage = bundle.get("coverage") if isinstance(bundle.get("coverage"), Mapping) else {}
    code = str(summary.get("current_code") or "")
    n_doc = int(summary.get("n_same_doctor") or 0)
    n_spec = int(summary.get("n_same_specialty") or 0)
    n_all = int(summary.get("n_visits") or 0)
    first = str(coverage.get("first_date") or "")
    period = f" Период склада: с {first}." if first else ""
    codes_doc = summary.get("codes_same_doctor") if isinstance(summary.get("codes_same_doctor"), dict) else {}
    code_times = int(codes_doc.get(code) or 0) if code else 0
    tier = str(bundle.get("tier") or "")
    if tier == TIER_INSUFFICIENT:
        return "Истории недостаточно (нет patient_key) - историческое МО не штрафует."
    if tier == TIER_FIRST_CONTACT:
        extra = ""
        if n_spec:
            top = ", ".join(list((summary.get("codes_same_specialty") or {}).keys())[:3])
            extra = f" У других врачей этой специальности - {n_spec} визит(ов)" + (
                f" с кодами {top}." if top else "."
            )
        return (
            f"К этому врачу пациент впервые (визитов у врача: 0). "
            f"На складе до этого случая - {n_all} визит(ов).{extra}{period}"
        )
    if tier == TIER_KNOWN_DOCTOR:
        return (
            f"У этого врача по пациенту уже {n_doc} визит(ов); "
            f"код {code or '—'} ставился {code_times} раз(а). "
            f"У других этой специальности - ещё {n_spec} визит(ов). "
            f"Всего на складе до случая: {n_all}.{period}"
        )
    if tier == TIER_KNOWN_SPECIALTY:
        return (
            f"У этого врача по пациенту код {code or '—'} ещё не ставился "
            f"({n_doc} визит(ов) с другими кодами); у коллег специальности код уже был. "
            f"Всего визитов до случая: {n_all}.{period}"
        )
    return (
        f"У этого врача по пациенту уже {n_doc} визит(ов); код {code or '—'} новый "
        f"для профиля врача/специальности по этому пациенту. "
        f"Всего визитов до случая: {n_all}.{period}"
    )


def evaluate_history_mo(
    bundle: Mapping[str, Any] | None,
    *,
    current_code: str = "",
) -> list[dict[str, Any]]:
    """0 или 1 finding. insufficient → пусто (не штрафуем)."""
    if not isinstance(bundle, Mapping):
        return []
    tier = str(bundle.get("tier") or TIER_INSUFFICIENT)
    if tier == TIER_INSUFFICIENT:
        return []
    summary = dict(bundle.get("summary") or {})
    if current_code and not summary.get("current_code"):
        summary["current_code"] = current_code.strip().upper()
        bundle = {**bundle, "summary": summary, "tier": _tier_from_summary(summary)}
        tier = str(bundle.get("tier") or tier)
    severity = "P3"
    if tier in {TIER_NEW_PROFILE, TIER_FIRST_CONTACT}:
        severity = "P2"
    title = {
        TIER_KNOWN_DOCTOR: "История: код уже был у этого врача",
        TIER_KNOWN_SPECIALTY: "История: код был у коллег специальности",
        TIER_NEW_PROFILE: "История: новый код для профиля пациента",
        TIER_FIRST_CONTACT: "История: первый контакт с этим врачом",
    }.get(tier, "История пациента")
    return [
        {
            "code": FINDING_CODE,
            "axis": "patient_history",
            "severity": severity,
            "passed": tier == TIER_KNOWN_DOCTOR,
            "title_ru": title,
            "detail_ru": _history_detail(bundle),
            "evidence": "",
            "source_ref": _SOURCE,
            "needs_human": tier in {TIER_NEW_PROFILE, TIER_FIRST_CONTACT},
            "shadow": True,
            "engine": ENGINE,
            "history_tier": tier,
            "linked_fields": ["clinical_diagnosis", "mkb_code_main"],
            "link_hint_ru": "Сверьте текущий диагноз с лентой визитов пациента",
        }
    ]


def attach_bundle_to_case(
    case: dict[str, Any],
    *,
    warehouse: Path | str | sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Положить бандл в case['_patient_history'] (один раз)."""
    if not isinstance(case, dict):
        return empty_bundle(reason="bad_case")
    existing = case.get("_patient_history")
    if isinstance(existing, Mapping) and existing.get("engine") == ENGINE:
        return dict(existing)

    patient_id = str(
        case.get("patient_id")
        or case.get("patientId")
        or (case.get("raw") or {}).get("patient_id")
        or ""
    ).strip()
    patient_key = str(case.get("patient_key") or "").strip() or patient_key_for(patient_id)
    as_of = str(case.get("visit_date") or case.get("date") or "")[:10]
    doctor_id = str(
        case.get("doctor_id")
        or case.get("specialist_id_from_visit")
        or (case.get("raw") or {}).get("doctor_id")
        or (case.get("raw") or {}).get("specialist_id_from_visit")
        or ""
    ).strip()
    doctor_fio = str(case.get("doctor_fio") or (case.get("raw") or {}).get("doctor_fio") or "")
    from clinical_knowledge.mo_daily import doctor_key_for

    doctor_key = str(case.get("doctor_key") or "").strip() or doctor_key_for(doctor_fio)
    specialty = str(
        case.get("specialty")
        or case.get("doctor_specialization")
        or (case.get("raw") or {}).get("doctor_specialization")
        or ""
    )
    current_code = str(
        case.get("diagnosis_code")
        or case.get("mkb_code_main")
        or (case.get("raw") or {}).get("mkb_code_main")
        or ""
    ).strip().upper()
    exclude = {
        str(case.get("mis_id") or case.get("id") or ""),
        str(case.get("visit_id") or ""),
    } - {""}

    bundle = build_patient_history_bundle(
        patient_id=patient_id,
        patient_key=patient_key,
        as_of_date=as_of,
        doctor_id=doctor_id,
        doctor_key=doctor_key,
        specialty=specialty,
        current_code=current_code,
        exclude_ids=exclude,
        warehouse=warehouse,
    )
    case["_patient_history"] = bundle
    case["_patient_history_summary"] = history_summary_for_analyzers(bundle)
    return bundle


def evaluate_mo_patient_history(
    case: dict[str, Any],
    *,
    warehouse: Path | str | sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    if not patient_history_enabled():
        return []
    bundle = attach_bundle_to_case(case, warehouse=warehouse)
    code = str(case.get("diagnosis_code") or case.get("mkb_code_main") or "").strip().upper()
    return evaluate_history_mo(bundle, current_code=code)


def merge_patient_history_into_findings(
    findings: list[dict[str, Any]] | None,
    case: dict[str, Any],
    *,
    warehouse: Path | str | sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    base = [dict(f) for f in (findings or []) if isinstance(f, Mapping)]
    if not patient_history_enabled():
        return base
    # не дублировать
    if any(str(f.get("code") or "") == FINDING_CODE for f in base):
        return base
    extra = evaluate_mo_patient_history(case, warehouse=warehouse)
    if patient_history_primary_enabled():
        for item in extra:
            item = {**item, "shadow": False}
            base.append(item)
    else:
        base.extend(extra)
    return base


def name_match_threshold_delta(summary: Mapping[str, Any] | None) -> float:
    """B1: сдвиг порога name_review. Отрицательный = мягче (легче ok)."""
    if not isinstance(summary, Mapping):
        return 0.0
    if summary.get("current_code_seen_by_doctor"):
        return -0.05
    tier = str(summary.get("tier") or "")
    if tier in {TIER_NEW_PROFILE, TIER_FIRST_CONTACT}:
        return 0.05
    return 0.0


def upsert_history_cache(
    db: sqlite3.Connection,
    *,
    patient_key: str,
    as_of_date: str,
    bundle: Mapping[str, Any],
) -> None:
    """A6: нарастающий кэш summary (одна строка на patient_key)."""
    if not patient_key:
        return
    cols = {row[1] for row in db.execute("PRAGMA table_info(fact_mo_patient_history_cache)")}
    if not cols:
        return
    summary = history_summary_for_analyzers(bundle)
    visit_index = []
    for shelf in ("same_doctor", "same_specialty", "other"):
        for row in bundle.get(shelf) or []:
            if isinstance(row, Mapping):
                visit_index.append(
                    {
                        "mis_id": row.get("mis_id"),
                        "visit_date": row.get("visit_date"),
                        "doctor_key": row.get("doctor_key"),
                        "specialty": row.get("specialty"),
                        "diagnosis_code": row.get("diagnosis_code"),
                        "overall_pct": row.get("overall_pct"),
                        "shelf": shelf,
                    }
                )
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.execute(
        """INSERT INTO fact_mo_patient_history_cache
           (patient_key, lookback_days, as_of_date, n_visits, summary_json,
            visit_index_json, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(patient_key) DO UPDATE SET
           lookback_days=excluded.lookback_days,
           as_of_date=excluded.as_of_date,
           n_visits=excluded.n_visits,
           summary_json=excluded.summary_json,
           visit_index_json=excluded.visit_index_json,
           updated_at=excluded.updated_at""",
        (
            patient_key,
            summary.get("coverage", {}).get("lookback_days")
            if isinstance(summary.get("coverage"), dict)
            else None,
            as_of_date[:10],
            int(summary.get("n_visits") or 0),
            json.dumps(summary, ensure_ascii=False),
            json.dumps(visit_index[:200], ensure_ascii=False),
            now,
        ),
    )


def short_diagnosis_text_for_warehouse(case: Mapping[str, Any] | None) -> str:
    """A2: короткий текст Dx для ленты."""
    try:
        from clinical_knowledge.mo_icd_resolve import resolve_diagnosis_text_from_mo

        text = str(resolve_diagnosis_text_from_mo(dict(case or {})).get("text") or "").strip()
    except Exception:  # noqa: BLE001
        text = ""
    if not text:
        for key in ("clinical_diagnosis", "mis_diagnos", "diagnosis_main_text", "diagnosis_short"):
            text = str((case or {}).get(key) or "").strip()
            if text:
                break
    return text[:200]
