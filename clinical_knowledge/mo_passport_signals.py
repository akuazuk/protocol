"""P2: теневые сигналы паспорта. По умолчанию не двигают overall_grade.

Канон: docs/plans/2026-09-27-mo-client-passport-score-ui-v2.md волна P2.
Флаг MO_PASSPORT_IN_SCORE=0 (тень). Тексты КЗ и ФИО наружу не отдаём.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

ENGINE = "mo_passport_signals_v1"
CODE_REPEAT_PLAN = "B_repeat_same_plan"
CODE_CROSS_SPEC = "B_cross_spec_episode"
CODE_LAB_AFTER_PLAN = "B_lab_result_after_plan"
CODES = (CODE_REPEAT_PLAN, CODE_CROSS_SPEC, CODE_LAB_AFTER_PLAN)
CROSS_SPEC_DAYS = 180
EMPTY_EXAM_CHARS = 40


def passport_signals_enabled() -> bool:
    raw = (os.environ.get("MO_PASSPORT") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def passport_in_score_enabled() -> bool:
    raw = (os.environ.get("MO_PASSPORT_IN_SCORE") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def icd_root(code: Any) -> str:
    text = str(code or "").strip().upper().replace(" ", "")
    if len(text) < 3:
        return ""
    head = text[:3]
    if head[0].isalpha() and head[1:].isdigit():
        return head
    return ""


def plan_fingerprint(value: Any) -> str:
    text = " ".join(str(value or "").lower().split())
    if len(text) < 24:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _iso(value: Any) -> str:
    return _norm(value)[:10]


def _parse_day(value: Any) -> date | None:
    raw = _iso(value)
    if len(raw) != 10:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _shadow_finding(
    code: str,
    *,
    title_ru: str,
    detail_ru: str,
    axis: str,
    linked_fields: Sequence[str],
) -> dict[str, Any]:
    shadow = not passport_in_score_enabled()
    return {
        "code": code,
        "axis": axis,
        "severity": "P2",
        "passed": False,
        "title_ru": title_ru,
        "detail_ru": detail_ru,
        "evidence": "",
        "source_ref": ENGINE,
        "needs_human": False,
        "shadow": shadow,
        "is_shadow": shadow,
        "engine": ENGINE,
        "linked_fields": list(linked_fields),
        "link_hint_ru": "Сигнал паспорта, пока в тени",
    }


def evaluate_passport_signals(
    *,
    prior_cards: Sequence[Mapping[str, Any]] | None = None,
    prior_plans: Sequence[Mapping[str, Any]] | None = None,
    lab_dates: Sequence[str] | None = None,
    case: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Считать сигналы по уже собранным картам. Без PHI в результате."""
    if not passport_signals_enabled():
        return []
    rec = dict(case or {})
    as_of = _parse_day(rec.get("visit_date") or rec.get("date"))
    if as_of is None:
        return []
    current_spec = _norm(rec.get("specialty") or rec.get("doctor_specialization")).lower()
    current_root = icd_root(rec.get("diagnosis_code") or rec.get("mkb_code_main"))
    current_fp = plan_fingerprint(
        rec.get("treatment_recommendations") or rec.get("plan_text") or rec.get("treatment")
    )
    exam_text = _norm(rec.get("exam_data") or rec.get("exam_recommendations") or rec.get("exam_text"))
    mention = " ".join(
        _norm(rec.get(key))
        for key in (
            "complaints",
            "anamnesis",
            "objective_status",
            "clinical_diagnosis",
            "exam_data",
            "diagnosis_text",
        )
    ).lower()
    out: list[dict[str, Any]] = []

    if current_fp and current_spec and current_root:
        for prior in prior_plans or []:
            if icd_root(prior.get("diagnosis_code")) != current_root:
                continue
            if _norm(prior.get("specialty")).lower() != current_spec:
                continue
            if plan_fingerprint(prior.get("treatment_recommendations") or prior.get("plan_text")) != current_fp:
                continue
            prior_day = _iso(prior.get("visit_date"))
            out.append(
                _shadow_finding(
                    CODE_REPEAT_PLAN,
                    title_ru="Повтор того же плана при том же коде",
                    detail_ru=(
                        f"Тот же корень МКБ и специальность, план совпал с визитом {prior_day or 'ранее'}."
                    ),
                    axis="plan",
                    linked_fields=["treatment_recommendations"],
                )
            )
            break

    window_start = as_of - timedelta(days=CROSS_SPEC_DAYS)
    if current_root:
        for card in prior_cards or []:
            card_day = _parse_day(card.get("visit_date"))
            if card_day is None or card_day >= as_of or card_day < window_start:
                continue
            if icd_root(card.get("diagnosis_code")) != current_root:
                continue
            other_spec = _norm(card.get("specialty"))
            if not other_spec or other_spec.lower() == current_spec:
                continue
            if other_spec.lower() in mention:
                continue
            out.append(
                _shadow_finding(
                    CODE_CROSS_SPEC,
                    title_ru="Эпизод у другого специалиста не отражён",
                    detail_ru=(
                        f"За 180 дней тот же корень МКБ вёл другой специалист "
                        f"({_iso(card.get('visit_date'))}), в текущем тексте этого нет."
                    ),
                    axis="diagnosis",
                    linked_fields=["clinical_diagnosis", "anamnesis"],
                )
            )
            break

    prev_same_spec: date | None = None
    for card in prior_cards or []:
        card_day = _parse_day(card.get("visit_date"))
        if card_day is None or card_day >= as_of:
            continue
        if current_spec and _norm(card.get("specialty")).lower() != current_spec:
            continue
        if prev_same_spec is None or card_day > prev_same_spec:
            prev_same_spec = card_day
    if prev_same_spec is not None and len(exam_text) < EMPTY_EXAM_CHARS:
        for raw in lab_dates or []:
            lab_day = _parse_day(raw)
            if lab_day is None:
                continue
            if prev_same_spec < lab_day <= as_of:
                out.append(
                    _shadow_finding(
                        CODE_LAB_AFTER_PLAN,
                        title_ru="Результат анализов после прошлого плана не разобран",
                        detail_ru=(
                            f"После визита {prev_same_spec.isoformat()} есть анализ "
                            f"{lab_day.isoformat()}, слот обследований текущего визита пуст."
                        ),
                        axis="plan",
                        linked_fields=["exam_data", "exam_recommendations"],
                    )
                )
                break
    return out


def _load_prior_cards(
    db: sqlite3.Connection,
    *,
    patient_key: str,
    as_of: str,
    exclude_visit: str,
) -> list[dict[str, Any]]:
    tables = {str(r[0]) for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "fact_mo_visit_index" not in tables or not patient_key:
        return []
    rows = db.execute(
        """
        SELECT visit_id, visit_date, specialty, diagnosis_code
          FROM fact_mo_visit_index
         WHERE patient_key = ?
           AND visit_date < ?
           AND TRIM(COALESCE(visit_id,'')) != ?
         ORDER BY visit_date DESC
         LIMIT 80
        """,
        (patient_key, as_of, exclude_visit),
    ).fetchall()
    return [
        {
            "visit_id": str(row[0] or ""),
            "visit_date": str(row[1] or "")[:10],
            "specialty": str(row[2] or ""),
            "diagnosis_code": str(row[3] or ""),
        }
        for row in rows
    ]


def _load_prior_plans(
    db: sqlite3.Connection,
    *,
    patient_key: str,
    as_of: str,
    exclude_ids: set[str],
) -> list[dict[str, Any]]:
    cols = {str(row[1]) for row in db.execute("PRAGMA table_info(fact_mo_case)")}
    if "patient_key" not in cols or "treatment_recommendations" not in cols:
        return []
    rows = db.execute(
        """
        SELECT visit_id, visit_date, specialty, diagnosis_code, treatment_recommendations, mis_id
          FROM fact_mo_case
         WHERE patient_key = ?
           AND visit_date < ?
         ORDER BY visit_date DESC
         LIMIT 40
        """,
        (patient_key, as_of),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        visit_id = str(row[0] or "")
        mis_id = str(row[5] or "")
        if visit_id in exclude_ids or mis_id in exclude_ids:
            continue
        out.append(
            {
                "visit_id": visit_id,
                "visit_date": str(row[1] or "")[:10],
                "specialty": str(row[2] or ""),
                "diagnosis_code": str(row[3] or ""),
                "treatment_recommendations": str(row[4] or ""),
            }
        )
    return out


def _load_lab_dates(
    db: sqlite3.Connection,
    *,
    patient_key: str,
    lab_path: Path | None = None,
) -> list[str]:
    dates: list[str] = []
    tables = {str(r[0]) for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "fact_mo_lab_coverage" in tables:
        row = db.execute(
            "SELECT last_date FROM fact_mo_lab_coverage WHERE patient_key=?",
            (patient_key,),
        ).fetchone()
        if row and row[0]:
            dates.append(str(row[0])[:10])
    if lab_path is not None and Path(lab_path).is_file():
        with sqlite3.connect(str(lab_path)) as lab:
            lab_tables = {str(r[0]) for r in lab.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "fact_mo_lab" in lab_tables:
                for item in lab.execute(
                    """
                    SELECT DISTINCT test_date FROM fact_mo_lab
                     WHERE patient_key=? AND TRIM(COALESCE(test_date,'')) != ''
                     ORDER BY test_date DESC LIMIT 40
                    """,
                    (patient_key,),
                ):
                    dates.append(str(item[0])[:10])
    return sorted({d for d in dates if d})


def evaluate_mo_passport_signals(
    case: Mapping[str, Any],
    *,
    warehouse: Path | str | sqlite3.Connection | None = None,
    lab_path: Path | None = None,
) -> list[dict[str, Any]]:
    if not passport_signals_enabled() or not isinstance(case, Mapping):
        return []
    rec = dict(case)
    patient_key = _norm(rec.get("patient_key"))
    as_of = _iso(rec.get("visit_date") or rec.get("date"))
    if not patient_key or not as_of:
        return evaluate_passport_signals(
            prior_cards=list(rec.get("_passport_prior_cards") or []),
            prior_plans=list(rec.get("_passport_prior_plans") or []),
            lab_dates=list(rec.get("_passport_lab_dates") or []),
            case=rec,
        )
    exclude = { _norm(rec.get("visit_id")), _norm(rec.get("mis_id")), _norm(rec.get("id")) } - {""}
    cards: list[dict[str, Any]] = list(rec.get("_passport_prior_cards") or [])
    plans: list[dict[str, Any]] = list(rec.get("_passport_prior_plans") or [])
    labs: list[str] = list(rec.get("_passport_lab_dates") or [])
    own_db = False
    db: sqlite3.Connection | None = None
    if isinstance(warehouse, sqlite3.Connection):
        db = warehouse
    elif warehouse:
        path = Path(warehouse)
        if path.is_file():
            db = sqlite3.connect(str(path))
            own_db = True
    try:
        if db is not None:
            if not cards:
                cards = _load_prior_cards(
                    db, patient_key=patient_key, as_of=as_of, exclude_visit=_norm(rec.get("visit_id"))
                )
            if not plans:
                plans = _load_prior_plans(db, patient_key=patient_key, as_of=as_of, exclude_ids=exclude)
            if not labs:
                labs = _load_lab_dates(db, patient_key=patient_key, lab_path=lab_path)
    finally:
        if own_db and db is not None:
            db.close()
    return evaluate_passport_signals(prior_cards=cards, prior_plans=plans, lab_dates=labs, case=rec)


def merge_passport_signals_into_findings(
    findings: Sequence[Mapping[str, Any]] | None,
    case: Mapping[str, Any],
    *,
    warehouse: Path | str | sqlite3.Connection | None = None,
    lab_path: Path | None = None,
) -> list[dict[str, Any]]:
    base = [dict(item) for item in (findings or []) if isinstance(item, Mapping)]
    if not passport_signals_enabled():
        return base
    existing = {str(item.get("code") or "") for item in base}
    extra = evaluate_mo_passport_signals(case, warehouse=warehouse, lab_path=lab_path)
    for item in extra:
        code = str(item.get("code") or "")
        if code and code not in existing:
            base.append(item)
            existing.add(code)
    return base
