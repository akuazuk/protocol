"""Загрузка агрегатов L1/L2 mis_protocol и выборочный Gemini-разбор для методиста."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _candidate_summary_paths(month: str | None = None) -> list[Path]:
    month = (month or "").strip() or "2026-07"
    name = f"kz_l1_{month}_summary.json"
    env = (os.environ.get("MIS_KZ_SUMMARY_PATH") or "").strip()
    out: list[Path] = []
    if env:
        out.append(Path(env))
    out.append(Path("/var/data/mis_protocol") / name)
    out.append(ROOT / "data" / "mis_protocol" / name)
    for d in (Path("/var/data/mis_protocol"), ROOT / "data" / "mis_protocol"):
        if d.is_dir():
            out.extend(sorted(d.glob("kz_l1_*_summary.json"), reverse=True))
    return out


def _gemini_reviews_path(month: str) -> Path:
    env = (os.environ.get("MIS_KZ_GEMINI_PATH") or "").strip()
    if env:
        return Path(env)
    name = f"kz_l1_{month}_gemini_reviews.json"
    disk = Path("/var/data/mis_protocol") / name
    if disk.parent.is_dir():
        return disk
    return ROOT / "data" / "mis_protocol" / name


def _csv_path_for_month(month: str) -> Path | None:
    candidates = [
        Path("/var/data/mis_protocol") / f"mis_protocol_{month}.csv",
        ROOT / "data" / "mis_protocol" / f"mis_protocol_{month}.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_mis_kz_summary(*, month: str | None = None) -> dict[str, Any] | None:
    seen: set[str] = set()
    for path in _candidate_summary_paths(month):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        data = dict(data)
        data["_source_path"] = str(path)
        return data
    return None


def load_gemini_reviews(*, month: str | None = None) -> dict[str, Any]:
    month = (month or "").strip() or "2026-07"
    path = _gemini_reviews_path(month)
    if not path.is_file():
        return {"reviews": [], "meta": {}, "path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"reviews": [], "meta": {}, "path": str(path)}
    if not isinstance(data, dict):
        return {"reviews": [], "meta": {}, "path": str(path)}
    return {
        "reviews": data.get("reviews") or [],
        "meta": data.get("meta") or {},
        "path": str(path),
    }


def save_gemini_reviews(*, month: str, reviews: list[dict], meta: dict | None = None) -> Path:
    path = _gemini_reviews_path(month)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "month": month,
        "updated_at": _utc(),
        "meta": meta or {},
        "reviews": reviews,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _recompute_avg_from_groups(groups: list[dict[str, Any]]) -> float | None:
    """Взвешенное среднее overall по агрегатам с полем n / avg_overall_pct."""
    num = den = 0.0
    for g in groups:
        n = g.get("n")
        avg = g.get("avg_overall_pct")
        if isinstance(n, (int, float)) and n > 0 and isinstance(avg, (int, float)):
            num += float(avg) * float(n)
            den += float(n)
    return round(num / den, 1) if den else None


def build_month_compare(*, base_month: str, compare_month: str) -> dict[str, Any] | None:
    """Сравнение двух месяцев по клиническим специальностям (янв vs июль и т.п.)."""
    from .clinical_specialties import filter_clinical_rows

    a = load_mis_kz_summary(month=base_month)
    b = load_mis_kz_summary(month=compare_month)
    if a is None or b is None:
        return {
            "available": False,
            "base_month": base_month,
            "compare_month": compare_month,
            "missing": [
                m
                for m, s in ((base_month, a), (compare_month, b))
                if s is None
            ],
            "hint_ru": "Нет summary за один или оба месяца - дождитесь L1-батча.",
        }

    specs_a = {
        str(r.get("specialization") or ""): r
        for r in filter_clinical_rows(a.get("specialties") or [])
    }
    specs_b = {
        str(r.get("specialization") or ""): r
        for r in filter_clinical_rows(b.get("specialties") or [])
    }
    names = sorted(set(specs_a) | set(specs_b), key=lambda n: -(specs_b.get(n) or specs_a.get(n) or {}).get("n") or 0)
    by_spec: list[dict[str, Any]] = []
    up = down = flat = 0
    for name in names:
        ra, rb = specs_a.get(name) or {}, specs_b.get(name) or {}
        avg_a = ra.get("avg_overall_pct")
        avg_b = rb.get("avg_overall_pct")
        delta = None
        direction = "na"
        if isinstance(avg_a, (int, float)) and isinstance(avg_b, (int, float)):
            delta = round(float(avg_b) - float(avg_a), 1)
            if delta > 0.5:
                direction = "up"
                up += 1
            elif delta < -0.5:
                direction = "down"
                down += 1
            else:
                direction = "flat"
                flat += 1
        by_spec.append({
            "specialization": name,
            "n_base": ra.get("n"),
            "n_compare": rb.get("n"),
            "avg_base": avg_a,
            "avg_compare": avg_b,
            "delta": delta,
            "direction": direction,
            "core_base": ra.get("avg_core_overall_pct"),
            "core_compare": rb.get("avg_core_overall_pct"),
        })

    clin_a = filter_clinical_rows(a.get("specialties") or [])
    clin_b = filter_clinical_rows(b.get("specialties") or [])
    avg_a = _recompute_avg_from_groups(clin_a)
    avg_b = _recompute_avg_from_groups(clin_b)
    n_a = sum(int(r.get("n") or 0) for r in clin_a)
    n_b = sum(int(r.get("n") or 0) for r in clin_b)
    delta_avg = (
        round(float(avg_b) - float(avg_a), 1)
        if isinstance(avg_a, (int, float)) and isinstance(avg_b, (int, float))
        else None
    )

    blocks_a = a.get("block_avg") or {}
    blocks_b = b.get("block_avg") or {}
    block_keys = sorted(set(blocks_a) | set(blocks_b))
    blocks: list[dict[str, Any]] = []
    for k in block_keys:
        va, vb = blocks_a.get(k), blocks_b.get(k)
        d = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = round(float(vb) - float(va), 1)
        blocks.append({"block": k, "avg_base": va, "avg_compare": vb, "delta": d})

    return {
        "available": True,
        "base_month": base_month,
        "compare_month": compare_month,
        "clinical_only": True,
        "n_base": n_a,
        "n_compare": n_b,
        "avg_base": avg_a,
        "avg_compare": avg_b,
        "delta_avg": delta_avg,
        "specialties_up": up,
        "specialties_down": down,
        "specialties_flat": flat,
        "specialties": by_spec,
        "blocks": blocks,
        "note_ru": (
            f"Сравнение клинических специальностей: {base_month} → {compare_month}. "
            "Исключены стоматологи, медсёстры, логопед, лаборатория и пустые роли."
        ),
    }


def build_mis_kz_quality_view(
    *,
    month: str | None = None,
    compare_month: str | None = "2026-01",
) -> dict[str, Any]:
    from .clinical_specialties import (
        filter_clinical_doctors,
        filter_clinical_rows,
        filter_clinical_visits,
    )

    summary = load_mis_kz_summary(month=month)
    if summary is None:
        return {
            "ok": False,
            "available": False,
            "error": "summary_not_found",
            "hint_ru": (
                "Нет kz_l1_*_summary.json. Запустите "
                "scripts/run_mis_protocol_l1_batch.py на Render и положите summary "
                "в /var/data/mis_protocol/ или data/mis_protocol/."
            ),
            "month": month or "2026-07",
        }
    month_s = str(summary.get("month") or month or "2026-07")
    gem = load_gemini_reviews(month=month_s)
    doctors_all = summary.get("doctors") or []
    doctors = filter_clinical_doctors(doctors_all)
    specialties = filter_clinical_rows(summary.get("specialties") or [])
    top_doctors = filter_clinical_doctors(summary.get("top_doctors") or doctors[:15])
    bottom_doctors = filter_clinical_doctors(summary.get("bottom_doctors") or [])
    worst_visits = filter_clinical_visits(summary.get("worst_visits") or [])
    # Пересчёт KPI только по клиническим специальностям (взвешенно по n).
    avg_clinical = _recompute_avg_from_groups(specialties)
    n_clinical = sum(int(r.get("n") or 0) for r in specialties)
    n_all_spec = sum(
        int(r.get("n") or 0)
        for r in (summary.get("specialties") or [])
        if isinstance(r, dict)
    )
    excluded_n = max(0, n_all_spec - n_clinical)

    compare = None
    if compare_month and str(compare_month).strip() and str(compare_month) != month_s:
        compare = build_month_compare(base_month=str(compare_month).strip(), compare_month=month_s)

    return {
        "ok": True,
        "available": True,
        "month": summary.get("month"),
        "tier": summary.get("tier") or "L1",
        "generated_at": summary.get("generated_at"),
        "source_path": summary.get("_source_path"),
        "n_cases": summary.get("n_cases"),
        "n_ok": n_clinical if n_clinical else summary.get("n_ok"),
        "n_ok_all": summary.get("n_ok"),
        "n_clinical": n_clinical,
        "n_excluded_nonclinical": excluded_n,
        "n_errors": summary.get("n_errors"),
        "avg_overall_pct": avg_clinical if avg_clinical is not None else summary.get("avg_overall_pct"),
        "avg_overall_pct_all": summary.get("avg_overall_pct"),
        "median_overall_pct": summary.get("median_overall_pct"),
        "score_histogram": summary.get("score_histogram") or {},
        "status_counts": summary.get("status_counts") or {},
        "block_avg": summary.get("block_avg") or {},
        "block_avg_when_filled": summary.get("block_avg_when_filled") or {},
        "field_fill_rate": summary.get("field_fill_rate") or {},
        "avg_regulatory_compliance_pct": summary.get("avg_regulatory_compliance_pct"),
        "reg55_p0_defect_n": summary.get("reg55_p0_defect_n"),
        "reg55_scored_n": summary.get("reg55_scored_n"),
        "reg55_top_failed": summary.get("reg55_top_failed") or [],
        "reg55_meta": summary.get("reg55_meta") or {},
        "avg_core_overall_pct": summary.get("avg_core_overall_pct"),
        "n_multi_kz_visits": summary.get("n_multi_kz_visits"),
        "n_multi_kz_extra_rows": summary.get("n_multi_kz_extra_rows"),
        "doctors": doctors,
        "specialties": specialties,
        "specialties_n": len(specialties),
        "clinical_filter": True,
        "filials": summary.get("filials") or [],
        "pay_types": summary.get("pay_types") or [],
        "top_services": summary.get("top_services") or [],
        "top_doctors": top_doctors[:15] if top_doctors else doctors[:15],
        "bottom_doctors": bottom_doctors,
        "worst_visits": worst_visits,
        "worst_visits_meta": summary.get("worst_visits_meta") or {},
        "excluded_breakdown": summary.get("excluded_breakdown") or {},
        "llm_review_queue": summary.get("llm_review_queue") or {},
        "gemini_reviews": gem.get("reviews") or summary.get("gemini_reviews") or [],
        "gemini_meta": {
            "note_ru": "Выборочный LLM-разбор качества КЗ.",
            "storage_path": gem.get("path"),
        },
        "month_compare": compare,
        "deep_eval": summary.get("deep_eval"),
        "notes": [
            str(n).replace("Gemini", "LLM").replace("gemini", "LLM")
            for n in (summary.get("notes") or [])
        ]
        + [
            "В отчёте только клинические специальности врачей "
            f"({len(specialties)} шт.): без стоматологов, медсестёр, логопеда, лаборатории и пустых ролей."
            + (f" Исключено визитов: {excluded_n}." if excluded_n else ""),
        ],
        "doctors_n": len(doctors),
    }


# --------------------------------------------------------------------------- #
# Дашборд §7Б: кейсы с deep-скором, фильтры, диаграммы, динамика
# --------------------------------------------------------------------------- #

def _candidate_cases_paths(month: str | None = None) -> list[Path]:
    month = (month or "").strip() or "2026-07"
    name = f"kz_l1_{month}_cases.jsonl"
    env = (os.environ.get("MIS_KZ_CASES_PATH") or "").strip()
    out: list[Path] = []
    if env:
        out.append(Path(env))
    # deep-only (локальная разработка дашборда) - самые богатые (с deep-блоком)
    out.append(ROOT / "data" / "ml" / "reports" / "deep_eval" / name)
    out.append(Path("/var/data/mis_protocol") / name)
    out.append(ROOT / "data" / "mis_protocol" / name)
    return out


_CASES_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CSV_BY_VISIT_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}


def load_kz_cases(*, month: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
    """Загрузить cases.jsonl (с deep-блоком) с кэшем по mtime."""
    for path in _candidate_cases_paths(month):
        if not path.is_file():
            continue
        key = str(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        cached = _CASES_CACHE.get(key)
        if cached and cached[0] == mtime:
            return cached[1], key
        cases: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and not obj.get("error"):
                        cases.append(obj)
        except OSError:
            continue
        _CASES_CACHE[key] = (mtime, cases)
        return cases, key
    return [], None


def _load_csv_by_visit_cached(month: str) -> dict[str, dict]:
    path = _csv_path_for_month(month)
    if path is None:
        return {}
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    cached = _CSV_BY_VISIT_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        _, load_csv_by_visit = _load_batch_helpers()
        by_visit = load_csv_by_visit(path)
    except Exception:  # noqa: BLE001
        by_visit = {}
    _CSV_BY_VISIT_CACHE[key] = (mtime, by_visit)
    return by_visit


def icd10_chapter(code: str) -> tuple[str, str]:
    """Глава МКБ-10 по коду (напр. 'H66.1' -> ('VIII', 'H60-H95 Ухо'))."""
    c = (code or "").strip().upper()
    m = re.match(r"([A-Z])\s*(\d{1,2})", c)
    if not m:
        return ("", "без кода")
    letter, num = m.group(1), int(m.group(2))
    table = {
        "A": ("I", "A00-B99 Инфекционные"),
        "B": ("I", "A00-B99 Инфекционные"),
        "C": ("II", "C00-D48 Новообразования"),
        "E": ("IV", "E00-E90 Эндокринные"),
        "F": ("V", "F00-F99 Психические"),
        "G": ("VI", "G00-G99 Нервные"),
        "I": ("IX", "I00-I99 Кровообращение"),
        "J": ("X", "J00-J99 Дыхание"),
        "K": ("XI", "K00-K93 Пищеварение"),
        "L": ("XII", "L00-L99 Кожа"),
        "M": ("XIII", "M00-M99 Костно-мышечная"),
        "N": ("XIV", "N00-N99 Мочеполовая"),
        "O": ("XV", "O00-O99 Беременность"),
        "P": ("XVI", "P00-P96 Перинатальные"),
        "Q": ("XVII", "Q00-Q99 Врождённые"),
        "R": ("XVIII", "R00-R99 Симптомы/признаки"),
        "S": ("XIX", "S00-T98 Травмы/отравления"),
        "T": ("XIX", "S00-T98 Травмы/отравления"),
        "V": ("XX", "V01-Y98 Внешние причины"),
        "W": ("XX", "V01-Y98 Внешние причины"),
        "X": ("XX", "V01-Y98 Внешние причины"),
        "Y": ("XX", "V01-Y98 Внешние причины"),
        "Z": ("XXI", "Z00-Z99 Факторы здоровья"),
        "U": ("XXII", "U00-U99 Особые"),
    }
    if letter == "D":
        return ("II", "C00-D48 Новообразования") if num <= 48 else ("III", "D50-D89 Кровь")
    if letter == "H":
        return ("VII", "H00-H59 Глаз") if num <= 59 else ("VIII", "H60-H95 Ухо")
    return table.get(letter, ("?", "прочее"))


_ICD_CODE_RX = re.compile(r"\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b")


def extract_main_icd(row: dict[str, Any]) -> str:
    """Основной код МКБ КЗ без БД: слот 22 (`diagnosis_list`, главный по
    `diagnosis_main_index`), fallback - свободный текст `clinical_diagnosis`.

    Покрытие на выгрузке ~86% КЗ (см. план 2026-07-27-mis-kz-dashboard-rubric).
    """
    dl = (row.get("diagnosis_list") or "").strip()
    if dl:
        parts = [p for p in dl.split("|") if p.strip()]
        if parts:
            idx = 0
            raw_idx = str(row.get("diagnosis_main_index") or "").strip()
            if raw_idx.isdigit() and int(raw_idx) < len(parts):
                idx = int(raw_idx)
            m = _ICD_CODE_RX.search(parts[idx])
            if not m:
                for p in parts:
                    m = _ICD_CODE_RX.search(p)
                    if m:
                        break
            if m:
                return m.group(1).upper()
    m = _ICD_CODE_RX.search(row.get("clinical_diagnosis") or "")
    return m.group(1).upper() if m else ""


def _age_group(age: Any) -> str:
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return "неизв."
    if a < 18:
        return "дети (<18)"
    if a >= 65:
        return "пожилые (65+)"
    return "взрослые (18-64)"


def _score_band(pct: Any) -> str:
    if not isinstance(pct, (int, float)):
        return "нет скора"
    if pct < 50:
        return "<50"
    if pct < 75:
        return "50-75"
    if pct < 90:
        return "75-90"
    return "≥90"


_AXIS_RU = {
    "documentation": "оформление",
    "clinical_concordance": "согласованность",
    "safety": "безопасность",
    "regulatory": "регуляторика",
}


def _deep_status_calibrated(deep: dict) -> Any:
    """Пересчёт deep-статуса из overall+axes+severity через каноническую risk-gate
    движка с текущим config/deep_thresholds.yaml (Э4-калибровка). Отражает калибровку
    без перегенерации датасета. Фолбэк - сохранённый статус."""
    overall = deep.get("overall_pct")
    axes = deep.get("axes") or {}
    sev = deep.get("n_by_severity") or {}
    if overall is None and not sev:
        return deep.get("status")
    try:
        from .kz_deep_eval import _apply_risk_gate

        findings = [{"severity": s, "passed": False}
                    for s, cnt in sev.items() for _ in range(int(cnt or 0))]
        _, status = _apply_risk_gate(overall, findings, axes=axes)
        return status
    except Exception:  # noqa: BLE001
        return deep.get("status")


def _flat_case(case: dict[str, Any], csvrow: dict | None) -> dict[str, Any]:
    deep = case.get("deep") or {}
    axes = deep.get("axes") or {}
    sev = deep.get("n_by_severity") or {}
    findings = [f for f in (deep.get("findings") or []) if isinstance(f, dict) and not f.get("passed")]
    row = csvrow or {}
    code_main = (row.get("mkb_code_main") or "").strip()
    if not code_main:
        code_main = extract_main_icd(row)
    chap_key, chap_label = icd10_chapter(code_main)
    overall = deep.get("overall_pct")
    if overall is None:
        overall = case.get("overall_pct")
    finding_axes = sorted({str(f.get("axis") or "") for f in findings if f.get("axis")})
    diag = (case.get("diagnosis_short") or (row.get("clinical_diagnosis") or "").strip())
    diag = re.sub(r"\s+", " ", diag).strip()[:160]
    return {
        "visit_id": str(case.get("visit_id") or ""),
        "patient_id": str(case.get("patient_id") or row.get("patient_id") or ""),
        "date": (case.get("date") or row.get("visit_date") or "")[:10],
        "doctor_fio": case.get("doctor_fio") or (row.get("doctor_fio") or "").strip() or " - ",
        "specialization": case.get("doctor_specialization") or (row.get("doctor_specialization") or "").strip() or " - ",
        "filial": case.get("filial") or (row.get("filial") or "").strip() or " - ",
        "kz_kind": (row.get("kz_kind") or "").strip() or "kz",
        "mkb_code_main": code_main,
        "icd_chapter": chap_key,
        "icd_chapter_label": chap_label,
        "diagnosis_short": diag,
        "overall_pct": overall,
        "l1_overall_pct": case.get("overall_pct"),
        "deep_status": _deep_status_calibrated(deep),
        "status": _deep_status_calibrated(deep) or case.get("status") or "unknown",
        "axis_documentation": axes.get("documentation"),
        "axis_concordance": axes.get("clinical_concordance"),
        "axis_safety": axes.get("safety"),
        "axis_regulatory": axes.get("regulatory"),
        "p0": int(sev.get("P0", 0) or 0),
        "p1": int(sev.get("P1", 0) or 0),
        "p2": int(sev.get("P2", 0) or 0),
        "p3": int(sev.get("P3", 0) or 0),
        "n_findings": deep.get("n_findings"),
        "has_potential_harm": bool(deep.get("has_potential_harm")),
        "needs_human": any(f.get("needs_human") for f in findings),
        "finding_axes": finding_axes,
        "mkb_code_agreement": (row.get("mkb_code_agreement") or "").strip() or "unknown",
        "age_group": _age_group(row.get("patient_age_years")),
        "patient_age_years": row.get("patient_age_years"),
        "pay_type": (row.get("pay_type") or "").strip(),
        "date_mismatch": str(row.get("date_mismatch") or "0").strip(),
        "parse_ok": str(row.get("parse_ok") or "1").strip(),
        "protocol_used": bool(deep.get("protocol_used")),
        "score_band": _score_band(overall),
    }


def _match_filters(rec: dict[str, Any], flt: dict[str, Any]) -> bool:
    def eq(field: str, key: str) -> bool:
        v = flt.get(key)
        return not v or str(rec.get(field) or "") == str(v)

    if not eq("specialization", "specialization"):
        return False
    if not eq("filial", "filial"):
        return False
    if not eq("kz_kind", "kz_kind"):
        return False
    if not eq("icd_chapter", "mkb_chapter"):
        return False
    if not eq("mkb_code_agreement", "mkb_agreement"):
        return False
    if not eq("age_group", "age_group"):
        return False
    if flt.get("_no_mkb_code") and rec.get("icd_chapter"):
        return False
    if not eq("status", "status"):
        return False
    if not eq("score_band", "score_band"):
        return False
    doctor = (flt.get("doctor") or "").strip().lower()
    if doctor and doctor not in str(rec.get("doctor_fio") or "").lower():
        return False
    q = (flt.get("q") or "").strip().lower()
    if q:
        hay = f"{rec.get('diagnosis_short','')} {rec.get('doctor_fio','')} {rec.get('mkb_code_main','')}".lower()
        if q not in hay:
            return False
    fa = (flt.get("finding_axis") or "").strip()
    if fa and fa not in (rec.get("finding_axes") or []):
        return False
    if flt.get("needs_human") and not rec.get("needs_human"):
        return False
    if flt.get("potential_harm") and not rec.get("has_potential_harm"):
        return False
    sev = (flt.get("min_severity") or "").strip().upper()
    if sev == "P0" and rec.get("p0", 0) < 1:
        return False
    if sev == "P1" and (rec.get("p0", 0) + rec.get("p1", 0)) < 1:
        return False
    if flt.get("date_mismatch") and rec.get("date_mismatch") not in ("1", "true", "True"):
        return False
    df, dt = (flt.get("date_from") or "").strip(), (flt.get("date_to") or "").strip()
    d = rec.get("date") or ""
    if df and d and d < df:
        return False
    if dt and d and d > dt:
        return False
    return True


def _apply_preset(flt: dict[str, Any]) -> dict[str, Any]:
    preset = (flt.get("preset") or "").strip()
    if preset == "p0":
        flt["min_severity"] = "P0"
    elif preset == "dx_no_code":
        flt["_no_mkb_code"] = True
    elif preset == "treatment_off_protocol":
        flt["finding_axis"] = "clinical_concordance"
    elif preset == "exams_gap":
        flt["finding_axis"] = "clinical_concordance"
    elif preset == "needs_human":
        flt["needs_human"] = True
    return flt


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else None


def _facets(records: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    def top(field: str, limit: int = 60) -> list[dict[str, Any]]:
        c = Counter(str(r.get(field) or "") for r in records if r.get(field))
        return [{"value": k, "n": n} for k, n in c.most_common(limit)]

    chapters: dict[str, dict[str, Any]] = {}
    for r in records:
        key = r.get("icd_chapter") or ""
        if not key:
            continue
        entry = chapters.setdefault(key, {"key": key, "label": r.get("icd_chapter_label"), "n": 0})
        entry["n"] += 1
    return {
        "specialties": top("specialization"),
        "filials": top("filial"),
        "kz_kinds": top("kz_kind"),
        "statuses": top("status"),
        "score_bands": top("score_band"),
        "age_groups": top("age_group"),
        "agreements": top("mkb_code_agreement"),
        "mkb_chapters": sorted(chapters.values(), key=lambda x: -x["n"]),
        "finding_axes": [
            {"value": k, "label": _AXIS_RU.get(k, k), "n": n}
            for k, n in __import__("collections").Counter(
                a for r in records for a in (r.get("finding_axes") or [])
            ).most_common()
        ],
    }


def _filtered_agg(records: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    n = len(records)
    sev_tot = {s: sum(int(r.get(s.lower(), 0) or 0) for r in records) for s in ("P0", "P1", "P2", "P3")}
    status_dist = dict(Counter(str(r.get("status") or "unknown") for r in records))
    band_dist = dict(Counter(str(r.get("score_band") or "нет скора") for r in records))
    axis_dist = dict(Counter(a for r in records for a in (r.get("finding_axes") or [])))
    n_harm = sum(1 for r in records if r.get("has_potential_harm"))
    n_bad = sum(1 for r in records if isinstance(r.get("overall_pct"), (int, float)) and r["overall_pct"] < 75)

    by_spec: dict[str, dict[str, Any]] = {}
    for r in records:
        sp = r.get("specialization") or " - "
        e = by_spec.setdefault(sp, {"specialization": sp, "n": 0, "_ov": [], "n_bad": 0, "p0": 0})
        e["n"] += 1
        e["_ov"].append(r.get("overall_pct"))
        if isinstance(r.get("overall_pct"), (int, float)) and r["overall_pct"] < 75:
            e["n_bad"] += 1
        e["p0"] += int(r.get("p0", 0) or 0)
    spec_rows = []
    for e in by_spec.values():
        spec_rows.append({
            "specialization": e["specialization"],
            "n": e["n"],
            "avg_overall": _mean(e["_ov"]),
            "bad_pct": round(100 * e["n_bad"] / e["n"], 1) if e["n"] else 0.0,
            "p0": e["p0"],
        })
    spec_rows.sort(key=lambda x: (x["avg_overall"] if x["avg_overall"] is not None else 999, -x["n"]))

    icd_bad = Counter()
    for r in records:
        if isinstance(r.get("overall_pct"), (int, float)) and r["overall_pct"] < 75 and r.get("mkb_code_main"):
            icd_bad[(r["mkb_code_main"], r.get("diagnosis_short", ""))] += 1
    top_bad_icd = [
        {"mkb_code": k[0], "diagnosis": k[1], "n": v}
        for k, v in icd_bad.most_common(15)
    ]

    return {
        "n": n,
        "avg_overall": _mean([r.get("overall_pct") for r in records]),
        "axis_means": {
            "documentation": _mean([r.get("axis_documentation") for r in records]),
            "clinical_concordance": _mean([r.get("axis_concordance") for r in records]),
            "safety": _mean([r.get("axis_safety") for r in records]),
            "regulatory": _mean([r.get("axis_regulatory") for r in records]),
        },
        "severity_totals": sev_tot,
        "status_distribution": status_dist,
        "score_band_distribution": band_dist,
        "finding_axis_distribution": axis_dist,
        "n_potential_harm": n_harm,
        "n_bad": n_bad,
        "pct_bad": round(100 * n_bad / n, 1) if n else 0.0,
        "by_specialty": spec_rows,
        "top_bad_icd": top_bad_icd,
    }


_SORT_FIELDS = {
    "overall": "overall_pct",
    "date": "date",
    "p0": "p0",
    "n_findings": "n_findings",
    "documentation": "axis_documentation",
    "concordance": "axis_concordance",
    "safety": "axis_safety",
    "regulatory": "axis_regulatory",
}


def build_kz_cases_view(
    *,
    month: str | None = None,
    page: int = 1,
    page_size: int = 50,
    sort_by: str = "overall",
    sort_dir: str = "asc",
    **filters: Any,
) -> dict[str, Any]:
    """Таблица КЗ с deep-скором: фильтры по всем столбцам + facets + агрегат под фильтр."""
    month_s = (month or "").strip() or "2026-07"
    cases, path = load_kz_cases(month=month_s)
    if not cases:
        return {
            "ok": False,
            "available": False,
            "error": "cases_not_found",
            "hint_ru": (
                "Нет kz_l1_*_cases.jsonl с deep-блоком. Запустите батч с --deep-eval "
                "(или --deep-only) и положите cases.jsonl рядом с summary."
            ),
            "month": month_s,
        }
    csv_by_visit = _load_csv_by_visit_cached(month_s)
    records = [_flat_case(c, csv_by_visit.get(str(c.get("visit_id") or ""))) for c in cases]

    facets = _facets(records)

    flt = _apply_preset({k: v for k, v in filters.items() if v not in (None, "")})
    filtered = [r for r in records if _match_filters(r, flt)]

    field = _SORT_FIELDS.get(sort_by, "overall_pct")
    reverse = str(sort_dir or "asc").lower() == "desc"

    def _key(r: dict[str, Any]):
        v = r.get(field)
        if isinstance(v, (int, float)):
            return (0, v)
        if field == "date":
            return (0, str(v or ""))
        return (1, 0)

    filtered.sort(key=_key, reverse=reverse)

    agg = _filtered_agg(filtered)

    # Флаги доступности данных: фронт честно гасит фильтры без данных вместо пустых списков.
    deep_available = any(
        (r.get("axis_documentation") is not None)
        or (r.get("n_findings") or 0)
        or r.get("finding_axes")
        or r.get("has_potential_harm")
        for r in records
    )
    mkb_available = any(r.get("icd_chapter") for r in records)

    page = max(1, int(page or 1))
    page_size = max(1, min(200, int(page_size or 50)))
    total = len(filtered)
    n_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    rows = filtered[start:start + page_size]

    return {
        "ok": True,
        "available": True,
        "deep_available": deep_available,
        "mkb_available": mkb_available,
        "month": month_s,
        "source_path": path,
        "n_total_cases": len(records),
        "total": total,
        "page": page,
        "page_size": page_size,
        "n_pages": n_pages,
        "rows": rows,
        "facets": facets,
        "filtered_agg": agg,
        "applied_filters": flt,
        "sort_by": sort_by,
        "sort_dir": "desc" if reverse else "asc",
    }


def _available_months() -> list[str]:
    months: set[str] = set()
    dirs = [
        ROOT / "data" / "ml" / "reports" / "deep_eval",
        Path("/var/data/mis_protocol"),
        ROOT / "data" / "mis_protocol",
    ]
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.glob("kz_l1_*_summary.json"):
            mm = re.search(r"kz_l1_(\d{4}-\d{2})_summary\.json", p.name)
            if mm:
                months.add(mm.group(1))
        for p in d.glob("kz_l1_*_cases.jsonl"):
            mm = re.search(r"kz_l1_(\d{4}-\d{2})_cases\.jsonl", p.name)
            if mm:
                months.add(mm.group(1))
    return sorted(months)


def build_kz_case_detail(*, month: str | None = None, visit_id: str) -> dict[str, Any]:
    """Полный deep-разбор одного КЗ (экран B §7Б.4): оси, находки, block_scores."""
    month_s = (month or "").strip() or "2026-07"
    vid = str(visit_id or "").strip()
    if not vid:
        return {"ok": False, "error": "empty_visit_id"}
    cases, path = load_kz_cases(month=month_s)
    case = next((c for c in cases if str(c.get("visit_id") or "") == vid), None)
    if case is None:
        return {"ok": False, "error": "visit_not_found", "month": month_s, "visit_id": vid}
    csv_by_visit = _load_csv_by_visit_cached(month_s)
    rec = _flat_case(case, csv_by_visit.get(vid))
    deep = case.get("deep") or {}
    findings = [f for f in (deep.get("findings") or []) if isinstance(f, dict)]
    sev_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    findings.sort(key=lambda f: (sev_order.get(f.get("severity"), 9), bool(f.get("passed"))))
    return {
        "ok": True,
        "month": month_s,
        "visit_id": vid,
        "source_path": path,
        "record": rec,
        "axes": deep.get("axes") or {},
        "deep_overall_pct": deep.get("overall_pct"),
        "deep_status": _deep_status_calibrated(deep),
        "n_by_severity": deep.get("n_by_severity") or {},
        "has_potential_harm": bool(deep.get("has_potential_harm")),
        "protocol_used": bool(deep.get("protocol_used")),
        "findings": findings,
        "block_scores": case.get("block_scores") or {},
        "reg55": deep.get("reg55") or {},
    }


def _load_summary_prefer_deep(month: str) -> dict[str, Any] | None:
    """Summary, предпочитая тот, где есть deep_eval (deep_eval-dir важнее для §7Б)."""
    name = f"kz_l1_{month}_summary.json"
    paths = [
        ROOT / "data" / "ml" / "reports" / "deep_eval" / name,
        Path("/var/data/mis_protocol") / name,
        ROOT / "data" / "mis_protocol" / name,
    ]
    fallback: dict[str, Any] | None = None
    for p in paths:
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        data = dict(data)
        data["_source_path"] = str(p)
        if data.get("deep_eval"):
            return data
        if fallback is None:
            fallback = data
    return fallback


def build_scoring_info() -> dict[str, Any]:
    """Объяснимость скора: критерии осей, risk-gate, действующие пороги (Э4-конфиг).

    Используется дашбордом (вкладка «Как считается скор») - методист видит, из чего
    складывается overall и почему выставлен статус.
    """
    cfg = {}
    try:
        from .kz_deep_eval import load_deep_config

        cfg = load_deep_config()
    except Exception:  # noqa: BLE001
        cfg = {"t_good": 80.0, "t_acc": 60.0, "min_axis_review": None}
    t_good = cfg.get("t_good", 80.0)
    t_acc = cfg.get("t_acc", 60.0)
    min_axis = cfg.get("min_axis_review")
    return {
        "ok": True,
        "overall_rule": (
            "Overall = среднее доступных осей (оси без данных не штрафуются - объективность). "
            "Затем применяется risk-gate: критичные находки ограничивают итог независимо от среднего."
        ),
        "axes": [
            {"key": "documentation", "label": "Оформление (A)",
             "desc": "Полнота КЗ: жалобы, анамнез, объективный статус, диагноз, рекомендации по обследованию и лечению, наблюдение."},
            {"key": "clinical_concordance", "label": "Согласованность (B)",
             "desc": "Диагноз опирается на жалобы/анамнез/статус; валидность и совпадение кода МКБ; покрытие обязательных обследований и диагностических критериев протокола МЗ; лечение соответствует протоколу."},
            {"key": "safety", "label": "Безопасность (C)",
             "desc": "Red flags без маршрутизации, дубли НПВС, лекарственные взаимодействия (DDInter), препараты высокого риска без дозы/мониторинга (ISMP), STOPP/Beers у пожилых."},
            {"key": "regulatory", "label": "Регуляторика (D)",
             "desc": "Соответствие требованиям Пост. №55 (обязательные реквизиты и разделы КЗ)."},
        ],
        "severity": [
            {"key": "P0", "label": "Критично", "desc": "Потенциальный вред пациенту. Ограничивает overall до 40, статус «критично»."},
            {"key": "P1", "label": "Клинический дефект", "desc": "Серьёзное несоответствие. Ограничивает overall до 60, статус «на разбор»/«плохо»."},
            {"key": "P2", "label": "Документирование", "desc": "Умеренный дефект оформления/согласованности."},
            {"key": "P3", "label": "Формальное", "desc": "Незначительное замечание."},
        ],
        "risk_gate": [
            "Есть находка P0 → overall ≤ 40, статус «критично».",
            "Есть находка P1 → overall ≤ 60, статус «на разбор» (или «плохо» при <50).",
            (f"Любая ось ниже {int(min_axis)} → статус не выше «на разбор» "
             "(сильная ось не маскирует провал другой)." if min_axis is not None
             else "Правило min-axis отключено."),
            f"Иначе overall ≥ {int(t_good)} → «хорошо»; ≥ {int(t_acc)} → «приемлемо»; ниже → «на разбор».",
        ],
        "thresholds": {"good": t_good, "acceptable": t_acc, "min_axis_review": min_axis},
        "status_labels": {
            "good": "хорошо", "acceptable": "приемлемо", "review": "на разбор",
            "poor": "плохо", "critical": "критично", "insufficient_data": "мало данных",
        },
        "source": "config/deep_thresholds.yaml (Э4-калибровка на LLM-прокси); движок clinical_knowledge/kz_deep_eval.py",
    }


def build_kz_dynamics(*, months: list[str] | None = None) -> dict[str, Any]:
    """Динамика deep-оценки по месяцам (для линий/спарклайнов дашборда)."""
    ms = months or _available_months()
    series: list[dict[str, Any]] = []
    for m in ms:
        summary = _load_summary_prefer_deep(m)
        if not summary:
            continue
        deep = summary.get("deep_eval") or {}
        n = deep.get("n") or summary.get("n_cases") or 0
        sev = deep.get("severity_totals") or {}
        p0 = int(sev.get("P0", 0) or 0)
        axis_means = deep.get("axis_means") or {}
        status_dist = deep.get("status_distribution") or {}
        overall_vals = [v for v in axis_means.values() if isinstance(v, (int, float))]
        # «плохие» = статусы, требующие внимания методиста
        n_bad = sum(int(status_dist.get(k, 0) or 0) for k in ("review", "poor", "critical"))
        n_harm = deep.get("n_potential_harm")
        series.append({
            "month": m,
            "n": n,
            "avg_overall": round(sum(overall_vals) / len(overall_vals), 1) if overall_vals else summary.get("avg_overall_pct"),
            "axis_means": axis_means,
            "p0": p0,
            "p0_per_100": round(100 * p0 / n, 2) if n else 0.0,
            "n_potential_harm": n_harm,
            "n_bad": n_bad,
            "pct_bad": round(100 * n_bad / n, 1) if n else 0.0,
            "status_distribution": status_dist,
            "has_deep": bool(deep),
        })
    return {
        "ok": True,
        "available": bool(series),
        "months": [s["month"] for s in series],
        "series": series,
    }


def _parse_gemini_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _load_batch_helpers():
    import importlib.util

    batch_path = ROOT / "scripts" / "run_mis_protocol_l1_batch.py"
    spec = importlib.util.spec_from_file_location("run_mis_protocol_l1_batch", batch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("batch_script_missing")
    batch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch_mod)
    return batch_mod.build_kz_text, batch_mod.load_csv_by_visit


def _protocol_title(path: str) -> str:
    name = Path(str(path)).name if path else ""
    return name[:120] if name else str(path)[:120]


def _extract_l2_context(result: dict[str, Any]) -> dict[str, Any]:
    sa = result.get("structured_analysis") or {}
    comp = sa.get("compliance") if isinstance(sa, dict) else {}
    if not isinstance(comp, dict):
        comp = {}
    overall = result.get("overall_score")
    if overall is None:
        overall = comp.get("overall_score")
    status = result.get("overall_status") or comp.get("overall_status")

    protocols: list[dict[str, Any]] = []
    seen: set[str] = set()
    for src in (
        comp.get("matched_protocols") or [],
        result.get("retrieval_paths") or [],
        ((result.get("alignment") or {}) if isinstance(result.get("alignment"), dict) else {}).get(
            "protocol_paths"
        )
        or [],
    ):
        if not isinstance(src, list):
            continue
        for it in src:
            if isinstance(it, dict):
                p = str(it.get("path") or it.get("protocol_path") or it.get("title") or "").strip()
            else:
                p = str(it or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            protocols.append({"path": p, "title": _protocol_title(p)})
            if len(protocols) >= 6:
                break
        if len(protocols) >= 6:
            break

    gaps: list[str] = []
    for item in comp.get("critical_issues") or comp.get("issues") or []:
        if isinstance(item, dict):
            txt = str(item.get("message_ru") or item.get("text") or item.get("issue") or "").strip()
        else:
            txt = str(item).strip()
        if txt:
            gaps.append(txt[:220])
        if len(gaps) >= 8:
            break
    align = result.get("alignment") if isinstance(result.get("alignment"), dict) else {}
    for card in align.get("alignment_cards") or []:
        if not isinstance(card, dict):
            continue
        for g in card.get("gaps_ru") or []:
            t = str(g).strip()
            if t:
                gaps.append(t[:220])
            if len(gaps) >= 10:
                break
        if len(gaps) >= 10:
            break

    block_scores: dict[str, Any] = {}
    alignment_blocks = comp.get("alignment_by_block") or {}
    if isinstance(alignment_blocks, dict):
        for bid, val in alignment_blocks.items():
            if isinstance(val, dict):
                sc = val.get("score")
                if sc is None:
                    sc = val.get("alignment_score")
                block_scores[str(bid)] = sc
            elif isinstance(val, (int, float)):
                block_scores[str(bid)] = float(val)

    evidence_snippets: list[str] = []
    ep = result.get("evidence_pack")
    if isinstance(ep, dict):
        blocks = ep.get("blocks") or {}
        if isinstance(blocks, dict):
            for items in blocks.values():
                if not isinstance(items, list):
                    continue
                for it in items[:2]:
                    if not isinstance(it, dict):
                        continue
                    snip = str(it.get("text") or it.get("excerpt") or it.get("snippet") or "").strip()
                    title = _protocol_title(str(it.get("protocol_path") or ""))
                    if snip:
                        evidence_snippets.append(f"[{title}] {snip[:280]}")
                    if len(evidence_snippets) >= 8:
                        break
                if len(evidence_snippets) >= 8:
                    break

    review = result.get("review") if isinstance(result.get("review"), dict) else {}
    summary_l2 = str(
        result.get("summary_ru")
        or review.get("summary_ru")
        or ""
    ).strip()

    try:
        overall_f = round(float(overall), 1) if overall is not None else None
    except (TypeError, ValueError):
        overall_f = None

    return {
        "l2_overall_pct": overall_f,
        "l2_status": status,
        "l2_summary": summary_l2[:500] if summary_l2 else None,
        "protocols": protocols,
        "gaps_l2": gaps[:10],
        "block_scores": block_scores,
        "evidence_snippets": evidence_snippets,
        "render_l2_limited": bool(result.get("render_l2_limited")),
    }


def _build_full_llm_prompt(*, row: dict, visit_id: str, text: str, l2_ctx: dict) -> str:
    proto_lines = []
    for p in l2_ctx.get("protocols") or []:
        proto_lines.append(f"- {p.get('title') or p.get('path')}")
    gap_lines = [f"- {g}" for g in (l2_ctx.get("gaps_l2") or [])[:8]]
    ev_lines = [f"- {e}" for e in (l2_ctx.get("evidence_snippets") or [])[:6]]
    blocks = l2_ctx.get("block_scores") or {}
    block_line = ", ".join(f"{k}={v}" for k, v in list(blocks.items())[:10])
    return (
        "Ты методист клиники и аудитор качества КЗ по клиническим протоколам Минздрава РБ.\n"
        "Сделай ИНДИВИДУАЛЬНЫЙ полный разбор консультативного заключения.\n"
        "Опирайся на найденные протоколы МЗ и замечания L2; не выдумывай протоколы вне списка.\n"
        "Верни ТОЛЬКО JSON без markdown со схемой:\n"
        "{\n"
        '  "overall_pct": 0-100,\n'
        '  "status": "non_compliant|partially_compliant|mostly_compliant|compliant|manual_review_required",\n'
        '  "executive_summary_ru": "3-6 предложений: итог для методиста",\n'
        '  "protocol_review": [{"protocol": "название", "compliance_ru": "насколько КЗ соответствует", "gaps_ru": ["пробел"]}],\n'
        '  "block_review": [{"block": "жалобы|анамнез|объективный статус|диагноз|обследования|лечение|наблюдение", '
        '"score_pct": 0-100, "comment_ru": "что не так / что хорошо"}],\n'
        '  "critical_gaps_ru": ["критичный пробел"],\n'
        '  "recommendations_ru": ["конкретное действие врачу/методисту"],\n'
        '  "mz_notes_ru": "кратко про соответствие требованиям оформления и протоколам МЗ"\n'
        "}\n\n"
        f"Врач: {(row.get('doctor_fio') or '').strip()}\n"
        f"Специальность: {(row.get('doctor_specialization') or '').strip()}\n"
        f"Филиал: {(row.get('filial') or '').strip()}\n"
        f"Дата: {(row.get('date') or '')[:19]}\n"
        f"Visit ID: {visit_id}\n"
        f"Patient ID: {str(row.get('patient_id') or '').strip()}\n"
        f"L2 overall: {l2_ctx.get('l2_overall_pct')} / {l2_ctx.get('l2_status')}\n"
        f"L2 summary: {l2_ctx.get('l2_summary') or ' - '}\n"
        f"Баллы блоков L2: {block_line or ' - '}\n"
        "Протоколы МЗ (кандидаты):\n"
        + ("\n".join(proto_lines) if proto_lines else "- (не найдены)")
        + "\nЗамечания L2:\n"
        + ("\n".join(gap_lines) if gap_lines else "- (нет)")
        + "\nВыдержки из протоколов:\n"
        + ("\n".join(ev_lines) if ev_lines else "- (нет)")
        + f"\n\nТекст КЗ:\n{text[:11000]}"
    )


def _format_full_report_text(parsed: dict[str, Any], l2_ctx: dict[str, Any]) -> str:
    parts: list[str] = []
    exec_s = str(parsed.get("executive_summary_ru") or parsed.get("comment_ru") or "").strip()
    if exec_s:
        parts.append(exec_s)
    mz = str(parsed.get("mz_notes_ru") or "").strip()
    if mz:
        parts.append("МЗ / оформление: " + mz)
    crit = parsed.get("critical_gaps_ru") or []
    if isinstance(crit, list) and crit:
        parts.append("Критичные пробелы: " + "; ".join(str(x)[:120] for x in crit[:5]))
    rec = parsed.get("recommendations_ru") or []
    if isinstance(rec, list) and rec:
        parts.append("Рекомендации: " + "; ".join(str(x)[:120] for x in rec[:5]))
    protos = parsed.get("protocol_review") or []
    if isinstance(protos, list) and protos:
        bits = []
        for p in protos[:4]:
            if not isinstance(p, dict):
                continue
            bits.append(
                f"{p.get('protocol') or 'протокол'}: {str(p.get('compliance_ru') or '')[:140]}"
            )
        if bits:
            parts.append("Протоколы: " + " | ".join(bits))
    if not parts and l2_ctx.get("l2_summary"):
        parts.append(str(l2_ctx["l2_summary"]))
    return "\n\n".join(parts)[:2500]


def upsert_llm_review(*, month: str, item: dict[str, Any]) -> dict[str, Any]:
    existing = load_gemini_reviews(month=month)
    by_vid = {
        str(r.get("visit_id") or ""): r
        for r in (existing.get("reviews") or [])
        if isinstance(r, dict)
    }
    vid = str(item.get("visit_id") or "")
    if vid:
        by_vid[vid] = item
    reviews = sorted(
        by_vid.values(),
        key=lambda r: (
            r.get("overall_pct") if isinstance(r.get("overall_pct"), (int, float)) else 999,
            str(r.get("ts") or ""),
        ),
    )
    meta = {
        **(existing.get("meta") or {}),
        "note_ru": "Выборочный полный LLM-разбор КЗ с опорой на протоколы МЗ.",
        "last_batch_at": _utc(),
        "last_visit_id": vid,
    }
    path = save_gemini_reviews(month=month, reviews=reviews, meta=meta)
    summary = load_mis_kz_summary(month=month)
    if summary and summary.get("_source_path"):
        try:
            sp = Path(str(summary["_source_path"]))
            if sp.is_file():
                data = json.loads(sp.read_text(encoding="utf-8"))
                data["gemini_reviews"] = reviews
                data["gemini_meta"] = {
                    "note_ru": meta["note_ru"],
                    "last_batch_at": meta.get("last_batch_at"),
                }
                sp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return {"reviews": reviews, "meta": meta, "path": str(path)}


def review_one_visit_full(*, month: str, visit_id: str) -> dict[str, Any]:
    """Один визит: L2-контекст по протоколам МЗ + полный LLM-отчёт."""
    month = (month or "").strip() or "2026-07"
    vid = str(visit_id or "").strip()
    if not vid:
        return {"ok": False, "error": "empty_visit_id"}

    csv_path = _csv_path_for_month(month)
    if csv_path is None:
        return {
            "ok": False,
            "error": "csv_not_found",
            "hint_ru": f"Нет mis_protocol_{month}.csv на /var/data или data/mis_protocol",
        }

    import os as _os

    import rag_server as rs
    from clinical_knowledge.gemini_model_config import methodist_gemini_model_name

    build_kz_text, load_csv_by_visit = _load_batch_helpers()
    csv_by_visit = load_csv_by_visit(csv_path)
    row = csv_by_visit.get(vid)
    if not row:
        item = {
            "visit_id": vid,
            "error": "visit_not_in_csv",
            "comment": "Визит не найден в CSV",
            "ts": _utc(),
            "report_kind": "full",
        }
        stored = upsert_llm_review(month=month, item=item)
        return {"ok": False, "error": "visit_not_in_csv", "item": item, **stored}

    model_name, model_warn = methodist_gemini_model_name()
    model = rs.get_methodist_gemini()
    if model is None:
        return {
            "ok": False,
            "error": "llm_unavailable",
            "hint_ru": "LLM недоступен (нет ключа или модели).",
        }

    text = build_kz_text(row)
    _os.environ.setdefault("CONSULT_L2_FAST", "1")
    _os.environ.setdefault("CONSULT_RENDER_L2_SKIP_LLM", "0")

    stages: list[str] = ["l2_start"]
    l2_ctx: dict[str, Any] = {}
    try:
        l2_result = rs._consult_review_from_tier_or_pipeline(
            tier="L2",
            text=text,
            bundle=None,
            consultation_id=f"mis-llm-{vid}",
            category_slugs="",
            require_rag_for_l2=False,
            l2_narrative=False,
        )
        l2_ctx = _extract_l2_context(l2_result if isinstance(l2_result, dict) else {})
        stages.append("l2_done")
    except Exception as e:
        stages.append("l2_fail")
        l2_ctx = {
            "l2_overall_pct": None,
            "l2_status": None,
            "l2_summary": f"L2 недоступен: {e}"[:200],
            "protocols": [],
            "gaps_l2": [],
            "block_scores": {},
            "evidence_snippets": [],
            "render_l2_limited": False,
            "l2_error": str(e)[:300],
        }

    prompt = _build_full_llm_prompt(row=row, visit_id=vid, text=text, l2_ctx=l2_ctx)
    stages.append("llm_start")
    try:
        resp = rs.generate_gemini_methodist_ai_review(model, prompt)
        raw = rs._extract_gemini_text(resp)
        parsed = _parse_gemini_json(raw)
        if not parsed:
            parsed = {
                "overall_pct": l2_ctx.get("l2_overall_pct"),
                "status": l2_ctx.get("l2_status"),
                "executive_summary_ru": (
                    (l2_ctx.get("l2_summary") or "").strip()
                    or "LLM вернул ответ без валидного JSON - показан контекст L2."
                ),
                "critical_gaps_ru": l2_ctx.get("gaps_l2") or [],
                "recommendations_ru": [],
                "protocol_review": [
                    {"protocol": p.get("title"), "compliance_ru": "см. L2", "gaps_ru": []}
                    for p in (l2_ctx.get("protocols") or [])[:3]
                ],
                "block_review": [],
                "mz_notes_ru": "",
            }
        overall = parsed.get("overall_pct")
        try:
            overall_f = round(float(overall), 1) if overall is not None else l2_ctx.get("l2_overall_pct")
        except (TypeError, ValueError):
            overall_f = l2_ctx.get("l2_overall_pct")
        # Если парсер вернул сырой JSON в executive_summary - переразбираем.
        exec_s = str(parsed.get("executive_summary_ru") or "").strip()
        if exec_s.startswith("{"):
            repaired = _parse_gemini_json(exec_s)
            if repaired.get("executive_summary_ru"):
                parsed = {**parsed, **{k: repaired[k] for k in repaired if repaired.get(k) is not None}}
                exec_s = str(parsed.get("executive_summary_ru") or "").strip()
                if isinstance(parsed.get("overall_pct"), (int, float)):
                    overall_f = round(float(parsed["overall_pct"]), 1)
        report_text = _format_full_report_text(parsed, l2_ctx)
        item = {
            "visit_id": vid,
            "patient_id": str(row.get("patient_id") or "").strip(),
            "date": (row.get("date") or "")[:19],
            "doctor_fio": (row.get("doctor_fio") or "").strip(),
            "doctor_specialization": (row.get("doctor_specialization") or "").strip(),
            "filial": (row.get("filial") or "").strip(),
            "diagnosis_short": ((row.get("clinical_diagnosis") or "").strip())[:160],
            "overall_pct": overall_f,
            "status": parsed.get("status") or l2_ctx.get("l2_status"),
            "comment": report_text[:600],
            "report_full_ru": report_text,
            "executive_summary_ru": str(parsed.get("executive_summary_ru") or "")[:1200],
            "protocol_review": parsed.get("protocol_review") or [],
            "block_review": parsed.get("block_review") or [],
            "critical_gaps_ru": parsed.get("critical_gaps_ru") or [],
            "recommendations_ru": parsed.get("recommendations_ru") or [],
            "mz_notes_ru": str(parsed.get("mz_notes_ru") or "")[:800],
            "protocols_mz": l2_ctx.get("protocols") or [],
            "l2_overall_pct": l2_ctx.get("l2_overall_pct"),
            "l2_status": l2_ctx.get("l2_status"),
            "l2_gaps": l2_ctx.get("gaps_l2") or [],
            "block_scores": l2_ctx.get("block_scores") or {},
            "report_kind": "full",
            "stages": stages + ["llm_done"],
            "ts": _utc(),
            "error": None,
        }
        # do not expose vendor model names to UI clients
        item.pop("model", None)
        stored = upsert_llm_review(month=month, item=item)
        return {
            "ok": True,
            "month": month,
            "visit_id": vid,
            "item": item,
            "stages": item["stages"],
            "reviews": stored["reviews"],
            "storage_path": stored["path"],
        }
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "spend" in low or "spending cap" in low:
            friendly = (
                "LLM-разбор недоступен: исчерпан месячный лимит расходов проекта Gemini "
                "(spend cap). Поднимите лимит в Google AI Studio (ai.studio/spend) или "
                "дождитесь начала нового месяца. Детерминированная оценка (оси A/B/C/D, "
                "находки, risk-gate) в дашборде работает без LLM."
            )
        elif "429" in msg or "resource_exhausted" in low or "rate limit" in low or "quota" in low:
            friendly = (
                "LLM временно недоступен: превышен лимит запросов Gemini (429). "
                "Повторите позже. Детерминированная оценка доступна без LLM."
            )
        else:
            friendly = f"Ошибка LLM: {msg}"[:300]
        item = {
            "visit_id": vid,
            "patient_id": str(row.get("patient_id") or "").strip(),
            "date": (row.get("date") or "")[:19],
            "doctor_fio": (row.get("doctor_fio") or "").strip(),
            "error": str(e)[:300],
            "comment": friendly[:300],
            "report_full_ru": friendly[:500],
            "l2_overall_pct": l2_ctx.get("l2_overall_pct"),
            "protocols_mz": l2_ctx.get("protocols") or [],
            "report_kind": "full",
            "stages": stages + ["llm_fail"],
            "ts": _utc(),
        }
        stored = upsert_llm_review(month=month, item=item)
        return {
            "ok": False,
            "error": "llm_spend_cap" if ("spend" in low or "spending cap" in low) else "llm_failed",
            "hint_ru": friendly[:300],
            "item": item,
            "reviews": stored["reviews"],
            "storage_path": stored["path"],
        }


def review_visits_with_gemini(
    *,
    month: str,
    visit_ids: list[str],
    max_visits: int = 20,
) -> dict[str, Any]:
    """Пакетный прогон (совместимость): полный разбор по каждому visit_id."""
    month = (month or "").strip() or "2026-07"
    ids = [str(v).strip() for v in visit_ids if str(v).strip()]
    ids = ids[: max(1, int(max_visits))]
    if not ids:
        return {"ok": False, "error": "empty_visit_ids", "reviews": []}

    batch: list[dict[str, Any]] = []
    last_reviews: list[dict] = []
    storage_path = ""
    for vid in ids:
        out = review_one_visit_full(month=month, visit_id=vid)
        if out.get("item"):
            batch.append(out["item"])
        last_reviews = out.get("reviews") or last_reviews
        storage_path = out.get("storage_path") or storage_path
    ok_n = sum(1 for x in batch if not x.get("error"))
    return {
        "ok": ok_n > 0,
        "month": month,
        "storage_path": storage_path,
        "reviews": last_reviews,
        "batch": batch,
        "hint_ru": None if ok_n else "Не удалось разобрать выбранные визиты",
    }
