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


def build_mis_kz_quality_view(*, month: str | None = None) -> dict[str, Any]:
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
    doctors = summary.get("doctors") or []
    from clinical_knowledge.gemini_model_config import methodist_gemini_model_name

    model_name, model_warn = methodist_gemini_model_name()
    return {
        "ok": True,
        "available": True,
        "month": summary.get("month"),
        "tier": summary.get("tier") or "L1",
        "generated_at": summary.get("generated_at"),
        "source_path": summary.get("_source_path"),
        "n_cases": summary.get("n_cases"),
        "n_ok": summary.get("n_ok"),
        "n_errors": summary.get("n_errors"),
        "avg_overall_pct": summary.get("avg_overall_pct"),
        "median_overall_pct": summary.get("median_overall_pct"),
        "score_histogram": summary.get("score_histogram") or {},
        "status_counts": summary.get("status_counts") or {},
        "block_avg": summary.get("block_avg") or {},
        "doctors": doctors,
        "specialties": summary.get("specialties") or [],
        "filials": summary.get("filials") or [],
        "top_doctors": summary.get("top_doctors") or doctors[:15],
        "bottom_doctors": summary.get("bottom_doctors") or [],
        "worst_visits": summary.get("worst_visits") or [],
        "worst_visits_meta": summary.get("worst_visits_meta") or {},
        "gemini_reviews": gem.get("reviews") or summary.get("gemini_reviews") or [],
        "gemini_meta": {
            **(summary.get("gemini_meta") or {}),
            **(gem.get("meta") or {}),
            "model": model_name,
            "model_warn": model_warn,
            "storage_path": gem.get("path"),
        },
        "notes": summary.get("notes") or [],
        "doctors_n": len(doctors),
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


def review_visits_with_gemini(
    *,
    month: str,
    visit_ids: list[str],
    max_visits: int = 20,
) -> dict[str, Any]:
    """Выборочный разбор КЗ через methodist Gemini (обычно gemini-2.5-pro)."""
    month = (month or "").strip() or "2026-07"
    ids = [str(v).strip() for v in visit_ids if str(v).strip()]
    ids = ids[: max(1, int(max_visits))]
    if not ids:
        return {"ok": False, "error": "empty_visit_ids", "reviews": []}

    csv_path = _csv_path_for_month(month)
    if csv_path is None:
        return {
            "ok": False,
            "error": "csv_not_found",
            "hint_ru": f"Нет mis_protocol_{month}.csv на /var/data или data/mis_protocol",
            "reviews": [],
        }

    from clinical_knowledge.gemini_model_config import methodist_gemini_model_name
    import importlib.util
    import rag_server as rs

    batch_path = ROOT / "scripts" / "run_mis_protocol_l1_batch.py"
    spec = importlib.util.spec_from_file_location("run_mis_protocol_l1_batch", batch_path)
    if spec is None or spec.loader is None:
        return {"ok": False, "error": "batch_script_missing", "reviews": []}
    batch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch_mod)
    build_kz_text = batch_mod.build_kz_text
    load_csv_by_visit = batch_mod.load_csv_by_visit

    model_name, model_warn = methodist_gemini_model_name()
    model = rs.get_methodist_gemini()
    if model is None:
        return {
            "ok": False,
            "error": "gemini_unavailable",
            "hint_ru": "Gemini недоступен (нет ключа или модели).",
            "model": model_name,
            "model_warn": model_warn,
            "reviews": [],
        }

    csv_by_visit = load_csv_by_visit(csv_path)
    existing = load_gemini_reviews(month=month)
    by_vid = {
        str(r.get("visit_id") or ""): r
        for r in (existing.get("reviews") or [])
        if isinstance(r, dict)
    }

    new_rows: list[dict[str, Any]] = []
    for vid in ids:
        row = csv_by_visit.get(vid)
        if not row:
            item = {
                "visit_id": vid,
                "error": "visit_not_in_csv",
                "comment": "Визит не найден в CSV",
                "model": model_name,
                "ts": _utc(),
            }
            by_vid[vid] = item
            new_rows.append(item)
            continue
        text = build_kz_text(row)
        prompt = (
            "Ты методист клиники. Оцени качество консультативного заключения (КЗ).\n"
            "Верни ТОЛЬКО JSON без markdown:\n"
            '{"overall_pct": 0-100, "status": "non_compliant|partially_compliant|mostly_compliant|compliant",'
            ' "comment_ru": "2-4 предложения: что не так и что исправить",'
            ' "gaps_ru": ["краткий пробел 1", "пробел 2"]}\n\n'
            f"Врач: {(row.get('doctor_fio') or '').strip()}\n"
            f"Специальность: {(row.get('doctor_specialization') or '').strip()}\n"
            f"Дата: {(row.get('date') or '')[:19]}\n"
            f"Visit ID: {vid}\n"
            f"Patient ID: {str(row.get('patient_id') or '').strip()}\n\n"
            f"Текст КЗ:\n{text[:12000]}"
        )
        try:
            resp = rs.generate_gemini_methodist_ai_review(model, prompt)
            raw = rs._extract_gemini_text(resp)
            parsed = _parse_gemini_json(raw)
            overall = parsed.get("overall_pct")
            try:
                overall_f = round(float(overall), 1) if overall is not None else None
            except (TypeError, ValueError):
                overall_f = None
            gaps = parsed.get("gaps_ru") or []
            if not isinstance(gaps, list):
                gaps = []
            comment = str(parsed.get("comment_ru") or "").strip()
            if gaps and not comment:
                comment = "; ".join(str(g) for g in gaps[:4])
            elif gaps:
                comment = comment + " | Пробелы: " + "; ".join(str(g) for g in gaps[:3])
            item = {
                "visit_id": vid,
                "patient_id": str(row.get("patient_id") or "").strip(),
                "date": (row.get("date") or "")[:19],
                "doctor_fio": (row.get("doctor_fio") or "").strip(),
                "doctor_specialization": (row.get("doctor_specialization") or "").strip(),
                "filial": (row.get("filial") or "").strip(),
                "diagnosis_short": ((row.get("clinical_diagnosis") or "").strip())[:160],
                "overall_pct": overall_f,
                "status": parsed.get("status"),
                "comment": comment[:600] or (raw or "")[:400],
                "gaps_ru": [str(g)[:160] for g in gaps[:5]],
                "model": model_name,
                "model_warn": model_warn,
                "ts": _utc(),
                "error": None,
            }
        except Exception as e:
            item = {
                "visit_id": vid,
                "patient_id": str(row.get("patient_id") or "").strip(),
                "date": (row.get("date") or "")[:19],
                "doctor_fio": (row.get("doctor_fio") or "").strip(),
                "error": str(e)[:300],
                "comment": f"Ошибка Gemini: {e}"[:300],
                "model": model_name,
                "model_warn": model_warn,
                "ts": _utc(),
            }
        by_vid[vid] = item
        new_rows.append(item)

    reviews = sorted(
        by_vid.values(),
        key=lambda r: (
            r.get("overall_pct") if isinstance(r.get("overall_pct"), (int, float)) else 999,
            str(r.get("ts") or ""),
        ),
    )
    meta = {
        "model": model_name,
        "model_warn": model_warn,
        "note_ru": "Gemini 3.6 в API нет; используется methodist-модель (обычно gemini-2.5-pro).",
        "last_batch_n": len(new_rows),
        "last_batch_at": _utc(),
    }
    path = save_gemini_reviews(month=month, reviews=reviews, meta=meta)

    summary = load_mis_kz_summary(month=month)
    if summary and summary.get("_source_path"):
        try:
            sp = Path(str(summary["_source_path"]))
            if sp.is_file():
                data = json.loads(sp.read_text(encoding="utf-8"))
                data["gemini_reviews"] = reviews
                data["gemini_meta"] = meta
                sp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    return {
        "ok": True,
        "month": month,
        "model": model_name,
        "model_warn": model_warn,
        "storage_path": str(path),
        "reviews": reviews,
        "batch": new_rows,
    }
