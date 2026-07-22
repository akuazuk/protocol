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
        item = {
            "visit_id": vid,
            "patient_id": str(row.get("patient_id") or "").strip(),
            "date": (row.get("date") or "")[:19],
            "doctor_fio": (row.get("doctor_fio") or "").strip(),
            "error": str(e)[:300],
            "comment": f"Ошибка LLM: {e}"[:300],
            "report_full_ru": f"Ошибка LLM: {e}"[:500],
            "l2_overall_pct": l2_ctx.get("l2_overall_pct"),
            "protocols_mz": l2_ctx.get("protocols") or [],
            "report_kind": "full",
            "stages": stages + ["llm_fail"],
            "ts": _utc(),
        }
        stored = upsert_llm_review(month=month, item=item)
        return {
            "ok": False,
            "error": "llm_failed",
            "hint_ru": str(e)[:300],
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
