"""Ночной цикл качества B2C: агрегация, LLM-рекомендации, pending snippets, email."""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from .feedback_store import feedback_dir
from .patient_email import send_report_email
from .patient_specialty import PENDING_DIR, SPECIALTY_DIR

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "data" / "ml" / "reports"
LATEST_JSON = REPORTS_DIR / "patient_nightly_latest.json"
LATEST_MD = REPORTS_DIR / "patient_nightly_latest.md"

SYSTEM_PATIENT_NIGHTLY = """Ты аудитор качества B2C-модуля «Protocol · после приёма» (отчёт для пациента после консультации).

На вход - агрегированная статистика за период и примеры quality_flags (без ПДн, без текста КЗ).

Задача:
1) Кратко оценить здоровье продукта (2-4 предложения).
2) Предложить 3-7 конкретных улучшений для кода/шаблонов (не «врачу дописать КЗ»).
3) Для каждого улучшения - specialty (default|dermatology|neurology|...) и тип (code|yaml_snippet|test).
4) Отметить, были ли автоматически применены изменения (applied=false всегда - только черновики).

Верни ОДИН JSON:
{
  "summary_ru": "...",
  "health_score": 1-5,
  "improvements": [
    {"title_ru": "...", "detail_ru": "...", "specialty": "default", "change_type": "code|yaml_snippet|test", "priority": "high|medium|low"}
  ],
  "applied_changes_ru": ["..."],
  "no_action_ru": "..."
}"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iter_patient_events(fb_root: Path | None = None) -> Iterator[dict[str, Any]]:
    root = fb_root or feedback_dir()
    for name in ("patient_review.jsonl", "patient_ui.jsonl", "patient_nightly.jsonl"):
        path = root / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row["_log"] = name
                yield row


def aggregate_patient_feedback(*, fb_root: Path | None = None, days: int = 7) -> dict[str, Any]:
    flag_counts: Counter[str] = Counter()
    specialty_counts: Counter[str] = Counter()
    build_counts: Counter[str] = Counter()
    ui_events: Counter[str] = Counter()
    reviews = 0
    clarity_sum = 0
    clarity_n = 0

    for row in iter_patient_events(fb_root):
        if row.get("event_type") == "patient_review":
            reviews += 1
            for f in row.get("quality_flags") or []:
                flag_counts[str(f)] += 1
            spec = (row.get("context") or {}).get("specialty_inferred") or "unknown"
            specialty_counts[str(spec)] += 1
            build_counts[str(row.get("build_version") or "unknown")] += 1
            pct = (row.get("scores") or {}).get("patient_clarity")
            if isinstance(pct, (int, float)):
                clarity_sum += int(pct)
                clarity_n += 1
        elif row.get("event_type") == "patient_ui":
            ui_events[str(row.get("event") or "unknown")] += 1

    return {
        "period_days": days,
        "generated_at": _utc_now(),
        "review_count": reviews,
        "avg_patient_clarity": round(clarity_sum / clarity_n, 1) if clarity_n else None,
        "quality_flags": dict(flag_counts.most_common(20)),
        "specialties": dict(specialty_counts.most_common(12)),
        "build_versions": dict(build_counts.most_common(8)),
        "ui_events": dict(ui_events.most_common(20)),
    }


def _sample_flag_snapshots(fb_root: Path | None, flag: str, limit: int = 2) -> list[dict[str, Any]]:
    from .feedback_store import analyses_dir

    out: list[dict[str, Any]] = []
    snap_dir = analyses_dir() / "patient"
    if not snap_dir.is_dir():
        return out
    for row in iter_patient_events(fb_root):
        if row.get("event_type") != "patient_review":
            continue
        flags = row.get("quality_flags") or []
        if flag not in flags:
            continue
        th = str(row.get("text_hash") or "")
        snap_path = snap_dir / f"{th.replace(':', '_')}.json"
        if snap_path.is_file():
            try:
                snap = json.loads(snap_path.read_text(encoding="utf-8"))
                out.append({"text_hash": th, "flags": flags, "questions": snap.get("questions_for_doctor")})
            except json.JSONDecodeError:
                pass
        if len(out) >= limit:
            break
    return out


def run_llm_nightly_review(
    stats: dict[str, Any],
    *,
    llm_call: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    top_flags = list((stats.get("quality_flags") or {}).keys())[:5]
    samples = []
    for f in top_flags[:3]:
        samples.extend(_sample_flag_snapshots(None, f, limit=1))

    user_payload = {
        "stats": stats,
        "sample_snapshots": samples,
    }
    prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)

    if llm_call is None:
        try:
            import rag_server as rs

            model = rs.get_gemini()
            if model is None:
                return _deterministic_review(stats)

            def _llm(system: str, user: str) -> str:
                resp = rs.generate_gemini(model, f"{system}\n\n---\n\n{user}")
                return rs._extract_gemini_text(resp) if resp else ""

            llm_call = _llm
        except Exception:
            return _deterministic_review(stats)

    try:
        raw = llm_call(SYSTEM_PATIENT_NIGHTLY, prompt)
        import rag_server as rs

        parsed = rs._try_parse_json(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return _deterministic_review(stats)


def _deterministic_review(stats: dict[str, Any]) -> dict[str, Any]:
    flags = stats.get("quality_flags") or {}
    improvements: list[dict[str, Any]] = []
    if flags.get("no_mri_in_source_but_in_summary"):
        improvements.append(
            {
                "title_ru": "Убрать МРТ из narrative без назначения",
                "detail_ru": "Проверить patient_narrative и quality_flags.",
                "specialty": "default",
                "change_type": "code",
                "priority": "high",
            }
        )
    if flags.get("false_imaging_in_understood"):
        improvements.append(
            {
                "title_ru": "Ужесточить extract_exams word boundaries",
                "detail_ru": "Добавить regression test на креатинин/трансфузии.",
                "specialty": "default",
                "change_type": "test",
                "priority": "high",
            }
        )
    if stats.get("review_count", 0) == 0:
        return {
            "summary_ru": "За период нет событий patient_review - телеметрия не пишется или диск не подключён.",
            "health_score": 3,
            "improvements": [],
            "applied_changes_ru": [],
            "no_action_ru": "Включите PATIENT_TELEMETRY=1 и ML_FEEDBACK_DIR на Persistent Disk.",
        }
    return {
        "summary_ru": f"Обработано {stats.get('review_count')} проверок B2C. Топ флаги: {', '.join(list(flags.keys())[:3]) or 'нет'}.",
        "health_score": 4 if not improvements else 3,
        "improvements": improvements,
        "applied_changes_ru": [],
        "no_action_ru": "",
    }


def write_pending_snippet_proposals(llm_review: dict[str, Any]) -> list[str]:
    """Черновики YAML в _pending - не применяются автоматически в prod."""
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    for i, imp in enumerate(llm_review.get("improvements") or []):
        if not isinstance(imp, dict):
            continue
        if imp.get("change_type") != "yaml_snippet":
            continue
        spec = str(imp.get("specialty") or "default")
        fname = f"{ts}_{spec}_{i}.yaml"
        path = PENDING_DIR / fname
        payload = {
            "specialty": spec,
            "created_at": _utc_now(),
            "summary_ru": imp.get("title_ru") or "",
            "proposed_changes": [imp.get("detail_ru") or ""],
            "status": "pending_review",
        }
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        created.append(str(path.relative_to(ROOT)))
    return created


def build_markdown_report(stats: dict[str, Any], llm_review: dict[str, Any], pending: list[str]) -> str:
    lines = [
        f"# Protocol B2C - ночной отчёт качества ({stats.get('generated_at', '')})",
        "",
        f"**Проверок за период:** {stats.get('review_count', 0)}",
        f"**Средняя понятность:** {stats.get('avg_patient_clarity') or '-'}%",
        f"**Health score (LLM):** {llm_review.get('health_score', '-')}/5",
        "",
        "## Резюме",
        str(llm_review.get("summary_ru") or ""),
        "",
        "## Quality flags",
    ]
    for flag, cnt in (stats.get("quality_flags") or {}).items():
        lines.append(f"- `{flag}`: {cnt}")
    if not stats.get("quality_flags"):
        lines.append("- (нет)")
    lines.extend(["", "## Рекомендации"])
    for imp in llm_review.get("improvements") or []:
        if isinstance(imp, dict):
            lines.append(f"- **{imp.get('title_ru')}** ({imp.get('priority', 'medium')}): {imp.get('detail_ru')}")
    if llm_review.get("applied_changes_ru"):
        lines.extend(["", "## Автоматически подготовлено"])
        for a in llm_review["applied_changes_ru"]:
            lines.append(f"- {a}")
    if pending:
        lines.extend(["", "## Черновики snippet (методист)"])
        for p in pending:
            lines.append(f"- `{p}`")
    if llm_review.get("no_action_ru"):
        lines.extend(["", "## Примечание", str(llm_review["no_action_ru"])])
    lines.append("")
    return "\n".join(lines)


def run_patient_nightly_quality(
    *,
    fb_root: Path | None = None,
    send_email: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    stats = aggregate_patient_feedback(fb_root=fb_root)
    llm_review = run_llm_nightly_review(stats)
    pending = write_pending_snippet_proposals(llm_review)
    md = build_markdown_report(stats, llm_review, pending)

    if not dry_run:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        LATEST_MD.write_text(md, encoding="utf-8")
        payload = {
            "generated_at": stats.get("generated_at"),
            "stats": stats,
            "llm_review": llm_review,
            "pending_snippets": pending,
            "markdown_path": str(LATEST_MD.relative_to(ROOT)),
        }
        LATEST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        from .patient_feedback_store import _append_jsonl, PATIENT_NIGHTLY_LOG

        _append_jsonl(
            PATIENT_NIGHTLY_LOG,
            {
                "event_type": "patient_nightly",
                "ts": stats.get("generated_at"),
                "review_count": stats.get("review_count"),
                "health_score": llm_review.get("health_score"),
                "pending_count": len(pending),
            },
        )

    email_result: dict[str, Any] = {"skipped": True}
    if send_email and not dry_run:
        subject = f"Protocol B2C nightly · health {llm_review.get('health_score', '?')}/5 · {stats.get('review_count', 0)} reviews"
        html = md.replace("\n", "<br>\n").replace("## ", "<h2>").replace("# ", "<h1>")
        email_result = send_report_email(subject=subject, body_text=md, body_html=f"<html><body>{html}</body></html>")

    return {
        "ok": True,
        "stats": stats,
        "llm_review": llm_review,
        "pending_snippets": pending,
        "email": email_result,
        "report_md": str(LATEST_MD),
    }


def load_latest_nightly_report() -> dict[str, Any]:
    if not LATEST_JSON.is_file():
        return {"ok": False, "error": "no_report"}
    try:
        data = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
        data["ok"] = True
        return data
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid_json"}


def build_methodist_patient_quality_view() -> dict[str, Any]:
    """Ответ для GET /api/methodist/patient-quality: live stats + последний отчёт или черновик."""
    from .patient_specialty import list_pending_snippet_updates

    live_stats = aggregate_patient_feedback()
    pending_detail = list_pending_snippet_updates()
    pending_paths = [str(p.get("path") or "") for p in pending_detail if p.get("path")]
    report = load_latest_nightly_report()

    if report.get("ok"):
        llm_review = report.get("llm_review") or _deterministic_review(live_stats)
        stats_for_md = report.get("stats") or live_stats
        snippet_paths = report.get("pending_snippets") or pending_paths
        if LATEST_MD.is_file():
            markdown_ru = LATEST_MD.read_text(encoding="utf-8")[:12000]
        else:
            markdown_ru = build_markdown_report(stats_for_md, llm_review, snippet_paths)
        nightly_generated_at = report.get("generated_at") or stats_for_md.get("generated_at")
    else:
        llm_review = _deterministic_review(live_stats)
        markdown_ru = build_markdown_report(live_stats, llm_review, pending_paths)
        nightly_generated_at = None

    return {
        "ok": True,
        "live_stats": live_stats,
        "stats": live_stats,
        "llm_review": llm_review,
        "pending_snippets_detail": pending_detail,
        "markdown_ru": markdown_ru,
        "nightly_available": bool(report.get("ok")),
        "nightly_generated_at": nightly_generated_at,
    }
