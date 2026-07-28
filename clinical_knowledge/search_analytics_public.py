"""Публичная аналитика: поиск протоколов + обезличенные метрики КЗ (без ПДн)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SEARCH_USAGE_TIPS_RU = [
    "Указывайте код МКБ-10 в запросе (например I10, J20.9) - подбор точнее, чем только текст жалобы.",
    "Если кода нет, нажмите «Подобрать коды МКБ-10», затем «Добавить в поле и найти протоколы».",
    "Отметьте 1-2 рубрики каталога, если знаете профиль (кардиология, ЛОР, гастро…).",
    "Заполните возраст и пол - сервис отделит детские и взрослые протоколы.",
    "Оценку «соответствие» смотрите вместе с выдержками PDF; при сомнении откройте полный протокол на сайте МЗ.",
    "Дифференциальные гипотезы - для уточнения запроса, не окончательный диагноз.",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_public_search_analytics(
    *,
    corpus: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    version: str | None = None,
    rag_ready: bool | None = None,
) -> dict[str, Any]:
    from clinical_knowledge.pilot_analytics_public import build_public_pilot_analytics
    from clinical_knowledge.search_telemetry import aggregate_protocol_search, iter_protocol_search_events

    corpus = corpus or {}
    quality = quality or {}
    events = iter_protocol_search_events()
    telemetry = aggregate_protocol_search(events)
    has_search_live = telemetry["total_searches"] > 0

    pilot = build_public_pilot_analytics(corpus=corpus, version=version, rag_ready=rag_ready)
    pilot_charts = pilot.get("charts") or {}
    has_kz_live = bool(pilot.get("live"))

    categories = (corpus.get("categories_top") or [])[:12]
    years_top = (corpus.get("years_top") or [])[:8]
    post_mz = int(corpus.get("protocols_post_mz") or 0)
    total_index = int(corpus.get("protocols_in_index") or 0)
    post_mz_chart: list[dict[str, Any]] = []
    if total_index > 0:
        post_mz_chart = [
            {"label": "пост МЗ", "count": post_mz},
            {"label": "прочие PDF", "count": max(0, total_index - post_mz)},
        ]

    passed = int(quality.get("queries_passed") or 0)
    total_q = int(quality.get("queries_total") or 0)
    failed = max(0, total_q - passed) if total_q else 0

    verdict = pilot.get("verdict_breakdown") or {}
    verdict_chart = [
        {"label": k or "?", "count": int(v)}
        for k, v in verdict.items()
        if v and int(v) > 0
    ]

    events_by_type = pilot_charts.get("events_by_type") or []
    # Переименуем для UI
    event_labels = {
        "protocol_search": "поиск протокола",
        "kz_analysis": "прогон КЗ",
        "analysis_review": "оценка методиста",
        "retrieval_fix": "retrieval_fix",
        "methodist_override": "override правила",
    }
    events_chart = [
        {
            "type": row.get("type") or "",
            "label": event_labels.get(str(row.get("type") or ""), str(row.get("type") or "")),
            "count": int(row.get("count") or 0),
        }
        for row in events_by_type
        if int(row.get("count") or 0) > 0
    ]

    return {
        "generated_at": _utc_now(),
        "live": has_search_live or has_kz_live,
        "version": version,
        "rag_ready": rag_ready,
        "corpus": {
            "protocols_in_index": corpus.get("protocols_in_index"),
            "protocols_post_mz": corpus.get("protocols_post_mz"),
            "rubrics_in_index": corpus.get("rubrics_in_index") or corpus.get("specialties_catalog"),
            "chunks_loaded": corpus.get("chunks_loaded"),
            "manifest_paths": corpus.get("manifest_paths"),
            "startup_mode": corpus.get("startup_mode"),
            "index_csv_updated_utc": corpus.get("index_csv_updated_utc"),
        },
        "quality_benchmark": {
            "pass_rate_pct": quality.get("pass_rate_pct"),
            "queries_passed": passed,
            "queries_total": total_q,
            "updated": quality.get("updated"),
            "title": quality.get("title"),
        },
        "telemetry": telemetry,
        "kz": {
            "kz_runs_total": pilot.get("kz_runs_total"),
            "unique_kz": pilot.get("unique_kz"),
            "reviews_total": pilot.get("reviews_total"),
            "avg_compliance_pct": pilot.get("avg_compliance_pct"),
            "risk_zone_rate_pct": pilot.get("risk_zone_rate_pct"),
            "protocol_hit_at_3_pct": pilot.get("protocol_hit_at_3_pct"),
            "readiness_overall_pct": pilot.get("readiness_overall_pct"),
            "retrieval_fix_count": pilot.get("retrieval_fix_count"),
            "priority_cases": pilot.get("priority_cases"),
            "engine_releases_count": pilot.get("engine_releases_count"),
        },
        "charts": {
            "categories_top": categories,
            "years_top": years_top,
            "post_mz_breakdown": post_mz_chart,
            "search_activity_by_day": telemetry.get("activity_by_day") or [],
            "benchmark_pass_fail": [
                {"label": "эталон OK", "count": passed},
                {"label": "не прошли", "count": failed},
            ]
            if total_q
            else [],
            "confidence_buckets": telemetry.get("confidence_buckets") or [],
            "icd_usage": telemetry.get("icd_usage") or [],
            "rubric_filter_usage": telemetry.get("rubric_filter_usage") or [],
            "protocols_returned_buckets": telemetry.get("protocols_returned_buckets") or [],
            "audience_breakdown": telemetry.get("audience_breakdown") or [],
            "kz_compliance_buckets": pilot_charts.get("compliance_buckets") or [],
            "rating_histogram": pilot_charts.get("rating_histogram") or [],
            "rubric_kz_runs": pilot_charts.get("rubric_kz_runs") or [],
            "kz_activity_by_day": pilot_charts.get("activity_by_day") or [],
            "tags_top": pilot_charts.get("tags_top") or [],
            "readiness_items": pilot_charts.get("readiness_items") or [],
            "events_by_type": events_chart,
            "verdict_breakdown": verdict_chart,
            "block_overrides_top": pilot_charts.get("block_overrides_top") or [],
            "kz_protocol_match": pilot_charts.get("kz_protocol_match") or [],
        },
        "tips_ru": SEARCH_USAGE_TIPS_RU,
        "note_ru": (
            "Данные обновляются при каждом запросе; тексты запросов и КЗ не показываются."
            if (has_search_live or has_kz_live)
            else "После первых поисков и прогонов КЗ здесь появятся обезличенные метрики."
        ),
        "period_label": pilot.get("period_label"),
    }
