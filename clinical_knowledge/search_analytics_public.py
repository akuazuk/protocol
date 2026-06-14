"""Публичная аналитика поиска протоколов для вкладки «Поиск» (без ПДн)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SEARCH_USAGE_TIPS_RU = [
    "Указывайте код МКБ-10 в запросе (например I10, J20.9) — подбор точнее, чем только текст жалобы.",
    "Если кода нет, нажмите «Подобрать коды МКБ-10», затем «Добавить в поле и найти протоколы».",
    "Отметьте 1–2 рубрики каталога, если знаете профиль (кардиология, ЛОР, гастро…).",
    "Заполните возраст и пол — сервис отделит детские и взрослые протоколы.",
    "Оценку «соответствие» смотрите вместе с выдержками PDF; при сомнении откройте полный протокол на сайте МЗ.",
    "Дифференциальные гипотезы — для уточнения запроса, не окончательный диагноз.",
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
    from clinical_knowledge.search_telemetry import aggregate_protocol_search, iter_protocol_search_events

    corpus = corpus or {}
    quality = quality or {}
    events = iter_protocol_search_events()
    telemetry = aggregate_protocol_search(events)
    has_live = telemetry["total_searches"] > 0

    categories = (corpus.get("categories_top") or [])[:10]
    passed = int(quality.get("queries_passed") or 0)
    total_q = int(quality.get("queries_total") or 0)
    failed = max(0, total_q - passed) if total_q else 0

    return {
        "generated_at": _utc_now(),
        "live": has_live,
        "version": version,
        "rag_ready": rag_ready,
        "corpus": {
            "protocols_in_index": corpus.get("protocols_in_index"),
            "protocols_post_mz": corpus.get("protocols_post_mz"),
            "rubrics_in_index": corpus.get("rubrics_in_index") or corpus.get("specialties_catalog"),
            "chunks_loaded": corpus.get("chunks_loaded"),
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
        "charts": {
            "categories_top": categories,
            "activity_by_day": telemetry.get("activity_by_day") or [],
            "benchmark_pass_fail": [
                {"label": "эталон OK", "count": passed},
                {"label": "не прошли", "count": failed},
            ]
            if total_q
            else [],
            "confidence_buckets": telemetry.get("confidence_buckets") or [],
        },
        "tips_ru": SEARCH_USAGE_TIPS_RU,
        "note_ru": (
            "Статистика поиска обновляется при каждом запросе; тексты запросов не сохраняются."
            if has_live
            else "После первых поисков здесь появятся обезличенные метрики использования."
        ),
    }
