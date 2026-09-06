"""Две оценки + min как итог допуска (wave 4)."""
from __future__ import annotations

from typing import Any, Mapping


def dual_admission_scores(
    *,
    clinical_pct: float | int | None,
    document_ready_pct: float | int | None,
) -> dict[str, Any]:
    """Клиническая оценка и готовность документа; итог = минимум.

    Не заменяет overall SSOT №55 - отдельный контракт для UI/статьи.
    """
    clinical = None if clinical_pct is None else round(float(clinical_pct), 1)
    ready = None if document_ready_pct is None else round(float(document_ready_pct), 1)
    if clinical is None or ready is None:
        admission = None
    else:
        admission = round(min(clinical, ready), 1)
    return {
        "clinical_pct": clinical,
        "document_ready_pct": ready,
        "admission_pct": admission,
        "status": "incomplete" if admission is None else "complete",
        "rule": "min(clinical, document_ready)",
        "note_ru": (
            "Итог допуска - худшая из двух оценок, а не среднее. "
            "Не смешивать с №55 без явного решения владельца."
        ),
    }


def dual_scores_from_result(result: Mapping[str, Any] | None) -> dict[str, Any]:
    result = result or {}
    axes = result.get("axes") if isinstance(result.get("axes"), Mapping) else {}
    clinical_parts = [
        axes.get(key)
        for key in ("concordance", "safety", "documentation")
        if axes.get(key) is not None
    ]
    clinical = (
        round(sum(float(x) for x in clinical_parts) / len(clinical_parts), 1)
        if clinical_parts
        else result.get("overall_pct")
    )
    ready = None
    for key in (
        "document_ready_pct",
        "cisz_ready_pct",
        "readiness_pct",
        "fhir_ready_pct",
    ):
        if result.get(key) is not None:
            ready = result.get(key)
            break
    if ready is None and isinstance(result.get("readiness"), Mapping):
        ready = result["readiness"].get("pct")
        if ready is None:
            ready = result["readiness"].get("score")
    return dual_admission_scores(clinical_pct=clinical, document_ready_pct=ready)


def form_content_matrix(
    *,
    clinical_pct: float | int | None,
    document_ready_pct: float | int | None,
    clinical_threshold: float = 70.0,
    ready_threshold: float = 70.0,
) -> dict[str, Any]:
    """Матрица 2×2 «лечение × документы» из статьи РЗ."""
    if clinical_pct is None or document_ready_pct is None:
        return {
            "cell": "unknown",
            "title_ru": "Недостаточно данных для сравнения",
            "action_ru": "Проверьте наличие обеих оценок и полноту расчёта.",
            "clinical_high": None if clinical_pct is None else float(clinical_pct) >= clinical_threshold,
            "document_high": None if document_ready_pct is None else float(document_ready_pct) >= ready_threshold,
            "clinical_pct": clinical_pct,
            "document_ready_pct": document_ready_pct,
        }
    c_ok = clinical_pct is not None and float(clinical_pct) >= clinical_threshold
    d_ok = document_ready_pct is not None and float(document_ready_pct) >= ready_threshold
    if c_ok and d_ok:
        cell = "stable"
        title_ru = "Всё в порядке"
        action_ru = "Выборочный контроль."
    elif c_ok and not d_ok:
        cell = "integration"
        title_ru = "Проблема у программистов, а не у врачей"
        action_ru = "Задача информационной службе: справочники и закрытие визитов."
    elif (not c_ok) and d_ok:
        cell = "form_over_content"
        title_ru = "Отчётность красивая, лечение хромает"
        action_ru = "Разбор случаев и обучение по повторяющимся ошибкам."
    else:
        cell = "both"
        title_ru = "Проблемы и там, и там"
        action_ru = "Наставник + настройка программы; нужна внешняя помощь."
    return {
        "cell": cell,
        "title_ru": title_ru,
        "action_ru": action_ru,
        "clinical_high": c_ok,
        "document_high": d_ok,
        "clinical_pct": clinical_pct,
        "document_ready_pct": document_ready_pct,
    }
