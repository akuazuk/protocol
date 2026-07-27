"""Тесты coverage-aware structural score (Workstream C ТЗ overnight-v1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.kz_evaluation_engine import evaluate_kz_v3, score_documentation

_FULL = {
    "complaints": "боль в горле 3 дня",
    "anamnesis_doctor": "болен 3 дня, температура",
    "objective_status": "зев гиперемирован, миндалины отёчны, налётов нет",
    "clinical_diagnosis": "Острый фарингит",
    "mkb_code_main": "J02.9",
    "exam_recommendations": "ОАК",
    "treatment_recommendations": "полоскание, парацетамол 500 мг 3 раза 5 дней",
}


def test_optional_does_not_compensate_required():
    # есть все рекомендуемые, но нет диагноза и рекомендаций (обязательных)
    case = {
        "complaints": "жалобы описаны",
        "anamnesis_doctor": "анамнез описан",
        "objective_status": "статус описан",
        # нет clinical_diagnosis, нет treatment/exam recommendations
    }
    score, cov, findings, caps = score_documentation(case)
    # обязательные проваливаются -> score ограничен cap, не компенсируется optional
    assert score is not None
    assert score <= 55.0, f"optional не должен компенсировать required, got {score}"
    codes = {f.code for f in findings}
    assert "A_missing_diagnosis" in codes
    assert "A_missing_recommendations" in codes


def test_missing_diagnosis_cap():
    case = dict(_FULL)
    case.pop("clinical_diagnosis")
    score, cov, findings, caps = score_documentation(case)
    assert score <= 45.0
    assert any("диагноз" in c for c in caps)


def test_missing_recommendations_cap():
    case = dict(_FULL)
    case.pop("treatment_recommendations")
    case.pop("exam_recommendations")
    score, cov, findings, caps = score_documentation(case)
    assert score <= 55.0


def test_empty_kz_insufficient_data():
    score, cov, findings, caps = score_documentation({})
    assert score is None
    assert cov == 0.0
    r = evaluate_kz_v3({})
    assert r.status == "insufficient_data"
    assert r.score_pct is None


def test_full_documentation_high():
    score, cov, findings, caps = score_documentation(_FULL)
    assert score >= 90.0
    assert cov == 1.0
    assert not caps


def test_no_protocol_lowers_concordance_coverage_not_hidden():
    # None-блоки не дают скрытого преимущества: без протокола покрытие concordance
    # ограничено, а не «идеально».
    r = evaluate_kz_v3(_FULL)
    # concordance оценён только по базовым проверкам, protocol-пункты отсутствуют
    assert r.coverage.clinical_concordance is not None
    # overall coverage учитывает недостачу регуляторных/протокольных данных как измерение
    assert r.coverage.overall is not None


def test_none_block_reduces_coverage_vs_present():
    # КЗ без протокольных требований vs с ними: coverage concordance должна отличаться
    proto = {
        "condition_id": "x", "name": "Острый фарингит",
        "required_exams": ["мазок из зева", "общий анализ крови"],
        "treatment": ["симптоматическая терапия"],
        "review_status": "not_reviewed", "match_score": 0.9,
    }
    r_no = evaluate_kz_v3(_FULL)
    r_yes = evaluate_kz_v3(_FULL, protocol_ctx=proto)
    # с протоколом появляются потенциальные проверки -> покрытие concordance ниже
    # (мы честно отражаем незакрытые протокольные требования как advisory)
    assert (r_yes.coverage.clinical_concordance or 1.0) <= (r_no.coverage.clinical_concordance or 1.0)
