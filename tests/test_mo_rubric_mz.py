"""Shadow-рубрика МЗ «Как оценивать»."""
from __future__ import annotations

from clinical_knowledge.mo_rubric_mz import evaluate_mo_rubric_mz, load_rubric_config


def test_rubric_config_has_thirteen_criteria() -> None:
    cfg = load_rubric_config()
    ids = [c["id"] for c in cfg["criteria"]]
    assert len(ids) == 13
    assert "complaints" in ids
    assert "exam_correction" in ids
    assert cfg["primary"] is False


def test_rich_case_scores_high_and_dynamics_na_without_prior() -> None:
    result = evaluate_mo_rubric_mz(
        clinical={
            "complaints": "Боль в горле локально справа, характер ноющий, длительность 3 дня, интенсивность умеренная",
            "anamnesis_doctor": (
                "Болеет третий день. Температура до 37.5. Аллергологический анамнез не отягощён. "
                "Курение отрицает. Наследственность не отягощена. Ранее ангины ежегодно."
            ),
            "objective_status": (
                "Состояние удовлетворительное. Кожные покровы чистые. Дыхание везикулярное. "
                "ЧСС 72. А/Д 120/80. Живот мягкий. Локальный статус: гиперемия миндалин."
            ),
            "clinical_diagnosis": "Острый тонзиллит J03.9",
            "exam_recommendations": "ОАК, мазок из зева",
            "treatment_recommendations": "Полоскание, контроль через 5 дней, явка к терапевту",
            "exam_data": "ОАК от 01.08: лейкоциты 9.1",
        },
        meta={"visit_date": "2026-08-02", "visit_time": "10:30", "diagnosis_code": "J03.9"},
        block_scores={"exams": 70, "treatment": 65},
    )
    assert result["ok"] is True
    assert result["primary"] is False
    assert result["rubric_pct"] is not None
    assert result["rubric_pct"] >= 70
    by_id = {c["id"]: c for c in result["criteria"]}
    assert by_id["datetime"]["score"] == 1.0
    assert by_id["diagnosis"]["score"] == 1.0
    assert by_id["exam_correction"]["score"] is None
    assert by_id["treatment_correction"]["score"] is None
    assert by_id["exam_correction"]["score_label"] == "n/a"


def test_empty_case_scores_low() -> None:
    result = evaluate_mo_rubric_mz(clinical={}, meta={})
    by_id = {c["id"]: c for c in result["criteria"]}
    assert by_id["complaints"]["score"] == 0.0
    assert by_id["mo_complete"]["score"] == 0.0
    assert by_id["exam_data"]["score"] is None


def test_dynamics_detects_plan_change() -> None:
    result = evaluate_mo_rubric_mz(
        clinical={"exam_recommendations": "МРТ вместо УЗИ", "treatment_recommendations": "Смена терапии"},
        meta={"visit_date": "2026-08-02"},
        prior_clinical={
            "exam_recommendations": "УЗИ",
            "treatment_recommendations": "Прежняя терапия",
        },
    )
    by_id = {c["id"]: c for c in result["criteria"]}
    assert by_id["exam_correction"]["score"] == 1.0
    assert by_id["treatment_correction"]["score"] == 1.0


def test_follow_up_interval_scores_full() -> None:
    result = evaluate_mo_rubric_mz(
        clinical={"treatment_recommendations": "Явка через 7 дней на контроль"},
        meta={"visit_date": "2026-08-02"},
    )
    by_id = {c["id"]: c for c in result["criteria"]}
    assert by_id["follow_up"]["score"] == 1.0


def test_summarize_rubric_batch_ranks_failures() -> None:
    from clinical_knowledge.mo_rubric_mz import summarize_rubric_batch

    a = evaluate_mo_rubric_mz(clinical={}, meta={})
    b = evaluate_mo_rubric_mz(
        clinical={"complaints": "Боль в горле локально, длительность 2 дня, интенсивность сильная"},
        meta={"visit_date": "2026-08-01", "visit_time": "09:00", "diagnosis_code": "J03.9"},
    )
    summary = summarize_rubric_batch([a, b])
    assert summary["cases_n"] == 2
    assert summary["top_fail"]
    assert summary["top_fail"][0]["fail_pct"] >= summary["top_fail"][-1]["fail_pct"]
    with_spec = summarize_rubric_batch([a, b], specialties=["Терапевт", "Хирург"])
    assert with_spec["by_specialty"]
