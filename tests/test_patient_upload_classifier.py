"""B2C: определение не того документа."""
from __future__ import annotations

from clinical_knowledge.patient_upload_classifier import (
    build_upload_joke_report,
    check_consult_document,
    check_patient_uploads,
    classify_kz_upload,
    classify_lab_upload,
    is_b2c_lab_filename,
)

RECIPE = """
Рецепт борща. Ингредиенты: свёкла 300 г, капуста 200 г, картофель.
Нарезать овощи, варить в кастрюле 40 минут. Подавать со сметаной.
"""

KZ = """
Консультативное заключение
Врач: кардиолог Иванов И.И.
Жалобы: боли в груди при нагрузке.
Диагноз: I20.8 Стенокардия.
Рекомендации по лечению: нитроглицерин по потребности.
Контрольная явка через 1 месяц.
"""

LAB = """
ИНВИТРО
Биохимический анализ крови
Глюкоза 5.2 ммоль/л
Креатинин 78 мкмоль/л
АСТ 22 Ед/л
АЛТ 19 Ед/л
"""


def test_classify_recipe_as_not_kz() -> None:
    g = classify_kz_upload(RECIPE)
    assert not g.is_expected
    assert g.kind == "recipe"


def test_classify_real_kz() -> None:
    g = classify_kz_upload(KZ)
    assert g.is_expected
    assert g.kind == "kz"


def test_lab_in_kz_slot() -> None:
    g = classify_kz_upload(LAB)
    assert not g.is_expected
    assert g.kind == "lab_in_kz"


def test_kz_in_lab_slot() -> None:
    g = classify_lab_upload(KZ)
    assert not g.is_expected
    assert g.kind == "kz_in_lab"


def test_lab_upload_ok() -> None:
    g = classify_lab_upload(LAB)
    assert g.is_expected


def test_joke_report_shape() -> None:
    g = classify_kz_upload(RECIPE)
    rep = build_upload_joke_report(g)
    assert rep["upload_mismatch"] is True
    assert rep["upload_joke"]["emoji"]
    assert rep["upload_joke"]["title_ru"]
    assert rep["headline_ru"]
    assert len(rep.get("questions_structured") or []) >= 3
    assert len(rep.get("action_checklist") or []) >= 3
    assert all("?" in q.get("text", "") for q in rep["questions_structured"])


def test_joke_questions_diverse_by_kind() -> None:
    from clinical_knowledge.patient_upload_classifier import _pick_joke_doctor_questions

    recipe_guess = classify_kz_upload(RECIPE)
    lab_guess = classify_kz_upload(LAB)
    rq = _pick_joke_doctor_questions(recipe_guess, limit=4)
    lq = _pick_joke_doctor_questions(lab_guess, limit=4)
    assert len(rq) == 4
    assert len(lq) == 4
    assert {q["text"] for q in rq} != {q["text"] for q in lq}
    again = _pick_joke_doctor_questions(recipe_guess, limit=4)
    assert [q["text"] for q in rq] == [q["text"] for q in again]


PROTOCOL_PDF = """
КЛИНИЧЕСКИЙ ПРОТОКОЛ
диагностики и лечения пациентов с заболеваниями кожи
УТВЕРЖДЕН
приказом Министерства здравоохранения Республики Беларусь
1. Общие положения
Настоящий клинический протокол устанавливает порядок диагностики и лечения.
2. Диагностика
Рекомендуется осмотр и назначение обследований по показаниям.
"""


def test_classify_minzdrav_protocol_pdf_as_not_kz() -> None:
    g = classify_kz_upload(PROTOCOL_PDF)
    assert not g.is_expected
    assert g.kind == "protocol_pdf"


def test_joke_report_has_no_protocol_match() -> None:
    g = classify_kz_upload(RECIPE)
    rep = build_upload_joke_report(g)
    assert rep["matched_protocols_count"] == 0
    assert not rep.get("protocol_links")
    assert not rep.get("protocol_context")


def test_check_patient_uploads_priority_kz() -> None:
    mismatch = check_patient_uploads(kz_text=RECIPE, lab_text=LAB)
    assert mismatch is not None
    assert mismatch.slot == "kz"


def test_b2c_lab_filename_prefix_case_insensitive() -> None:
    assert is_b2c_lab_filename("a_1")
    assert is_b2c_lab_filename("A_2.pdf")
    assert is_b2c_lab_filename("clients_consult/a_3.pdf")
    assert is_b2c_lab_filename("А_1")
    assert is_b2c_lab_filename("а_2.pdf")
    assert not is_b2c_lab_filename("report_n_1")
    assert not is_b2c_lab_filename("gastro_1")


def test_consult_document_rejects_a_prefix_without_scoring() -> None:
    from clinical_knowledge.consult_tiering import run_consult_by_tier

    out = run_consult_by_tier(
        tier="L1",
        text=LAB,
        consultation_id="A_2",
    )
    assert out.get("upload_mismatch") is True
    assert out.get("overall_score") is None
    assert out.get("wrong_document_kind") == "lab_in_kz"
    assert out.get("review", {}).get("upload_joke")


def test_check_consult_by_filename() -> None:
    guess = check_consult_document(KZ, consultation_id="a_99")
    assert guess is not None
    assert guess.kind == "lab_in_kz"


MED_EXAM = """
МЕДИЦИНСКИЙ ОСМОТР ЛОР-врач первичный
Дата и время проведения медицинского осмотра: 01.06.2026 10:30
Ф.И.О: Иванов Иван Иванович, 02.12.1992.
Жалобы пациента: На боль в левом ухе
Анамнез заболевания: Болеет около 4 суток
Данные результатов медицинского осмотра: слизистая розовая.
Диагноз: H60. Наружный отит.
Рекомендации: капли в ухо 7 дней.
"""


def test_medical_exam_accepted_as_clinical_visit() -> None:
    g = classify_kz_upload(MED_EXAM, filename="osmotr.pdf")
    assert g.is_expected is True
    assert g.kind == "kz"
    assert "осмотр" in g.label_ru.lower() or "заключен" in g.label_ru.lower()
    assert check_consult_document(MED_EXAM, filename="osmotr.pdf") is None


def test_consult_title_without_kz_word_accepted() -> None:
    text = (
        "Консультация терапевта\n"
        "Дата: 10.01.2025\n"
        "Жалобы: кашель 3 дня.\n"
        "Анамнез: ОРВИ в анамнезе.\n"
        "Диагноз: J06.9.\n"
        "Рекомендации: покой, обильное питьё.\n"
    )
    g = classify_kz_upload(text)
    assert g.is_expected is True
    assert check_consult_document(text) is None
