"""B2C: определение не того документа."""
from __future__ import annotations

from clinical_knowledge.patient_upload_classifier import (
    build_upload_joke_report,
    check_patient_uploads,
    classify_kz_upload,
    classify_lab_upload,
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


def test_check_patient_uploads_priority_kz() -> None:
    mismatch = check_patient_uploads(kz_text=RECIPE, lab_text=LAB)
    assert mismatch is not None
    assert mismatch.slot == "kz"
