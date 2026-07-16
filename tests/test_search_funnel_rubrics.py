"""Рубрики воронки: без ложных инфекций от «без лихорадки»."""
from __future__ import annotations

from clinical_knowledge.search_funnel import _infer_rubric_choices


def test_hip_limp_rubrics_not_infectious() -> None:
    q = (
        "боль в бедре больше месяца у ребенка 9 лет и хромота "
        "хромота нарушение походки без лихорадки без травмы "
        "тазобедренный сустав ТБС"
    )
    ids = [c["id"] for c in _infer_rubric_choices(q, ["M91.1"])]
    assert ids[0] == "travmatologiya-ortopediya"
    assert "revmatologiya" in ids
    assert "infektsionnye-zabolevaniya" not in ids
    assert "pulmonologiya-ftiziatriya" not in ids
    assert "pediatriya" not in ids


def test_fever_cough_still_maps_respiratory() -> None:
    ids = [c["id"] for c in _infer_rubric_choices("температура 39 и кашель", [])]
    assert "pulmonologiya-ftiziatriya" in ids or "infektsionnye-zabolevaniya" in ids
