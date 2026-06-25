"""Фильтры респираторных путей PDF."""
from __future__ import annotations

from clinical_knowledge.respiratory_path_filters import (
    filter_paths_for_respiratory_context,
    is_urti_icd_only,
    path_has_respiratory_wrong_markers,
    urti_path_rank,
)


def test_is_urti_icd_only():
    assert is_urti_icd_only(["J06.9"])
    assert not is_urti_icd_only(["J45.9"])


def test_wrong_path_markers():
    p = "minzdrav_protocols/pulmonologiya-ftiziatriya/КП_диагностика_лечение_саркоидозом.pdf"
    assert path_has_respiratory_wrong_markers(p)


def test_filter_prefers_bronchitis():
    paths = [
        "minzdrav_protocols/pulmonologiya-ftiziatriya/КП_диагностика_лечение_саркоидозом.pdf",
        "minzdrav_protocols/pulmonologiya-ftiziatriya/КП диагностики и лечения острого и хронического бронхита.pdf",
    ]
    out = filter_paths_for_respiratory_context(paths, limit=4)
    assert len(out) == 1
    assert "бронхит" in out[0].lower()
    assert urti_path_rank(out[0]) > 0
