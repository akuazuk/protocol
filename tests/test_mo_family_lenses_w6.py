"""W6: family dashboards share the cohort query; doctors open by grade."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend/web/methodist/mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")


def test_family_dashboard_uses_cohort_query() -> None:
    assert "function familyCohortQuery(extra)" in APP
    assert "familyCohortQuery({ family: family })" in APP
    assert 'familyCohortQuery({ finding_family: family })' in APP
    assert 'request("/drugs-labs-kpis?" + familyCohortQuery().toString())' in APP
    assert 'kpiQuery.delete("statuses")' not in APP


def test_doctors_open_by_grade_not_only_zone() -> None:
    assert 'id="doctor-grade-filter"' in HTML
    assert 'data-doctor-grade="poor"' in HTML
    assert 'data-doctor-grade="good"' in HTML
    assert 'data-doctor-grade="all"' in HTML
    assert "function doctorOpenFilters()" in APP
    assert 'overallGrade: "poor|important|critical"' in APP
    assert 'overallGrade: "good"' in APP


if __name__ == "__main__":
    test_family_dashboard_uses_cohort_query()
    test_doctors_open_by_grade_not_only_zone()
    print("ok")
