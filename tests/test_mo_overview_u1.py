"""U1: Overview rings share the evaluation legend and drop №55 as a fourth score."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")
HTML = (ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html").read_text(
    encoding="utf-8"
)


def test_overview_legend_matches_grade_chips() -> None:
    assert 'aria-label="Шкала оценки"' in APP
    for label in ("Хорошо", "С замечанием", "Слабо", "Важно", "Критично", "Нет оценки"):
        assert label in APP
    assert "score-grade-legend" in CSS


def test_plan_kpi_reads_zone_bands_from_overview_dash() -> None:
    assert "planKpiMeta(a, (opts.zones || {}).zone2b)" in APP
    assert "bands.na && bands.na.n" in APP


def test_trend_has_three_zones_without_reg55_series() -> None:
    assert 'data: ["Оформление", "Диагноз", "План"]' in APP
    assert 'series("№55", "reg55_avg"' not in APP
    assert "Средние % трёх зон по дням выбранного периода" in APP
    assert "Средние % трёх зон по дням выбранного периода" in HTML


def test_rings_are_three_columns_not_four() -> None:
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in CSS
    assert "grid-template-columns: repeat(4, minmax(0, 1fr))" not in CSS
