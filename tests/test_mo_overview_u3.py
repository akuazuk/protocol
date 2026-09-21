"""U3: Overview chart clicks use the same Find MO chips, not a second filter set."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")


def test_ring_click_still_opens_find_mo() -> None:
    assert "function openZoneBandCases" in APP
    chunk = APP.split("function openZoneBandCases", 1)[1][:400]
    assert 'page: "documents"' in chunk
    assert "zoneBandFilter" in chunk
    assert "openZoneBandCases(meta.key, band)" in APP


def test_trend_click_opens_find_mo_with_day_and_zone() -> None:
    assert "function openTrendDayCases" in APP
    chunk = APP.split("function openTrendDayCases", 1)[1][:500]
    assert 'page: "documents"' in chunk
    assert 'period: "custom"' in chunk
    assert "dateFrom: day" in chunk
    assert "zoneFilter: zoneKeyFromSeriesName(seriesName)" in chunk
    assert 'zoneBandFilter: ""' in chunk
    assert "openTrendDayCases(dates[params.dataIndex], params.seriesName)" in APP
    dynamics = APP.split("function renderScoreDynamics", 1)[1].split(
        "function renderYesterdayScoreDashboard", 1
    )[0]
    assert "filtersChanged();" not in dynamics
    assert "клик открывает Найти МО" in APP
