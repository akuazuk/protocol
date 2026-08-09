"""Visual refresh: muted charts + Today fallback to last data day."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")
TOKENS = (ROOT / "frontend/web/shared/mo-tokens.css").read_text(encoding="utf-8")
CSS = (ROOT / "frontend/web/shared/mo-ui.css").read_text(encoding="utf-8")
CHARTS = (ROOT / "frontend/web/shared/mo-charts.js").read_text(encoding="utf-8")


def test_zone_trend_uses_echarts_not_table() -> None:
    assert "function renderZoneTrendHost" in APP
    assert 'type: "line"' in APP
    assert "zone-trend-chart" in APP
    assert "Оформление" in APP


def test_today_falls_back_to_data_through() -> None:
    assert "function resolveTodayWorkingDay" in APP
    assert "Показан последний день с данными" in APP


def test_palette_is_muted() -> None:
    assert "--chart-1:" in TOKENS
    assert "#7c3aed" not in TOKENS  # old bright purple removed
    assert "#be123c" not in TOKENS  # old bright crimson removed from light tokens
    assert 'token("--chart-1"' in CHARTS
    assert "attention-tile--zone1" in CSS
