"""W3: overview grain uses the same window for tiles, rings and queue."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")


def test_overview_tiles_read_score_dashboard_window() -> None:
    assert 'renderAttentionStrip("yesterday-attention", (dash && dash.attention)' in APP
    assert 'kpi("Критично в очереди"' in APP
    assert '"то же окно, не рабочий день"' in APP
    assert 'source, "рабочий день"' not in APP
    assert "Окно фильтров:" in APP
    assert "Показан последний день с данными" in APP


def test_rings_use_scale_words_and_dynamics_show_n() -> None:
    assert 'assessedN > 0 ? (zoneLabels[dominant.band]' in APP
    assert "if (row.n_evaluated != null) lines.push" in APP
    assert "connectNulls: false" in APP
    assert "Все цифры этого экрана:" in APP
