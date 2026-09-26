"""Smoke: меню 11 видимых пунктов без «Ещё»; legacy charts не на hero Обзора."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend/web/methodist/mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")


def _nav_block() -> str:
    return HTML.split('id="app-nav"')[1].split("</ul>")[0]


def test_left_menu_has_eleven_visible_pages() -> None:
    nav = _nav_block()
    visible = [line for line in nav.splitlines() if 'class="nav-button"' in line]
    # 7 рабочих + 4 служебных, включая Справку; ничего не спрятано за «Ещё».
    assert len(visible) == 11
    for page in ("yesterday", "queue", "documents", "mis", "doctors", "medications", "labs", "reports", "kp-sync", "rceth-sync", "settings"):
        assert f'data-page="{page}"' in nav
    assert 'data-page="overview"' not in nav
    assert 'id="nav-more"' not in nav
    assert "Безопасность" not in nav
    assert "Специальности" not in nav


def test_period_hero_keeps_zones_not_heatmap() -> None:
    overview = HTML.split('id="page-overview"')[1].split('id="page-yesterday"')[0]
    assert "month-zone-trend" in overview
    assert "month-look-where" in overview
    assert "month-attention" in overview
    assert 'id="month-heatmap-chart" hidden' in overview
    assert 'id="month-pareto-chart" hidden' in overview
    assert 'id="month-funnel-chart" hidden' in overview
    assert 'Подробнее: №55' in overview
    assert 'id="month-reg55"' in overview
    assert 'id="month-rubric-mz"' in overview


def test_today_hero_has_table_and_score_rings() -> None:
    today = HTML.split('id="page-yesterday"')[1].split('id="page-queue"')[0]
    assert "yesterday-action-rows" in today
    assert "yesterday-score-rings" in today
    assert "yesterday-score-dynamics" in today
    assert "yesterday-score-kpis" in today
    assert "yesterday-zone-trend" in today
    assert 'id="yesterday-index-cards" hidden' in today
    assert "hostActive" in APP
    assert "renderScoreRings" in APP
    assert "renderScoreDynamics" in APP
    assert "/score-dashboard?" in APP
    assert "reg55KpiHtml" not in APP
    assert "data-look-doctor" in APP


if __name__ == "__main__":
    test_left_menu_has_eleven_visible_pages()
    test_period_hero_keeps_zones_not_heatmap()
    test_today_hero_has_table_and_score_rings()
    print("ok")
