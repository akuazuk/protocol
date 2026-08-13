"""D6: hidden legacy pages removed; nav stays 7 + settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend/web/methodist/mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")


def test_nav_only_seven_visible_and_settings_hidden() -> None:
    nav = HTML.split('id="app-nav"')[1].split("</ul>")[0]
    visible = [l for l in nav.splitlines() if "nav-button" in l and "<li hidden>" not in l]
    assert len(visible) == 7
    assert 'data-page="settings"' in nav
    assert "<li hidden>" in nav


def test_legacy_pages_removed_from_dom() -> None:
    for pid in (
        "page-specialties",
        "page-diagnoses",
        "page-safety",
        "page-doctor-cabinet",
        "page-access-log",
        "page-data-quality",
    ):
        assert f'id="{pid}"' not in HTML
    assert 'id="page-settings"' in HTML
    assert 'id="access-log-content"' in HTML
    assert "REMOVED_PAGES" in APP
    assert 'specialties: "Специальности"' not in APP


if __name__ == "__main__":
    test_nav_only_seven_visible_and_settings_hidden()
    test_legacy_pages_removed_from_dom()
    print("ok")
