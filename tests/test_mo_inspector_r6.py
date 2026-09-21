"""R6: №55 evidence is pack items and p.13 words, not a hero percentage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")


def test_reg55_section_leads_with_band_not_percent() -> None:
    chunk = APP.split("function renderReg55(", 1)[1][:7000]
    assert "<h3>№55, раздел V</h3>" in chunk
    assert "Градация п.13 словами" in chunk
    assert "№127 - опора, не второй балл" in chunk
    assert "font-size:1.35rem" not in chunk
    assert "Средний балл (п.12)" not in chunk
    assert "reg55-fail-list" in chunk
    assert "Все пункты pack, сначала невыполненные" in chunk
    assert "<th>Цитата</th>" in chunk


def test_why_includes_reg55_fail_when_noncompliant() -> None:
    chunk = APP.split("function renderCaseWhy(", 1)[1][:2500]
    assert 'regBand === "noncompliant"' in chunk
    assert "№55: не выполнен пункт" in chunk
    assert "zone2b" in chunk


def test_accordion_opens_reg55_when_noncompliant() -> None:
    chunk = APP.split("function pickOpenEvidenceId(", 1)[1][:900]
    assert 'if (regBand === "noncompliant") return "evidence-reg55"' in chunk
    assert 'if (z2b === "bad") return "evidence-kp-plan"' in chunk


def test_zones_hero_has_no_reg55_percent() -> None:
    chunk = APP.split("function renderZonesHero(", 1)[1].split("function assessmentStatusLabel", 1)[0]
    assert "reg55_section_pct" not in chunk
    assert "№55" not in chunk


def test_reg55_css() -> None:
    assert ".reg55-fail-list" in CSS
    assert ".reg55-checklist-table" in CSS
