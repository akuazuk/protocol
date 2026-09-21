"""R4: lab evidence shows timeline, reconcile-to-plan, and honest missing reference."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")


def test_lab_reconcile_links_to_plan() -> None:
    assert "Связано с диагнозом / планом" in APP
    assert "есть на складе, в тексте МО не названы" in APP


def test_lab_table_has_honest_reference() -> None:
    assert "<th>Референс</th>" in APP
    assert "референса в складе нет" in APP
    assert "MO_LAB_IN_PRIMARY=1" not in APP
