"""R3: episode history is a timeline, not three counts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")


def test_history_compact_has_timeline_and_statuses() -> None:
    assert "function renderHistoryTimeline(" in APP
    assert "function historyCorrectionStatus(" in APP
    assert "первый контакт - коррекции плана не оцениваются" in APP
    assert "коррекция оценивается" in APP
    assert "нет сравнимого плана" in APP
    assert "другой эпизод" in APP
    assert "Предыдущий визит эпизода:" in APP
    assert "class=\"history-timeline\"" in APP


def test_first_contact_is_visible_without_details() -> None:
    chunk = APP.split("function renderHistoryCompact", 1)[1].split("function renderLabReconcile", 1)[0]
    assert "Первый контакт с этим врачом" in chunk
    assert chunk.find("first_contact") < chunk.find("<details open>")
    assert "patient_id" not in chunk
    assert "К этому врачу:" not in chunk


def test_history_timeline_css() -> None:
    assert ".history-timeline" in CSS
    assert ".history-timeline-now" in CSS
