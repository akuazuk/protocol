"""R5: medication evidence shows Rceth/KP, not a false draft badge."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")


def test_meds_table_has_kp_rceth_risk_columns() -> None:
    chunk = APP.split("function renderMedicationNormativeCards", 1)[1][:4500]
    assert "<th>Препарат</th><th>Доза</th><th>КП</th><th>Rceth</th><th>Риск</th>" in chunk
    assert "по инструкции реестра" in chunk
    assert "инструкции в скачанном реестре нет - проверка по КП и DDI" in chunk
    assert "DDI и high-alert - полоса риска" in chunk
    assert "badge--shadow" not in chunk
    assert 'if (status === "in_scheme") return "в схеме"' in APP
    assert "нет в схеме КП" in APP
    assert "в КП нет препаратов" in APP


def test_false_draft_badge_is_conditional() -> None:
    chunk = APP.split("function renderMedicationNormativeCards", 1)[1][:4500]
    assert "card.draft === true" in chunk
    assert "по схеме КП" in chunk
    assert chunk.find("черновик") > chunk.find("card.draft === true")


def test_ddi_stays_risk_not_plan_zone() -> None:
    chunk = APP.split("function medicationRiskFindings", 1)[1][:1800]
    assert 'code !== "C_ddi"' in chunk
    assert 'code !== "C_high_alert_no_dose"' in chunk
    assert "zone2b" not in chunk
    assert "high-alert" in APP.split("function renderMedicationNormativeCards", 1)[1][:4500]


def test_meds_table_css() -> None:
    assert ".med-normative-table" in CSS
    assert ".med-normative-risk" in CSS
