"""Wave 1: unused lab findings + canons (≥15 panels)."""
from __future__ import annotations

import os

from clinical_knowledge.lab_canons import lab_panels
from clinical_knowledge.lab_unused_findings import (
    CODE_UNUSED_DX,
    CODE_UNUSED_PLAN,
    lab_unused_primary_enabled,
    unused_lab_findings,
)
from clinical_knowledge.mo_finding_labels_ru import FINDING_TITLE_RU


def test_lab_canons_have_at_least_15_panels() -> None:
    panels = lab_panels()
    assert len(panels) >= 15
    ids = {p["id"] for p in panels}
    assert {"oak", "glucose", "crp", "alt", "creatinine", "tsh"} <= ids


def test_unused_in_dx_and_plan_from_exam_data() -> None:
    case = {
        "exam_data": "ОАК: Hb 132 г/л. Глюкоза 5.4 ммоль/л.",
        "clinical_diagnosis": "Остеохондроз поясничного отдела",
        "treatment_recommendations": "НПВП курсом 5 дней",
        "exam_recommendations": "",
    }
    findings = unused_lab_findings(case)
    codes = {f["code"] for f in findings}
    assert CODE_UNUSED_DX in codes
    assert CODE_UNUSED_PLAN in codes
    assert all(f.get("shadow") and f.get("is_shadow") for f in findings)
    assert lab_unused_primary_enabled() is False


def test_unused_not_fired_when_dx_and_plan_mention() -> None:
    case = {
        "exam_data": "Глюкоза 7.8 ммоль/л",
        "clinical_diagnosis": "Сахарный диабет 2 типа, глюкоза повышена",
        "treatment_recommendations": "Контроль глюкозы, метформин",
    }
    findings = unused_lab_findings(case)
    assert findings == []


def test_unused_primary_flag_promotes(monkeypatch) -> None:
    monkeypatch.setenv("MO_LAB_UNUSED_PRIMARY", "1")
    case = {
        "exam_data": "СРБ 24 мг/л",
        "clinical_diagnosis": "ОРВИ",
        "treatment_recommendations": "симптоматически",
    }
    findings = unused_lab_findings(case)
    assert findings
    assert all(f.get("shadow") is False for f in findings)
    monkeypatch.delenv("MO_LAB_UNUSED_PRIMARY", raising=False)


def test_unused_labels_ru_present() -> None:
    assert "Готовый анализ" in FINDING_TITLE_RU[CODE_UNUSED_DX]
    assert "плане" in FINDING_TITLE_RU[CODE_UNUSED_PLAN].lower()
