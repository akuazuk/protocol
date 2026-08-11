"""Очередь разбора: только точные сигналы оценки."""
from __future__ import annotations

from clinical_knowledge.mo_action_queue_select import (
    finding_eligible_for_action_queue,
    pick_primary_queue_finding,
    queue_reason_ru,
    signal_band_for_finding,
    strip_pn_tokens,
)


def test_reg55_findings_never_enter_queue() -> None:
    assert not finding_eligible_for_action_queue(
        {"finding_code": "D_reg55_gap", "severity": "P1", "finding_title": "№55"}
    )
    assert not finding_eligible_for_action_queue(
        {"finding_code": "D_reg55_p0", "severity": "P0", "finding_title": "критический"}
    )


def test_icd_and_missing_blocks_excluded() -> None:
    assert not finding_eligible_for_action_queue(
        {"finding_code": "B_icd_invalid", "severity": "P2"}
    )
    assert not finding_eligible_for_action_queue(
        {"finding_code": "A_missing_diagnosis", "severity": "P2"}
    )
    assert not finding_eligible_for_action_queue(
        {"finding_code": "E_template_copy", "severity": "P1"}
    )


def test_red_flag_and_dx_support_included() -> None:
    assert signal_band_for_finding({"finding_code": "C_red_flag", "severity": "P1"}) == "critical"
    assert (
        signal_band_for_finding({"finding_code": "B_dx_no_support", "severity": "P1"})
        == "important"
    )


def test_ddi_major_in_queue_moderate_out() -> None:
    major = {
        "finding_code": "C_ddi",
        "severity": "P1",
        "finding_title": "Лекарственное взаимодействие (Major): escitalopram + sumatriptan",
    }
    moderate = {
        "finding_code": "C_ddi",
        "severity": "P2",
        "finding_title": "Лекарственное взаимодействие (Moderate): a + b",
    }
    assert signal_band_for_finding(major) == "critical"
    assert finding_eligible_for_action_queue(major)
    assert signal_band_for_finding(moderate) is None


def test_topical_major_ddi_not_queue_critical() -> None:
    """Ксарелто + диклофенак гель: не полоса Критично (кейс 3665385)."""
    topical = {
        "finding_code": "C_ddi",
        "severity": "P1",
        "finding_title": "Лекарственное взаимодействие (Major): ксарелто / rivaroxaban + диклофенак / diclofenac",
        "evidence": (
            "ксарелто / rivaroxaban + диклофенак / diclofenac. Фрагмент плана: "
            "Ксарелто 20 мг. Местно: диклофенак гель."
        ),
    }
    assert signal_band_for_finding(topical) is None
    assert not finding_eligible_for_action_queue(topical)

    marked = {
        "finding_code": "C_ddi",
        "severity": "P2",
        "topical_ddi": True,
        "finding_title": "Лекарственное взаимодействие (Major, топический путь - понижено): a + b",
    }
    assert signal_band_for_finding(marked) is None


def test_pick_primary_prefers_critical_over_important() -> None:
    picked = pick_primary_queue_finding(
        [
            {"finding_code": "B_dx_no_support", "severity": "P1"},
            {
                "finding_code": "C_ddi",
                "severity": "P1",
                "finding_title": "Major: a + b",
            },
        ]
    )
    assert picked is not None
    assert picked["finding_code"] == "C_ddi"
    assert picked["_queue_band"] == "critical"


def test_reason_and_strip_have_no_pn_tokens() -> None:
    reason = queue_reason_ru(
        band="important",
        finding_title="Важно: взаимодействие P1 не P0",
        finding_code="C_ddi",
    )
    assert "P0" not in reason
    assert "P1" not in reason
    assert "P0" not in strip_pn_tokens("не P0 · P2")
    assert "P2" not in strip_pn_tokens("не P0 · P2")
