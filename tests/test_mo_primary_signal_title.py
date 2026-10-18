"""Заголовок главного замечания: список и «Почему так» читают одно правило."""
from __future__ import annotations

from clinical_knowledge.mo_primary_signal_title import (
    STALE_TITLE_RU,
    apply_primary_signal_titles,
    primary_signal_title,
)
from clinical_knowledge.mo_zone_scores import zones_api_payload


def _finding(code: str, title: str, severity: str = "P1", **extra: object) -> dict:
    row = {
        "finding_code": code,
        "title_ru": title,
        "severity": severity,
        "passed": 0,
        "is_shadow": 0,
    }
    row.update(extra)
    return row


def test_zone2a_keeps_diagnosis_sentence() -> None:
    signal = primary_signal_title(
        attention_primary="zone2a",
        attention_reason_ru="Диагноз и/или МКБ отсутствуют",
        findings=[_finding("C_nsaid_dup", "Два НПВП: Альфацин и Бетапрол")],
    )
    assert signal["title_ru"] == "Диагноз и/или МКБ отсутствуют"
    assert signal["stale"] is False
    assert signal["code"] == ""


def test_major_ddi_outranks_nsaid_dup() -> None:
    signal = primary_signal_title(
        attention_primary="safety",
        attention_reason_ru="Есть важный сигнал риска",
        findings=[
            _finding("C_nsaid_dup", "Два НПВП: Альфацин и Бетапрол"),
            _finding(
                "C_ddi",
                "Лекарственное взаимодействие (Major): Альфацин + Бетапрол",
            ),
        ],
    )
    assert signal["code"] == "C_ddi"
    assert "Альфацин" in signal["title_ru"]
    assert "Бетапрол" in signal["title_ru"]
    assert signal["title_ru"] != "Есть важный сигнал риска"
    assert signal["severity_label_ru"] == "Важно"
    assert signal["layer_ru"] == "Риск"


def test_nsaid_title_when_it_is_the_only_live_signal() -> None:
    signal = primary_signal_title(
        attention_primary="safety",
        attention_reason_ru="Есть важный сигнал риска",
        findings=[_finding("C_nsaid_dup", "Два НПВП: Альфацин и Бетапрол")],
    )
    assert signal["code"] == "C_nsaid_dup"
    assert signal["title_ru"].startswith("Два НПВП")


def test_stale_when_safety_has_no_open_c_signal() -> None:
    signal = primary_signal_title(
        attention_primary="safety",
        attention_reason_ru="Есть важный сигнал риска",
        findings=[
            _finding("C_ddi", "Умеренное взаимодействие", severity="P2"),
            _finding("C_nsaid_dup", "Закрыто", passed=1),
            _finding("C_high_alert_no_dose", "Черновик", is_shadow=1),
        ],
    )
    assert signal["title_ru"] == STALE_TITLE_RU
    assert signal["stale"] is True
    assert "Есть важный сигнал риска" not in signal["title_ru"]


def test_unknown_findings_do_not_invent_stale() -> None:
    signal = primary_signal_title(
        attention_primary="safety",
        attention_reason_ru="Есть важный сигнал риска",
        findings=None,
    )
    assert signal["title_ru"] == ""
    assert signal["stale"] is False


def test_topical_major_does_not_outrank_high_alert() -> None:
    signal = primary_signal_title(
        attention_primary="safety",
        attention_reason_ru="Есть важный сигнал риска",
        findings=[
            _finding(
                "C_ddi",
                "Лекарственное взаимодействие (Major): крем топический",
                topical_ddi=True,
            ),
            _finding("C_high_alert_no_dose", "Гепарин учебный без дозы"),
        ],
    )
    assert signal["code"] == "C_high_alert_no_dose"


def test_red_flag_outranks_major() -> None:
    signal = primary_signal_title(
        attention_primary="safety",
        attention_reason_ru="Есть критичный риск (безопасность)",
        findings=[
            _finding("C_ddi", "Лекарственное взаимодействие (Major): Альфацин + Бетапрол"),
            _finding("C_red_flag", "Красный флаг без маршрута", severity="P0"),
        ],
    )
    assert signal["code"] == "C_red_flag"
    assert signal["severity_label_ru"] == "Критично"


def test_clip_keeps_word_boundary() -> None:
    long_title = "Лекарственное взаимодействие (Major): " + ("Альфацин " * 20)
    signal = primary_signal_title(
        attention_primary="safety",
        attention_reason_ru="Есть важный сигнал риска",
        findings=[_finding("C_ddi", long_title.strip())],
    )
    assert len(signal["title_ru"]) <= 143
    assert not signal["title_ru"].endswith("Альфац...")
    assert signal["title_full_ru"].startswith("Лекарственное взаимодействие")
    assert len(signal["title_full_ru"]) > len(signal["title_ru"])


def test_list_rows_keep_grade_and_skip_unqueried_safety() -> None:
    rows = [
        {
            "mis_id": "1",
            "attention_primary": "safety",
            "attention_reason_ru": "Есть важный сигнал риска",
            "overall_grade": "important",
        },
        {
            "mis_id": "2",
            "attention_primary": "zone2a",
            "attention_reason_ru": "Диагноз и/или МКБ отсутствуют",
            "overall_grade": "important",
        },
    ]
    apply_primary_signal_titles(rows, {}, safety_queried=False)
    assert "primary_finding_title_ru" not in rows[0]
    assert rows[0]["overall_grade"] == "important"
    assert rows[1]["primary_finding_title_ru"] == "Диагноз и/или МКБ отсутствуют"
    apply_primary_signal_titles(
        rows,
        {"1": [_finding("C_nsaid_dup", "Два НПВП: Альфацин и Бетапрол")]},
        safety_queried=True,
    )
    assert rows[0]["primary_finding_title_ru"].startswith("Два НПВП")
    assert rows[0]["overall_grade"] == "important"


def test_zones_payload_leads_with_finding_not_generic_reason() -> None:
    payload = zones_api_payload(
        {
            "ok": True,
            "attention_primary": "safety",
            "attention_reason_ru": "Есть важный сигнал риска",
            "overall_grade": {
                "grade": "important",
                "label_ru": "Важно",
                "reason_ru": "Важный сигнал риска",
            },
        },
        findings=[_finding("C_uncertainty_unrouted", "Неясный диагноз без плана обследования")],
    )
    assert payload["primary_finding_title_ru"] == "Неясный диагноз без плана обследования"
    assert payload["overall_grade"]["grade"] == "important"
    assert payload["overall_grade"]["reason_ru"] == "Важный сигнал риска"
    assert payload["primary_finding_stale"] is False
