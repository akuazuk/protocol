"""Статические структурные/accessibility-проверки UX-редизайна (ТЗ №2 §12, §15).

Без браузера: проверяем инварианты приёмки прямо по исходникам HTML/CSS/JS.
Полные browser/axe/visual проверки выполняются на deploy (см. отчёт, раздел
«Известные ограничения»).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def index_html() -> str:
    return _read("index.html")


@pytest.fixture(scope="module")
def patient_html() -> str:
    return _read("patient.html")


@pytest.fixture(scope="module")
def ux_css() -> str:
    return _read("ux-redesign.css")


# --- §D2/D3: проверка КЗ упрощена ---

def test_ux_css_linked(index_html: str) -> None:
    assert '/ux-redesign.css' in index_html


def test_kz_single_checkbox_and_advanced_details(index_html: str) -> None:
    assert 'id="consult-deep-check"' in index_html
    # уровни L0/L1/L2 - внутри раскрываемого <details>
    assert 'id="consult-advanced-tier"' in index_html
    m = re.search(r'<details[^>]*id="consult-advanced-tier"', index_html)
    assert m, "advanced tier должен быть <details>"


def test_kz_single_primary_cta(index_html: str) -> None:
    # основной CTA переименован, конкурирующая кнопка L2 скрыта
    assert '>Проверить заключение<' in index_html
    l2 = re.search(r'id="consult-btn-run-l2"[^>]*', index_html)
    assert l2 and "hidden" in l2.group(0), "кнопка «Сверка L2» должна быть hidden"
    assert 'Проанализировать (L1)' not in index_html


# --- §P0-B: расширенный фильтр не пустой ---

def test_advanced_drawer_has_title(index_html: str) -> None:
    assert 'id="search-settings-drawer-title"' in index_html
    assert 'aria-labelledby="search-settings-drawer-title"' in index_html


# --- §G1/G2: design tokens и минимальные размеры ---

def test_design_tokens_defined(ux_css: str) -> None:
    for tok in ("--ux-text-muted", "--ux-action", "--ux-status-error", "--ux-touch-min", "--ux-font-body"):
        assert tok in ux_css, f"нет токена {tok}"


def test_reduced_motion_supported(ux_css: str) -> None:
    assert "prefers-reduced-motion" in ux_css


def test_touch_target_min(ux_css: str) -> None:
    assert "--ux-touch-min: 44px" in ux_css
    assert "min-height: var(--ux-touch-min)" in ux_css


def test_no_tiny_working_font_in_ux_css(ux_css: str) -> None:
    # рабочие тексты >= 14px: в ux-redesign.css нет явных font-size меньше 14px/0.875rem
    for m in re.finditer(r"font-size:\s*([0-9.]+)px", ux_css):
        assert float(m.group(1)) >= 14, f"слишком мелкий шрифт {m.group(1)}px в ux-redesign.css"


# --- §E1/E2: пациентский сценарий ---

def test_patient_onboard_collapsible(patient_html: str) -> None:
    m = re.search(r'<details[^>]*id="onboard"', patient_html)
    assert m, "онбординг «Как это работает» должен быть раскрываемым <details>"


def test_patient_upload_before_or_near_top(patient_html: str) -> None:
    # зона загрузки (kz-drop) идёт сразу после свёрнутого онбординга
    onboard = patient_html.index('id="onboard"')
    drop = patient_html.index('id="kz-drop"')
    between = patient_html[onboard:drop]
    # между онбордингом и загрузкой не должно быть тяжёлых секций (нет второй <section class="card">)
    assert between.count('<section class="card"') <= 1


def test_playful_tone_gated(patient_html: str) -> None:
    js = _read("patient-ui.js")
    assert "__PATIENT_PLAYFUL_TONE__" in js
    # по умолчанию serious
    assert 'selectedQuestionTone = "serious"' in js
