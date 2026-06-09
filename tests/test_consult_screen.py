"""L0-скрининг КЗ."""
from __future__ import annotations

from clinical_knowledge.consult_screen import run_compliance_screen

KZ = """\
Врач: флеболог
Дата консультации: 12.04.2024
Дата рождения: 15.08.1970
Пол: женский
Диагноз: I80.1 Флеботромбоз поверхностных вен нижней конечности.
Рекомендации по лечению: ривароксабан 20 мг 1 раз в день постоянно.
"""


def test_compliance_screen_returns_send_gate():
    out = run_compliance_screen(text=KZ, consultation_id="t-screen")
    assert out["ok"] is True
    assert out["screen_level"] == "L0"
    assert "send_gate" in out
    assert "gate_allowed" in out["send_gate"]
