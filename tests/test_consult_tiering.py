"""L0/L1/L2 tiering."""
from __future__ import annotations

from clinical_knowledge.consult_tiering import resolve_tier, run_consult_by_tier

KZ = """\
Врач: флеболог
Дата консультации: 12.04.2024
Дата рождения: 15.08.1970
Пол: женский
Диагноз: I80.1 Флеботромбоз поверхностных вен нижней конечности.
Рекомендации по лечению: ривароксабан 20 мг 1 раз в день постоянно.
"""


def test_resolve_tier_defaults_l2():
    assert resolve_tier(None) == "L2"
    assert resolve_tier("l0") == "L0"


def test_l0_tier_screen():
    out = run_consult_by_tier(tier="L0", text=KZ, consultation_id="t-tier-l0")
    assert out["review_tier"] == "L0"
    assert out.get("delegate_full_pipeline") is not True


def test_l1_tier_structured():
    out = run_consult_by_tier(tier="L1", text=KZ, consultation_id="t-tier-l1")
    assert out["review_tier"] == "L1"
    assert out.get("structured_analysis")
    assert out.get("llm_used") is False


def test_l2_delegates_pipeline():
    out = run_consult_by_tier(tier="L2", text=KZ)
    assert out["delegate_full_pipeline"] is True
