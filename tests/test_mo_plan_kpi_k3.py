"""K3: honest Overview caption when plan was not compared to a protocol."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")


def test_plan_kpi_uses_uncompared_caption_when_na_dominates() -> None:
    assert "function planKpiMeta(attn)" in APP
    assert "план не сравнивался с КП:" in APP
    assert 'tile("План плохо", a.zone2b_bad, planKpiMeta(a)' in APP


def test_plan_ring_center_says_not_compared_when_na_dominates() -> None:
    assert 'center = "не сравнивался с КП"' in APP
    assert "meta.key === \"zone2b\"" in APP
