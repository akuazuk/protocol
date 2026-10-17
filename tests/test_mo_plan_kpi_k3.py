"""K3: honest Overview caption when plan was not compared to a protocol."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")


def test_plan_kpi_uses_uncompared_caption_when_na_dominates() -> None:
    assert "function planKpiMeta(attn, zoneHint)" in APP
    assert "план не сравнивался с КП:" in APP
    assert 'tile("План плохо", a.zone2b_bad, planKpiMeta(a, (opts.zones || {}).zone2b)' in APP


def test_plan_ring_center_stays_good_share() -> None:
    # Крупная цифра - доля «хорошо» среди сравнённых. Доля без КП только в подписи.
    assert 'centerSub = "без КП"' not in APP
    assert 'var centerSub = assessedN > 0 ? "хорошо" : "не оценено";' in APP
    assert 'ringSub += " · без сравнения с КП: " + naN + " из " + zoneN;' in APP
    assert 'ringSub += " · плохо среди сравнённых: " + badN;' in APP
    assert "meta.key === \"zone2b\"" in APP
