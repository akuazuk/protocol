"""K3: honest Overview caption when plan was not compared to a protocol."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")


def test_plan_kpi_uses_uncompared_caption_when_na_dominates() -> None:
    assert "function planKpiMeta(attn, zoneHint)" in APP
    assert "план не сравнивался с КП:" in APP
    assert 'tile("План плохо", a.zone2b_bad, planKpiMeta(a, (opts.zones || {}).zone2b)' in APP


def test_plan_ring_center_shows_na_share_when_na_dominates() -> None:
    # Центр кольца - короткая доля «без КП», длинная фраза уходит в подпись под кольцом.
    assert 'centerSub = "без КП"' in APP
    assert 'ringSub = "план не сравнивался с КП: " + naN + " из " + zoneN' in APP
    assert "meta.key === \"zone2b\"" in APP
