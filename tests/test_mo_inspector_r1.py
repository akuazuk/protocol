"""R1: unmatched protocol must not look like a plan fail."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")


def test_unmatched_plan_chip_does_not_require_na_band() -> None:
    assert "function protocolIsMatched(kpStatus)" in APP
    assert "if (kpStatus != null && String(kpStatus).trim() !== \"\" && !protocolIsMatched(kpStatus))" in APP
    assert 'return \'<span class="status muted">протокол не подобран</span>\';' in APP


def test_unmatched_plan_hides_fail_criteria() -> None:
    assert "planUnmatched && String(c.zone || \"\") === \"plan\"" in APP
    assert "протокол не подобран - критерий плана не штрафуем" in APP
    assert "unmatchedPlan" in APP
    assert "pair[0] === \"zone2b\" && !protocolIsMatched((zones.zone2b || {}).kp_status)" in APP
