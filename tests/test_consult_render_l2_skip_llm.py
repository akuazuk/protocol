"""Render L2 без LLM: tier=L2 → фактически L1, без pipeline OOM."""
from __future__ import annotations


def test_l2_skip_llm_routes_to_l1(monkeypatch) -> None:
    import rag_server as rs

    monkeypatch.setenv("RENDER", "1")
    monkeypatch.setenv("CONSULT_RENDER_L2_SKIP_LLM", "1")
    monkeypatch.setattr(rs, "_consult_render_l2_skip_llm", lambda: True)

    seen: dict[str, str] = {}

    def fake_run(tier, **kwargs):
        seen["tier"] = tier
        return {
            "ok": True,
            "review_tier": "L1",
            "review": {"criteria": [], "overall_compliance_pct": 81},
            "overall_score": 81,
        }

    import clinical_knowledge.consult_tiering as ct

    monkeypatch.setattr(ct, "run_consult_by_tier", fake_run)

    out = rs._consult_review_from_tier_or_pipeline(
        tier="L2",
        text="Консультативное заключение\nДиагноз: J20",
        bundle=None,
        consultation_id="t",
        category_slugs="",
        require_rag_for_l2=False,
    )
    assert seen["tier"] == "L1"
    assert out.get("review_tier") == "L2"
    assert out.get("render_l2_limited") is True
    assert "512 MiB" in (out.get("review") or {}).get("limitations_ru", "")
