"""Render-safe L2-lite consult pipeline."""
from __future__ import annotations

import consult_review_pipeline as crp


def test_render_l2_lite_branch(monkeypatch) -> None:
    import rag_server as rs

    rs._consult_review_cache.clear()
    seen: dict[str, bool] = {"lite": False}

    def _fake_lite(**kwargs):
        seen["lite"] = True
        yield crp._progress_tuple("synthesize", 80, "тест", {})
        yield (
            "done",
            {
                "ok": True,
                "review_tier": "L2",
                "review": {"criteria": [], "overall_compliance_pct": 70},
                "retrieval_paths": [],
            },
        )

    monkeypatch.setattr(crp, "_iter_consult_review_render_l2_lite", _fake_lite)
    monkeypatch.setattr(rs, "_consult_render_l2_lite_enabled", lambda: True)

    out = crp.run_consult_review_pipeline(
        full_text="Консультативное заключение\nДиагноз: J20",
        n_files=1,
        consult_docs_meta=[{"filename": "t.txt"}],
        pdf_warnings=[],
        content_signature="sig",
        category_slugs="",
    )
    assert seen["lite"] is True
    assert out.get("review_tier") == "L2"
    assert out.get("review", {}).get("overall_compliance_pct") == 70
