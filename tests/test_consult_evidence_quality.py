"""Фильтры evidence pack и block gaps."""
from __future__ import annotations

from clinical_knowledge.consult_evidence_quality import (
    is_kp_checklist_item,
    is_usable_evidence_excerpt,
    normalize_gap_text,
)
from clinical_knowledge.consult_l2_review import extract_block_gaps


def test_normalize_gap_text_strips_bullet() -> None:
    assert normalize_gap_text("— МРТ грудной клетки") == "МРТ грудной клетки"


def test_is_kp_checklist_item_rejects_toc() -> None:
    assert not is_kp_checklist_item("— клинический протокол диагностики и лечения инфаркта")
    assert is_kp_checklist_item("МРТ органов грудной клетки")


def test_is_usable_evidence_excerpt_rejects_months() -> None:
    assert not is_usable_evidence_excerpt("июня; октября")


def test_extract_block_gaps_collapses_exams_bullets() -> None:
    align = {
        "alignment_cards": [
            {
                "block_id": "exams",
                "name_ru": "Обследование",
                "gaps_ru": [
                    "— клинический протокол диагностики",
                    "— МРТ",
                ],
                "comment_ru": "КП «Тромбоз»: в КЗ отражено 0 из 12 рекомендуемых обследований.",
                "score_pct": 20,
            }
        ]
    }
    gaps = extract_block_gaps(align)
    assert len(gaps) == 1
    assert "0 из 12" in gaps[0]["gap_ru"]
