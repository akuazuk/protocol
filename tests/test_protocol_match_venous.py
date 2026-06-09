"""Матчер протоколов: приоритет венозных КП при I80."""
from __future__ import annotations

from clinical_knowledge.protocol_match import compute_match_score


def _card(title: str, path: str, icd: list[str]) -> dict:
    return {
        "title": title,
        "source_path": path,
        "icd10_primary": icd,
        "population": "adult",
        "status": "active",
    }


def test_venous_icd_prefers_thrombosis_over_heart_failure():
    icd = ["I80.1"]
    thrombosis = _card(
        "Диагностика и лечение пациентов с тромбозом глубоких вен",
        "bolezni/tromboz_glv.pdf",
        ["I80"],
    )
    heart = _card(
        "Клинический протокол заболеваний, осложненных сердечной недостаточностью",
        "bolezni/heart_failure.pdf",
        ["I50"],
    )
    s_thromb = compute_match_score(
        thrombosis, icd_list=icd, audience="adult", hints=set(),
        specialty_slug="bolezni-sistemy-krovoobrashcheniya", diag_text="флеботромбоз", complaints=[], performed_exams=[],
    )
    s_heart = compute_match_score(
        heart, icd_list=icd, audience="adult", hints=set(),
        specialty_slug="bolezni-sistemy-krovoobrashcheniya", diag_text="флеботромбоз", complaints=[], performed_exams=[],
    )
    assert s_thromb > s_heart
