"""Проверки лекарственной безопасности (взаимодействия, дубли групп)."""
from __future__ import annotations

import re

from .consult_schema import ConsultationDocument, SafetyAssessment
from .medication_parser import looks_like_medication_item

_NSAID_PATTERN = re.compile(
    r"(?:"
    r"а[еэ]ртал|aertal|aceclofenac|ацеклофенак|"
    r"дексалгин|dexalgin|кеторолак|ketorolac|"
    r"аркоксия|arcoxia|эторикоксиб|etoricoxib|"
    r"диклофенак|diclofenac|вольтарен|voltaren|"
    r"ибупрофен|ibuprofen|нимесулид|nimesulide|"
    r"мелоксикам|meloxicam|кетопрофен|ketoprofen|"
    r"напроксен|naproxen|"
    r"целебрекс|celecoxib|"
    r"кетанов"
    r")",
    re.I,
)


def nsaid_labels_in_text(text: str) -> list[str]:
    return [m.group(0).lower() for m in _NSAID_PATTERN.finditer(text or "")]


def detect_concurrent_nsaids(doc: ConsultationDocument) -> SafetyAssessment | None:
    """Два и более НПВП в одном КЗ — критическая ошибка назначения."""
    found: set[str] = set()
    for m in doc.medications:
        if not looks_like_medication_item(m):
            continue
        blob = " ".join(x for x in (m.drug_name, m.raw_text) if x)
        found.update(nsaid_labels_in_text(blob))
    treat = doc.sections.recommendations_treatment or ""
    found.update(nsaid_labels_in_text(treat))
    if len(found) < 2:
        return None
    names = ", ".join(sorted(found)[:8])
    return SafetyAssessment(
        issue_type="drug_safety",
        severity="critical",
        finding_text=(
            f"Одновременно назначены два и более НПВП ({names}) — "
            "повышенный риск побочных эффектов (ЖКТ, почки)."
        ),
        expected_action="Исключить дублирование НПВП; оставить один препарат с контролем переносимости.",
        actual_action=None,
        status="not_handled",
    )
