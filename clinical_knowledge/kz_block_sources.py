"""Источники доказательств по блокам анализа КЗ."""
from __future__ import annotations

from typing import Literal

SourceKind = Literal["mkb", "kp", "completeness", "regulation", "limitations"]

# block_id -> как оценивать (не искать в КП то, чего там нет)
KZ_BLOCK_SOURCES: dict[str, dict[str, str]] = {
    "documentation": {"source": "completeness", "label": "Полнота КЗ"},
    "patient_data": {"source": "completeness", "label": "Полнота КЗ"},
    "complaints": {"source": "completeness", "label": "Полнота КЗ"},
    "anamnesis": {"source": "completeness", "label": "Полнота КЗ"},
    "objective_status": {"source": "completeness", "label": "Полнота КЗ"},
    "diagnosis": {"source": "mkb", "label": "МКБ-10"},
    "exams": {"source": "kp", "label": "КП"},
    "treatment": {"source": "kp", "label": "КП"},
    "follow_up": {"source": "regulation", "label": "НПА / КП"},
    "safety": {"source": "completeness", "label": "Полнота КЗ"},
    "protocol_applicability": {"source": "kp", "label": "КП"},
}

ALIGNMENT_CARD_ORDER: tuple[str, ...] = (
    "diagnosis",
    "complaints",
    "anamnesis",
    "objective_status",
    "exams",
    "treatment",
    "follow_up",
    "limitations",
)

ALIGNMENT_CARD_TITLES: dict[str, str] = {
    "diagnosis": "Диагноз и коды МКБ-10",
    "complaints": "Жалобы",
    "anamnesis": "Анамнез",
    "objective_status": "Объективный статус",
    "exams": "Обследование",
    "treatment": "Лечение и назначения",
    "follow_up": "Наблюдение и контроль",
    "limitations": "Ограничения проверки",
}

SOURCE_KIND_LABELS: dict[str, str] = {
    "mkb": "МКБ-10",
    "kp": "КП",
    "completeness": "Полнота КЗ",
    "regulation": "НПА",
    "limitations": "—",
}
