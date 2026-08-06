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
    r"диклофенак|diclofenac|вольтарен|voltaren|дикловит|diclovit|"
    r"ибупрофен|ibuprofen|нимесулид|nimesulide|найз|nise|"
    r"мелоксикам|meloxicam|кетопрофен|ketoprofen|"
    r"напроксен|naproxen|"
    r"целебрекс|celecoxib|"
    r"кетанов"
    r")",
    re.I,
)

# Альтернативы в скобках: ("Кетопрофен", "Найз" и т.п.) / (или ибупрофен)
_ALT_PAREN = re.compile(
    r"\([^)]{0,160}?(?:или|либо|и\s*т\.?\s*п\.?|и\s*др\.?|/)[^)]{0,160}?\)",
    re.I,
)
# Скобки только со списком препаратов после основного назначения
_PAREN_NSAID_LIST = re.compile(
    r"\((?:[^)]*?(?:" + _NSAID_PATTERN.pattern + r")[^)]*?,){1,}[^)]*?\)",
    re.I,
)
_TOPICAL_NEAR = re.compile(
    r"(?:гель|мазь|крем|спрей|пластырь|смазывать|местно|наружно|"
    r"\bgel\b|\bcream\b|\bointment\b|\btopical\b)",
    re.I,
)


def _strip_nsaid_alternatives(text: str) -> str:
    """Убрать скобки с альтернативами НПВП, чтобы не считать их одновременным приёмом."""
    cleaned = _ALT_PAREN.sub(" ", text or "")
    cleaned = _PAREN_NSAID_LIST.sub(" ", cleaned)
    return cleaned


def _canonical_nsaid(label: str) -> str:
    raw = (label or "").lower().strip()
    aliases = {
        "aertal": "аэртал",
        "aceclofenac": "ацеклофенак",
        "ацеклофенак": "аэртал",
        "voltaren": "диклофенак",
        "вольтарен": "диклофенак",
        "diclofenac": "диклофенак",
        "diclovit": "диклофенак",
        "дикловит": "диклофенак",
        "ibuprofen": "ибупрофен",
        "ketoprofen": "кетопрофен",
        "nimesulide": "нимесулид",
        "nise": "нимесулид",
        "найз": "нимесулид",
        "meloxicam": "мелоксикам",
        "ketorolac": "кеторолак",
        "dexalgin": "дексалгин",
    }
    return aliases.get(raw, raw)


def nsaid_mentions_in_text(text: str) -> list[dict[str, str | bool]]:
    """НПВП в тексте с признаком топического пути введения."""
    cleaned = _strip_nsaid_alternatives(text or "")
    out: list[dict[str, str | bool]] = []
    seen: set[tuple[str, bool]] = set()
    for match in _NSAID_PATTERN.finditer(cleaned):
        label = _canonical_nsaid(match.group(0))
        start = max(0, match.start() - 28)
        end = min(len(cleaned), match.end() + 28)
        window = cleaned[start:end]
        topical = bool(_TOPICAL_NEAR.search(window))
        key = (label, topical)
        if key in seen:
            continue
        seen.add(key)
        out.append({"label": label, "topical": topical})
    return out


def nsaid_labels_in_text(text: str) -> list[str]:
    """Системные (не топические) НПВП после очистки альтернатив в скобках."""
    return [str(item["label"]) for item in nsaid_mentions_in_text(text) if not item["topical"]]


def concurrent_systemic_nsaids(text: str) -> list[str]:
    """Список системных НПВП при реальном дубле (≥2 разных)."""
    labels = sorted(set(nsaid_labels_in_text(text)))
    return labels if len(labels) >= 2 else []


def detect_concurrent_nsaids(doc: ConsultationDocument) -> SafetyAssessment | None:
    """Два и более системных НПВП в одном КЗ - критическая ошибка назначения.

    Топический гель/мазь + пероральный НПВП не считается дублем.
    Альтернативы в скобках («или», «и т.п.») не считаются одновременным приёмом.
    """
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
            f"Одновременно назначены два и более НПВП ({names}) - "
            "повышенный риск побочных эффектов (ЖКТ, почки)."
        ),
        expected_action="Исключить дублирование НПВП; оставить один препарат с контролем переносимости.",
        actual_action=None,
        status="not_handled",
    )
