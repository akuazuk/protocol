"""Проверки лекарственной безопасности (взаимодействия, дубли групп)."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .consult_schema import ConsultationDocument, SafetyAssessment
from .medication_parser import looks_like_medication_item

_ROOT = Path(__file__).resolve().parents[1]
_CLASS_PATH = _ROOT / "data" / "drug_safety" / "therapeutic_classes.json"

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
    r"свеч(?:и|а|е|ей)?|суппозитор|"
    r"\bgel\b|\bcream\b|\bointment\b|\btopical\b|\bsuppositor)",
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


def drug_mention_is_topical(text: str, needle: str, *, window: int = 36) -> bool:
    """True, если упоминание ЛС рядом с гель/мазь/местно/свечи и т.п."""
    blob = str(text or "")
    token = str(needle or "").strip()
    if not blob or not token or len(token) < 3:
        return False
    pattern = re.compile(re.escape(token), re.I)
    for match in pattern.finditer(blob):
        start = max(0, match.start() - window)
        end = min(len(blob), match.end() + window)
        if _TOPICAL_NEAR.search(blob[start:end]):
            return True
    return False


def ddi_pair_has_topical_partner(
    text: str,
    *,
    surfaces: list[str] | tuple[str, ...] = (),
    inns: list[str] | tuple[str, ...] = (),
) -> bool:
    """Хотя бы один участник DDI в тексте выглядит как топический путь."""
    needles: list[str] = []
    for value in list(surfaces) + list(inns):
        raw = str(value or "").strip()
        if not raw:
            continue
        needles.append(raw)
        # surface вида «ксарелто / rivaroxaban» - проверяем обе части
        if "/" in raw:
            needles.extend(part.strip() for part in raw.split("/") if part.strip())
    seen: set[str] = set()
    for needle in needles:
        key = needle.lower()
        if key in seen:
            continue
        seen.add(key)
        if drug_mention_is_topical(text, needle):
            return True
    return False


def finding_suggests_topical_ddi(row: Mapping[str, Any] | None) -> bool:
    """Для уже сохранённых findings: топический DDI по evidence/title."""
    row = row or {}
    if row.get("topical_ddi") or str(row.get("route") or "").lower() == "topical":
        return True
    blob = " ".join(
        str(row.get(key) or "")
        for key in (
            "finding_title",
            "title_ru",
            "detail",
            "detail_ru",
            "evidence",
            "why_important",
            "reason",
        )
    )
    if not blob:
        return False
    low = blob.lower()
    if "топическ" in low or "topical" in low:
        return True
    # Участники пары в title: «…: a + b» или surface/INN
    for match in re.finditer(
        r"([A-Za-zА-Яа-яЁё-]{4,})(?:\s*/\s*[A-Za-zА-Яа-яЁё-]{3,})?",
        blob,
    ):
        for part in re.split(r"\s*/\s*", match.group(0)):
            if drug_mention_is_topical(blob, part.strip()):
                return True
    return False


def class_dup_primary_enabled() -> bool:
    raw = (os.environ.get("MO_CLASS_DUP_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def class_dup_shadow_only() -> bool:
    """Wave 2 default: class dups (кроме уже primary NSAID) остаются shadow."""
    return not class_dup_primary_enabled()


@lru_cache(maxsize=1)
def load_therapeutic_classes() -> list[dict[str, Any]]:
    if not _CLASS_PATH.is_file():
        return []
    data = json.loads(_CLASS_PATH.read_text(encoding="utf-8"))
    return list(data.get("classes") or [])


def _strip_class_alternatives(text: str) -> str:
    cleaned = _ALT_PAREN.sub(" ", text or "")
    return cleaned


def concurrent_systemic_class(
    text: str,
    class_id: str,
) -> list[str]:
    """≥2 разных системных препарата одного терапевтического класса."""
    classes = {c.get("id"): c for c in load_therapeutic_classes() if c.get("id")}
    spec = classes.get(class_id)
    if not spec:
        return []
    aliases = [str(a).lower() for a in (spec.get("aliases") or []) if a]
    if not aliases:
        return []
    # longest first to prefer full names
    aliases_sorted = sorted(aliases, key=len, reverse=True)
    pattern = re.compile(
        r"(?:" + "|".join(re.escape(a) for a in aliases_sorted) + r")",
        re.I,
    )
    canon_map = {
        str(k).lower(): str(v).lower()
        for k, v in (spec.get("canonical") or {}).items()
    }
    cleaned = _strip_class_alternatives(text or "")
    found: set[str] = set()
    for match in pattern.finditer(cleaned):
        raw = match.group(0).lower().replace("ё", "е")
        label = canon_map.get(raw, raw)
        start = max(0, match.start() - 28)
        end = min(len(cleaned), match.end() + 28)
        if _TOPICAL_NEAR.search(cleaned[start:end]):
            continue
        found.add(label)
    labels = sorted(found)
    return labels if len(labels) >= 2 else []


def all_therapeutic_class_dups(text: str) -> list[dict[str, Any]]:
    """Список дублей по всем классам словаря (включая nsaid для единообразия)."""
    out: list[dict[str, Any]] = []
    for spec in load_therapeutic_classes():
        cid = str(spec.get("id") or "")
        if not cid:
            continue
        labels = concurrent_systemic_class(text, cid)
        if len(labels) < 2:
            continue
        out.append(
            {
                "class_id": cid,
                "label_ru": spec.get("label_ru") or cid,
                "finding_code": spec.get("finding_code") or f"C_{cid}_dup",
                "labels": labels,
            }
        )
    return out


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
