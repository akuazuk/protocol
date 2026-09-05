"""Обезличивание перед отправкой в Gemini.

Зачем
-----
Gemini - обработчик за пределами Беларуси, поэтому объём персональных данных
в запросе должен быть минимальным по смыслу задачи (закон 99-З, принцип
минимизации). Аудит показал, что часть промптов несла прямые идентификаторы,
которые модели не нужны для оценки качества: ФИО врача, Visit ID, Patient ID,
точную дату рождения.

Что делаем
----------
1. Прямые идентификаторы заменяем на устойчивый псевдоним. Устойчивость важна:
 методист должен видеть, что два разбора относятся к одному случаю, не получая
 самого идентификатора. Псевдоним - HMAC на ключе, то есть по нему нельзя
 восстановить исходное значение без ключа.
2. ФИО сводим к инициалам, ИНП и телефоны скрываем, точную дату рождения
 заменяем на возраст: для клинической оценки нужен возраст, а не дата.
3. Клинический текст оставляем как есть - он и есть предмет оценки. Убирать
 из него симптомы и диагнозы означало бы сломать саму задачу.

Ключ
----
`PHI_PSEUDONYM_KEY` (в проде - Secret Manager). Без него псевдонимы считаются
на резервной соли: детерминированность сохраняется, стойкость - нет, поэтому
модуль один раз пишет предупреждение в лог.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from datetime import date, datetime
from typing import Any, Mapping

from .privacy import name_to_initials, redact_kz_text_for_display

_LOG = logging.getLogger(__name__)

_FALLBACK_SALT = b"protocol-phi-fallback-salt"
_warned_no_key = False

# Прямые идентификаторы: сами по себе не несут клинического смысла для модели.
IDENTIFIER_KEYS = frozenset(
    {
        "case_id",
        "visit_id",
        "patient_id",
        "patient_key",
        "mis_id",
        "doctor_id",
        "doctor_key",
        "specialist_id",
        "specialist_id_from_visit",
    }
)

# ФИО в структурированных полях.
NAME_KEYS = frozenset(
    {
        "doctor_fio",
        "doctor_name",
        "patient_fio",
        "patient_name",
        "specialist_name",
        "fio",
        "doctor",
    }
)

_RE_PHONE = re.compile(r"(?<!\d)(?:\+375|375|8)[\s\-(]*\d{2}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)")
_RE_DOB_LINE = re.compile(
    r"(?im)^(?P<prefix>[^\n]{0,40}?(?:дата\s+рождения|д\.?\s*р\.?)\s*[:\-]?\s*)"
    r"(?P<dob>\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4})(?P<tail>.*)$"
)


def _key() -> bytes:
    global _warned_no_key
    raw = (os.environ.get("PHI_PSEUDONYM_KEY") or "").strip()
    if raw:
        return raw.encode("utf-8")
    if not _warned_no_key:
        _warned_no_key = True
        _LOG.warning(
            "PHI_PSEUDONYM_KEY не задан: псевдонимы считаются на резервной соли. "
            "Детерминированность сохранена, стойкость к подбору - нет."
        )
    return _FALLBACK_SALT


def pseudonym(value: Any, *, prefix: str = "id") -> str:
    """Устойчивый псевдоним вида `case-3f9a2c71`; пустое значение остаётся пустым."""
    text = str(value or "").strip()
    if not text:
        return ""
    digest = hmac.new(_key(), text.encode("utf-8"), hashlib.sha256).hexdigest()[:8]
    return f"{prefix}-{digest}"


def age_from_dob(dob: str | date | datetime, *, on: date | None = None) -> int | None:
    """Полных лет на дату; None, если дату разобрать не удалось."""
    ref = on or date.today()
    if isinstance(dob, datetime):
        born: date | None = dob.date()
    elif isinstance(dob, date):
        born = dob
    else:
        born = None
        for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                born = datetime.strptime(str(dob).strip(), fmt).date()
                break
            except ValueError:
                continue
    if born is None:
        return None
    years = ref.year - born.year - ((ref.month, ref.day) < (born.month, born.day))
    return years if 0 <= years <= 130 else None


def redact_text_for_llm(text: str, *, on: date | None = None) -> str:
    """Текст для модели: ФИО - инициалы, ИНП и телефоны скрыты, дата рождения - возраст."""
    if not text or not str(text).strip():
        return ""
    out = redact_kz_text_for_display(str(text))
    out = _RE_PHONE.sub("[телефон скрыт]", out)

    def _dob_repl(m: re.Match[str]) -> str:
        years = age_from_dob(m.group("dob"), on=on)
        replacement = f"возраст {years} лет" if years is not None else "[дата скрыта]"
        return m.group("prefix") + replacement + (m.group("tail") or "")

    return _RE_DOB_LINE.sub(_dob_repl, out)


def redact_mapping_for_llm(row: Mapping[str, Any]) -> dict[str, Any]:
    """Копия словаря без прямых идентификаторов и ФИО.

    Клинические поля не трогаем: они и есть предмет оценки.
    """
    out: dict[str, Any] = {}
    for key, value in row.items():
        low = str(key).lower()
        if low in IDENTIFIER_KEYS:
            out[key] = pseudonym(value, prefix=low.replace("_id", "").replace("_key", "") or "id")
        elif low in NAME_KEYS:
            out[key] = name_to_initials(str(value)) if value else ""
        else:
            out[key] = value
    return out


def contains_identifier_label(prompt: str) -> list[str]:
    """Явные подписи идентификаторов в готовом промпте (для тестов и проверок)."""
    labels = [
        r"Visit\s*ID",
        r"Patient\s*ID",
        r"MIS\s*ID",
        r"дата\s+рождения\s*[:\-]?\s*\d",
    ]
    return [lab for lab in labels if re.search(lab, prompt, re.IGNORECASE)]
