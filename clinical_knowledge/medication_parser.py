"""Разбор лекарственных назначений из КЗ (ТЗ разделы 9, 23, 24).

Извлекает: препарат, дозу (значение+единица), кратность, длительность, путь.
Длинные схемы снижения дозы («С 12.08.24 - Преднизолон 5 мг по 11 таб») собираются
в ``schedule``.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .consult_schema import MedicationItem, MedicationScheduleStep
from .date_parser import parse_date

# Разделители в тексте КZ/PDF (типографские тире не нормализуются в regex-скрипте).
_RE_DASH = r"[\-\u2013\u2014]"
_RE_DASH_COLON = r"[\-:\u2013\u2014]"
_STRIP_LEAD = " \t-:.,\u2013\u2014"

_DOSE_UNITS = r"мкг|мг|г|мл|л|ммоль|ме|ед|таб(?:л(?:етк[аи])?)?|т|капс(?:ул[аы])?|к|кап(?:ель|ли)?|доз[аы]?"
RE_DOSE = re.compile(rf"(\d+(?:[.,]\d+)?)\s*({_DOSE_UNITS})\b", re.I)
RE_FREQ = re.compile(
    r"(\d+\s*раз[ауы]?\s*(?:в|/)\s*(?:день|сутки|сут|неделю|нед)"
    r"|раз\s*(?:в|/)\s*(?:день|сутки|сут)"
    r"|по\s+требованию"
    r"|\d+\s*р/?(?:д|сут))",
    re.I,
)
RE_DURATION = re.compile(
    rf"(\d+\s*(?:{_RE_DASH}\s*\d+\s*)?(?:дн(?:я|ей|евн)?|сут(?:ок|ки)?|нед(?:ел[ьия])?|мес(?:яц[ева]*)?)"
    r"|постоянно|пожизненно|длительно|курс[а-я]*)",
    re.I,
)
RE_ROUTE = re.compile(
    r"\b(внутрь|перорально|в/в|в/м|п/к|внутривенно|внутримышечно|подкожно|местно|наружно|ингаляц\w*)\b",
    re.I,
)
RE_FORM = re.compile(
    r"\b(таб(?:л(?:етка|етки|еток)?)?|капс(?:ула|улы|ул)?|"
    r"р-?р|раствор|гель|мазь|крем|спрей|суппозитор|свечи?|ампула|амп)\b",
    re.I,
)
RE_PAST = re.compile(
    r"\b(ранее|до обращения|в прошлом|принимал[аи]?|получал[аи]?|"
    r"отмен[её]н|отменить|прекращ[её]н|заверш[её]н)\b",
    re.I,
)
RE_NEGATED = re.compile(r"\b(не принимает|не получа(?:ет|л[аи]?)|без при[её]ма)\b", re.I)
RE_HYPOTHESIS = re.compile(r"\b(возможно|предположительно|рассмотреть)\b", re.I)
RE_FAMILY = re.compile(
    r"\b(мать|отец|родственник|семейн(?:ый|ая|ом)|у родственник)\w*\b",
    re.I,
)
RE_CONTEXT_PREFIX = re.compile(
    r"^\s*(?:ранее|до обращения|в прошлом|принимал[аи]?|получал[аи]?|"
    r"отмен[её]н|отменить|прекращ[её]н|заверш[её]н)\s*[:,-]?\s*",
    re.I,
)
RE_SCHEDULE_PREFIX = re.compile(
    rf"^\s*с\s+(\d{{1,2}}[.\-/]\d{{1,2}}[.\-/]\d{{2,4}})\s*{_RE_DASH_COLON}\s*(.+)$",
    re.I,
)
RE_SCHEDULE_SUFFIX = re.compile(
    r"^(.+?)\s+с\s+(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})\s*$",
    re.I,
)
_DOSE_LEADIN = re.compile(r"\b(по|на)\b", re.I)

RE_DRUG_MARKERS = re.compile(
    r"\b(?:таб\.?|капс(?:\.|ул)?|р-?р\.?|амп\.?|супп\.?|"
    r"мг|мкг|мл|г\b|ме\b|ед\.?|"
    r"№\s*\d+|"
    r"\d+\s*(?:мг|мкг|мл|г\b|таб|капс|доз)|"
    r"(?:\d+\s*)?раз\s*(?:в|/)|р/с|р/д|"
    r"в/м|в/мыш|в/в|п/к|перорально|внутрь)\b",
    re.I,
)
RE_NON_DRUG_LINE = re.compile(
    r"(?:^|\d+\.\s*)"
    r"(?:"
    r"консультац\w*\s+(?:физиотерап|специалист|врач)|"
    r"(?:ручн\w*\s+)?(?:классическ\w*\s+)?массаж|"
    r"иглорефлекс|"
    r"курс\s+корпоральн|"
    r"контрольн\w*\s+явк|"
    r"л/н\b|"
    r"лист\s+нетруд|"
    r"физиотерап\w*|"
    r"^лфк\b|"
    r"немедикамент|"
    r"^врач\s*:"
    r")",
    re.I,
)
RE_SICK_LEAVE_PERIOD = re.compile(
    r"^\s*с\s+\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\s+по\s+\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\s*$",
    re.I,
)

_mid = 0


def _next_id() -> str:
    global _mid
    _mid += 1
    return f"med{_mid}"


def is_non_drug_prescription_line(raw: str) -> bool:
    """Строка назначения без лекарственного препарата (процедуры, явка, л/н)."""
    s = (raw or "").strip()
    if len(s) < 3:
        return True
    if RE_SICK_LEAVE_PERIOD.match(s):
        return True
    if RE_NON_DRUG_LINE.search(s):
        return True
    return False


def looks_like_medication_item(item: MedicationItem) -> bool:
    """True только для строк с признаками лекарственного назначения."""
    raw = item.raw_text or ""
    if is_non_drug_prescription_line(raw):
        return False
    if item.dose_value is not None:
        return True
    if item.frequency or item.duration or item.route or item.schedule:
        return True
    return bool(RE_DRUG_MARKERS.search(raw))


def _extract_drug_name(raw: str) -> str | None:
    """Имя препарата = ведущие слова до первой дозы/числа."""
    s = raw.strip()
    s = re.sub(
        r"^\s*(?:местно|наружно|внутрь|перорально|в/в|в/м|п/к)\s*:\s*",
        "",
        s,
        flags=re.I,
    )
    m = RE_DOSE.search(s)
    head = s[: m.start()] if m else s
    head = re.split(r"\d", head)[0]
    head = re.split(
        r"\b(?:по схеме|по требованию|внутрь|перорально|местно|наружно|"
        r"в/в|в/м|п/к|таб(?:л)?|капс(?:ул)?|гель|мазь|крем|спрей|раствор)\b",
        head,
        maxsplit=1,
        flags=re.I,
    )[0]
    head = head.strip(_STRIP_LEAD)
    return head or None


def _parse_one(raw: str, *, source_section: str | None = None) -> MedicationItem | None:
    s = (raw or "").strip()
    if len(s) < 3:
        return None
    if is_non_drug_prescription_line(s):
        return None
    family = bool(RE_FAMILY.search(s))
    negated = bool(RE_NEGATED.search(s))
    hypothesis = bool(RE_HYPOTHESIS.search(s))
    past = source_section == "medication_history" or bool(RE_PAST.search(s))
    assertion = (
        "family"
        if family
        else "negated"
        if negated
        else "hypothesis"
        if hypothesis
        else "confirmed"
    )
    subject = "family" if family else "patient"
    activity = "past" if past or negated else "active"
    fact_time = "past" if past else "current"
    parse_text = RE_CONTEXT_PREFIX.sub("", s)
    dose_value = None
    dose_unit = None
    mdose = RE_DOSE.search(parse_text)
    if mdose:
        try:
            dose_value = float(mdose.group(1).replace(",", "."))
        except ValueError:
            dose_value = None
        dose_unit = mdose.group(2).lower()
    freq = None
    mf = RE_FREQ.search(parse_text)
    if mf:
        freq = re.sub(r"\s+", " ", mf.group(1).strip())
    dur = None
    md = RE_DURATION.search(parse_text)
    if md:
        dur = re.sub(r"\s+", " ", md.group(1).strip())
    route = None
    mr = RE_ROUTE.search(parse_text)
    if mr:
        route = mr.group(1).lower()
    return MedicationItem(
        medication_id=_next_id(),
        raw_text=s,
        drug_name=_extract_drug_name(parse_text),
        dose_value=dose_value,
        dose_unit=dose_unit,
        form=(RE_FORM.search(parse_text).group(1).lower() if RE_FORM.search(parse_text) else None),
        frequency=freq,
        duration=dur,
        route=route,
        source_section=source_section,
        activity_status=activity,
        assertion=assertion,
        subject=subject,
        fact_time=fact_time,
    )


def _schedule_step_suffix(raw: str) -> tuple[MedicationScheduleStep, str] | None:
    """Строка «препарат … с DD.MM.YY» в конце."""
    m = RE_SCHEDULE_SUFFIX.match(raw.strip())
    if not m:
        return None
    rest, date_raw = m.group(1).strip(), m.group(2)
    start = parse_date(date_raw)
    mdose = RE_DOSE.search(rest)
    dose_text = rest[mdose.start():].strip() if mdose else rest
    mf = RE_FREQ.search(rest)
    freq_text = mf.group(1).strip() if mf else None
    drug = _extract_drug_name(rest) or ""
    return (
        MedicationScheduleStep(
            start_date=start, dose_text=dose_text or rest, frequency_text=freq_text,
        ),
        drug.lower(),
    )


def _schedule_step(raw: str) -> tuple[MedicationScheduleStep, str] | None:
    """Если строка вида «С DD.MM.YY - <назначение>», вернуть (шаг, имя_препарата)."""
    m = RE_SCHEDULE_PREFIX.match(raw)
    if not m:
        return None
    start = parse_date(m.group(1))
    rest = m.group(2).strip()
    mdose = RE_DOSE.search(rest)
    dose_text = rest
    if mdose:
        # текст дозы - от препарата/«по» до конца
        lead = _DOSE_LEADIN.search(rest)
        dose_text = rest[lead.start():].strip() if lead else rest[mdose.start():].strip()
    mf = RE_FREQ.search(rest)
    freq_text = mf.group(1).strip() if mf else None
    drug = _extract_drug_name(rest) or ""
    return (
        MedicationScheduleStep(
            start_date=start, dose_text=dose_text or rest, frequency_text=freq_text,
        ),
        drug.lower(),
    )


def parse_medications(text: str, *, source_section: str | None = None) -> list[MedicationItem]:
    """Разбирает блок назначений в список MedicationItem.

    Последовательные строки «С <дата> - <препарат> ...» с тем же препаратом
    объединяются в ``schedule`` одного MedicationItem.
    """
    if not text:
        return []
    items: list[MedicationItem] = []
    schedule_acc: dict[str, MedicationItem] = {}

    for raw_line in re.split(r"[\n]+", text):
        for segment in re.split(r"\s*;\s*|(?<=\.)\s+(?=[А-ЯA-Z])", raw_line):
            line = segment.strip(_STRIP_LEAD + "\t•")
            if len(line) < 3:
                continue
            step = _schedule_step(line) or _schedule_step_suffix(line)
            if step is not None:
                sched_step, drug_key = step
                if not drug_key and len(schedule_acc) == 1:
                    drug_key = next(iter(schedule_acc.keys()))
                if drug_key and drug_key in schedule_acc:
                    schedule_acc[drug_key].schedule.append(sched_step)
                    continue
                m_prefix = RE_SCHEDULE_PREFIX.match(line)
                rest = m_prefix.group(2).strip() if m_prefix else line
                base = _parse_one(rest, source_section=source_section)
                if base is None:
                    continue
                if drug_key and not base.drug_name:
                    base.drug_name = drug_key.capitalize()
                base.raw_text = line
                base.schedule.append(sched_step)
                items.append(base)
                if drug_key:
                    schedule_acc[drug_key] = base
                continue
            item = _parse_one(line, source_section=source_section)
            if item is None:
                continue
            items.append(item)
            if item.drug_name:
                schedule_acc[item.drug_name.lower()] = item
    return items


def medication_assignments_from_case(
    case: Mapping[str, Any] | None,
    *,
    include_inactive: bool = True,
) -> list[dict[str, Any]]:
    """Build normalized, per-line assignments without dose bleed between drugs."""
    if not isinstance(case, Mapping):
        return []
    from .drug_normalizer import normalize_drug
    from .medication_safety import strip_treatment_alternatives

    sources = (
        ("treatment_recommendations", "treatment_recommendations"),
        ("medication_history", "medication_history"),
    )
    out: list[dict[str, Any]] = []
    for key, section in sources:
        text = str(case.get(key) or "").strip()
        if not text:
            continue
        if key == "treatment_recommendations":
            text = strip_treatment_alternatives(text)
        for item in parse_medications(text, source_section=section):
            row = item.model_dump(mode="json")
            normalized = normalize_drug(item.drug_name or item.raw_text)
            if normalized:
                row.update(
                    {
                        "surface": normalized.get("surface") or item.drug_name,
                        "inn": normalized.get("inn"),
                        "confidence": normalized.get("confidence"),
                        "normalization_method": normalized.get("method"),
                    }
                )
            if include_inactive or (
                row["activity_status"] == "active"
                and row["assertion"] == "confirmed"
                and row["subject"] == "patient"
            ):
                out.append(row)
    return out


def active_medication_assignments(case: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return medication_assignments_from_case(case, include_inactive=False)
