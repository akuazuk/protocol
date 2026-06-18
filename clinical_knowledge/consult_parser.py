"""Структурный парсер консультативного заключения (ТЗ раздел 10).

Принимает сырой текст КЗ (извлечённый из PDF/DOCX/TXT) и возвращает
ConsultationDocument с разобранными секциями, диагнозами, обследованиями,
лекарствами, датой повторной явки и оценкой качества извлечения.

Парсер устойчив к «грязным» КЗ: любое непрошедшее извлечение фиксируется
в ExtractionQuality.warnings, но не роняет разбор (ТЗ 4.6).
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Any

from corpus_pipeline.entities_extract import extract_icd10

from . import age_sex_resolver as asr
from .consult_schema import (
    ConsultationDocument,
    ConsultationSections,
    ExamItem,
    ExtractionQuality,
    FollowUpItem,
    PatientContext,
)
from .date_parser import parse_date, parse_time
from .diagnosis_parser import parse_diagnoses
from .medication_parser import parse_medications
from .template_parser import parse_template_blocks

RE_FOLLOW_UP_INLINE = re.compile(
    r"контрольн\w*\s+явк\w*(?:\s+(?:на\s+)?(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}))?",
    re.I,
)

# Заголовок секции -> поле ConsultationSections. Порядок важен (более специфичные выше).
_SECTION_HEADERS: list[tuple[str, str]] = [
    (r"аллерг(?:оанамнез|ия\s+на\s+лс|ологическ\w*\s+анамнез)", "allergy_history"),
    (r"цель\s+консультац\w*|повод\s+обращен\w*", "consultation_purpose"),
    (r"анамнез\s+жизни|фактор\w*\s+риска", "life_history"),
    (r"маршрутизац\w*|направлен\w*\s+на\s+консультац\w*", "routing"),
    (r"информированн\w*\s+соглас\w*|отказ\s+от\s+лечен\w*", "consent_text"),
    (r"немедикаментозн\w*\s+рекоменд\w*", "non_drug_recommendations"),
    (r"подпис\w*\s+врач\w*|врач\s*[:\-]|зав\.?\s*отделен\w*", "doctor_signature"),
    (r"лекарственн\w*\s+анамнез", "medication_history"),
    (r"хирургическ\w*\s+анамнез|оперативн\w*\s+анамнез", "surgical_history"),
    (r"объективн\w*\s+статус|объективно|status\s+praesens", "objective_status"),
    (r"локальн\w*\s+статус|locus\s+morbi|st\.?\s*localis", "local_status"),
    (r"данн\w*\s+обследован\w*|результат\w*\s+обследован\w*|данн\w*\s+лаборатор\w*", "exam_results"),
    (r"рекомендац\w*\s+по\s+обследован\w*|план\s+обследован\w*", "recommendations_exams"),
    (r"рекомендац\w*\s+по\s+лечен\w*|назначен\w*\s+лечен\w*|лечение\s+рекомендован\w*", "recommendations_treatment"),
    (r"общие\s+рекомендац\w*", "general_recommendations"),
    (r"дата\s+повторн\w*\s+явк\w*|повторн\w*\s+явк\w*|повторн\w*\s+консультац\w*|контрольн\w*\s+явк\w*", "follow_up_text"),
    (r"жалоб\w*", "complaints"),
    (r"анамнез\s+жизни", "life_history"),
    (r"анамнез\s+заболеван\w*", "anamnesis"),
    (r"анамнез", "anamnesis"),
    (r"диагноз(?:\s+клиническ\w*|\s+заключительн\w*|\s+основн\w*)?", "diagnosis_text"),
    (r"рекомендац\w*", "general_recommendations"),
]
_HEADER_RE = re.compile(
    r"(?im)^\s*(" + "|".join(h for h, _ in _SECTION_HEADERS) + r")\s*[:\- - ]",
)

RE_SPECIALTY = re.compile(
    r"(?:специальность|врач[\-\s]*)?\s*"
    r"(гастроэнтеролог|дерматолог|дерматовенеролог|кардиолог|флеболог|ангиолог|аритмолог|невролог|нейрохирург|"
    r"эндокринолог|пульмонолог|ревматолог|уролог|нефролог|гематолог|онколог|"
    r"хирург|травматолог|ортопед|офтальмолог|оториноларинголог|лор|психиатр|нарколог|"
    r"акушер[\-\s]*гинеколог|гинеколог|аллерголог|иммунолог|инфекционист|стоматолог|"
    r"анестезиолог|реаниматолог|терапевт|педиатр|фтизиатр)\w*",
    re.I,
)
RE_DOCTOR_NAME = re.compile(
    r"\b([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.)",
)
RE_CLINIC = re.compile(
    r"((?:ООО|УП|ГУ|УЗ|ГБУЗ|ЗАО|ОАО|медицинск\w*\s+центр|клиник\w*|центр)\b[^\n]{0,80})",
    re.I,
)
RE_CONSULT_DATE = re.compile(
    r"(?:дата\s+(?:консультац\w*|приёма|приема|осмотра)|консультац\w*\s+от)\s*[:\-]?\s*"
    r"(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})",
    re.I,
)
# Простой вариант «Дата: 14.07.2024 …» в шапке КЗ (не «Дата рождения» и не «повторной явки»).
RE_CONSULT_DATE_SIMPLE = re.compile(
    r"(?im)^[ \t]*дата\s*[:\-]?\s*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})",
)
# Дата рождения на строке ФИО: «Ф.И.О: Иванов Иван Иванович, 12.07.1976».
RE_FIO_DOB = re.compile(
    r"(?:ф\.?\s*и\.?\s*о\.?|фио|пациент\w*)\s*[:\-]?[^\n]*?(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4})",
    re.I,
)
# ФИО пациента: «Ф.И.О: Кузавка Павел Леонидович».
RE_FIO_NAME = re.compile(
    r"(?:ф\.?\s*и\.?\s*о\.?|фио)\s*[:\-]?\s*"
    r"([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})",
    re.I,
)


def _join_section_text(a: str | None, b: str | None) -> str | None:
    parts = [p.strip() for p in (a, b) if p and p.strip()]
    return "\n".join(parts) if parts else None


_INLINE_ANAM_SPLIT = re.compile(
    r"(?i)(?:^|[\s.;])(?:анамнез|aнамнез)\s*[:\-]\s*",
)
_LIFE_ANAM_MARKERS = re.compile(
    r"(?i)аллергоанамнез|онкоанамнез|гемотрансфуз|оперативн\w*\s+вмешательств|"
    r"перен[её]с\w*\s+заболеван|хроническ\w*\s+заболеван|не\s+отягощен|"
    r"отрицает|курит|алкогол|вредн\w*\s+привычк",
)


def _refine_complaints_anamnesis_sections(sections: ConsultationSections) -> None:
    """Отделить жалобы от встроенного анамнеза (частый формат КЗ в одной строке)."""
    for extra_field in ("allergy_history", "surgical_history", "medication_history"):
        extra = getattr(sections, extra_field, None)
        if extra and str(extra).strip():
            sections.life_history = _join_section_text(sections.life_history, str(extra).strip())
            setattr(sections, extra_field, None)

    complaints = (sections.complaints or "").strip()
    if not complaints:
        return

    m = _INLINE_ANAM_SPLIT.search(complaints)
    if m:
        head = complaints[: m.start()].strip(" \n:;.-")
        tail = complaints[m.end() :].strip()
        sections.complaints = head or None
        if tail:
            if _LIFE_ANAM_MARKERS.search(tail) and not re.search(
                r"(?i)болеет|с\s+\d+\s+(?:дн|нед|мес|лет)|давност|развивал",
                tail,
            ):
                sections.life_history = _join_section_text(sections.life_history, tail)
            else:
                sections.anamnesis = _join_section_text(sections.anamnesis, tail)
        return

    if _LIFE_ANAM_MARKERS.search(complaints):
        cm = re.match(
            r"(?i)^(?:жалоб\w*\s*[:\-]\s*)?(.{5,120}?)(?:\s*(?:анамнез|аллерго|онко))",
            complaints,
        )
        if cm:
            sections.complaints = cm.group(1).strip(" .;:-") or None
            rest = complaints[cm.end() :].strip() or complaints[cm.start(1) + len(cm.group(1)) :].strip()
            if rest:
                sections.life_history = _join_section_text(sections.life_history, rest)


def _split_sections(text: str) -> dict[str, str]:
    """Разбивает текст на секции по распознанным заголовкам."""
    sections: dict[str, str] = {}
    matches = list(_HEADER_RE.finditer(text or ""))
    if not matches:
        return sections
    header_lookup = [(re.compile(h, re.I), field) for h, field in _SECTION_HEADERS]
    for i, m in enumerate(matches):
        head_text = m.group(1).lower()
        field = None
        for rx, fld in header_lookup:
            if rx.match(head_text):
                field = fld
                break
        if field is None:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip(" \n\t: - -")
        if field in sections and body:
            sections[field] = (sections[field] + "\n" + body).strip()
        elif body:
            sections[field] = body
    return sections


def _exam_items_from_text(text: str, status: str) -> list[ExamItem]:
    out: list[ExamItem] = []
    if not text:
        return out
    idx = 0
    for raw in re.split(r"[\n;,]+", text):
        line = raw.strip(" \t-•.\u2013\u2014")
        if len(line) < 3:
            continue
        idx += 1
        out.append(
            ExamItem(
                exam_id=f"ex_{status}_{idx}",
                exam_name=line[:200],
                status=status,  # type: ignore[arg-type]
                source_section=status,
            )
        )
    return out


def _extract_follow_up_from_text(text: str) -> tuple[str, list[FollowUpItem]]:
    """Убирает строки контрольной явки из блока рекомендаций."""
    if not text:
        return text, []
    kept: list[str] = []
    items: list[FollowUpItem] = []
    fu_idx = 0
    for raw in re.split(r"[\n]+", text):
        line = raw.strip()
        if not line:
            continue
        m = RE_FOLLOW_UP_INLINE.search(line)
        if m:
            fu_idx += 1
            fu_date = parse_date(m.group(1) or line)
            items.append(
                FollowUpItem(
                    follow_up_id=f"fu_inline_{fu_idx}",
                    raw_text=line[:300],
                    date=fu_date,
                    source_section="general_recommendations",
                )
            )
            continue
        kept.append(line)
    return "\n".join(kept), items


def _detect_specialty(text: str) -> str | None:
    m = RE_SPECIALTY.search(text or "")
    return m.group(0).strip() if m else None


def parse_consultation(
    raw_text: str,
    *,
    consultation_id: str = "consult",
    source_file: str = "",
    source_file_type: str = "",
    demographics_meta: dict[str, Any] | None = None,
) -> ConsultationDocument:
    """Главная функция: сырой текст КЗ → ConsultationDocument."""
    text = raw_text or ""
    warnings: list[str] = []
    errors: list[str] = []

    sections_map = _split_sections(text)
    sections = ConsultationSections(**{
        k: v for k, v in sections_map.items()
        if k in ConsultationSections.model_fields
    })
    _refine_complaints_anamnesis_sections(sections)

    # --- метаданные врача/клиники/даты ---
    demo = demographics_meta or {}
    consult_date: _dt.date | None = None
    mcd = RE_CONSULT_DATE.search(text)
    if mcd:
        consult_date = parse_date(mcd.group(1))
    if consult_date is None:
        for ms in RE_CONSULT_DATE_SIMPLE.finditer(text[:1200]):
            line_start = text.rfind("\n", 0, ms.start()) + 1
            line = text[line_start:ms.end()].lower()
            if "рожд" in line or "явк" in line or "повтор" in line:
                continue
            consult_date = parse_date(ms.group(1))
            if consult_date is not None:
                break
    if consult_date is None and demo.get("consultation_date"):
        consult_date = demo.get("consultation_date")
    consult_dt = None
    t = parse_time(text[:600])
    if consult_date and t:
        consult_dt = _dt.datetime.combine(consult_date, t)

    specialty = _detect_specialty(text[:1500]) or _detect_specialty(text)
    mdoc = RE_DOCTOR_NAME.search(text)
    doctor_name = mdoc.group(1) if mdoc else None
    mclinic = RE_CLINIC.search(text[:1000])
    clinic = mclinic.group(1).strip() if mclinic else None

    # --- демография ---
    birth_date = demo.get("birth_date") or asr.parse_birth_date(text)
    if birth_date is None:
        mfio = RE_FIO_DOB.search(text)
        if mfio:
            cand = parse_date(mfio.group(1))
            # ДР должна быть в прошлом относительно консультации (или просто исторической).
            if cand is not None and (consult_date is None or cand < consult_date):
                birth_date = cand
    age_info = asr.resolve_age(
        text, birth_date=birth_date, consultation_date=consult_date,
    )
    warnings.extend(age_info["warnings"])
    full_name = demo.get("full_name")
    if not full_name:
        mfn0 = RE_FIO_NAME.search(text)
        if mfn0:
            full_name = mfn0.group(1).strip()
    # Приоритет: явные данные → явный маркер пола в тексте → отчество пациента.
    sex = demo.get("sex")
    if sex not in ("male", "female"):
        sex = asr.detect_sex(text)
    if sex not in ("male", "female"):
        sex = asr.detect_sex_from_name(full_name)
    pregnancy = asr.detect_pregnancy(text)

    patient = PatientContext(
        full_name=full_name,
        birth_date=birth_date,
        age_years=age_info["age_years"],
        age_months=age_info["age_months"],
        sex=sex if sex in ("male", "female") else "unknown",
        age_group=age_info["age_group"],
        adult_or_child=age_info["adult_or_child"],
        pregnancy=pregnancy if pregnancy else None,
    )

    # --- диагнозы ---
    diag_block = sections.diagnosis_text or ""
    diagnoses = parse_diagnoses(diag_block) if diag_block else []
    # Подставляем код болезни из словаря, если в тексте диагноза кода нет
    # (иначе эвристика может дать симптом-код вроде R21.9 и увести подбор КП).
    from .diagnosis_icd import lookup_disease_icd

    for d in diagnoses:
        if not d.icd10_code:
            lex = lookup_disease_icd(d.diagnosis_name or d.raw_text)
            if lex:
                d.icd10_code = lex[0]
    if not diagnoses:
        # fallback: словарь нозологий по всему тексту, затем ICD из текста.
        # Приоритет - коды болезней, симптом-коды (R..) уходят в конец.
        from .diagnosis_icd import prioritize_codes

        codes = prioritize_codes(lookup_disease_icd(text[:120_000]) + extract_icd10(text[:120_000]))
        if codes:
            from .consult_schema import ConsultationDiagnosis

            diagnoses = [
                ConsultationDiagnosis(
                    diagnosis_id="dx1",
                    raw_text=f"(ICD из текста) {codes[0]}",
                    icd10_code=codes[0],
                    source_section="auto_icd",
                )
            ]

    # --- обследования ---
    performed = _exam_items_from_text(sections.exam_results or "", "performed")
    recommended = _exam_items_from_text(sections.recommendations_exams or "", "recommended")

    # --- лекарства ---
    inline_follow_up: list[FollowUpItem] = []
    med_parts: list[str] = []
    for field in ("recommendations_treatment", "general_recommendations"):
        chunk = getattr(sections, field)
        if not chunk:
            continue
        cleaned, fu = _extract_follow_up_from_text(chunk)
        setattr(sections, field, cleaned or None)
        inline_follow_up.extend(fu)
        if cleaned:
            med_parts.append(cleaned)
    med_text = "\n".join(med_parts)
    medications = parse_medications(med_text) if med_text else []

    # --- повторная явка ---
    follow_up: list[FollowUpItem] = list(inline_follow_up)
    if sections.follow_up_text:
        fu_date = parse_date(sections.follow_up_text)
        follow_up.append(
            FollowUpItem(
                follow_up_id="fu1",
                raw_text=sections.follow_up_text[:300],
                date=fu_date,
                source_section="follow_up_text",
            )
        )

    # --- автошаблонные блоки ---
    template_blocks = parse_template_blocks(text)

    # --- качество ---
    low = text.lower()
    has_undefined = "undefined" in low
    has_qmark = any(d.certainty == "suspected" for d in diagnoses) or "?" in diag_block
    if has_undefined:
        warnings.append("Обнаружено значение 'undefined' - проблема качества документа.")
    if birth_date is None:
        warnings.append("Дата рождения не распознана.")
    if consult_date is None:
        warnings.append("Дата консультации не распознана.")
    if specialty is None:
        warnings.append("Специальность врача не распознана.")

    parsed_count = sum(
        1 for f in ConsultationSections.model_fields
        if getattr(sections, f)
    )
    confidence = round(min(1.0, 0.2 + 0.1 * parsed_count), 3) if parsed_count else 0.1

    quality = ExtractionQuality(
        raw_text_length=len(text),
        parsed_sections_count=parsed_count,
        confidence=confidence,
        warnings=warnings,
        errors=errors,
        has_undefined=has_undefined,
        has_question_mark_diagnosis=bool(has_qmark),
        has_unparsed_medication_schedule=False,
        has_missing_birth_date=birth_date is None,
        has_missing_consultation_date=consult_date is None,
        has_missing_doctor_specialty=specialty is None,
    )

    return ConsultationDocument(
        consultation_id=consultation_id,
        source_file=source_file,
        source_file_type=source_file_type,
        raw_text=text,
        clinic_name=clinic,
        doctor_specialty=specialty,
        doctor_name=doctor_name,
        consultation_date=consult_date,
        consultation_datetime=consult_dt,
        patient=patient,
        sections=sections,
        diagnoses=diagnoses,
        medications=medications,
        planned_exams=recommended,
        performed_exams=performed,
        follow_up=follow_up,
        template_blocks=template_blocks,
        extraction_quality=quality,
    )
