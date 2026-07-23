"""Парсер mis_protocol.result по схеме epam/scheme_mis_protocols.docx.

Индекс N в схеме = parts[N] после split('::').
Поле 22 (диагноз) дополнительно: split('##'), [0]=список через '|', [3]=индекс основного.

Плюс детерминированная классификация строки `classify_kz_kind`: КЗ / справка /
диагностика (УЗИ, рентген, функц., эндоскопия, лаборатория) / неклиническое / пустое.
Модуль намеренно на чистом stdlib (без clinical_knowledge.__init__), т.к. импортируется
экспортёром через importlib.
"""
from __future__ import annotations

import datetime as _dt
import re

# Индекс в ::-массиве → имя столбца (латиница, для parquet/csv).
RESULT_FIELD_MAP: dict[int, str] = {
    1: "visit_date_text",
    2: "specialty_id",
    3: "complaints",
    4: "objective_status",
    5: "clinical_diagnosis",
    6: "exam_recommendations",
    7: "doctor_id",
    8: "doctor_extra",
    10: "anamnesis_doctor",
    11: "exam_data",
    12: "complaints_print",
    13: "anamnesis_print",
    14: "objective_print",
    15: "exam_data_print",
    16: "diagnosis_print",
    17: "exam_recommendations_print",
    21: "visit_time",
    22: "diagnosis_structured_raw",
    23: "manipulations_print",
    24: "manipulations",
    25: "treatment_recommendations_print",
    26: "treatment_recommendations",
    27: "dispensary_print",
    28: "dispensary_info",
    35: "weight",
    36: "height",
    37: "bmi",
    38: "temperature",
    39: "bp_1",
    40: "bp_2",
    41: "anamnesis_auto",
    50: "return_date_print",
    51: "return_date",
    52: "disability_expertise",
    53: "sick_leave",
    54: "scabies_check",
    55: "pediculosis_check",
    56: "ped_scab_abs",
    60: "resp_rate",
    61: "heart_rate",
    62: "midlevel_exam",
    63: "hide_primary_repeat",
    102: "doctor2_id",
    103: "doctor2_extra",
}

# Максимальный индекс схемы: если слотов меньше - строка обрезана/битая (parse_ok=False).
RESULT_MAX_INDEX = max(RESULT_FIELD_MAP)

# --- Коды МКБ-10 в структурном диагнозе (слот 22) -------------------------
# Каждый диагноз в списке (`|`) имеет вид «K29.3. Хронический гастрит» - код в начале.
# МКБ-10: буква (кроме U) + 2 цифры + опц. «.цифры».
_RE_MKB_HEAD = re.compile(r"^\s*([A-TV-Z][0-9]{2}(?:\.[0-9]{1,2})?)\b")
_RE_MKB_ANY = re.compile(r"\b([A-TV-Z][0-9]{2}(?:\.[0-9]{1,2})?)\b")


def extract_mkb_code(text: str | None) -> str:
    """Первый код МКБ-10 из строки диагноза (предпочтительно в начале)."""
    s = (text or "").strip()
    if not s:
        return ""
    m = _RE_MKB_HEAD.match(s)
    if m:
        return m.group(1).upper()
    m = _RE_MKB_ANY.search(s)
    return m.group(1).upper() if m else ""


def _diagnosis_entries(diagnosis_list: str | None) -> list[str]:
    return [p.strip() for p in (diagnosis_list or "").split("|") if p.strip()]


# --- Нормализация дат визита ----------------------------------------------
_RE_DMY = re.compile(r"^\s*(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})")
_RE_ISO = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})")


def to_iso_date(value) -> str:
    """Дата в ISO 'YYYY-MM-DD' из date/datetime, ДД.ММ.ГГГГ или ISO-строки. Иначе ''."""
    if value is None:
        return ""
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.strftime("%Y-%m-%d")
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    m = _RE_ISO.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = _RE_DMY.match(s)
        if not m:
            return ""
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y = 2000 + y if y <= 69 else 1900 + y
    try:
        return _dt.date(y, mo, d).isoformat()
    except ValueError:
        return ""


# Клинические поля КЗ (для анализа / consult-review).
KZ_CORE_FIELDS: tuple[str, ...] = (
    "specialty_id",
    "visit_date_text",
    "visit_time",
    "complaints",
    "anamnesis_doctor",
    "anamnesis_auto",
    "objective_status",
    "exam_data",
    "manipulations",
    "clinical_diagnosis",
    "diagnosis_list",
    "diagnosis_main_index",
    "exam_recommendations",
    "treatment_recommendations",
    "doctor_id",
    "doctor_extra",
    "return_date",
)


# --- Классификация строки: КЗ vs не-КЗ ------------------------------------

# Диагностические / лабораторные специальности: их протоколы - НЕ КЗ, не оцениваем.
_DIAGNOSTIC_RE = re.compile(
    r"(ультразвук|рентген|лучев|эндоскоп|гастроскоп|колоноскоп|"
    r"функциональн\w*\s+диагност|лаборат|цитолог|патоморфолог|"
    r"патологоанат|гистолог)",
    re.IGNORECASE,
)

# Неклинические роли (нет врачебного заключения).
_NON_CLINICAL_RE = re.compile(
    r"(стоматолог|зубн|медсестр|медицинск\w*\s+сестр|логопед|фельдшер|регистратор)",
    re.IGNORECASE,
)

# Только пунктуация / пробелы / прочерки - пустая специальность.
_ONLY_DASH_RE = re.compile(r"^[\s\-\u2013\u2014\u2212\u2011._]+$")

_EMPTY_FIELD_TOKENS = frozenset({"", "on", "off", "0", "1", "nan", "none", "null"})


def is_diagnostic_specialty(name: str | None) -> bool:
    """True для диагностических/лабораторных специальностей (УЗИ, рентген, лаба и т.п.)."""
    return bool(_DIAGNOSTIC_RE.search((name or "").strip().lower()))


def _field_nonempty(row: dict, *names: str) -> bool:
    for name in names:
        v = str(row.get(name) or "").strip().lower()
        if v and v not in _EMPTY_FIELD_TOKENS:
            return True
    return False


def _norm_pay_code(raw) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return s


# Валидные значения kz_kind (в оценку идут только KZ_SCORED_KINDS).
KZ_SCORED_KINDS: frozenset[str] = frozenset({"kz", "certificate"})


def classify_kz_kind(row: dict) -> tuple[str, str]:
    """Определить тип строки mis_protocol и причину.

    Возвращает (kz_kind, reason_ru):
      - "kz"           клиническое консультативное заключение (оцениваем);
      - "certificate"  справка/профосмотр pay_type=12 (оцениваем отдельной рубрикой);
      - "diagnostic"   УЗИ/рентген/функц./эндоскопия/лаборатория (НЕ оцениваем);
      - "non_clinical" медсестра/стоматология/логопед/пустая спец. (НЕ оцениваем);
      - "empty"        нет клинического содержания (НЕ оцениваем).

    Порядок: диагностика → неклиническое → справка → контентный guard → КЗ.
    Опирается на doctor_specialization (автор), pay_type и клинические поля строки.
    """
    spec = str(row.get("doctor_specialization") or "").strip()
    low = spec.lower()

    if is_diagnostic_specialty(spec):
        return "diagnostic", f"диагностическая специальность: {spec}"
    if _NON_CLINICAL_RE.search(low):
        return "non_clinical", f"неклиническая специальность: {spec}"
    if not spec or _ONLY_DASH_RE.match(spec):
        # Специальность неизвестна - решаем по содержанию (guard ниже),
        # но помечаем как неклиническое, если содержания тоже нет.
        spec = ""

    if _norm_pay_code(row.get("pay_type")) == "12":
        return "certificate", "справка/профосмотр (pay_type=12)"

    has_dx = _field_nonempty(row, "clinical_diagnosis", "diagnosis_list")
    has_subjective = _field_nonempty(
        row, "complaints", "anamnesis_doctor", "anamnesis_auto", "objective_status"
    )
    has_plan = _field_nonempty(row, "exam_recommendations", "treatment_recommendations")
    if not (has_dx or has_subjective or has_plan):
        return "empty", "нет клинического содержания (диагноз/жалобы/статус/рекомендации пусты)"
    if not spec:
        return "kz", "специальность не распознана; есть клиническое содержание"
    return "kz", ""


def parse_result(result: str | None) -> dict[str, str]:
    """Развернуть result в именованные поля + разбор диагноза ##."""
    parts = (result or "").split("::")
    out: dict[str, str] = {}
    for idx, name in RESULT_FIELD_MAP.items():
        out[name] = parts[idx].strip() if idx < len(parts) else ""
    raw = out.get("diagnosis_structured_raw") or ""
    if "##" in raw:
        dparts = raw.split("##")
        out["diagnosis_list"] = (dparts[0] if dparts else "").strip()
        out["diagnosis_main_index"] = (dparts[3] if len(dparts) > 3 else "").strip()
    else:
        out["diagnosis_list"] = ""
        out["diagnosis_main_index"] = ""

    # Коды МКБ-10 из структурного диагноза (inline в тексте каждого диагноза).
    entries = _diagnosis_entries(out.get("diagnosis_list"))
    codes = [c for c in (extract_mkb_code(e) for e in entries) if c]
    out["mkb_codes"] = "|".join(codes)
    # Основной диагноз по индексу (слот 22, ##[3]); fallback - первый.
    main_idx = 0
    idx_raw = (out.get("diagnosis_main_index") or "").strip()
    if idx_raw.isdigit():
        main_idx = int(idx_raw)
    if entries and 0 <= main_idx < len(entries):
        out["diagnosis_main_text"] = entries[main_idx]
        out["mkb_code_main"] = extract_mkb_code(entries[main_idx])
    elif entries:
        out["diagnosis_main_text"] = entries[0]
        out["mkb_code_main"] = extract_mkb_code(entries[0])
    else:
        out["diagnosis_main_text"] = ""
        out["mkb_code_main"] = ""

    out["result_slots"] = str(len(parts))
    # Строка считается корректно разобранной, если все поля схемы присутствуют
    # (в живых КЗ ~300 слотов; < RESULT_MAX_INDEX+1 => обрезано/битый ::).
    out["parse_ok"] = "1" if len(parts) > RESULT_MAX_INDEX else "0"
    return out
