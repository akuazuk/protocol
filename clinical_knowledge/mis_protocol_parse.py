"""Парсер mis_protocol.result по схеме epam/scheme_mis_protocols.docx.

Индекс N в схеме = parts[N] после split('::').
Поле 22 (диагноз) дополнительно: split('##'), [0]=список через '|', [3]=индекс основного.
"""
from __future__ import annotations

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
    out["result_slots"] = str(len(parts))
    return out
