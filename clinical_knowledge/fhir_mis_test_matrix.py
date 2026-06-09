"""Матрица проверок готовности FHIR-пакета к ЦИСЗ.

Источники:
- data/Программа испытаний МИС v.1.3-4 - Амбулаторный профиль.pdf (содержимое сценариев)
- data/Протокол информационного взаимодействия МИС ОЗ с ЦИСЗ_v_1.4.pdf (структура пакета)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ScenarioId = Literal["primary_ambulatory", "specialist_consult", "medication", "auto"]


@dataclass(frozen=True)
class MisCheckDef:
    check_id: str
    title_ru: str
    weight: float
    table_ref: str
    critical: bool = False


# 3.2.1 – первичный амбулаторный приём (табл. 12–17)
PRIMARY_AMBULATORY_CHECKS: tuple[MisCheckDef, ...] = (
    MisCheckDef("patient", "Пациент (Patient)", 8, "3.2.1 / Patient"),
    MisCheckDef("encounter", "Визит (EncounterGeneral)", 10, "табл. 17", critical=True),
    MisCheckDef("encounter_completed", "Статус визита completed", 5, "табл. 17"),
    MisCheckDef("encounter_period", "Период визита actualPeriod", 5, "табл. 17"),
    MisCheckDef("encounter_participant", "Участник визита (врач)", 5, "табл. 17"),
    MisCheckDef("encounter_diagnosis_link", "Ссылка визита на диагноз", 7, "табл. 17", critical=True),
    MisCheckDef("complaints", "Жалобы (ObservationSubjective)", 10, "табл. 12"),
    MisCheckDef("vitals", "Жизненно важные показатели (VitalSignsBy)", 10, "табл. 14"),
    MisCheckDef("anthropometrics", "Антропометрия (рост/вес/ИМТ)", 5, "табл. 13"),
    MisCheckDef("objective_exam", "Объективный осмотр по системам", 10, "табл. 15"),
    MisCheckDef("diagnosis_icd10", "Код МКБ-10 в FinalDiagnosis", 12, "табл. 16", critical=True),
    MisCheckDef("diagnosis_final_kind", "Вид диагноза: заключительный", 10, "табл. 16", critical=True),
    MisCheckDef("diagnosis_clinical_status", "Клинический статус диагноза", 3, "табл. 16"),
    MisCheckDef("diagnosis_author", "Автор диагноза (PractitionerRole)", 5, "табл. 16"),
    MisCheckDef("bundle_links", "Связи Patient–Encounter–Condition", 5, "3.2.1"),
)

# 3.13.1 – консультация специалиста (табл. 118, 125–126)
SPECIALIST_CONSULT_CHECKS: tuple[MisCheckDef, ...] = (
    MisCheckDef("patient", "Пациент (Patient)", 10, "3.13.1"),
    MisCheckDef("service_request_consult", "Направление ServiceRequestConsult", 20, "табл. 118", critical=True),
    MisCheckDef("encounter", "Визит консультации (EncounterGeneral)", 15, "табл. 125", critical=True),
    MisCheckDef("encounter_referral_link", "Визит ссылается на направление", 15, "табл. 125", critical=True),
    MisCheckDef("encounter_completed", "Статус визита completed", 5, "табл. 125"),
    MisCheckDef("encounter_participant", "Врач-консультант", 5, "табл. 125"),
    MisCheckDef("diagnosis_icd10", "Код МКБ-10", 15, "табл. 126", critical=True),
    MisCheckDef("diagnosis_final_kind", "Заключительный диагноз", 10, "табл. 126", critical=True),
    MisCheckDef("diagnosis_author", "Автор диагноза", 5, "табл. 126"),
)

# Дополнительно при наличии назначений (3.3)
MEDICATION_EXTRA_CHECKS: tuple[MisCheckDef, ...] = (
    MisCheckDef("medication_request", "Назначение лекарства (MedicationRequest)", 15, "3.3"),
    MisCheckDef("medication_encounter_link", "Назначение привязано к визиту", 10, "табл. 49"),
)

# Протокол взаимодействия МИС ОЗ – ЦИСЗ v.1.4 (§4.1, §5.1.1, §5.2.2) — обёртка пакета
PROTOCOL_V14_BUNDLE_CHECKS: tuple[MisCheckDef, ...] = (
    MisCheckDef("bundle_type_document", "Bundle.type = document", 8, "ПИ МИС–ЦИСЗ §5.1.1", critical=True),
    MisCheckDef(
        "bundle_profile_package",
        "meta.profile пакета (MedicationDocument / пакет МИО)",
        7,
        "ПИ v1.4 §4.1",
        critical=True,
    ),
    MisCheckDef("bundle_identifier", "Идентификатор пакета/документа", 4, "ПИ v1.4 §5.1.1"),
    MisCheckDef("bundle_timestamp", "Bundle.timestamp (дата формирования)", 3, "ПИ v1.4 §5.1.1"),
)

PROTOCOL_V14_COMPOSITION_CHECKS: tuple[MisCheckDef, ...] = (
    MisCheckDef("composition_first_entry", "Composition — первый entry Bundle", 10, "ПИ v1.4 §5.1.1", critical=True),
    MisCheckDef("composition_present", "Ресурс Composition (CompDocument)", 8, "ПИ v1.4 §4.1", critical=True),
    MisCheckDef("composition_status", "Composition.status", 3, "ПИ v1.4"),
    MisCheckDef("composition_type", "Composition.type (НСИ CompositionType)", 5, "ПИ v1.4"),
    MisCheckDef("composition_subject", "Composition.subject → Patient", 7, "ПИ v1.4", critical=True),
    MisCheckDef("composition_encounter", "Composition.encounter → Encounter", 5, "ПИ v1.4"),
    MisCheckDef("composition_author", "Composition.author (медработник)", 8, "ПИ v1.4 §5.2.2", critical=True),
    MisCheckDef("composition_custodian", "Composition.custodian (ОЗ)", 8, "ПИ v1.4 §5.2.2", critical=True),
    MisCheckDef("composition_date", "Composition.date (дата составления)", 5, "ПИ v1.4 §5.1.1"),
    MisCheckDef("composition_event_links", "Composition.event → Condition/Observation", 5, "ПИ v1.4 / fhir.by"),
)

PROTOCOL_V14_CHECKS: tuple[MisCheckDef, ...] = (
    PROTOCOL_V14_BUNDLE_CHECKS + PROTOCOL_V14_COMPOSITION_CHECKS
)

SCENARIO_CHECKS: dict[str, tuple[MisCheckDef, ...]] = {
    "primary_ambulatory": PRIMARY_AMBULATORY_CHECKS,
    "specialist_consult": SPECIALIST_CONSULT_CHECKS,
    "medication": PRIMARY_AMBULATORY_CHECKS + MEDICATION_EXTRA_CHECKS,
}


def checks_for_scenario(
    scenario: str,
    *,
    include_medication: bool = False,
    include_protocol_v14: bool = False,
) -> tuple[MisCheckDef, ...]:
    if scenario == "specialist_consult":
        base: tuple[MisCheckDef, ...] = SPECIALIST_CONSULT_CHECKS
    elif scenario == "medication":
        base = SCENARIO_CHECKS["medication"]
    else:
        base = PRIMARY_AMBULATORY_CHECKS
        if include_medication:
            base = base + MEDICATION_EXTRA_CHECKS
    if include_protocol_v14:
        return PROTOCOL_V14_CHECKS + base
    return base
