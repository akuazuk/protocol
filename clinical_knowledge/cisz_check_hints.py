"""Пояснения к проверкам cisz_readiness для врача и интегратора МИС."""
from __future__ import annotations

from typing import Any

# layer: A = обёртка Bundle, B = Composition v1.4, C = ресурсы приёма (программа испытаний)
CHECK_HINTS: dict[str, dict[str, str]] = {
    "bundle_type_document": {
        "layer": "A",
        "layer_label_ru": "Обёртка пакета",
        "mis_field_ru": "Формирует МИС автоматически",
        "explain_ru": "Пакет должен иметь type=document, иначе ЦИСЗ не примет его как электронный документ.",
        "fix_ru": "Проверить настройку экспорта Bundle в МИС; при ошибке - к ИТ/интегратору.",
    },
    "bundle_profile_package": {
        "layer": "A",
        "layer_label_ru": "Обёртка пакета",
        "mis_field_ru": "Профиль пакета (MedicationDocument и др.)",
        "explain_ru": "В meta.profile указан профиль пакета по fhir.by / Протокол v.1.4 §4.1.",
        "fix_ru": "Обновить шаблон Bundle в МИС до актуального профиля раздела ЦИСЗ.",
    },
    "bundle_identifier": {
        "layer": "A",
        "layer_label_ru": "Обёртка пакета",
        "mis_field_ru": "Идентификатор документа",
        "explain_ru": "У пакета или Composition должен быть identifier для трассировки в ЦИСЗ.",
        "fix_ru": "Убедиться, что МИС присваивает id документа перед подписью.",
    },
    "bundle_timestamp": {
        "layer": "A",
        "layer_label_ru": "Обёртка пакета",
        "mis_field_ru": "Дата формирования пакета",
        "explain_ru": "Bundle.timestamp фиксирует момент сборки пакета.",
        "fix_ru": "Проверить генерацию timestamp при сохранении приёма.",
    },
    "composition_first_entry": {
        "layer": "B",
        "layer_label_ru": "Composition v1.4",
        "mis_field_ru": "Порядок ресурсов в пакете",
        "explain_ru": "Composition должен быть первым ресурсом в Bundle.entry (Протокол v.1.4 §5.1.1).",
        "fix_ru": "Исправить порядок entry в шаблоне экспорта МИС.",
    },
    "composition_present": {
        "layer": "B",
        "layer_label_ru": "Composition v1.4",
        "mis_field_ru": "Ресурс Composition",
        "explain_ru": "В пакете отсутствует Composition (CompDocument) - обложка электронного документа.",
        "fix_ru": "Включить Composition в экспорт; без него импорт невозможен.",
    },
    "composition_status": {
        "layer": "B",
        "layer_label_ru": "Composition v1.4",
        "mis_field_ru": "Статус документа",
        "explain_ru": "Composition.status (final, registered и т.д.) обязателен по профилю.",
        "fix_ru": "Заполнить статус при финализации приёма.",
    },
    "composition_type": {
        "layer": "B",
        "layer_label_ru": "Composition v1.4",
        "mis_field_ru": "Тип документа (CompositionType)",
        "explain_ru": "Тип из справочника НСИ CompositionType.",
        "fix_ru": "Выбрать корректный тип медицинского документа в МИС.",
    },
    "composition_subject": {
        "layer": "B",
        "layer_label_ru": "Composition v1.4",
        "mis_field_ru": "Пациент в документе",
        "explain_ru": "Composition.subject должен ссылаться на Patient.",
        "fix_ru": "Привязать документ к карте пациента в МИС.",
    },
    "composition_encounter": {
        "layer": "B",
        "layer_label_ru": "Composition v1.4",
        "mis_field_ru": "Визит в документе",
        "explain_ru": "Composition.encounter связывает документ с Encounter.",
        "fix_ru": "Убедиться, что приём сохранён как визит и связан с документом.",
    },
    "composition_author": {
        "layer": "B",
        "layer_label_ru": "Composition v1.4",
        "mis_field_ru": "Автор (подписывающий врач)",
        "explain_ru": "Composition.author - медработник; при импорте должен совпасть с OAuth-токеном.",
        "fix_ru": "Подписывать под своей учётной записью; author = вы.",
    },
    "composition_custodian": {
        "layer": "B",
        "layer_label_ru": "Composition v1.4",
        "mis_field_ru": "Организация (ОЗ)",
        "explain_ru": "Composition.custodian - организация здравоохранения; должна совпасть с токеном.",
        "fix_ru": "Проверить, что выбрано правильное ОЗ/филиал в МИС.",
    },
    "composition_date": {
        "layer": "B",
        "layer_label_ru": "Composition v1.4",
        "mis_field_ru": "Дата составления",
        "explain_ru": "Composition.date - дата составления документа.",
        "fix_ru": "Указать дату консультации / составления КЗ.",
    },
    "composition_event_links": {
        "layer": "B",
        "layer_label_ru": "Composition v1.4",
        "mis_field_ru": "Ссылки на события приёма",
        "explain_ru": "Composition.event.detail связывает документ с Condition и Observation.",
        "fix_ru": "Проверить, что диагноз и ключевые наблюдения включены в event.",
    },
    "patient": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Карта пациента, ИНП, ФИО, ДР, пол",
        "explain_ru": "Ресурс Patient или блок данных пациента в тексте КЗ.",
        "fix_ru": "Проверить ФИО, дату рождения, пол; в МИС - ИНП пациента.",
    },
    "encounter": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Визит / приём",
        "explain_ru": "EncounterGeneral фиксирует факт амбулаторного приёма (табл. 17, сценарий 3.2.1).",
        "fix_ru": "Завершить приём в МИС; в PDF - указать дату консультации.",
    },
    "encounter_completed": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Статус визита",
        "explain_ru": "Статус визита должен быть completed перед отправкой.",
        "fix_ru": "Закрыть визит (завершить приём), не оставлять черновик.",
    },
    "encounter_period": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Дата и время приёма",
        "explain_ru": "actualPeriod.start/end - когда проходил приём.",
        "fix_ru": "Указать дату и время консультации в форме.",
    },
    "encounter_participant": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Врач приёма",
        "explain_ru": "У визита указан участник - лечащий/консультирующий врач.",
        "fix_ru": "Указать ФИО и специальность врача в шапке КЗ.",
    },
    "encounter_diagnosis_link": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Диагноз визита",
        "explain_ru": "Encounter.diagnosis связывает визит с Condition - без связи пакет «без диагноза приёма».",
        "fix_ru": "Заполнить блок «Диагноз» и привязать к визиту в МИС.",
    },
    "encounter_referral_link": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Связь с направлением",
        "explain_ru": "Визит консультации должен ссылаться на ServiceRequest-направление (3.13.1).",
        "fix_ru": "Указать направление и связать с визитом консультации.",
    },
    "service_request_consult": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Направление на консультацию",
        "explain_ru": "ServiceRequestConsult - направление для сценария консультации специалиста.",
        "fix_ru": "Оформить направление в МИС или указать в тексте КЗ.",
    },
    "complaints": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Жалобы",
        "explain_ru": "ObservationSubjective или раздел «Жалобы» в КЗ.",
        "fix_ru": "Заполнить жалобы пациента своими словами, не оставлять пустым.",
    },
    "vitals": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "АД, ЧСС, температура и др.",
        "explain_ru": "VitalSignsBy - жизненно важные показатели (табл. 14).",
        "fix_ru": "Внести АД, пульс, температуру или отметить причину отсутствия.",
    },
    "anthropometrics": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Рост, вес, ИМТ",
        "explain_ru": "AnthropometricDataBy (табл. 13) - обязательно при первичном приёме.",
        "fix_ru": "Заполнить рост и вес; ИМТ может рассчитываться МИС.",
    },
    "objective_exam": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Объективный статус",
        "explain_ru": "ObservationObjective - осмотр по системам (табл. 15).",
        "fix_ru": "Описать объективный статус, не ограничиваться шаблоном «без изменений».",
    },
    "diagnosis_icd10": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Код МКБ-10",
        "explain_ru": "FinalDiagnosis с coding из справочника InternClassificDiseases10.",
        "fix_ru": "Указать код МКБ-10, согласованный с формулировкой диагноза.",
    },
    "diagnosis_final_kind": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Вид диагноза",
        "explain_ru": "Диагноз не должен быть предварительным («?») если отправляется как заключительный.",
        "fix_ru": "Уточнить диагноз или явно указать «предварительный» без отправки как final.",
    },
    "diagnosis_clinical_status": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Клинический статус диагноза",
        "explain_ru": "clinicalStatus Condition (active и т.д.).",
        "fix_ru": "Заполнить статус диагноза в структурированной форме МИС.",
    },
    "diagnosis_author": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Автор диагноза",
        "explain_ru": "Кто установил диагноз (PractitionerRole / recorder).",
        "fix_ru": "Указать врача, установившего диагноз.",
    },
    "bundle_links": {
        "layer": "C",
        "layer_label_ru": "Ресурсы приёма",
        "mis_field_ru": "Связи Patient-Encounter-Condition",
        "explain_ru": "Пациент, визит и диагноз должны быть связаны ссылками в пакете.",
        "fix_ru": "Проверить целостность данных в МИС перед экспортом.",
    },
    "medication_request": {
        "layer": "C",
        "layer_label_ru": "Назначения (3.3)",
        "mis_field_ru": "Лекарственные назначения",
        "explain_ru": "MedicationRequest при наличии назначений в приёме.",
        "fix_ru": "Оформить назначения структурно: препарат, доза, курс.",
    },
    "medication_encounter_link": {
        "layer": "C",
        "layer_label_ru": "Назначения (3.3)",
        "mis_field_ru": "Связь назначения с визитом",
        "explain_ru": "MedicationRequest.encounter ссылается на текущий визит.",
        "fix_ru": "Привязать назначение к приёму в МИС.",
    },
}


def enrich_check_item(
    check_id: str,
    *,
    passed: bool,
    source: str,
) -> dict[str, str]:
    """Добавляет пояснения к одной проверке."""
    hint = CHECK_HINTS.get(check_id, {})
    out: dict[str, str] = {
        "layer": hint.get("layer", ""),
        "layer_label_ru": hint.get("layer_label_ru", ""),
        "explain_ru": hint.get("explain_ru", ""),
        "fix_ru": hint.get("fix_ru", ""),
        "mis_field_ru": hint.get("mis_field_ru", ""),
    }
    if source == "text" and hint.get("layer") in ("A", "B"):
        out["explain_ru"] = (
            "При проверке PDF не оценивается - нужен FHIR Bundle из МИС. "
            + (hint.get("explain_ru") or "")
        )
    return out


def build_critical_gaps(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Список критических пробелов с расшифровкой."""
    gaps: list[dict[str, Any]] = []
    for chk in checks:
        if chk.get("passed") or not chk.get("critical"):
            continue
        gaps.append({
            "check_id": chk.get("check_id"),
            "title_ru": chk.get("title_ru"),
            "layer": chk.get("layer"),
            "layer_label_ru": chk.get("layer_label_ru"),
            "explain_ru": chk.get("explain_ru"),
            "fix_ru": chk.get("fix_ru"),
            "mis_field_ru": chk.get("mis_field_ru"),
            "table_ref": chk.get("table_ref"),
        })
    return gaps


def build_decode_ru(
    *,
    checks: list[dict[str, Any]],
    critical_gaps: list[dict[str, Any]],
    source: str,
    scenario_label_ru: str,
) -> str:
    """Текстовая расшифровка для UI."""
    failed = [c for c in checks if not c.get("passed")]
    if not failed and not critical_gaps:
        return ""

    parts: list[str] = []
    if source == "text":
        parts.append(
            "Оценка по тексту PDF (эвристика сценария «"
            + scenario_label_ru
            + "»). Слои A-B (Bundle/Composition) проверяются только при FHIR Bundle из МИС."
        )
    elif source == "fhir_bundle":
        parts.append(
            "Оценка по FHIR Bundle: слой A (пакет), B (Composition v1.4), C (ресурсы приёма)."
        )

    if critical_gaps:
        parts.append(
            "Критические пробелы ("
            + str(len(critical_gaps))
            + ") - высокий риск отклонения в ЦИСЗ или неверных данных в реестре:"
        )
        for g in critical_gaps:
            line = "• " + (g.get("title_ru") or g.get("check_id") or "")
            if g.get("mis_field_ru"):
                line += " - в МИС/КЗ: " + g["mis_field_ru"]
            if g.get("fix_ru"):
                line += ". " + g["fix_ru"]
            parts.append(line)

    other = [c for c in failed if not c.get("critical")]
    if other:
        parts.append("Также доработать (" + str(len(other)) + "):")
        for c in other[:6]:
            line = "• " + (c.get("title_ru") or "")
            if c.get("fix_ru"):
                line += ": " + c["fix_ru"]
            parts.append(line)
        if len(other) > 6:
            parts.append("• … ещё " + str(len(other) - 6) + " пунктов (см. таблицу ниже).")

    return "\n".join(parts)
