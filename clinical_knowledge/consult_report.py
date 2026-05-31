"""Генерация отчётов по оценке КЗ: JSON и Markdown (ТЗ раздел 20).

Каждый блок отчёта опирается на проверяемые данные (assessments + source_refs).
"""
from __future__ import annotations

from typing import Any

from .consult_schema import ComplianceReport, ConsultationDocument, SourceRef


def _src_line(ref: SourceRef) -> str:
    parts: list[str] = []
    if ref.local_path:
        parts.append(ref.local_path)
    if ref.section_title:
        parts.append(f"раздел: {ref.section_title}")
    if ref.page_start:
        pg = f"с. {ref.page_start}"
        if ref.page_end and ref.page_end != ref.page_start:
            pg += f"-{ref.page_end}"
        parts.append(pg)
    if ref.quote:
        parts.append(f"«{ref.quote[:160]}»")
    return " — ".join(parts) if parts else (ref.protocol_id or "источник не указан")


def report_to_json(
    report: ComplianceReport,
    doc: ConsultationDocument | None = None,
) -> dict[str, Any]:
    """JSON-отчёт по ТЗ раздел 20."""
    patient_summary: dict[str, Any] = {}
    doctor_specialty = None
    consultation_date = None
    if doc is not None:
        patient_summary = {
            "age_years": doc.patient.age_years,
            "sex": doc.patient.sex,
            "adult_or_child": doc.patient.adult_or_child,
            "pregnancy": doc.patient.pregnancy,
        }
        doctor_specialty = doc.doctor_specialty
        consultation_date = doc.consultation_date.isoformat() if doc.consultation_date else None

    return {
        "consultation_id": report.consultation_id,
        "patient_summary": patient_summary,
        "doctor_specialty": doctor_specialty,
        "consultation_date": consultation_date,
        "matched_protocols": [m.model_dump(mode="json") for m in report.protocol_matches],
        "score_breakdown": report.score_breakdown.model_dump(mode="json"),
        "overall_status": report.overall_status,
        "critical_issues": [i.model_dump(mode="json") for i in report.critical_issues],
        "warnings": [i.model_dump(mode="json") for i in report.warnings],
        "missing_required_items": [i.model_dump(mode="json") for i in report.missing_required_items],
        "diagnosis_assessments": [a.model_dump(mode="json") for a in report.diagnosis_assessments],
        "exam_assessments": [a.model_dump(mode="json") for a in report.exam_assessments],
        "treatment_assessments": [a.model_dump(mode="json") for a in report.treatment_assessments],
        "safety_assessments": [a.model_dump(mode="json") for a in report.safety_assessments],
        "source_refs": [r.model_dump(mode="json") for r in report.source_refs],
    }


def _fmt_pct(v: float | None) -> str:
    return f"{v:.0f}%" if isinstance(v, (int, float)) else "—"


def report_to_markdown(
    report: ComplianceReport,
    doc: ConsultationDocument | None = None,
) -> str:
    """Markdown-отчёт по структуре ТЗ раздел 20 (8 разделов)."""
    L: list[str] = []
    bd = report.score_breakdown

    L.append("# Оценка консультативного заключения")
    L.append("")
    # 1. Резюме
    L.append("## 1. Краткое резюме")
    if doc is not None:
        p = doc.patient
        L.append(f"- Пациент: {doc.patient.full_name or '—'}")
        L.append(f"- Возраст: {p.age_years if p.age_years is not None else '—'} ({p.age_group})")
        L.append(f"- Пол: {p.sex}")
        L.append(f"- Дата консультации: {doc.consultation_date.isoformat() if doc.consultation_date else '—'}")
        L.append(f"- Специальность врача: {doc.doctor_specialty or '—'}")
        diags = ", ".join(
            f"{d.icd10_code or ''} {d.diagnosis_name or d.raw_text}".strip()
            for d in doc.diagnoses
        ) or "—"
        L.append(f"- Основные диагнозы: {diags}")
    protos = ", ".join(
        f"{m.document_title or m.protocol_id} [{m.applicability}]"
        for m in report.protocol_matches[:5]
    ) or "—"
    L.append(f"- Подобранные протоколы: {protos}")
    L.append(f"- Общая оценка: {_fmt_pct(report.overall_score)} — статус **{report.overall_status}**")
    L.append("")

    # 2. Применимость протокола
    L.append("## 2. Применимость протокола")
    if report.protocol_matches:
        for m in report.protocol_matches[:5]:
            L.append(f"- **{m.document_title or m.protocol_id}** — {m.applicability}")
            for r in m.match_reasons:
                L.append(f"  - + {r}")
            for r in m.mismatch_reasons:
                L.append(f"  - − {r}")
    else:
        L.append("- Подходящие протоколы не подобраны.")
    L.append("")

    # 3. Диагноз
    L.append("## 3. Оценка диагноза")
    L.append(f"_Балл блока: {_fmt_pct(bd.diagnosis_score)}_")
    for a in report.diagnosis_assessments:
        L.append(f"- {a.icd10_code or ''} {a.diagnosis_text} — **{a.status}**")
        for e in a.evidence_found:
            L.append(f"  - подтверждено: {e}")
        for e in a.evidence_missing:
            L.append(f"  - не хватает: {e}")
    L.append("")

    # 4. Обследования
    L.append("## 4. Оценка обследований")
    L.append(f"_Балл блока: {_fmt_pct(bd.required_exams_score)}_")
    if report.exam_assessments:
        for e in report.exam_assessments:
            L.append(f"- {e.exam_name} — **{e.status}**" + (f" ({e.reason})" if e.reason else ""))
    else:
        L.append("- Детерминированные правила по обследованиям не сработали (нет данных).")
    L.append("")

    # 5. Лечение
    L.append("## 5. Оценка лечения")
    L.append(f"_Балл блока: {_fmt_pct(bd.treatment_score)}_")
    if report.treatment_assessments:
        for t in report.treatment_assessments:
            L.append(f"- {t.treatment_text} — **{t.status}**")
            for iss in t.issues:
                L.append(f"  - {iss.message_ru}")
    else:
        L.append("- Назначения не распознаны.")
    L.append("")

    # 6. Red flags
    L.append("## 6. Красные флаги и безопасность")
    L.append(f"_Балл блока: {_fmt_pct(bd.safety_score)}_")
    if report.safety_assessments:
        for s in report.safety_assessments:
            L.append(f"- [{s.severity}] {s.finding_text} — **{s.status}**")
            if s.expected_action:
                L.append(f"  - ожидаемые действия: {s.expected_action}")
    else:
        L.append("- Красные флаги не обнаружены.")
    if report.overall_status == "manual_review_required":
        L.append("- ⚠️ Требуется ручное рассмотрение (критический red flag без маршрутизации).")
    L.append("")

    # 7. Качество КЗ
    L.append("## 7. Качество оформления КЗ")
    L.append(f"_Балл блока: {_fmt_pct(bd.documentation_quality_score)}_")
    sq = report.section_quality
    if sq.missing_sections:
        L.append(f"- Пропущенные разделы: {', '.join(sq.missing_sections)}")
    if sq.suspicious_placeholders:
        L.append(f"- Placeholder-значения: {', '.join(sq.suspicious_placeholders)}")
    if sq.extraction_warnings:
        for w in sq.extraction_warnings:
            L.append(f"- ⚠ {w}")
    if not (sq.missing_sections or sq.suspicious_placeholders or sq.extraction_warnings):
        L.append("- Замечаний по оформлению нет.")
    L.append("")

    # 8. Источники
    L.append("## 8. Ссылки на источники")
    refs = report.source_refs or []
    if refs:
        for r in refs:
            L.append(f"- {_src_line(r)}")
    else:
        L.append("- Источники не указаны.")
    L.append("")

    L.append(
        "> Оценка ориентировочная и не заменяет врача. "
        "Документ — инструмент экспертной проверки соответствия клиническим протоколам."
    )
    return "\n".join(L)
