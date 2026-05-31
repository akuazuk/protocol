"""Генерация отчётов по оценке КЗ: JSON и Markdown (ТЗ раздел 20).

Каждый блок отчёта опирается на проверяемые данные (assessments + source_refs).
"""
from __future__ import annotations

from html import escape as _esc
from typing import Any

from .consult_schema import ComplianceReport, ConsultationDocument, SourceRef
from .privacy import name_to_initials

# Русские подписи статусов/значений для человекочитаемого отчёта.
_OVERALL_RU = {
    "compliant": "соответствует",
    "mostly_compliant": "в основном соответствует",
    "partially_compliant": "частично соответствует",
    "non_compliant": "не соответствует",
    "insufficient_data": "недостаточно данных",
    "manual_review_required": "нужен ручной разбор",
}
_DIAG_RU = {
    "supported": "подтверждён протоколом",
    "partially_supported": "частично подтверждён",
    "not_supported": "не подтверждён протоколом",
    "suspected_needs_confirmation": "предварительный, нужно подтверждение",
    "insufficient_data": "недостаточно данных",
    "not_assessed": "не оценивался",
}
_EXAM_RU = {
    "present_performed": "выполнено",
    "present_recommended": "рекомендовано",
    "missing_required": "пропущено обязательное",
    "missing_conditional": "пропущено по условию",
    "not_applicable": "не применимо",
    "extra_not_assessed": "дополнительно, не оценивалось",
}
_TREAT_RU = {
    "matches_protocol": "соответствует протоколу",
    "partially_matches_protocol": "частично соответствует протоколу",
    "not_in_protocol": "нет в протоколе",
    "dose_mismatch": "несовпадение дозы",
    "duration_mismatch": "несовпадение длительности",
    "frequency_mismatch": "несовпадение кратности",
    "age_contraindication": "противопоказано по возрасту",
    "contraindication_warning": "предупреждение о противопоказании",
    "insufficient_data": "недостаточно данных",
    "not_assessed": "не оценивалось",
}
_SAFETY_RU = {
    "handled": "учтён в рекомендациях",
    "partially_handled": "частично учтён",
    "not_handled": "не учтён",
    "not_assessed": "не оценивался",
}
_SEVERITY_RU = {
    "low": "низкая",
    "medium": "средняя",
    "high": "высокая",
    "critical": "критическая",
}
_SEX_RU = {"male": "мужской", "female": "женский", "unknown": "не определён"}

# Цвета статусов для HTML-бейджей.
_STATUS_COLOR = {
    "compliant": ("#0d7a4a", "#e6f7ef"),
    "mostly_compliant": ("#1a7f5a", "#e8f5f0"),
    "partially_compliant": ("#b45309", "#fef3c7"),
    "non_compliant": ("#b91c1c", "#fee2e2"),
    "insufficient_data": ("#64748b", "#f1f5f9"),
    "manual_review_required": ("#9333ea", "#f3e8ff"),
}
_SCORE_KEYS = (
    ("protocol_match_score", "Подбор протокола", "#2563eb"),
    ("diagnosis_score", "Диагноз", "#0d9488"),
    ("required_exams_score", "Обследования", "#7c3aed"),
    ("treatment_score", "Лечение", "#db2777"),
    ("safety_score", "Безопасность", "#dc2626"),
    ("documentation_quality_score", "Оформление", "#64748b"),
)


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


def _e(s: Any) -> str:
    return _esc(str(s)) if s is not None else ""


def _status_badge_html(status: str, label: str | None = None) -> str:
    fg, bg = _STATUS_COLOR.get(status, ("#334155", "#f8fafc"))
    text = _e(label or _OVERALL_RU.get(status, status))
    return (
        f'<span class="cr-badge" style="color:{fg};background:{bg};'
        f'border:1px solid {fg}33">{text}</span>'
    )


def _score_bars_html(bd: Any) -> str:
    rows: list[str] = []
    for key, title, color in _SCORE_KEYS:
        val = getattr(bd, key, None) if bd is not None else None
        if val is None and isinstance(bd, dict):
            val = bd.get(key)
        pct = float(val) if isinstance(val, (int, float)) else 0.0
        width = max(0, min(100, pct))
        display = _fmt_pct(val)
        rows.append(
            f'<div class="cr-bar-row"><span class="cr-bar-label">{_e(title)}</span>'
            f'<div class="cr-bar-track"><div class="cr-bar-fill" style="width:{width:.0f}%;'
            f'background:{color}"></div></div>'
            f'<span class="cr-bar-val">{display}</span></div>'
        )
    return "".join(rows)


def report_to_html(
    report: ComplianceReport,
    doc: ConsultationDocument | None = None,
    rubric_specifics: dict | None = None,
) -> str:
    """Цветной HTML-отчёт для UI (обезличенный: ФИО → инициалы)."""
    bd = report.score_breakdown
    status = report.overall_status
    status_ru = _OVERALL_RU.get(status, status)
    fg, bg = _STATUS_COLOR.get(status, ("#334155", "#f8fafc"))

    parts: list[str] = [
        '<article class="consult-report-html">',
        '<header class="cr-header" style="background:linear-gradient(135deg,#f0f9f6,#e8f4fc);'
        'border:1px solid #cfe0db;border-radius:12px;padding:1rem 1.1rem;margin-bottom:1rem">',
        '<h2 class="cr-title" style="margin:0 0 0.5rem;font-size:1.15rem;color:#0f3d36">'
        "Оценка консультативного заключения</h2>",
        '<div class="cr-header-meta" style="display:flex;flex-wrap:wrap;gap:0.6rem;align-items:center">',
        _status_badge_html(status, status_ru),
    ]
    if report.overall_score is not None:
        parts.append(
            f'<span class="cr-overall-pct" style="font-size:1.4rem;font-weight:700;color:{fg}">'
            f"{_fmt_pct(report.overall_score)}</span>"
        )
    parts.append("</div></header>")

    # Резюме
    parts.append('<section class="cr-section"><h3 class="cr-section-title">Краткое резюме</h3><ul class="cr-kv">')
    if doc is not None:
        p = doc.patient
        initials = name_to_initials(p.full_name)
        parts.append(f"<li><strong>Пациент:</strong> {_e(initials)}</li>")
        parts.append(
            f"<li><strong>Возраст:</strong> {p.age_years if p.age_years is not None else '—'} "
            f"({_e(p.age_group)})</li>"
        )
        parts.append(f"<li><strong>Пол:</strong> {_e(_SEX_RU.get(p.sex, p.sex))}</li>")
        cdate = doc.consultation_date.isoformat() if doc.consultation_date else "—"
        parts.append(f"<li><strong>Дата консультации:</strong> {_e(cdate)}</li>")
        parts.append(f"<li><strong>Специальность врача:</strong> {_e(doc.doctor_specialty or '—')}</li>")
        diags = ", ".join(
            _e(f"{d.icd10_code or ''} {d.diagnosis_name or d.raw_text}".strip())
            for d in doc.diagnoses
        ) or "—"
        parts.append(f"<li><strong>Диагнозы:</strong> {diags}</li>")
    protos = ", ".join(
        _e(m.document_title or m.protocol_id)
        for m in report.protocol_matches[:5]
    ) or "—"
    parts.append(f"<li><strong>Протоколы:</strong> {protos}</li>")
    parts.append("</ul></section>")

    # Баллы
    parts.append(
        '<section class="cr-section"><h3 class="cr-section-title">Баллы по блокам</h3>'
        f'<div class="cr-bars">{_score_bars_html(bd)}</div></section>'
    )

    # Диагнозы
    if report.diagnosis_assessments:
        parts.append('<section class="cr-section"><h3 class="cr-section-title">Диагноз</h3><ul class="cr-list">')
        for a in report.diagnosis_assessments:
            st = _DIAG_RU.get(a.status, a.status)
            code = a.icd10_code or ""
            txt = a.diagnosis_text or ""
            raw_label = txt if (code and txt.upper().startswith(code.upper())) else f"{code} {txt}".strip()
            label = _e(raw_label)
            parts.append(f"<li><span class='cr-icd'>{label}</span> — {_status_badge_html(a.status, st)}</li>")
        parts.append("</ul></section>")

    # Обследования
    parts.append('<section class="cr-section"><h3 class="cr-section-title">Обследования</h3>')
    if report.exam_assessments:
        parts.append("<ul class='cr-list'>")
        for e in report.exam_assessments:
            st = _EXAM_RU.get(e.status, e.status)
            parts.append(f"<li>{_e(e.exam_name)} — <strong>{_e(st)}</strong></li>")
        parts.append("</ul>")
    else:
        parts.append("<p class='cr-muted'>Детерминированные правила не сработали (нет данных).</p>")
    parts.append("</section>")

    # Лечение
    parts.append('<section class="cr-section"><h3 class="cr-section-title">Лечение</h3>')
    if report.treatment_assessments:
        parts.append("<ul class='cr-list'>")
        for t in report.treatment_assessments:
            st = _TREAT_RU.get(t.status, t.status)
            parts.append(f"<li>{_e(t.treatment_text)} — <strong>{_e(st)}</strong></li>")
        parts.append("</ul>")
    else:
        parts.append("<p class='cr-muted'>Назначения не распознаны.</p>")
    parts.append("</section>")

    # Безопасность
    parts.append('<section class="cr-section cr-section--safety"><h3 class="cr-section-title">Красные флаги</h3>')
    if report.safety_assessments:
        parts.append("<ul class='cr-list'>")
        for s in report.safety_assessments:
            sev = _SEVERITY_RU.get(s.severity, s.severity)
            st = _SAFETY_RU.get(s.status, s.status)
            parts.append(
                f"<li><span class='cr-sev cr-sev--{_e(s.severity)}'>[{_e(sev)}]</span> "
                f"{_e(s.finding_text)} — <strong>{_e(st)}</strong></li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p class='cr-muted'>Красные флаги не обнаружены.</p>")
    parts.append("</section>")

    # Рубрика
    if rubric_specifics:
        by_rubric = rubric_specifics.get("by_rubric") or {}
        measurements = rubric_specifics.get("measurements") or {}
        if by_rubric or measurements:
            parts.append('<section class="cr-section"><h3 class="cr-section-title">Профиль рубрики</h3>')
            if by_rubric:
                parts.append("<ul class='cr-list'>")
                for slug, info in by_rubric.items():
                    title = (info or {}).get("title", slug)
                    cov = (info or {}).get("term_coverage_pct", 0)
                    parts.append(f"<li><strong>{_e(title)}</strong> — {cov}%</li>")
                parts.append("</ul>")
            if measurements:
                parts.append("<ul class='cr-list cr-list--meas'>")
                for name, m in measurements.items():
                    unit = (m or {}).get("unit") or ""
                    val = (m or {}).get("value") or ""
                    parts.append(f"<li>{_e(name)}: <strong>{_e(val)}</strong> {_e(unit)}</li>")
                parts.append("</ul>")
            parts.append("</section>")

    # Источники
    parts.append('<section class="cr-section cr-section--sources"><h3 class="cr-section-title">Источники</h3>')
    if report.source_refs:
        parts.append("<ul class='cr-list cr-list--sources'>")
        for r in report.source_refs:
            parts.append(f"<li>{_e(_src_line(r))}</li>")
        parts.append("</ul>")
    else:
        parts.append("<p class='cr-muted'>Источники не указаны.</p>")
    parts.append("</section>")

    parts.append(
        '<footer class="cr-disclaimer">Оценка ориентировочная и не заменяет врача. '
        "Инструмент экспертной проверки соответствия клиническим протоколам.</footer>"
    )
    parts.append("</article>")
    return "".join(parts)


def report_to_markdown(
    report: ComplianceReport,
    doc: ConsultationDocument | None = None,
    rubric_specifics: dict | None = None,
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
        L.append(f"- Пациент: {name_to_initials(p.full_name)}")
        L.append(f"- Возраст: {p.age_years if p.age_years is not None else '—'} ({p.age_group})")
        L.append(f"- Пол: {_SEX_RU.get(p.sex, p.sex)}")
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
    L.append(
        f"- Общая оценка: {_fmt_pct(report.overall_score)} — статус "
        f"**{_OVERALL_RU.get(report.overall_status, report.overall_status)}**"
    )
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
        L.append(f"- {a.icd10_code or ''} {a.diagnosis_text} — **{_DIAG_RU.get(a.status, a.status)}**")
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
            L.append(f"- {e.exam_name} — **{_EXAM_RU.get(e.status, e.status)}**" + (f" ({e.reason})" if e.reason else ""))
    else:
        L.append("- Детерминированные правила по обследованиям не сработали (нет данных).")
    L.append("")

    # 5. Лечение
    L.append("## 5. Оценка лечения")
    L.append(f"_Балл блока: {_fmt_pct(bd.treatment_score)}_")
    if report.treatment_assessments:
        for t in report.treatment_assessments:
            L.append(f"- {t.treatment_text} — **{_TREAT_RU.get(t.status, t.status)}**")
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
            L.append(f"- [{_SEVERITY_RU.get(s.severity, s.severity)}] {s.finding_text} — **{_SAFETY_RU.get(s.status, s.status)}**")
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

    # 7.1 Профильные показатели рубрики (ТЗ раздел 22)
    if rubric_specifics:
        by_rubric = rubric_specifics.get("by_rubric") or {}
        measurements = rubric_specifics.get("measurements") or {}
        if by_rubric or measurements:
            L.append("## 7.1 Профильные показатели рубрики")
            for slug, info in by_rubric.items():
                title = (info or {}).get("title", slug)
                matched = (info or {}).get("matched_terms") or []
                missing = (info or {}).get("missing_terms") or []
                cov = (info or {}).get("term_coverage_pct", 0)
                L.append(f"- **{title}** — покрытие профильных понятий {cov}%")
                if matched:
                    L.append(f"  - найдено: {', '.join(matched)}")
                if missing:
                    L.append(f"  - не отражено: {', '.join(missing)}")
            if measurements:
                L.append("- Числовые показатели:")
                for name, m in measurements.items():
                    unit = (m or {}).get("unit") or ""
                    val = (m or {}).get("value") or ""
                    L.append(f"  - {name}: {val} {unit}".rstrip())
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
