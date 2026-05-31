"""Генерация отчётов по оценке КЗ: JSON и Markdown (ТЗ раздел 20).

Каждый блок отчёта опирается на проверяемые данные (assessments + source_refs).
"""
from __future__ import annotations

from html import escape as _esc
from typing import Any

from .consult_schema import ComplianceReport, ConsultationDocument, SourceRef
from .privacy import name_to_initials
from .protocol_links import protocol_display_name, protocol_pdf_api_path, protocol_rubric_label

# Русские подписи статусов/значений для человекочитаемого отчёта.
_OVERALL_RU = {
    "compliant": "соответствует",
    "mostly_compliant": "в основном соответствует",
    "partially_compliant": "частично соответствует",
    "non_compliant": "не соответствует",
    "insufficient_data": "недостаточно данных",
    "insufficient_protocol_data": "нет данных протокола",
    "low_confidence": "низкая уверенность",
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
    "insufficient_protocol_data": ("#475569", "#f1f5f9"),
    "low_confidence": ("#7c3aed", "#f3e8ff"),
    "manual_review_required": ("#9333ea", "#f3e8ff"),
}
_SCORE_KEYS = (
    ("documentation_score", "Оформление КЗ", "#64748b", ("structural_score", "documentation_quality_score")),
    ("patient_data_score", "Данные пациента", "#0891b2", ()),
    ("protocol_applicability_score", "Применимость протокола", "#2563eb", ("protocol_match_score",)),
    ("diagnosis_score", "Диагноз", "#0d9488", ()),
    ("required_exams_score", "Обследования", "#7c3aed", ()),
    ("treatment_score", "Лечение", "#db2777", ()),
    ("safety_score", "Безопасность", "#dc2626", ()),
    ("follow_up_score", "Контроль", "#ea580c", ()),
)
_LAYER_KEYS = (
    ("documentation_score", "structural_score", "A. Оформление КЗ", "#64748b"),
    ("protocol_applicability_score", "protocol_match_score", "B. Протокол", "#2563eb"),
    ("diagnosis_score", None, "C. Клиника", "#0d9488"),
)


def _score_val(bd: Any, *keys: str) -> float | None:
    for k in keys:
        if not k:
            continue
        val = getattr(bd, k, None) if bd is not None else None
        if val is None and isinstance(bd, dict):
            val = bd.get(k)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _svg_donut(
    score: float | None,
    *,
    color: str,
    caption: str,
    size: int = 108,
) -> str:
    pct = max(0.0, min(100.0, float(score))) if isinstance(score, (int, float)) else 0.0
    r = 38
    c_len = 2 * 3.14159 * r
    dash = c_len * pct / 100.0
    display = _fmt_pct(score) if isinstance(score, (int, float)) else "—"
    return (
        f'<figure class="cr-donut-wrap" aria-label="{_e(caption)}: {display}">'
        f'<svg class="cr-donut" width="{size}" height="{size}" viewBox="0 0 88 88" role="img">'
        f'<circle cx="44" cy="44" r="{r}" fill="none" stroke="#e2e8f0" stroke-width="9"/>'
        f'<circle cx="44" cy="44" r="{r}" fill="none" stroke="{color}" stroke-width="9" '
        f'stroke-dasharray="{dash:.2f} {c_len:.2f}" stroke-linecap="round" '
        f'transform="rotate(-90 44 44)"/>'
        f'<text x="44" y="42" text-anchor="middle" font-size="17" font-weight="700" fill="{color}">'
        f'{display if display != "—" else "—"}</text>'
        f'<text x="44" y="56" text-anchor="middle" font-size="9" fill="#64748b">{_e(caption)}</text>'
        f"</svg></figure>"
    )


def _issue_chips_html(report: ComplianceReport) -> str:
    chips = [
        (len(report.critical_issues), "Критич.", "#b91c1c", "#fee2e2"),
        (len(report.major_issues), "Существен.", "#b45309", "#fef3c7"),
        (len(report.missing_required_items), "Пропуски", "#7c3aed", "#ede9fe"),
        (len(report.warnings), "Предупр.", "#0369a1", "#e0f2fe"),
    ]
    parts = ['<div class="cr-issue-chips">']
    for n, label, fg, bg in chips:
        if n <= 0:
            continue
        parts.append(
            f'<span class="cr-issue-chip" style="color:{fg};background:{bg};border-color:{fg}33">'
            f"<strong>{n}</strong> {_e(label)}</span>"
        )
    if len(parts) == 1:
        parts.append('<span class="cr-issue-chip cr-issue-chip--ok">Замечаний нет</span>')
    parts.append("</div>")
    return "".join(parts)


def _layer_cards_html(bd: Any) -> str:
    cards: list[str] = []
    for primary, fallback, title, color in _LAYER_KEYS:
        val = _score_val(bd, primary, fallback or "")
        if val is None and primary == "diagnosis_score":
            exams = _score_val(bd, "required_exams_score")
            treat = _score_val(bd, "treatment_score")
            if exams is not None or treat is not None:
                parts = [x for x in (exams, treat) if x is not None]
                val = sum(parts) / len(parts) if parts else None
        if val is None:
            continue
        w = max(0, min(100, val))
        cards.append(
            f'<div class="cr-layer-card" style="--layer-color:{color}">'
            f'<div class="cr-layer-card__title">{_e(title)}</div>'
            f'<div class="cr-layer-card__pct">{_fmt_pct(val)}</div>'
            f'<div class="cr-layer-card__track"><div class="cr-layer-card__fill" '
            f'style="width:{w:.0f}%"></div></div></div>'
        )
    if not cards:
        return ""
    return '<div class="cr-layer-grid">' + "".join(cards) + "</div>"


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


def _src_line_html(ref: SourceRef) -> str:
    parts: list[str] = []
    if ref.local_path:
        url = protocol_pdf_api_path(ref.local_path)
        name = _e(protocol_display_name(ref.local_path, ref.protocol_id or ""))
        if url:
            parts.append(
                f'<a class="cr-src-link" href="{url}" target="_blank" '
                f'rel="noopener noreferrer">{name}</a>'
            )
        else:
            parts.append(_e(ref.local_path))
    elif ref.protocol_id:
        parts.append(_e(ref.protocol_id))
    if ref.section_title:
        parts.append(f"раздел: {_e(ref.section_title)}")
    if ref.page_start:
        pg = f"с. {ref.page_start}"
        if ref.page_end and ref.page_end != ref.page_start:
            pg += f"-{ref.page_end}"
        parts.append(_e(pg))
    if ref.quote:
        parts.append(f"«{_e(ref.quote[:160])}»")
    return " — ".join(parts) if parts else "источник не указан"


def report_to_json(
    report: ComplianceReport,
    doc: ConsultationDocument | None = None,
) -> dict[str, Any]:
    """JSON-отчёт по ТЗ раздел 20."""
    patient_summary: dict[str, Any] = {}
    doctor_specialty = None
    consultation_date = None
    doctor_summary: dict[str, Any] = {}
    if doc is not None:
        patient_summary = {
            "full_name_initials": doc.patient.full_name,
            "age_years": doc.patient.age_years,
            "birth_date": doc.patient.birth_date.isoformat() if doc.patient.birth_date else None,
            "sex": doc.patient.sex,
            "adult_or_child": doc.patient.adult_or_child,
            "pregnancy": doc.patient.pregnancy,
        }
        doctor_specialty = doc.doctor_specialty
        doctor_summary = {
            "specialty": doc.doctor_specialty,
            "name": doc.doctor_name,
            "category": doc.doctor_category,
        }
        consultation_date = doc.consultation_date.isoformat() if doc.consultation_date else None

    all_issues = (
        list(report.critical_issues)
        + list(report.major_issues)
        + list(report.missing_required_items)
        + list(report.warnings)
    )
    return {
        "consultation_id": report.consultation_id,
        "source_file": report.source_file,
        "patient_summary": patient_summary,
        "doctor_summary": doctor_summary,
        "doctor_specialty": doctor_specialty,
        "consultation_date": consultation_date,
        "matched_protocols": [m.model_dump(mode="json") for m in report.protocol_matches],
        "not_applicable_protocols": [m.model_dump(mode="json") for m in report.not_applicable_protocols],
        "diagnoses": [d.model_dump(mode="json") for d in (doc.diagnoses if doc else [])],
        "overall_score": report.overall_score,
        "confidence_score": report.confidence_score,
        "score_source": report.score_source,
        "llm_score_ignored": report.llm_score_ignored,
        "score_breakdown": report.score_breakdown.model_dump(mode="json"),
        "overall_status": report.overall_status,
        "structural_assessment": report.structural_assessment.model_dump(mode="json"),
        "protocol_assessment": report.protocol_assessment.model_dump(mode="json"),
        "critical_issues": [i.model_dump(mode="json") for i in report.critical_issues],
        "major_issues": [i.model_dump(mode="json") for i in report.major_issues],
        "warnings": [i.model_dump(mode="json") for i in report.warnings],
        "missing_required_items": [i.model_dump(mode="json") for i in report.missing_required_items],
        "issues": [i.model_dump(mode="json") for i in all_issues],
        "diagnosis_assessments": [a.model_dump(mode="json") for a in report.diagnosis_assessments],
        "exam_assessments": [a.model_dump(mode="json") for a in report.exam_assessments],
        "treatment_assessments": [a.model_dump(mode="json") for a in report.treatment_assessments],
        "safety_assessments": [a.model_dump(mode="json") for a in report.safety_assessments],
        "evidence_map": [e.model_dump(mode="json") for e in report.evidence_map],
        "safety_cap": report.safety_cap.model_dump(mode="json"),
        "limitations": list(report.limitations),
        "source_refs": [r.model_dump(mode="json") for r in report.source_refs],
        "explanation": report.explanation,
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
    seen: set[str] = set()
    for key, title, color, fallbacks in _SCORE_KEYS:
        val = _score_val(bd, key, *fallbacks)
        if val is None:
            continue
        dedupe = key.replace("documentation_score", "doc").replace("protocol_applicability_score", "proto")
        if dedupe in seen:
            continue
        if key in ("documentation_score", "structural_score", "documentation_quality_score"):
            if "doc" in seen:
                continue
            seen.add("doc")
        elif key in ("protocol_applicability_score", "protocol_match_score"):
            if "proto" in seen:
                continue
            seen.add("proto")
        else:
            seen.add(key)
        pct = float(val)
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
    conf_color = "#7c3aed" if (report.confidence_score or 100) < 55 else "#0d9488"

    parts: list[str] = [
        '<article class="consult-report-html">',
        '<header class="cr-header cr-header--hero">',
        '<div class="cr-hero-grid">',
        '<div class="cr-hero-text">',
        '<h2 class="cr-title">Оценка консультативного заключения</h2>',
        '<div class="cr-header-meta">',
        _status_badge_html(status, status_ru),
    ]
    if report.safety_cap.applied:
        parts.append(
            f'<span class="cr-badge" style="color:#b91c1c;background:#fee2e2;border:1px solid #b91c1c44">'
            f"⚠ Safety cap {_fmt_pct(report.safety_cap.cap_value)}</span>"
        )
    parts.append("</div>")
    if report.limitations:
        parts.append('<ul class="cr-limitations">')
        for lim in report.limitations[:4]:
            parts.append(f"<li>{_e(lim)}</li>")
        parts.append("</ul>")
    parts.append("</div>")
    parts.append('<div class="cr-hero-charts">')
    parts.append(_svg_donut(report.overall_score, color=fg, caption="Соответствие"))
    if report.confidence_score is not None:
        parts.append(_svg_donut(report.confidence_score, color=conf_color, caption="Уверенность"))
    parts.append("</div></div>")
    parts.append(_issue_chips_html(report))
    parts.append(_layer_cards_html(bd))
    parts.append("</header>")

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
    protos_html: list[str] = []
    seen_proto: set[str] = set()
    for m in report.protocol_matches[:8]:
        sp = m.source_path or ""
        if sp in seen_proto:
            continue
        seen_proto.add(sp)
        title = _e(
            protocol_display_name(sp, m.protocol_id or "", registry_title=m.document_title)
        )
        url = protocol_pdf_api_path(sp)
        rub = protocol_rubric_label(sp)
        if url:
            inner = (
                f'<a class="cr-src-link" href="{url}" target="_blank" '
                f'rel="noopener noreferrer">{title}</a>'
            )
            if rub:
                inner += f' <span class="cr-proto-rubric">{_e(rub)}</span>'
            protos_html.append(inner)
        else:
            protos_html.append(title)
    parts.append(
        f"<li><strong>Протоколы:</strong> {', '.join(protos_html) if protos_html else '—'}</li>"
    )
    parts.append("</ul></section>")

    if report.protocol_matches:
        parts.append(
            '<section class="cr-section cr-section--protocols">'
            '<h3 class="cr-section-title">Подобранные протоколы</h3>'
            '<ul class="cr-list cr-list--sources cr-proto-cards">'
        )
        seen_proto = set()
        for m in report.protocol_matches[:8]:
            sp = m.source_path or ""
            if not sp or sp in seen_proto:
                continue
            seen_proto.add(sp)
            title = _e(
                protocol_display_name(sp, m.protocol_id or "", registry_title=m.document_title)
            )
            url = protocol_pdf_api_path(sp)
            rub = protocol_rubric_label(sp)
            parts.append('<li class="cr-proto-card">')
            if url:
                parts.append(
                    f'<a class="cr-src-link cr-proto-card__link" href="{url}" target="_blank" '
                    f'rel="noopener noreferrer">{title}</a>'
                )
            else:
                parts.append(f'<span class="cr-proto-card__link">{title}</span>')
            if rub:
                parts.append(f'<span class="cr-proto-rubric">{_e(rub)}</span>')
            parts.append("</li>")
        parts.append("</ul></section>")

    # Баллы
    bars = _score_bars_html(bd)
    if bars:
        parts.append(
            '<section class="cr-section cr-section--scores">'
            '<h3 class="cr-section-title">Детализация по блокам</h3>'
            f'<div class="cr-bars">{bars}</div></section>'
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

    if report.not_applicable_protocols:
        parts.append(
            '<section class="cr-section cr-section--muted">'
            '<h3 class="cr-section-title">Протоколы не применимы</h3><ul class="cr-list">'
        )
        for m in report.not_applicable_protocols[:6]:
            title = _e(m.document_title or m.protocol_id or "—")
            parts.append(f"<li>{title} — <em>{_e(m.applicability)}</em></li>")
        parts.append("</ul></section>")

    if report.evidence_map:
        parts.append(
            '<section class="cr-section cr-section--evidence">'
            '<h3 class="cr-section-title">Карта доказательств</h3>'
            '<div class="cr-evidence-grid">'
        )
        for ev in report.evidence_map[:16]:
            dec = ev.decision or "unknown"
            dec_color = {
                "satisfied": "#0d7a4a",
                "satisfied_by_recommendation": "#1a7f5a",
                "missing": "#b91c1c",
                "not_applicable": "#64748b",
                "manual_review": "#9333ea",
            }.get(dec, "#475569")
            parts.append(
                f'<div class="cr-evidence-card" style="border-left-color:{dec_color}">'
                f'<div class="cr-evidence-card__id">{_e(ev.rule_id)}</div>'
                f'<div class="cr-evidence-card__dec">{_e(dec)}</div>'
                f'<div class="cr-evidence-card__txt">{_e(ev.explanation or ev.required_item or "")}</div>'
                f"</div>"
            )
        parts.append("</div></section>")

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
            parts.append(f"<li>{_src_line_html(r)}</li>")
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
    """Markdown-отчёт по структуре ТЗ §15 (11 разделов)."""
    L: list[str] = []
    bd = report.score_breakdown

    L.append("# Оценка консультативного заключения")
    L.append("")
    # 1. Резюме
    L.append("## 1. Краткое резюме")
    if report.source_file:
        L.append(f"- Файл: {report.source_file}")
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
    if report.confidence_score is not None:
        L.append(f"- Уверенность разбора (confidence): {_fmt_pct(report.confidence_score)}")
    if report.score_source:
        L.append(f"- Источник оценки: {report.score_source}" + (" (LLM-score не используется)" if report.llm_score_ignored else ""))
    if report.safety_cap.applied:
        L.append(f"- ⚠️ Safety cap: {report.safety_cap.reason or 'применён'} (лимит {_fmt_pct(report.safety_cap.cap_value)})")
    if report.limitations:
        L.append("- Ограничения:")
        for lim in report.limitations:
            L.append(f"  - {lim}")
    L.append("")

    # 2. Структура КЗ (requirement checker)
    sa = report.structural_assessment
    L.append("## 2. Проверка структуры КЗ")
    L.append(f"_Балл: {_fmt_pct(sa.structural_score)}; данные пациента: {_fmt_pct(sa.patient_data_score)}_")
    if sa.filled_sections:
        L.append(f"- Заполнено: {', '.join(sa.filled_sections[:12])}")
    if sa.missing_required:
        L.append(f"- **Отсутствует (обязательное):** {', '.join(sa.missing_required)}")
    if sa.missing_conditional:
        L.append(f"- Отсутствует (условное): {', '.join(sa.missing_conditional)}")
    if sa.missing_recommended:
        L.append(f"- Рекомендуется добавить: {', '.join(sa.missing_recommended)}")
    sq = report.section_quality
    if sq.missing_sections or sq.suspicious_placeholders:
        L.append("- **Качество оформления:**")
        if sq.missing_sections:
            L.append(f"  - пропущенные разделы: {', '.join(sq.missing_sections)}")
        if sq.suspicious_placeholders:
            L.append(f"  - placeholder: {', '.join(sq.suspicious_placeholders)}")
    L.append("")

    # 3. Данные пациента
    L.append("## 3. Данные пациента")
    if doc is not None:
        p = doc.patient
        L.append(f"- Дата рождения: {p.birth_date.isoformat() if p.birth_date else '—'}")
        L.append(f"- Возраст на дату консультации: {p.age_years if p.age_years is not None else '—'}")
        L.append(f"- Пол: {_SEX_RU.get(p.sex, p.sex)}")
        if p.pregnancy is not None:
            L.append(f"- Беременность: {'да' if p.pregnancy else 'нет'}")
        if p.comorbidities:
            L.append(f"- Сопутствующие заболевания: {', '.join(p.comorbidities[:5])}")
        if p.allergies:
            L.append(f"- Аллергии: {', '.join(p.allergies[:5])}")
        if p.current_medications:
            L.append(f"- Текущие препараты: {', '.join(p.current_medications[:5])}")
    else:
        L.append("- Данные пациента не распознаны.")
    L.append("")

    # 4. Диагноз
    L.append("## 4. Проверка диагноза")
    if doc is not None and doc.diagnoses:
        primary = [d for d in doc.diagnoses if d.diagnosis_role == "primary"] or doc.diagnoses[:1]
        secondary = [d for d in doc.diagnoses if d.diagnosis_role != "primary" and d.certainty != "suspected"]
        suspected = [d for d in doc.diagnoses if d.certainty == "suspected"]
        if primary:
            L.append("- **Основной:** " + "; ".join(
                f"{d.icd10_code or '—'} {d.diagnosis_name or d.raw_text}" for d in primary
            ))
        if secondary:
            L.append("- Сопутствующие: " + "; ".join(d.raw_text[:80] for d in secondary[:3]))
        if suspected:
            L.append("- Подозрительные: " + "; ".join(d.raw_text[:80] for d in suspected[:3]))
    L.append(f"_Балл блока: {_fmt_pct(bd.diagnosis_score)}_")
    for a in report.diagnosis_assessments:
        L.append(f"- {a.icd10_code or ''} {a.diagnosis_text} — **{_DIAG_RU.get(a.status, a.status)}**")
        for e in a.evidence_found:
            L.append(f"  - подтверждено: {e}")
        for e in a.evidence_missing:
            L.append(f"  - не хватает: {e}")
    L.append("")

    # 5. Применимость протоколов
    pa = report.protocol_assessment
    L.append("## 5. Применимость протоколов")
    if pa.summary_ru:
        L.append(f"- {pa.summary_ru}")
    L.append(f"- Подобрано: {pa.matched_count}; применимо: {pa.applicable_count}")
    if report.protocol_matches:
        for m in report.protocol_matches[:5]:
            L.append(f"- **{m.document_title or m.protocol_id}** — {m.applicability}")
            if m.matched_condition:
                L.append(f"  - нозология: {m.matched_condition}")
            for r in m.match_reasons[:3]:
                L.append(f"  - + {r}")
            for r in m.mismatch_reasons[:3]:
                L.append(f"  - − {r}")
    else:
        L.append("- Подходящие протоколы не подобраны.")
    if report.not_applicable_protocols:
        L.append("- **Не применимы (возраст/пол/аудитория):**")
        for m in report.not_applicable_protocols[:5]:
            L.append(f"  - {m.document_title or m.protocol_id} — {m.applicability}")
            for r in m.mismatch_reasons[:2]:
                L.append(f"    - {r}")
    L.append("")

    # 6. Обследования
    L.append("## 6. Проверка обследований")
    L.append(f"_Балл блока: {_fmt_pct(bd.required_exams_score)}_")
    if doc is not None:
        if doc.performed_exams:
            L.append("- **Выполненные:** " + ", ".join(e.exam_name for e in doc.performed_exams[:8]))
        if doc.planned_exams:
            L.append("- **Рекомендованные:** " + ", ".join(e.exam_name for e in doc.planned_exams[:8]))
    if report.exam_assessments:
        for e in report.exam_assessments:
            L.append(f"- {e.exam_name} — **{_EXAM_RU.get(e.status, e.status)}**" + (f" ({e.reason})" if e.reason else ""))
    else:
        L.append("- Детерминированные правила по обследованиям не сработали (нет данных).")
    L.append("")

    # 7. Лечение
    L.append("## 7. Проверка лечения")
    L.append(f"_Балл блока: {_fmt_pct(bd.treatment_score)}_")
    if report.treatment_assessments:
        for t in report.treatment_assessments:
            L.append(f"- {t.treatment_text} — **{_TREAT_RU.get(t.status, t.status)}**")
            for iss in t.issues:
                L.append(f"  - {iss.message_ru}")
            if t.consultation_evidence:
                L.append(f"  - фрагмент КЗ: {t.consultation_evidence[0][:120]}")
    else:
        L.append("- Назначения не распознаны.")
    L.append("")

    # 8. Red flags
    L.append("## 8. Красные флаги и безопасность")
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

    # 9. Повторная явка и контроль
    L.append("## 9. Повторная явка и контроль")
    L.append(f"_Балл блока: {_fmt_pct(bd.follow_up_score)}_")
    if doc is not None:
        if doc.follow_up:
            for fu in doc.follow_up[:3]:
                L.append(f"- Повторная явка: {fu.date.isoformat() if fu.date else fu.raw_text or '—'}")
        elif doc.sections.follow_up_text:
            L.append(f"- {doc.sections.follow_up_text[:200]}")
        else:
            L.append("- Дата или срок повторной явки не указаны.")
    L.append("")

    # 9a. Evidence map
    if report.evidence_map:
        L.append("## 9a. Карта доказательств (evidence map)")
        for ev in report.evidence_map[:20]:
            L.append(f"- **{ev.rule_id}** ({ev.rule_type}): {ev.decision} — {ev.explanation or '—'}")
            if ev.consultation_evidence:
                L.append(f"  - КЗ: {ev.consultation_evidence[0][:120]}")
            if ev.protocol_evidence:
                L.append(f"  - протокол: {ev.protocol_evidence[0][:120]}")
        L.append("")

    # 10. Все замечания
    all_issues = (
        list(report.critical_issues)
        + list(report.major_issues)
        + list(report.missing_required_items)
        + list(report.warnings)
    )
    L.append("## 10. Все замечания")
    if all_issues:
        for sev_label, sevs in (
            ("Critical", ("critical",)),
            ("Major", ("high", "warning")),
            ("Minor", ("medium", "low")),
            ("Info", ("info",)),
        ):
            bucket = [i for i in all_issues if i.severity in sevs]
            if bucket:
                L.append(f"### {sev_label}")
                for iss in bucket[:15]:
                    ev = iss.consultation_evidence[0][:80] if iss.consultation_evidence else ""
                    line = f"- {iss.message_ru}"
                    if ev:
                        line += f" _(КЗ: {ev}…)_"
                    L.append(line)
    else:
        L.append("- Замечаний нет.")
    L.append("")

    # 7.1 Профильные показатели рубрики (ТЗ раздел 22)
    if rubric_specifics:
        by_rubric = rubric_specifics.get("by_rubric") or {}
        measurements = rubric_specifics.get("measurements") or {}
        if by_rubric or measurements:
            L.append("## 10.1 Профильные показатели рубрики")
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

    # 11. Источники
    L.append("## 11. Источники")
    if doc is not None and doc.raw_text:
        L.append("- **Фрагменты КЗ:**")
        for chunk in doc.raw_text.strip().split("\n\n")[:4]:
            if chunk.strip():
                L.append(f"  - «{chunk.strip()[:180]}…»")
    refs = report.source_refs or []
    if refs:
        L.append("- **Протоколы:**")
        for r in refs:
            L.append(f"  - {_src_line(r)}")
    elif not (doc and doc.raw_text):
        L.append("- Источники не указаны.")
    L.append("")

    L.append(
        "> Оценка ориентировочная и не заменяет врача. "
        "Документ — инструмент экспертной проверки соответствия клиническим протоколам."
    )
    return "\n".join(L)
