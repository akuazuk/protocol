"""Генерация отчётов по оценке КЗ: JSON и Markdown (ТЗ раздел 20).

Каждый блок отчёта опирается на проверяемые данные (assessments + source_refs).
"""
from __future__ import annotations

from html import escape as _esc
from typing import Any

from .consult_schema import ComplianceReport, ConsultationDocument, SourceRef
from .privacy import name_to_initials
from .protocol_links import protocol_display_name, protocol_nav_api_path, protocol_rubric_label
from .rule_labels_ru import decision_ru, rule_title_ru, rule_type_ru

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

# Пастельные цвета статусов для HTML-бейджей.
_STATUS_COLOR = {
    "compliant": ("#3d6b58", "#e8f3ed"),
    "mostly_compliant": ("#4a7262", "#eaf4ef"),
    "partially_compliant": ("#8a6f45", "#f5f0e6"),
    "non_compliant": ("#8f5a5a", "#f5ebeb"),
    "insufficient_data": ("#6b7280", "#f3f4f6"),
    "insufficient_protocol_data": ("#6b7280", "#f3f4f6"),
    "low_confidence": ("#6b5f8a", "#f0edf5"),
    "manual_review_required": ("#7a5f8a", "#f3eef5"),
}
_SCORE_KEYS = (
    ("documentation_score", "Оформление КЗ", "", ("structural_score", "documentation_quality_score")),
    ("patient_data_score", "Данные пациента", "", ()),
    ("protocol_applicability_score", "Применимость протокола", "", ("protocol_match_score",)),
    ("diagnosis_score", "Диагноз", "", ()),
    ("required_exams_score", "Обследования", "", ()),
    ("treatment_score", "Лечение", "", ()),
    ("safety_score", "Безопасность", "", ()),
    ("follow_up_score", "Контроль", "", ()),
)
_LAYER_KEYS = (
    ("documentation_score", "structural_score", "A. Оформление", "#8a9f8e"),
    ("protocol_applicability_score", "protocol_match_score", "B. Протокол", "#8fa8b8"),
    ("diagnosis_score", None, "C. Клиника", "#7aab96"),
)


def _score_bar_color(pct: float) -> str:
    if pct >= 75:
        return "#7aab96"
    if pct >= 50:
        return "#c4a574"
    return "#c99a9a"


def _score_verdict_ru(pct: float | None) -> tuple[str, str]:
    """Краткий вердикт для врача: (текст, css-класс)."""
    if not isinstance(pct, (int, float)):
        return " - ", "cr-verdict--na"
    p = float(pct)
    if p >= 85:
        return "Хорошо", "cr-verdict--good"
    if p >= 70:
        return "Приемлемо", "cr-verdict--ok"
    if p >= 50:
        return "Доработать", "cr-verdict--warn"
    return "Критично", "cr-verdict--low"


def _spoiler_section(title: str, inner_html: str, *, open_default: bool = True) -> str:
    open_attr = " open" if open_default else ""
    return (
        f'<details class="cr-spoiler"{open_attr}>'
        f'<summary class="cr-spoiler__summary">{_e(title)}</summary>'
        f'<div class="cr-spoiler__body">{inner_html}</div></details>'
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


def _mini_donut_svg(
    score: float | None,
    *,
    size: int = 52,
    stroke: str | None = None,
) -> str:
    pct = max(0.0, min(100.0, float(score))) if isinstance(score, (int, float)) else 0.0
    r = 18
    c_len = 2 * 3.14159 * r
    dash = c_len * pct / 100.0
    col = stroke or _score_bar_color(pct)
    display = f"{pct:.0f}" if isinstance(score, (int, float)) else " - "
    cx = size / 2
    return (
        f'<svg class="cr-mini-donut" width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'role="img" aria-hidden="true">'
        f'<circle cx="{cx}" cy="{cx}" r="{r}" fill="#f4f7f6" stroke="#e8eeec" stroke-width="1"/>'
        f'<circle cx="{cx}" cy="{cx}" r="{r}" fill="none" stroke="#e8eeec" stroke-width="5"/>'
        f'<circle cx="{cx}" cy="{cx}" r="{r}" fill="none" stroke="{col}" stroke-width="5" '
        f'stroke-dasharray="{dash:.2f} {c_len:.2f}" stroke-linecap="round" '
        f'transform="rotate(-90 {cx} {cx})"/>'
        f'<text x="{cx}" y="{cx + 4}" text-anchor="middle" font-size="11" font-weight="700" '
        f'fill="#2d4a42">{display}</text></svg>'
    )


def _svg_donut(
    score: float | None,
    *,
    color: str,
    caption: str,
    size: int = 120,
) -> str:
    pct = max(0.0, min(100.0, float(score))) if isinstance(score, (int, float)) else 0.0
    r = 40
    c_len = 2 * 3.14159 * r
    dash = c_len * pct / 100.0
    display = _fmt_pct(score) if isinstance(score, (int, float)) else " - "
    verdict, vcls = _score_verdict_ru(score if isinstance(score, (int, float)) else None)
    ring = _score_bar_color(pct) if isinstance(score, (int, float)) else "#cbd5e1"
    return (
        f'<figure class="cr-donut-wrap cr-donut-wrap--lg" aria-label="{_e(caption)}: {display}">'
        f'<svg class="cr-donut" width="{size}" height="{size}" viewBox="0 0 96 96" role="img">'
        f'<circle cx="48" cy="48" r="{r + 6}" fill="#f8fbfa" stroke="#e8f0ed" stroke-width="1"/>'
        f'<circle cx="48" cy="48" r="{r}" fill="none" stroke="#eef2f7" stroke-width="10"/>'
        f'<circle cx="48" cy="48" r="{r}" fill="none" stroke="{ring}" stroke-width="10" '
        f'stroke-dasharray="{dash:.2f} {c_len:.2f}" stroke-linecap="round" '
        f'transform="rotate(-90 48 48)"/>'
        f'<text x="48" y="44" text-anchor="middle" font-size="20" font-weight="800" fill="#1e3d34">'
        f'{display if display != " - " else " - "}</text>'
        f'<text x="48" y="58" text-anchor="middle" font-size="8.5" fill="#6b7f78" font-weight="600">'
        f'{_e(caption)}</text></svg>'
        f'<figcaption class="cr-donut-verdict {_e(vcls)}">{_e(verdict)}</figcaption></figure>'
    )


def _issue_chips_html(report: ComplianceReport) -> str:
    chips = [
        (len(report.critical_issues), "Критич.", "#8f5a5a", "#f5ebeb"),
        (len(report.major_issues), "Существен.", "#8a6f45", "#f5f0e6"),
        (len(report.missing_required_items), "Пропуски", "#7a6f8a", "#f0edf5"),
        (len(report.warnings), "Предупр.", "#5f7a8a", "#edf2f5"),
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
        verdict, vcls = _score_verdict_ru(val)
        cards.append(
            f'<div class="cr-layer-card" style="--layer-color:{color}">'
            f'<div class="cr-layer-card__donut">{_mini_donut_svg(val, stroke=color)}</div>'
            f'<div class="cr-layer-card__body">'
            f'<div class="cr-layer-card__title">{_e(title)}</div>'
            f'<div class="cr-layer-card__pct">{_fmt_pct(val)}</div>'
            f'<span class="cr-verdict {_e(vcls)}">{_e(verdict)}</span>'
            f'<div class="cr-layer-card__track"><div class="cr-layer-card__fill" '
            f'style="width:{w:.0f}%"></div></div></div></div>'
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
    return " - ".join(parts) if parts else (ref.protocol_id or "источник не указан")


def _src_line_html(ref: SourceRef) -> str:
    parts: list[str] = []
    if ref.local_path:
        url = protocol_nav_api_path(ref.local_path, section=ref.section_title)
        name = protocol_display_name(ref.local_path, ref.protocol_id or "")
        if url:
            parts.append(
                f'<a class="cr-src-link proto-nav-link--compact" href="{url}" target="_blank" '
                f'rel="noopener noreferrer" title="{_e(name)}">Протокол</a>'
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
    return " - ".join(parts) if parts else "источник не указан"


def _resolve_send_gate(
    report: ComplianceReport,
    *,
    send_gate: dict[str, Any] | None = None,
    headline_score: float | None = None,
) -> dict[str, Any]:
    if send_gate:
        return send_gate
    from .compliance_gate import evaluate_send_gate

    return evaluate_send_gate(report, headline_score=headline_score)


def _sign_decision_section_html(sg: dict[str, Any]) -> str:
    sd = sg.get("sign_decision") or "allowed"
    sd_ru = _e(sg.get("sign_decision_ru") or "Решение о подписи")
    sd_det = _e(sg.get("sign_decision_detail_ru") or "")
    sd_colors = {
        "allowed": ("#1a6b52", "#e8f5f1"),
        "allowed_with_warnings": ("#8a5a12", "#faf5eb"),
        "review_required": ("#8a5a12", "#faf5eb"),
        "blocked": ("#9a3030", "#faf0f0"),
    }
    sd_fg, sd_bg = sd_colors.get(sd, ("#334155", "#f8fafc"))
    return (
        f'<section class="cr-sign-decision" style="margin:0.65rem 0;padding:0.65rem 0.75rem;'
        f'border-radius:10px;border:1px solid {sd_fg}33;background:{sd_bg}">'
        f'<h3 style="margin:0 0 0.35rem;font-size:0.95rem;color:{sd_fg}">Решение о подписи</h3>'
        f'<p style="margin:0;font-weight:700;color:{sd_fg}">{sd_ru}</p>'
        f'<p style="margin:0.35rem 0 0;font-size:0.88rem;color:#4a5c56">{sd_det}</p>'
        "</section>"
    )


def patch_report_html_send_gate(html: str, send_gate: dict[str, Any]) -> str:
    """Подменяет блок «Решение о подписи» в уже собранном HTML-отчёте."""
    import re

    if not html or not send_gate:
        return html
    new_block = _sign_decision_section_html(send_gate)
    if 'class="cr-sign-decision"' not in html:
        return html
    return re.sub(
        r'<section class="cr-sign-decision"[^>]*>.*?</section>',
        new_block,
        html,
        count=1,
        flags=re.DOTALL,
    )


def report_to_json(
    report: ComplianceReport,
    doc: ConsultationDocument | None = None,
    *,
    send_gate: dict[str, Any] | None = None,
    headline_score: float | None = None,
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
    send_gate = _resolve_send_gate(
        report,
        send_gate=send_gate,
        headline_score=headline_score,
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
        "analysis_mode": report.analysis_mode,
        "protocol_summary_used": report.protocol_summary_used,
        "protocol_summary_status": report.protocol_summary_status,
        "fallback_to_legacy": report.fallback_to_legacy,
        "legacy_result_available": report.legacy_result_available,
        "summary_result_available": report.summary_result_available,
        "method_comparison": report.method_comparison,
        "summary_source_refs": [r.model_dump(mode="json") for r in report.summary_source_refs],
        "legacy_source_refs": [r.model_dump(mode="json") for r in report.legacy_source_refs],
        "summary_diagnostics": report.summary_diagnostics,
        "rules_count_by_source": report.rules_count_by_source,
        "send_gate": send_gate,
    }


def _fmt_pct(v: float | None) -> str:
    return f"{v:.0f}%" if isinstance(v, (int, float)) else " - "


def _e(s: Any) -> str:
    return _esc(str(s)) if s is not None else ""


def _status_badge_html(status: str, label: str | None = None) -> str:
    fg, bg = _STATUS_COLOR.get(status, ("#334155", "#f8fafc"))
    text = _e(label or _OVERALL_RU.get(status, status))
    return (
        f'<span class="cr-badge" style="color:{fg};background:{bg};'
        f'border:1px solid {fg}33">{text}</span>'
    )


def report_to_html(
    report: ComplianceReport,
    doc: ConsultationDocument | None = None,
    rubric_specifics: dict | None = None,
    *,
    send_gate: dict[str, Any] | None = None,
    headline_score: float | None = None,
) -> str:
    """Цветной HTML-отчёт для UI (обезличенный: ФИО → инициалы)."""
    bd = report.score_breakdown
    status = report.overall_status
    status_ru = _OVERALL_RU.get(status, status)
    fg, bg = _STATUS_COLOR.get(status, ("#334155", "#f8fafc"))
    conf_color = "#9a8fb8" if (report.confidence_score or 100) < 55 else "#7aab96"

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
            f'<span class="cr-badge" style="color:#8f5a5a;background:#f5ebeb;border:1px solid #c99a9a55">'
            f"Ограничение балла {_fmt_pct(report.safety_cap.cap_value)}</span>"
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

    sg = _resolve_send_gate(
        report,
        send_gate=send_gate,
        headline_score=headline_score,
    )
    parts.append(_sign_decision_section_html(sg))

    # Резюме (компактно, в спойлере)
    resume_inner: list[str] = ['<ul class="cr-kv">']
    if doc is not None:
        p = doc.patient
        initials = name_to_initials(p.full_name)
        resume_inner.append(f"<li><strong>Пациент:</strong> {_e(initials)}</li>")
        resume_inner.append(
            f"<li><strong>Возраст:</strong> {p.age_years if p.age_years is not None else ' - '} "
            f"({_e(p.age_group)})</li>"
        )
        resume_inner.append(f"<li><strong>Пол:</strong> {_e(_SEX_RU.get(p.sex, p.sex))}</li>")
        cdate = doc.consultation_date.isoformat() if doc.consultation_date else " - "
        resume_inner.append(f"<li><strong>Дата консультации:</strong> {_e(cdate)}</li>")
        resume_inner.append(f"<li><strong>Специальность врача:</strong> {_e(doc.doctor_specialty or ' - ')}</li>")
        diags = ", ".join(
            _e(f"{d.icd10_code or ''} {d.diagnosis_name or d.raw_text}".strip())
            for d in doc.diagnoses
        ) or " - "
        resume_inner.append(f"<li><strong>Диагнозы:</strong> {diags}</li>")
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
        url = protocol_nav_api_path(sp)
        rub = protocol_rubric_label(sp)
        if url:
            inner = (
                f'<a class="cr-src-link" href="{url}" target="_blank" '
                f'rel="noopener noreferrer" title="Открыть навигацию по протоколу">{title}</a>'
            )
            if rub:
                inner += f' <span class="cr-proto-rubric">{_e(rub)}</span>'
            protos_html.append(inner)
        else:
            protos_html.append(title)
    resume_inner.append(
        f"<li><strong>Протоколы:</strong> {', '.join(protos_html) if protos_html else ' - '}</li>"
    )
    resume_inner.append("</ul>")
    parts.append(_spoiler_section("Краткое резюме", "".join(resume_inner), open_default=True))

    if report.protocol_matches:
        proto_inner: list[str] = ['<ul class="cr-list cr-list--sources cr-proto-cards">']
        seen_proto = set()
        for m in report.protocol_matches[:8]:
            sp = m.source_path or ""
            if not sp or sp in seen_proto:
                continue
            seen_proto.add(sp)
            title = _e(
                protocol_display_name(sp, m.protocol_id or "", registry_title=m.document_title)
            )
            url = protocol_nav_api_path(sp)
            rub = protocol_rubric_label(sp)
            proto_inner.append('<li class="cr-proto-card">')
            if url:
                proto_inner.append(
                    f'<a class="cr-src-link cr-proto-card__link" href="{url}" target="_blank" '
                    f'rel="noopener noreferrer" title="Открыть навигацию по протоколу">{title}</a>'
                )
            else:
                proto_inner.append(f'<span class="cr-proto-card__link">{title}</span>')
            if rub:
                proto_inner.append(f'<span class="cr-proto-rubric">{_e(rub)}</span>')
            proto_inner.append("</li>")
        proto_inner.append("</ul>")
        parts.append(_spoiler_section("Подобранные протоколы (карточки каталога)", "".join(proto_inner)))

    # Диагнозы
    if report.diagnosis_assessments:
        diag_inner = ["<ul class='cr-list'>"]
        for a in report.diagnosis_assessments:
            st = _DIAG_RU.get(a.status, a.status)
            code = a.icd10_code or ""
            txt = a.diagnosis_text or ""
            raw_label = txt if (code and txt.upper().startswith(code.upper())) else f"{code} {txt}".strip()
            label = _e(raw_label)
            diag_inner.append(f"<li><span class='cr-icd'>{label}</span> - {_status_badge_html(a.status, st)}</li>")
        diag_inner.append("</ul>")
        parts.append(_spoiler_section("Диагноз", "".join(diag_inner), open_default=True))

    # Обследования
    if report.exam_assessments:
        exam_inner = ["<ul class='cr-list'>"]
        for e in report.exam_assessments:
            st = _EXAM_RU.get(e.status, e.status)
            exam_inner.append(f"<li>{_e(e.exam_name)} - <strong>{_e(st)}</strong></li>")
        exam_inner.append("</ul>")
        parts.append(_spoiler_section("Обследования", "".join(exam_inner)))
    else:
        parts.append(_spoiler_section("Обследования", "<p class='cr-muted'>Детерминированные правила не сработали (нет данных).</p>"))

    # Лечение
    if report.treatment_assessments:
        treat_inner = ["<ul class='cr-list'>"]
        for t in report.treatment_assessments:
            st = _TREAT_RU.get(t.status, t.status)
            treat_inner.append(f"<li>{_e(t.treatment_text)} - <strong>{_e(st)}</strong></li>")
        treat_inner.append("</ul>")
        parts.append(_spoiler_section("Лечение и назначения", "".join(treat_inner), open_default=True))
    else:
        parts.append(_spoiler_section("Лечение и назначения", "<p class='cr-muted'>Назначения не распознаны.</p>"))

    # Безопасность
    safety_inner: list[str]
    if report.safety_assessments:
        safety_inner = ["<ul class='cr-list'>"]
        for s in report.safety_assessments:
            sev = _SEVERITY_RU.get(s.severity, s.severity)
            st = _SAFETY_RU.get(s.status, s.status)
            safety_inner.append(
                f"<li><span class='cr-sev cr-sev--{_e(s.severity)}'>[{_e(sev)}]</span> "
                f"{_e(s.finding_text)} - <strong>{_e(st)}</strong></li>"
            )
        safety_inner.append("</ul>")
    else:
        safety_inner = ["<p class='cr-muted'>Красные флаги не обнаружены.</p>"]
    parts.append(
        _spoiler_section(
            "Красные флаги",
            "".join(safety_inner),
            open_default=bool(report.safety_assessments),
        )
    )

    if report.not_applicable_protocols:
        parts.append(
            '<section class="cr-section cr-section--muted">'
            '<h3 class="cr-section-title">Протоколы не применимы</h3><ul class="cr-list">'
        )
        for m in report.not_applicable_protocols[:6]:
            title = _e(m.document_title or m.protocol_id or " - ")
            parts.append(f"<li>{title} - <em>{_e(m.applicability)}</em></li>")
        parts.append("</ul></section>")

    if report.evidence_map:
        ev_inner = ['<div class="cr-evidence-grid">']
        for ev in report.evidence_map[:16]:
            dec = ev.decision or "unknown"
            dec_label = ev.decision_ru or decision_ru(dec)
            title = ev.title_ru or rule_title_ru(ev.rule_id, {})
            dec_color = {
                "satisfied": "#7aab96",
                "satisfied_by_recommendation": "#8ab5a3",
                "missing": "#c99a9a",
                "not_applicable": "#a8b0b8",
                "manual_review": "#b0a0c4",
            }.get(dec, "#9aa3ab")
            type_label = ev.rule_type_ru or rule_type_ru(ev.rule_type)
            ev_inner.append(
                f'<div class="cr-evidence-card" style="border-left-color:{dec_color}">'
                f'<div class="cr-evidence-card__id">{_e(title)}</div>'
                f'<div class="cr-evidence-card__dec">{_e(dec_label)}'
                f' · <span class="cr-evidence-card__type">{_e(type_label)}</span></div>'
                f'<div class="cr-evidence-card__txt">{_e(ev.explanation or ev.required_item or "")}</div>'
                f"</div>"
            )
        ev_inner.append("</div>")
        parts.append(_spoiler_section("Карта доказательств", "".join(ev_inner)))

    # Рубрика
    if rubric_specifics:
        by_rubric = rubric_specifics.get("by_rubric") or {}
        measurements = rubric_specifics.get("measurements") or {}
        if by_rubric or measurements:
            rub_inner: list[str] = []
            if by_rubric:
                rub_inner.append("<ul class='cr-list'>")
                for slug, info in by_rubric.items():
                    title = (info or {}).get("title", slug)
                    cov = (info or {}).get("term_coverage_pct", 0)
                    rub_inner.append(f"<li><strong>{_e(title)}</strong> - {cov}%</li>")
                rub_inner.append("</ul>")
            if measurements:
                rub_inner.append("<ul class='cr-list cr-list--meas'>")
                for name, m in measurements.items():
                    unit = (m or {}).get("unit") or ""
                    val = (m or {}).get("value") or ""
                    rub_inner.append(f"<li>{_e(name)}: <strong>{_e(val)}</strong> {_e(unit)}</li>")
                rub_inner.append("</ul>")
            parts.append(_spoiler_section("Профиль рубрики", "".join(rub_inner)))

    # Источники
    if report.source_refs:
        src_inner = ["<ul class='cr-list cr-list--sources'>"]
        for r in report.source_refs:
            src_inner.append(f"<li>{_src_line_html(r)}</li>")
        src_inner.append("</ul>")
        parts.append(_spoiler_section("Источники протоколов", "".join(src_inner)))
    else:
        parts.append(_spoiler_section("Источники протоколов", "<p class='cr-muted'>Источники не указаны.</p>"))

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
    mode_label = report.analysis_mode or "legacy"
    L.append(f"- **Режим анализа:** {mode_label}")
    L.append(f"- **Protocol Summary Cards:** {'да' if report.protocol_summary_used else 'нет'}")
    if report.protocol_summary_status:
        L.append(f"- **Статус карточки:** {report.protocol_summary_status}")
    if report.fallback_to_legacy:
        L.append("- **Fallback на legacy:** да")
    elif mode_label in ("summary", "hybrid") and not report.protocol_summary_used:
        L.append("- **Fallback на legacy:** нет (summary не найден - см. diagnostics)")
    if report.rules_count_by_source:
        L.append(
            f"- **Правила по источнику:** summary={report.rules_count_by_source.get('summary', 0)}, "
            f"legacy={report.rules_count_by_source.get('legacy', 0)}"
        )
    if report.summary_diagnostics:
        L.append("- **Diagnostics summary:**")
        for d in report.summary_diagnostics[:5]:
            reasons = ", ".join(d.get("match_reasons") or [])
            L.append(f"  - {d.get('protocol_id') or ' - '}: {reasons}")
    if report.method_comparison:
        mc = report.method_comparison
        L.append(
            f"- **Сравнение с legacy:** Δscore={mc.get('score_delta')}, "
            f"summary evidence={mc.get('summary_rules_in_evidence')}"
        )
    L.append("")
    # 1. Резюме
    L.append("## 1. Краткое резюме")
    if report.source_file:
        L.append(f"- Файл: {report.source_file}")
    if doc is not None:
        p = doc.patient
        L.append(f"- Пациент: {name_to_initials(p.full_name)}")
        L.append(f"- Возраст: {p.age_years if p.age_years is not None else ' - '} ({p.age_group})")
        L.append(f"- Пол: {_SEX_RU.get(p.sex, p.sex)}")
        L.append(f"- Дата консультации: {doc.consultation_date.isoformat() if doc.consultation_date else ' - '}")
        L.append(f"- Специальность врача: {doc.doctor_specialty or ' - '}")
        diags = ", ".join(
            f"{d.icd10_code or ''} {d.diagnosis_name or d.raw_text}".strip()
            for d in doc.diagnoses
        ) or " - "
        L.append(f"- Основные диагнозы: {diags}")
    protos = ", ".join(
        f"{m.document_title or m.protocol_id} [{m.applicability}]"
        for m in report.protocol_matches[:5]
    ) or " - "
    L.append(f"- Подобранные протоколы: {protos}")
    L.append(
        f"- Общая оценка: {_fmt_pct(report.overall_score)} - статус "
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
        L.append(f"- Дата рождения: {p.birth_date.isoformat() if p.birth_date else ' - '}")
        L.append(f"- Возраст на дату консультации: {p.age_years if p.age_years is not None else ' - '}")
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
                f"{d.icd10_code or ' - '} {d.diagnosis_name or d.raw_text}" for d in primary
            ))
        if secondary:
            L.append("- Сопутствующие: " + "; ".join(d.raw_text[:80] for d in secondary[:3]))
        if suspected:
            L.append("- Подозрительные: " + "; ".join(d.raw_text[:80] for d in suspected[:3]))
    L.append(f"_Балл блока: {_fmt_pct(bd.diagnosis_score)}_")
    for a in report.diagnosis_assessments:
        L.append(f"- {a.icd10_code or ''} {a.diagnosis_text} - **{_DIAG_RU.get(a.status, a.status)}**")
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
            L.append(f"- **{m.document_title or m.protocol_id}** - {m.applicability}")
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
            L.append(f"  - {m.document_title or m.protocol_id} - {m.applicability}")
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
            L.append(f"- {e.exam_name} - **{_EXAM_RU.get(e.status, e.status)}**" + (f" ({e.reason})" if e.reason else ""))
    else:
        L.append("- Детерминированные правила по обследованиям не сработали (нет данных).")
    L.append("")

    # 7. Лечение
    L.append("## 7. Проверка лечения")
    L.append(f"_Балл блока: {_fmt_pct(bd.treatment_score)}_")
    if report.treatment_assessments:
        for t in report.treatment_assessments:
            L.append(f"- {t.treatment_text} - **{_TREAT_RU.get(t.status, t.status)}**")
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
            L.append(f"- [{_SEVERITY_RU.get(s.severity, s.severity)}] {s.finding_text} - **{_SAFETY_RU.get(s.status, s.status)}**")
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
                L.append(f"- Повторная явка: {fu.date.isoformat() if fu.date else fu.raw_text or ' - '}")
        elif doc.sections.follow_up_text:
            L.append(f"- {doc.sections.follow_up_text[:200]}")
        else:
            L.append("- Дата или срок повторной явки не указаны.")
    L.append("")

    # 9a. Evidence map
    if report.evidence_map:
        L.append("## 9a. Карта доказательств (evidence map)")
        for ev in report.evidence_map[:20]:
            title = ev.title_ru or rule_title_ru(ev.rule_id, {})
            dec_label = ev.decision_ru or decision_ru(ev.decision)
            L.append(f"- **{title}** ({dec_label}): {ev.explanation or ' - '}")
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
                L.append(f"- **{title}** - покрытие профильных понятий {cov}%")
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
        "Документ - инструмент экспертной проверки соответствия клиническим протоколам."
    )
    return "\n".join(L)
