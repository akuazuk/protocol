"""Единый движок отчётов МО и тексты рассылок (фаза 8)."""
from __future__ import annotations

import html
from datetime import date, datetime, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

MINSK = ZoneInfo("Europe/Minsk")


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def report_footer(*, period: str, revision: Any, methodology: str, slice_url: str) -> str:
    return (
        f"<footer class='report-footer'>Период {_esc(period)} · ревизия {_esc(revision)} · "
        f"методика {_esc(methodology)} · "
        f"<a href='{_esc(slice_url)}'>открыть срез</a> · "
        f"сформировано {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</footer>"
    )


def render_day_briefing_html(
    report: Mapping[str, Any],
    *,
    base_url: str = "https://protocol-bimy.onrender.com",
    methodology: str = "v3",
) -> str:
    day = str(report.get("date") or "")
    summary = report.get("summary") or {}
    quality = report.get("quality") or {}
    queue = list(report.get("action_queue") or [])[:12]
    slice_url = f"{base_url.rstrip('/')}/methodist/mis-kz-quality.html#yesterday?date={day}"
    cards = "".join(
        f"<div class='kpi'><span>{_esc(label)}</span><b>{_esc(value)}</b></div>"
        for label, value in (
            ("Записей", summary.get("source_rows")),
            ("Оценено", summary.get("scored") or summary.get("eligible_rows")),
            ("Средняя", summary.get("avg_score")),
            ("Требуют внимания", summary.get("needs_attention")),
            ("Критические", summary.get("critical")),
        )
    )
    queue_rows = "".join(
        "<tr><td>{prio}</td><td>{doc}</td><td>{score}</td><td>{reason}</td>"
        "<td><a href='{link}'>МО</a> · <a href='{pdf}'>PDF</a></td></tr>".format(
            prio=_esc(item.get("priority")),
            doc=_esc(item.get("doctor_fio")),
            score=_esc(item.get("score")),
            reason=_esc(item.get("reason")),
            link=_esc(f"{base_url.rstrip('/')}/methodist/mis-kz-quality.html#queue?case={item.get('visit_id') or item.get('mis_id')}"),
            pdf=_esc(f"{base_url.rstrip('/')}/api/methodist/mo/cases/{item.get('visit_id') or item.get('mis_id')}/document"),
        )
        for item in queue
    ) or "<tr><td colspan='5'>Срочных случаев нет.</td></tr>"
    passed = "да" if (quality.get("passed") if isinstance(quality, dict) else True) else "нет"
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Брифинг МО { _esc(day) }</title>
<style>
body{{font:14px/1.5 Avenir Next,Segoe UI,sans-serif;color:#1f2a37;background:#f5f7fa;margin:0}}
main{{max-width:960px;margin:auto;padding:24px}}.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}
.kpi{{background:#fff;border:1px solid #d9e2ec;border-radius:14px;padding:12px}}.kpi span{{color:#66788a;display:block}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:14px;overflow:hidden}}
td,th{{border-bottom:1px solid #e4e9ef;padding:8px;text-align:left}}
.report-footer{{margin-top:18px;color:#66788a;font-size:12px}}
</style></head><body><main>
<h1>Утренний брифинг МО · {_esc(day)}</h1>
<p>Качество приёма: {_esc(passed)}. Откройте срез: <a href="{_esc(slice_url)}">{_esc(slice_url)}</a></p>
<div class="kpis">{cards}</div>
<h2>Требуют внимания / критические</h2>
<table><thead><tr><th>Приоритет</th><th>Врач</th><th>Оценка</th><th>Причина</th><th>Документ</th></tr></thead>
<tbody>{queue_rows}</tbody></table>
{report_footer(period=day, revision=report.get("revision"), methodology=methodology, slice_url=slice_url)}
</main></body></html>"""


def build_telegram_briefing(
    report: Mapping[str, Any],
    *,
    base_url: str = "https://protocol-bimy.onrender.com",
) -> str:
    day = str(report.get("date") or date.today().isoformat())
    summary = report.get("summary") or {}
    queue = list(report.get("action_queue") or [])
    critical = int(summary.get("critical") or 0)
    attention = int(summary.get("needs_attention") or 0)
    avg = summary.get("avg_score")
    slice_url = f"{base_url.rstrip('/')}/methodist/mis-kz-quality.html#yesterday?date={day}"
    lines = [
        f"МО брифинг за {day}",
        f"Средняя: {avg if avg is not None else 'н/д'}%",
        f"Оценено: {summary.get('scored') or summary.get('eligible_rows') or 'н/д'} из {summary.get('source_rows') or 'н/д'}",
        f"Требуют внимания: {attention}",
        f"Критические: {critical}",
        f"Срез: {slice_url}",
    ]
    for item in queue[:5]:
        case_id = item.get("visit_id") or item.get("mis_id")
        lines.append(
            f"- {item.get('priority')}: {item.get('doctor_fio')} · {item.get('score')}% · "
            f"{base_url.rstrip('/')}/api/methodist/mo/cases/{case_id}/document"
        )
    if not queue:
        lines.append("Срочной очереди нет.")
    return "\n".join(lines)


def build_anomaly_flags(daily_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Простые z-score флаги по средней оценке и объёму (фаза 8.6 baseline)."""
    if len(daily_rows) < 8:
        return []
    flags: list[dict[str, Any]] = []
    for metric in ("avg_score", "source_rows"):
        values = [float(row[metric]) for row in daily_rows if row.get(metric) is not None]
        if len(values) < 8:
            continue
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        std = var ** 0.5
        if std <= 1e-9:
            continue
        latest = daily_rows[-1]
        current = latest.get(metric)
        if current is None:
            continue
        z = (float(current) - mean) / std
        if abs(z) >= 2.5:
            flags.append(
                {
                    "metric": metric,
                    "date": latest.get("date") or latest.get("visit_date"),
                    "z_score": round(z, 2),
                    "value": current,
                    "mean": round(mean, 2),
                }
            )
    return flags
