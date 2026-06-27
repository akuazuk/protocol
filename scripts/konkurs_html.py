"""Print-ready HTML для пакета конкурса Белинфонд (PDF через Chrome headless)."""
from __future__ import annotations

import html
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from fill_konkurs_docx import (  # noqa: E402
    PASSPORT_ACHIEVEMENTS,
    PASSPORT_ADVANTAGES,
    PASSPORT_CONSUMERS,
    PASSPORT_DESC,
    PASSPORT_EXTRA,
    PASSPORT_PRODUCT_CERT,
    PASSPORT_TIMELINE,
    STRATEGY_5Y,
    STRATEGY_LEVELS,
    STRATEGY_METHODS,
    STRATEGY_YEAR,
    STRATEGY_YEAR_PLAN,
    TEAM,
)
from konkurs_bp_sections import RESUME, SECTIONS, TOC  # noqa: E402
from konkurs_finance import (  # noqa: E402
    B2C_AVG_PRICE,
    B2C_TIERS,
    CERTIFICATE_BYN,
    CLINIC_B2C_REV_Y1_K,
    CLINIC_B2C_REV_Y2_K,
    CLINIC_B2C_REV_Y3_K,
    CLINIC_B2C_REVSHARE,
    FIN_Y1,
    FIN_Y2,
    FIN_Y3,
    KRAVIRA_B2B_YEAR,
    KRAVIRA_KZ_MONTH,
    MARKET_KZ_MONTH,
    MARKET_KZ_YEAR,
    ROI_NET,
    ROI_PROTOCOL_COST,
    ROI_TOTAL_SAVING,
    SAM_KZ_YEAR,
    SOM_Y3_KZ_YEAR,
    TAM_B2B_CEILING_YEAR_K,
    TAM_REVENUE_YEAR,
    Y3_MARKET_SHARE,
    ebitda_k,
    ebitda_month_k,
    total_rev_k,
)
from konkurs_scenarios import (  # noqa: E402
    B2C_CONV_SENSITIVITY,
    B2C_PROTOCOL_PER_CHECK,
    CHANNEL_TABLE,
    MONTHLY_Y3_CAUTIOUS,
    PENETRATION_SENSITIVITY,
    SCENARIO_BASE,
    SCENARIO_CAUTIOUS,
    SCENARIO_COMPARE_TABLE,
    SCENARIO_OPTIMISTIC,
    TAM_BRIDGE,
)
from konkurs_expansion import (  # noqa: E402
    B2C_TAM_COMPARE_TABLE,
    EXPANSION_INTRO,
    EXPANSION_ML_PORTABILITY,
    EXPANSION_PUBLIC_SECTOR,
    EXPANSION_RISKS,
    EXPANSION_RUSSIA,
    BELSTAT_AMBULATORY_VISITS_YEAR,
)
from konkurs_expansion_intl import (  # noqa: E402
    CALC_METHODOLOGY,
    GTM_PHASES,
    GTM_STEPS,
    INTL_CORPUS_TABLE,
    INTL_ENTRY_COST_TABLE,
    INTL_EXPANSION_INTRO,
    INTL_FLYWHEEL,
    INTL_MARKET_BASE_TABLE,
    INTL_MARKET_CAUTIOUS_TABLE,
    INTL_RISKS,
    INTL_TIER1,
    INTL_TIER2,
    INTL_UPSIDE_BASE_K,
    INTL_UPSIDE_CAUTIOUS_K,
)
from konkurs_future import (  # noqa: E402
    COMMUNITY_VISION,
    FUTURE_RISKS,
    FUTURE_ROADMAP_TABLE,
    FUTURE_STREAMS,
    FUTURE_VISION,
    PLATFORM_INTEGRATIONS,
)
from konkurs_expansion_us import (  # noqa: E402
    US_B2C_BASE_K,
    US_CORPUS_TABLE,
    US_ENTRY_COST_K,
    US_EXPANSION_INTRO,
    US_GTM_PHASES,
    US_GTM_STEPS,
    US_INSURANCE_ECON_TABLE,
    US_INSURANCE_MODEL,
    US_PRODUCT_TABLE,
    US_REVENUE_STACK_TABLE,
    US_RISKS,
    US_STAKEHOLDER_TABLE,
    US_TAM_TABLE,
    US_TOTAL_UPSIDE_K,
    US_VALUE_PROP,
)
from konkurs_impact import (  # noqa: E402
    CISZ_CONTEXT,
    ECOSYSTEM_FLYWHEEL,
    ECOSYSTEM_GOVERNANCE,
    ECOSYSTEM_INTRO,
    GLOBAL_ANALOGUES,
    MARKET_CONTEXT,
    STAKEHOLDER_BENEFITS,
)
from konkurs_b2c_ux import (  # noqa: E402
    B2C_OUT_OF_SCOPE,
    B2C_PRICING_TABLE,
    B2C_REPORT_TABLE,
    B2C_REVSHARE_EXAMPLES,
    B2C_SCENARIOS,
    B2C_UX_INTRO,
    MARKET_SCOPE_NOTE,
)
from konkurs_market import CISZ_DRIVERS, COMPETITOR_MATRIX, INVESTMENT_PLAN, RB_MARKET_TABLE  # noqa: E402
from konkurs_monetization import (  # noqa: E402
    ALL_REVENUE_SCENARIOS_Y3,
    EXPANDED_Y3,
    MONETIZATION_INTRO,
    MONETIZATION_TABLE,
    WEIGHTED_EXTRA_Y3_K,
)
from konkurs_continuous_ml import (  # noqa: E402
    CONTINUOUS_ML_INTRO,
    ML_AB_INTERPRETATION,
    ML_AB_KZ_TABLE,
    ML_APPENDIX_TABLE,
    ML_COMPETITION_NOTE,
    ML_DATA_CYCLE,
    ML_EXPERIMENT_EMBEDDER_TABLE,
    ML_PRINCIPLES_TABLE,
    ML_ROADMAP_TABLE,
)
from konkurs_glossary import (  # noqa: E402
    FORMULA_TABLE,
    GLOSSARY_INTRO,
    GLOSSARY_TABLE,
    build_calc_audit_table,
)
from konkurs_org import (  # noqa: E402
    CONTACT_PERSON,
    DIRECTOR,
    EMAIL,
    NOMINATION,
    ORG_ADDRESS,
    ORG_NAME,
    ORG_SHORT,
    ORG_UNP,
    PASSPORT_IP,
    PHONE,
    PROJECT_NAME as ORG_PROJECT,
    WEB,
)

ROOT = Path(__file__).resolve().parents[1]
KONKURS_DIR = ROOT / "docs" / "konkurs"

PRINT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap');
:root {
  --sage: #7a9e8e; --sage-light: #d4e8de; --sage-pale: #eef5f1;
  --blue-pale: #dce8f0; --lavender: #e8e0f0; --peach: #f5ebe4;
  --text: #2d3d38; --muted: #5c6f68; --border: rgba(122,158,142,.22);
}
* { box-sizing: border-box; }
body { margin: 0; font-family: "DM Sans", system-ui, sans-serif; color: var(--text);
  font-size: 10.5pt; line-height: 1.58; background: #f7f5f2; }
.wrap { max-width: 210mm; margin: 0 auto; padding: 0; }
h1,h2,h3 { font-family: Outfit, sans-serif; color: #3d5c52; page-break-after: avoid; font-weight: 600; }
h1 { font-size: 22pt; margin: 0 0 4mm; letter-spacing: .01em; }
h2 { font-size: 13.5pt; margin: 8mm 0 3mm; padding-bottom: 2mm; border-bottom: 2px solid var(--sage-light); }
h3 { font-size: 11pt; margin: 5mm 0 2mm; color: var(--sage); }
p { margin: 0 0 3mm; text-align: justify; hyphens: auto; }
ul { margin: 2mm 0 4mm; padding-left: 5mm; }
li { margin-bottom: 1.5mm; }
.muted { color: var(--muted); font-size: 9pt; }
.cover {
  min-height: 267mm; display: flex; flex-direction: column; justify-content: center;
  text-align: center; padding: 20mm 18mm; page-break-after: always;
  background: linear-gradient(165deg, #fff 0%, var(--sage-pale) 50%, #fff 100%);
  border-bottom: 3px solid var(--sage-light);
}
.cover .badge { display: inline-block; font-size: 9pt; color: var(--muted); letter-spacing: .08em;
  text-transform: uppercase; margin-bottom: 6mm; }
.cover .title { font-size: 26pt; font-weight: 600; color: #3d5c52; margin: 4mm 0; }
.cover .subtitle { font-size: 12.5pt; color: var(--muted); margin-bottom: 8mm; }
.cover .project { font-size: 11.5pt; font-weight: 600; max-width: 140mm; margin: 0 auto 10mm; line-height: 1.45; }
.cover .meta { text-align: left; max-width: 150mm; margin: 10mm auto 0; font-size: 10pt; line-height: 1.65;
  background: #fff; padding: 6mm 8mm; border-radius: 4mm; border: 1px solid var(--border); }
.cover hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, var(--sage-light), transparent);
  margin: 8mm auto; width: 70%; }
.cover .logo-lockup { margin: 0 auto 8mm; }
.cover .logo-lockup img { height: 14mm; width: auto; max-width: 92%; }
.cover .logo-lockup--hero { margin: 0 auto 14mm; width: 168mm; max-width: 90%; }
.cover .logo-lockup--hero img { width: 100%; height: auto; display: block; }
.cover .logo-emblem { margin: 0 auto 4mm; }
.cover .logo-emblem img { height: 18mm; width: auto; }
.mission {
  page-break-after: always; padding: 14mm 16mm;
  background: linear-gradient(135deg, #6b9080 0%, #7a9e8e 100%); color: #fff;
}
.mission h2 { color: #fff; border-bottom-color: rgba(255,255,255,.3); font-size: 15pt; }
.mission p { text-align: left; opacity: .96; }
.glossary-block { page-break-inside: avoid; }
.glossary-block .abbr { font-weight: 700; color: var(--sage); white-space: nowrap; }
.section { padding: 10mm 16mm 4mm; page-break-inside: avoid; }
.section-body { padding: 0 16mm 6mm; }
.card { background: #fff; border: 1px solid var(--border); border-radius: 4mm; padding: 4mm 5mm;
  margin-bottom: 4mm; box-shadow: 0 1px 3px rgba(61,92,82,.05); page-break-inside: avoid; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
.stake { border-left: 3px solid var(--sage-light); padding-left: 4mm; background: var(--sage-pale); border-radius: 0 3mm 3mm 0; }
.stake h3 { margin-top: 0; }
.stake .tag { font-size: 9pt; color: var(--sage); font-weight: 600; margin-bottom: 2mm; }
table { width: 100%; border-collapse: collapse; font-size: 9pt; margin: 3mm 0 5mm; page-break-inside: avoid; }
th, td { padding: 2.4mm 3mm; border: 1px solid #e0ebe6; text-align: left; vertical-align: top; }
th { background: var(--sage-light); color: #2d4a3e; font-weight: 600; }
tr:nth-child(even) td { background: var(--sage-pale); }
.caption { font-size: 9pt; font-weight: 600; color: #3d5c52; margin: 4mm 0 2mm; }
.chart { text-align: center; margin: 5mm 0 7mm; page-break-inside: avoid; padding: 3mm;
  background: #fff; border-radius: 3mm; border: 1px solid var(--border); }
.chart img { max-width: 100%; height: auto; border-radius: 2mm; }
.toc { padding: 12mm 16mm; page-break-after: always; }
.toc pre { font-family: "DM Sans", sans-serif; font-size: 10pt; line-height: 1.7; white-space: pre-wrap;
  background: #fff; padding: 6mm; border-radius: 4mm; border: 1px solid var(--border); }
.highlight { background: var(--peach); border-left: 3px solid #c9a88a; padding: 4mm 5mm; margin: 4mm 0;
  border-radius: 0 3mm 3mm 0; font-size: 10pt; }
.stats { display: flex; flex-wrap: wrap; gap: 3mm; margin: 4mm 0 6mm; }
.stat { flex: 1 1 38mm; background: #fff; border: 1px solid var(--border); border-radius: 3mm;
  padding: 3mm 4mm; text-align: center; }
.stat b { display: block; font-size: 12.5pt; color: #3d5c52; }
.form-table td:first-child { width: 38%; font-weight: 600; background: var(--sage-pale); }
@media print {
  body { background: #fff; }
  .section, .section-body { padding-left: 12mm; padding-right: 12mm; }
}
@page { size: A4; margin: 12mm 10mm; }
.scenario { border: 1px solid var(--border); border-radius: 4mm; padding: 4mm 5mm; margin-bottom: 5mm;
  page-break-inside: avoid; background: #fff; }
.scenario h3 { margin-top: 0; color: #3d5c52; }
.scenario .tag { font-size: 8.5pt; color: var(--sage); font-weight: 600; margin-bottom: 2mm; }
.flow { font-family: monospace; font-size: 8.5pt; background: var(--sage-pale); padding: 3mm 4mm;
  border-radius: 2mm; margin: 2mm 0 3mm; word-break: break-word; }
.flow-cycle { font-size: 9pt; background: #fff; border: 1px solid var(--border); border-radius: 3mm;
  padding: 4mm 5mm; margin: 4mm 0 5mm; page-break-inside: avoid; }
.flow-cycle .step { margin-bottom: 3mm; padding-left: 3mm; border-left: 2px solid var(--sage); }
.flow-cycle .step b { color: #3d5c52; }
"""


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def _ps(text: str) -> str:
    parts = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return "".join(f"<p>{_e(p.replace(chr(10), ' '))}</p>" for p in parts)


def _table(headers: list[str], rows: list[tuple | list], *, caption: str = "", css_class: str = "") -> str:
    cap = f'<div class="caption">{_e(caption)}</div>' if caption else ""
    cls = f' class="{css_class}"' if css_class else ""
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{_e(str(c))}</td>" for c in row) + "</tr>"
    return f'{cap}<table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _page(title: str, body: str, *, extra_head: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8"/>
<title>{_e(title)}</title>
<style>{PRINT_CSS}</style>{extra_head}
</head><body><div class="wrap">{body}</div></body></html>"""


def _cover(title: str, doc_kind: str, *, show_emblem: bool = True, hero_logo: bool = False) -> str:
    logo_rel = "../../protocol-logo-wordmark.svg"
    emblem_rel = "../../protocol-logo.svg"
    emblem_html = (
        f'  <div class="logo-emblem"><img src="{emblem_rel}" alt="Protocol"/></div>\n'
        if show_emblem
        else ""
    )
    lockup_cls = "logo-lockup logo-lockup--hero" if hero_logo else "logo-lockup"
    return f"""
<div class="cover">
{emblem_html}  <div class="{lockup_cls}"><img src="{logo_rel}" alt="Protocol"/></div>
  <div class="badge">Белинфонд · 2026 · {_e(NOMINATION)}</div>
  <div class="title">{_e(title)}</div>
  <div class="subtitle">{_e(doc_kind)}</div>
  <hr/>
  <div class="project">{_e(ORG_PROJECT)}</div>
  <hr/>
  <div class="meta">
    <strong>Участник:</strong> {_e(ORG_NAME)} ({_e(ORG_SHORT)})<br/>
    {_e(ORG_ADDRESS)} · УНП {_e(ORG_UNP)}<br/>
    <strong>Руководитель:</strong> {_e(DIRECTOR)}<br/><br/>
    <strong>Контакт:</strong> {_e(CONTACT_PERSON)}<br/>
    {_e(PHONE)} · {_e(EMAIL)} · {_e(WEB)}<br/><br/>
    <span class="muted">Минск · 2026</span>
  </div>
</div>"""


def _mission_block() -> str:
    return f"""
<div class="mission">
  <h2>Миссия Protocol</h2>
  <p>Protocol - платформа, которая делает качество амбулаторной помощи <strong>измеримым и прозрачным</strong>
  для каждого консультативного заключения - до подписи врачом и до попадания данных в ЦИСЗ. Мы не заменяем врача
  и МЭЭ; мы даём инструмент соответствия <strong>478 клиническим протоколам Минздрава</strong> на потоке
  <strong>100% КЗ</strong>, а пациенту - право проверить заключение по тем же стандартам.</p>
  <p>История продукта начинается в МЦ «Кравира» - якорном медцентре, где Protocol прошёл первый production-пилот
  (25 000 КЗ/мес). Кравира - партнёр и участник конкурса; бренд и масштабирование - <strong>Protocol</strong>.</p>
  <p>{_e(MARKET_CONTEXT.replace(chr(10), ' '))}</p>
  <p>{_e(CISZ_CONTEXT.replace(chr(10), ' '))}</p>
</div>"""


def _stakeholders_html() -> str:
    cards = []
    for key in ("patients", "doctors", "clinics", "state"):
        s = STAKEHOLDER_BENEFITS[key]
        lis = "".join(f"<li>{_e(p)}</li>" for p in s["points"])
        cards.append(f"""
<div class="card stake">
  <span class="icon">{s['icon']}</span>
  <h3>{_e(s['title'])}</h3>
  <div class="tag">{_e(s['summary'])}</div>
  <ul>{lis}</ul>
</div>""")
    return f"""
<div class="section"><h2>Социальная значимость: кому и как помогает Protocol</h2></div>
<div class="section-body"><div class="grid2">{''.join(cards)}</div></div>"""


def _ecosystem_html() -> str:
    rows = [(a, b, c) for a, b, c in ECOSYSTEM_FLYWHEEL]
    return f"""
<div class="section"><h2>3.1. Экосистема B2C ↔ B2B ↔ государство: саморегулирующийся контур</h2></div>
<div class="section-body">
<p>{_e(ECOSYSTEM_INTRO.replace(chr(10), ' '))}</p>
{_table(['Участник / этап', 'Что происходит', 'Эффект'], rows, caption='Flywheel: как интересы пациента, клиники и государства усиливают друг друга')}
<p>{_e(ECOSYSTEM_GOVERNANCE.replace(chr(10), ' '))}</p>
<div class="highlight"><strong>Ключевая идея для инвестора и регулятора:</strong> массовое использование B2C пациентами
создаёт рыночное давление на клиники без отдельного административного приказа; клиники с B2B закрывают
проблемы до подписи ЭЦП; государство получает более чистые данные в ЦИСЗ и исполнение КП при роли
«контролёра нормативки», а не операционного аудитора каждого КЗ. Protocol постоянно развивает patient-блок
и B2B-ядро на одном evidence_map - цикл самоконтроля рынка.</div>
</div>"""


def _analogues_html() -> str:
    rows = [(a[0], a[1], a[2], a[4]) for a in GLOBAL_ANALOGUES]
    return f"""
<div class="section"><h2>Мировые аналоги и отличие Protocol</h2>
<p class="muted">CDSS (Clinical Decision Support) существуют более 40 лет (AHRQ/ONC); &gt;90% больниц США используют EHR с CDS.
Зарубежные решения не закрывают связку «КП конкретной страны + предподпись ЭЦП + FHIR + B2C для пациента» для Беларуси.</p></div>
<div class="section-body">
{_table(['Решение', 'Регион', 'Тип', 'Почему не заменяет Protocol'], rows, caption='Таблица. Сравнение с мировыми CDSS')}
</div>"""


SECTION_TITLES = {
    "4.": "4. Описание проекта",
    "5.": "5. Описание продукции (услуг)",
    "6.": "6. Анализ рынка и маркетинг",
    "7.": "7. Интеллектуальная собственность",
    "8.": "8. Потребители и сбыт",
    "9.": "9. Ценообразование",
    "10.": "10. Конкуренты",
    "11.": "11. Поставщики",
    "12.": "12. Производственный план",
    "13.": "13. Организационный план",
    "14.": "14. Риски",
    "15.": "15. Финансовый план",
    "16.": "16. Иные сведения",
    "17.": "17. Непрерывное дообучение моделей",
}


def _market_scope_html() -> str:
    tier_rows = [(t["name"], t["specialty"], f'{t["price"]:.2f}'.replace(".", ","), f'{t["mix"] * 100:.0f}%') for t in B2C_TIERS]
    return f"""
<div class="section"><h2>Масштаб рынка: все частные ОЗ и все пациенты РБ</h2></div>
<div class="section-body">
<p>{_e(MARKET_SCOPE_NOTE.replace(chr(10), ' '))}</p>
<div class="stats">
  <div class="stat"><b>~1%</b><span class="muted">якорный пилот Кравира</span></div>
  <div class="stat"><b>{MARKET_KZ_MONTH:,}</b><span class="muted">КЗ/мес частный сектор</span></div>
  <div class="stat"><b>8</b><span class="muted">клиник B2B к 2029</span></div>
  <div class="stat"><b>{CLINIC_B2C_REV_Y3_K}</b><span class="muted">тыс. BYN rev-share клиникам</span></div>
</div>
{_table(['Tier B2C', 'Specialty', 'BYN', 'Микс'], tier_rows, caption='Tier-цены B2C (средний чек пациента ~8,33 BYN)')}
</div>"""


def _sections_html() -> str:
    out = []
    for key, title in SECTION_TITLES.items():
        body = SECTIONS.get(key, "")
        out.append(f'<div class="section"><h2>{_e(title)}</h2></div>')
        out.append(f'<div class="section-body">{_ps(body)}</div>')
    return "".join(out)


def _b2c_ux_html() -> str:
    scenarios_html = []
    for s in B2C_SCENARIOS:
        landing = "".join(f"<li>{_e(x)}</li>" for x in s["landing"])
        l1 = "".join(f"<li>{_e(x)}</li>" for x in s["report_l1"])
        scenarios_html.append(f"""
<div class="scenario">
  <div class="tag">Сценарий {s['id']} · {_e(s['priority'])}</div>
  <h3>{_e(s['title'])}</h3>
  <p><strong>Триггер:</strong> {_e(s['trigger'])}</p>
  <div class="flow">{_e(s['path'])}</div>
  <p><strong>Landing:</strong></p><ul>{landing}</ul>
  <p><strong>Отчёт L1:</strong></p><ul>{l1}</ul>
  <p><strong>L2:</strong> {_e(s['report_l2'])}</p>
  <p class="muted"><strong>Scope:</strong> {_e(s['scope'])}</p>
</div>""")
    oos = "".join(f"<li>{_e(x)}</li>" for x in B2C_OUT_OF_SCOPE)
    return f"""
<div class="section"><h2>8.1. B2C: MVP, UX, tier-цены и rev-share (национальный рынок)</h2></div>
<div class="section-body">
<p>{_e(B2C_UX_INTRO.replace(chr(10), ' '))}</p>
{''.join(scenarios_html)}
{_table(['Tier', 'Specialty / приём', 'BYN', 'Микс'], B2C_PRICING_TABLE, caption='Tier-цены B2C по сложности приёма')}
{_table(['Продукт', 'Оплата пациента', 'Клинике 30%', 'Protocol 70%'], B2C_REVSHARE_EXAMPLES, caption='Rev-share при оплате по SMS/QR-ссылке клиники')}
{_table(['Блок отчёта', 'Содержание', 'Зачем'], B2C_REPORT_TABLE, caption='Patient view отчёта L1/L2')}
{_table(['Сценарий', 'Конверсия', 'Сложность'], [
  ('1. QR после визита', 'Средняя', 'Минимальная - любая ОЗ-партнёр'),
  ('2. SMS/email rev-share', 'Высокая', 'Основной канал масштаба'),
  ('3. Национальный SEO', 'Средняя', '2028+, без rev-share'),
], caption='Приоритет запуска B2C')}
<div class="highlight">B2C MVP развёрнут (patient.html, tier P1, API /api/patient/*). Коммерческий запуск QR + SMS rev-share и ERIP - Q4 2026 / 2027.
Rev-share {int(CLINIC_B2C_REVSHARE * 100)}% мотивирует клиники рассылать ссылку - доп. доход ~{CLINIC_B2C_REV_Y3_K} тыс. BYN/год к 2029 (осторожный сценарий, B2C {FIN_Y3['b2c_k']} тыс. Protocol).</div>
<p><strong>Вне scope (не делаем в 2026-2027):</strong></p><ul>{oos}</ul>
</div>"""


def _expansion_html() -> str:
    risk_rows = [(a, b, c) for a, b, c in EXPANSION_RISKS]
    tam_rows = [(a, b, c, d, e, f) for a, b, c, d, e, f in B2C_TAM_COMPARE_TABLE]
    return f"""
<div class="section"><h2>6.2. Расширение B2C: госполиклиники РБ и экспорт (РФ)</h2></div>
<div class="section-body">
<p>{_e(EXPANSION_INTRO.replace(chr(10), ' '))}</p>
<p>{_e(EXPANSION_PUBLIC_SECTOR.replace(chr(10), ' '))}</p>
<div class="highlight"><strong>Источник:</strong> Белстат - <strong>108,4 млн</strong> посещений врачей в РБ в 2025 г.
(амбулаторно + на дому), ~12 на жителя. Базовый план B2C использует знаменатель <strong>30 млн</strong> (частный сектор) -
это <strong>не противоречие</strong>, а сознательно осторожный горизонт 2029; расширенный TAM B2C - отдельный upside.</div>
{_table(['Знаменатель TAM B2C', 'Обращений/год', 'Конверсия', 'Проверок/год', 'B2C Protocol, тыс. BYN', 'Примечание'], tam_rows, caption='Сравнение TAM B2C: только выручка B2C Protocol (без B2B, API, OPEX и EBITDA)')}
<p>{_e(EXPANSION_RUSSIA.replace(chr(10), ' '))}</p>
<h3>Переносимость ML и корпуса протоколов</h3>
<p>{_e(EXPANSION_ML_PORTABILITY.replace(chr(10), ' '))}</p>
{_table(['Риск', 'Уровень', 'Митигация'], risk_rows, caption='Риски расширения B2C / экспорта')}
</div>"""


def _intl_expansion_html() -> str:
    caut_rows = list(INTL_MARKET_CAUTIOUS_TABLE)
    base_rows = list(INTL_MARKET_BASE_TABLE)
    corpus_rows = list(INTL_CORPUS_TABLE)
    cost_rows = list(INTL_ENTRY_COST_TABLE)
    gtm_rows = list(GTM_PHASES)
    step_rows = list(GTM_STEPS)
    intl_risk_rows = [(a, b, c) for a, b, c in INTL_RISKS]
    upside_caut = f"{INTL_UPSIDE_CAUTIOUS_K:,}".replace(",", " ")
    upside_base = f"{INTL_UPSIDE_BASE_K:,}".replace(",", " ")
    return f"""
<div class="section"><h2>6.3. Международная экспансия: рынки с гос. регламентами</h2></div>
<div class="section-body">
<p>{_e(INTL_EXPANSION_INTRO.replace(chr(10), ' '))}</p>
<p>{_e(INTL_TIER1.replace(chr(10), ' '))}</p>
<p>{_e(INTL_TIER2.replace(chr(10), ' '))}</p>
<p>{_e(INTL_FLYWHEEL.replace(chr(10), ' '))}</p>
<div class="highlight"><strong>Методика расчёта:</strong> {_e(CALC_METHODOLOGY.replace(chr(10), ' '))}</div>
<div class="stats">
  <div class="stat"><b>{upside_caut}</b><span class="muted">тыс. BYN B2C / 9 стран (осторожная conv.)</span></div>
  <div class="stat"><b>{upside_base}</b><span class="muted">тыс. BYN B2C / 9 стран (базовая conv.)</span></div>
  <div class="stat"><b>9</b><span class="muted">целевых рынков Tier 1-2</span></div>
</div>
{_table(['Страна', 'Tier', 'Обращений/год', 'Conv.', 'Проверок/год', 'B2C тыс. BYN', 'Запуск'], caut_rows, caption='Международный upside B2C · осторожная конверсия (не в EBITDA 2029)')}
{_table(['Страна', 'Tier', 'Обращений/год', 'Conv.', 'Проверок/год', 'B2C тыс. BYN', 'Запуск'], base_rows, caption='Международный upside B2C · базовая конверсия (сценарий 2033)')}
{_table(['Страна', 'Корпус регламентов', 'Портал', 'Язык', 'Tier'], corpus_rows, caption='Официальные репозитории клинических регламентов')}
{_table(['Фаза', 'География', 'Действия', 'B2C upside', 'Inv. тыс. BYN'], gtm_rows, caption='Дорожная карта выхода на рынки 2029-2033')}
{_table(['Шаг', 'Содержание', 'Срок', 'Критерий'], step_rows, caption='Playbook входа на новый рынок (7 шагов)')}
{_table(['Страна', 'Inv. год 1, тыс. BYN', 'B2C launch', 'B2B horizon'], cost_rows, caption='Оценка инвестиций входа по странам (corpus + legal + ML + marketing)')}
{_table(['Риск', 'Уровень', 'Митигация'], intl_risk_rows, caption='Риски международной экспансии')}
<p class="muted">Tier 3 EU/NICE и Китай - вне горизонта 2033. **США** - отдельный стратегический рынок (§6.5):
Epic/CDS, payer economics, Visit Prep; не смешивать с таблицами Tier 1-2 выше.</p>
</div>"""


def _us_expansion_html(assets_rel: str = "_assets") -> str:
    def img(name: str, cap: str) -> str:
        return f'<div class="chart"><div class="caption">{_e(cap)}</div><img src="{assets_rel}/{name}" alt=""/></div>'

    stake_rows = list(US_STAKEHOLDER_TABLE)
    corpus_rows = list(US_CORPUS_TABLE)
    product_rows = list(US_PRODUCT_TABLE)
    tam_rows = list(US_TAM_TABLE)
    rev_rows = list(US_REVENUE_STACK_TABLE)
    gtm_rows = list(US_GTM_PHASES)
    step_rows = list(US_GTM_STEPS)
    econ_rows = list(US_INSURANCE_ECON_TABLE)
    risk_rows = [(a, b, c) for a, b, c in US_RISKS]
    upside = f"{US_TOTAL_UPSIDE_K:,}".replace(",", " ")
    b2c_base = f"{US_B2C_BASE_K:,}".replace(",", " ")
    return f"""
<div class="section"><h2>6.5. Выход на рынок США: Visit Prep, payer flywheel и страховая экономика</h2></div>
<div class="section-body">
<p>{_e(US_EXPANSION_INTRO.replace(chr(10), ' '))}</p>
<p>{_e(US_VALUE_PROP.replace(chr(10), ' '))}</p>
<h3>Страховая и сетевая модель: кто выигрывает и как монетизируется</h3>
<p>{_e(US_INSURANCE_MODEL.replace(chr(10), ' '))}</p>
<div class="stats">
  <div class="stat"><b>990 млн</b><span class="muted">office visits/год (TAM)</span></div>
  <div class="stat"><b>{b2c_base}</b><span class="muted">тыс. BYN B2C (0,05% conv.)</span></div>
  <div class="stat"><b>{upside}</b><span class="muted">тыс. BYN stack 2037+</span></div>
  <div class="stat"><b>2033-37</b><span class="muted">GTM HIPAA → Epic → Payer</span></div>
</div>
{img('chart_us_stakeholders.png', 'Ценность Protocol для стейкholders США')}
{img('chart_us_insurance_flywheel.png', 'Payer flywheel: Prep Card снижает copay → клиника Verified → MLR ↓')}
{img('chart_us_tam.png', 'B2C upside: Visit Prep + AVS при разной конверсии')}
{img('chart_us_revenue_stack.png', 'Стек выручки USA: B2C + B2B Epic + Payer API + Employer PEPM')}
{_table(['Стейкholder', 'Без Protocol', 'С Protocol', 'Монетизация'], stake_rows, caption='Кому полезен Protocol в США: пациент, врач, клиника, payer, employer')}
{_table(['Сценарий', 'Визитов/год TAM', 'Conv.', 'Проверок/год', 'B2C тыс. BYN', 'Канал'], tam_rows, caption='TAM США: ambulatory office visits (~990 млн/год, CDC/NCHS). Не в EBITDA 2029.')}
{_table(['Показатель', 'Без Protocol', 'С Protocol', 'Эффект'], econ_rows, caption='Страховая экономика: copay, MLR, HEDIS, network tier (иллюстративная модель для actuarial pilot)')}
{_table(['Продукт', 'Канал', 'Цена USD', 'Когда', 'Суть'], product_rows, caption='Продуктовая линейка Protocol USA')}
{_table(['Корпус', 'Тип', 'Источник', 'Блок отчёта'], corpus_rows, caption='US guideline pack (не единый MOH; ingest + license where needed)')}
{_table(['Поток', 'Ориентир тыс. BYN/год', 'Горизонт'], rev_rows, caption='Суммарный upside USA (B2C + B2B + payer + employer)')}
{_table(['Фаза', 'Фокус', 'Deliverable', 'Upside', 'Inv. тыс. BYN'], gtm_rows, caption='GTM США 2033-2037 (после Tier 1-2 и platform API)')}
{_table(['Шаг', 'Содержание', 'Срок', 'Критерий'], step_rows, caption='Playbook входа на рынок США (7 шагов)')}
{_table(['Риск', 'Уровень', 'Митигация'], risk_rows, caption='Риски US expansion')}
<div class="highlight"><strong>Суть для конкурса:</strong> Protocol помогает пациенту <em>подготовиться к приёму</em>
(Visit Prep Card) и проверить выписку после визита - по тем же 8 блокам и evidence_map, что в РБ. Это
<strong>стимулирует качество клиники</strong> (L0 в Epic до sign note) и создаёт <strong>страховые рычаги</strong>:
payer снижает copay подготовленному пациенту и повышает fee schedule / network tier клинике с Protocol Verified;
клиника без Protocol в value-based сети несёт actuarial surcharge. Якорный пилот - МЦ «Кravira», РБ; США - масштаб
и payer economics, не замена локального пилота.</div>
</div>"""


def _future_html(assets_rel: str = "_assets") -> str:
    def img(name: str, cap: str) -> str:
        return f'<div class="chart"><div class="caption">{_e(cap)}</div><img src="{assets_rel}/{name}" alt=""/></div>'

    road_rows = [(a, b, c, d) for a, b, c, d in FUTURE_ROADMAP_TABLE]
    stream_rows = [(a, f"{b:,}".replace(",", " "), c) for a, b, c in FUTURE_STREAMS]
    risk_rows = [(a, b, c) for a, b, c in FUTURE_RISKS]
    return f"""
<div class="section"><h2>6.4. Будущее Protocol: платформа, Big Tech и health-community</h2></div>
<div class="section-body">
<p>{_e(FUTURE_VISION.replace(chr(10), ' '))}</p>
<h3>Интеграции Google, Meta, Apple</h3>
<p>{_e(PLATFORM_INTEGRATIONS.replace(chr(10), ' '))}</p>
<h3>Protocol Community - мини-соцсеть вокруг доказательной медицины</h3>
<p>{_e(COMMUNITY_VISION.replace(chr(10), ' '))}</p>
<div class="stats">
  <div class="stat"><b>2033+</b><span class="muted">Platform API</span></div>
  <div class="stat"><b>2034+</b><span class="muted">Care Rooms (Meta)</span></div>
  <div class="stat"><b>2035</b><span class="muted">Community Pro</span></div>
  <div class="stat"><b>3 слоя</b><span class="muted">ядро → каналы → платформа</span></div>
</div>
{img('chart_platform_layers.png', 'Три слоя развития Protocol: ядро, каналы, платформа')}
{img('chart_platform_roadmap.png', 'Дорожная карта Protocol 2026-2035')}
{img('chart_future_streams.png', 'Потоки выручки: от B2B РБ к международной платформе')}
{img('chart_ecosystem_flywheel.png', 'Flywheel: B2C давит на B2B, B2B улучшает данные для ЦИСЗ')}
{_table(['Период', 'Бренд', 'Веха', 'Фаза'], road_rows, caption='Дорожная карта развития Protocol (не EBITDA 2029)')}
{_table(['Поток', 'Ориентир тыс. BYN/год', 'Горизонт'], stream_rows, caption='Дополнительные потоки выручки платформенного этапа')}
{_table(['Риск', 'Уровень', 'Митигация'], risk_rows, caption='Риски платформенного и community-слоя')}
<div class="highlight"><strong>Для инвестора:</strong> Protocol строит не «локальный SaaS Кравиры», а
<strong>национальную и международную платформу</strong> контроля качества КЗ. Кравира остаётся якорным пилотом
и первым B2B/B2C-кейсом; дальше - сеть ОЗ, patient.html, locale-pack, API и community вокруг официальных
клинических протоколов. Big Tech - каналы дистрибуции и уведомлений, не замена evidence_map.</div>
</div>"""


def _continuous_ml_html() -> str:
    principles = ML_PRINCIPLES_TABLE
    roadmap = ML_ROADMAP_TABLE
    appendix = ML_APPENDIX_TABLE
    cycle = "".join(f"<li>{_e(step)}</li>" for step in ML_DATA_CYCLE)
    return f"""
<div class="section"><h2>17. Непрерывное дообучение моделей Protocol</h2></div>
<div class="section-body">
<p>{_e(CONTINUOUS_ML_INTRO.replace(chr(10), ' '))}</p>
<div class="highlight">{_e(ML_COMPETITION_NOTE.replace(chr(10), ' '))}</div>
<h3>Результаты эксперимента embedder (июнь 2026)</h3>
{_table(['Метрика', 'Baseline e5', 'Fine-tune', 'Δ'], ML_EXPERIMENT_EMBEDDER_TABLE, caption='Офлайн MRR на seed 313 пар (ml/experiments/embedder_exp_001)')}
<h3>A/B: baseline vs fine-tune в retrieve()</h3>
{_table(['Слой оценки', 'Baseline e5', 'Fine-tune', 'Δ'], ML_AB_KZ_TABLE, caption='consult_gold + golden RAG на полном корпусе КП')}
<p>{_e(ML_AB_INTERPRETATION.replace(chr(10), ' '))}</p>
<h3>Принцип: что обучается, а что нет</h3>
{_table(['Компонент', 'Механизм', 'Обучение'], principles, caption='Разделение детерминированного ядра и ML')}
<h3>Цикл MLOps</h3>
<ol>{cycle}</ol>
{_table(['Фаза', 'Срок', 'Deliverable', 'Эффект'], roadmap, caption='Дорожная карта ML 2026-2028')}
{_table(['Артефакт', 'Назначение'], appendix, caption='Приложение И. ML-контур в репозитории Protocol')}
<p class="muted">Экспорт датасетов: <code>python3 scripts/export_training_feedback.py</code> ·
конфиг: <code>ml/configs/default.json</code></p>
</div>"""


def _sections_html_with_b2c(assets_rel: str = "_assets") -> str:
    out = []
    for key, title in SECTION_TITLES.items():
        body = SECTIONS.get(key, "")
        out.append(f'<div class="section"><h2>{_e(title)}</h2></div>')
        out.append(f'<div class="section-body">{_ps(body)}</div>')
        if key == "6.":
            out.append(_expansion_html())
            out.append(_intl_expansion_html())
            out.append(_future_html(assets_rel))
            out.append(_us_expansion_html(assets_rel))
        if key == "8.":
            out.append(_b2c_ux_html())
        if key == "9.":
            out.append(_monetization_html())
    return "".join(out)


def _monetization_html() -> str:
    scen_rows = [
        (name, f"{rev:,}".replace(",", " "), f"+{ebitda:,}".replace(",", " "), note)
        for name, rev, ebitda, note in ALL_REVENUE_SCENARIOS_Y3
    ]
    base_rev = total_rev_k(FIN_Y3)
    return f"""
<div class="section"><h2>9.2. Дополнительная монетизация (рост выручки и прибыли)</h2></div>
<div class="section-body">
<p>{_e(MONETIZATION_INTRO.replace(chr(10), ' '))}</p>
<div class="stats">
  <div class="stat"><b>{base_rev}</b><span class="muted">тыс. BYN базовый 2029</span></div>
  <div class="stat"><b>+{EXPANDED_Y3['extra_rev_k']}</b><span class="muted">тыс. доп. каналы</span></div>
  <div class="stat"><b>{EXPANDED_Y3['total_rev_k']}</b><span class="muted">тыс. выручка расширен.</span></div>
  <div class="stat"><b>+{EXPANDED_Y3['ebitda_month_k']}</b><span class="muted">тыс. EBITDA/мес расшир.</span></div>
</div>
{_table(['Канал', 'Цена', 'План тыс.', 'Драйвер', 'Вероятн.', 'Старт'], MONETIZATION_TABLE, caption='8 дополнительных потоков дохода к 2029')}
{_table(['Сценарий 2029', 'Выручка', 'EBITDA', 'Комментарий'], scen_rows, caption='Сравнение сценариев: осторожный → расширенный')}
<div class="highlight"><strong>Расширенный сценарий:</strong> базовый план + L2 для методистов, подписка «Методслужба Pro»,
обучение врачей, OEM/API, медтуризм B2C, корпоративные аудиты, white-label Enterprise, аналитика для руководства ОЗ.
Взвешенный прогноз доп. каналов: ~{WEIGHTED_EXTRA_Y3_K} тыс. BYN/год (вероятность × план).</div>
</div>"""


def _glossary_html() -> str:
    gloss_rows = [(a, en, ru) for a, en, ru in GLOSSARY_TABLE]
    audit = build_calc_audit_table()
    return f"""
<div class="section glossary-block"><h2>Глоссарий, формулы и сверка расчётов</h2></div>
<div class="section-body glossary-block">
<p>{_e(GLOSSARY_INTRO)}</p>
{_table(['Сокращение', 'English', 'Расшифровка'], gloss_rows, caption='Термины: TAM, SAM, SOM, B2B, B2C, EBITDA и др.')}
{_table(['Показатель', 'Формула', 'Результат'], FORMULA_TABLE, caption='Ключевые формулы финмодели')}
{_table(['Проверка', 'Значение', 'Расчёт', 'Статус'], audit, caption='Автоматическая сверка показателей (скрипт финмодели)')}
</div>"""


def _finance_block(assets_rel: str) -> str:
    def img(name: str, cap: str) -> str:
        return f'<div class="chart"><div class="caption">{_e(cap)}</div><img src="{assets_rel}/{name}" alt=""/></div>'

    stats = f"""
<div class="stats">
  <div class="stat"><b>2,7 млрд</b><span class="muted">BYN рынок платных услуг</span></div>
  <div class="stat"><b>{MARKET_KZ_MONTH:,}</b><span class="muted">КЗ/мес TAM</span></div>
  <div class="stat"><b>{KRAVIRA_B2B_YEAR:,}</b><span class="muted">BYN/год якорь</span></div>
  <div class="stat"><b>+{ebitda_k(FIN_Y3)}</b><span class="muted">тыс. BYN EBITDA/год 2029</span></div>
  <div class="stat"><b>+{ebitda_month_k(FIN_Y3)}</b><span class="muted">тыс. BYN EBITDA/мес (8% TAM)</span></div>
</div>""".replace(",", " ")

    tam_rows = [
        ("TAM - весь рынок", f"{MARKET_KZ_MONTH:,} КЗ/мес".replace(",", " "), "2,5 млн - все частные ОЗ РБ"),
        ("TAM B2B потолок", f"{TAM_B2B_CEILING_YEAR_K:,} тыс. BYN/год".replace(",", " "), "30 млн КЗ × 0,75 BYN (теория)"),
        ("SAM - крупные ОЗ 5%", f"{SAM_KZ_YEAR:,} КЗ/год".replace(",", " "), "целевой B2B-сегмент"),
        ("SOM - план B2B 2029", f"{FIN_Y3['kz_month']:,} КЗ/мес ({Y3_MARKET_SHARE:.0%} TAM)".replace(",", " "), "осторожный сценарий"),
        ("B2C конверсия 2029", f"{FIN_Y3['b2c_checks']:,} проверок/год".replace(",", " "), f"{FIN_Y3['b2c_checks']/MARKET_KZ_YEAR:.2%} от частного TAM B2C (30 млн)"),
    ]
    fin_rows = [
        ("Доля TAM B2B", "1%", "3%", "8%"),
        ("Клиенты (ОЗ)", "1", "3", "8"),
        ("КЗ/мес B2B (не весь TAM)", f"{FIN_Y1['kz_month']:,}".replace(",", " "), f"{FIN_Y2['kz_month']:,}".replace(",", " "), f"{FIN_Y3['kz_month']:,}".replace(",", " ")),
        ("B2B Кравира, тыс.", str(FIN_Y1["b2b_kravira_k"]), str(FIN_Y2["b2b_kravira_k"]), str(FIN_Y3["b2b_kravira_k"])),
        ("B2B другие ОЗ, тыс.", str(FIN_Y1["b2b_other_k"]), str(FIN_Y2["b2b_other_k"]), str(FIN_Y3["b2b_other_k"])),
        ("B2C Protocol, тыс.", str(FIN_Y1["b2c_k"]), str(FIN_Y2["b2c_k"]), str(FIN_Y3["b2c_k"])),
        ("Rev-share клиникам, тыс.", str(CLINIC_B2C_REV_Y1_K), str(CLINIC_B2C_REV_Y2_K), str(CLINIC_B2C_REV_Y3_K)),
        ("API/МИС, тыс.", str(FIN_Y1["api_k"]), str(FIN_Y2["api_k"]), str(FIN_Y3["api_k"])),
        ("Итого выручка, тыс.", str(total_rev_k(FIN_Y1)), str(total_rev_k(FIN_Y2)), str(total_rev_k(FIN_Y3))),
        ("OPEX, тыс.", str(FIN_Y1["opex_k"]), str(FIN_Y2["opex_k"]), str(FIN_Y3["opex_k"])),
        ("EBITDA год, тыс.", str(ebitda_k(FIN_Y1)), f"+{ebitda_k(FIN_Y2)}", f"+{ebitda_k(FIN_Y3)}"),
        ("EBITDA мес, тыс.", f"{ebitda_month_k(FIN_Y1)}", f"+{ebitda_month_k(FIN_Y2)}", f"+{ebitda_month_k(FIN_Y3)}"),
    ]
    bridge_rows = [(name, f"{kz:,}".replace(",", " ") if kz else " - ", f"{rev:,}".replace(",", " "), note) for name, kz, rev, note in TAM_BRIDGE]
    pen_rows = [
        (f"{p}% TAM B2B (частный сектор)", f"{e:,}".replace(",", " "), f"~{int(MARKET_KZ_MONTH*p/100):,} КЗ/мес B2B".replace(",", " "))
        for p, e in PENETRATION_SENSITIVITY
    ]
    b2c_pen_rows = list(B2C_CONV_SENSITIVITY)
    return f"""
<div class="section"><h2>Финансовые приложения и графики</h2></div>
<div class="section-body">
{stats}
{_table(['Показатель рынка РБ', 'Значение', 'Источник'], RB_MARKET_TABLE, caption='Контекст рынка платных медуслуг РБ')}
<div class="highlight"><strong>Важно:</strong> TAM B2B = 2,5 млн КЗ/мес (частные ОЗ). TAM B2C в базовом плане = 30 млн КЗ/год (тот же частный сегмент); расширенный B2C TAM - 108,4 млн амбул. посещений (§6.2). В осторожном плане 2029: <strong>200 тыс. КЗ/мес B2B (8%)</strong> и <strong>{FIN_Y3['b2c_checks']:,} B2C-проверок/год (0,23% частного TAM B2C)</strong> - EBITDA <strong>+{ebitda_k(FIN_Y3)} тыс./год (~{ebitda_month_k(FIN_Y3)} тыс./мес)</strong>. Теор. потолок B2B ~{TAM_B2B_CEILING_YEAR_K:,} тыс./год. B2C Protocol/проверка: ~{B2C_PROTOCOL_PER_CHECK} BYN.</div>
{_table(['Этап', 'КЗ/год', 'Выручка тыс.', 'Пояснение'], bridge_rows, caption='Мост TAM → SAM → SOM → выручка Protocol')}
{_table(['Показатель', 'Значение', 'Комментарий'], tam_rows, caption='TAM / SAM / SOM - различие рынка и плана')}
{_table(['Показатель', '2027', '2028', '2029'], fin_rows, caption='Финансовый план · осторожный сценарий (тыс. BYN)')}
{_table(['Сценарий 2029', 'B2B TAM', 'B2C conv', 'Выручка', 'EBITDA/год', 'EBITDA/мес'], SCENARIO_COMPARE_TABLE, caption='Три сценария года 3: осторожный · базовый · оптimистичный')}
{_table(['Канал', 'Вероятность', 'План тыс.', 'Драйвер'], CHANNEL_TABLE, caption='Какой канал с большей вероятностью даст выручку')}
{_table(['Доля TAM B2B', 'EBITDA тыс./год', 'КЗ/мес B2B'], pen_rows, caption='Чувствительность EBITDA к проникновению B2B (B2C и API фикс. - осторожный сценарий 2029)')}
{_table(['Конверсия B2C', 'Проверок/год', 'B2C, тыс.', 'Выручка итого, тыс.', 'EBITDA, тыс.'], b2c_pen_rows, caption='Чувствительность EBITDA к конверсии B2C (B2B фикс. 8% = 1 800 тыс., API 75, OPEX 650; знаменатель 30 млн КЗ/год). При росте проверок EBITDA растёт на ту же величину, что и B2C: +1 тыс. B2C → +1 тыс. EBITDA.')}
<div class="highlight"><strong>Не путать с §6.2:</strong> таблица расширенного TAM (108,4 млн посещений) показывает только <em>выручку B2C</em> (686-1 592 тыс.) без B2B и OPEX - там нет столбца EBITDA. Столбец «B2C, тыс.» ниже - доля выручки, а EBITDA включает B2B (~1 800 тыс.) минус OPEX.</div>
{img('chart_revenue_ebitda.png', 'Выручка и EBITDA по годам (Plotly)')}
{img('chart_monetization.png', '8 дополнительных каналов монетизации 2029')}
{img('chart_all_scenarios.png', 'Все сценарии 2029: до 3,5 млн выручки')}
{img('chart_expanded_potential.png', 'Базовый vs расширенный потенциал 2029')}
{img('chart_scenarios_ebitda.png', 'EBITDA 2029: три сценария (год и месяц)')}
{img('chart_scenarios_revenue.png', 'Структура выручки 2029 по сценариям B2B/B2C/API')}
{img('chart_penetration.png', 'Чувствительность EBITDA к проникновению B2B (B2C фикс.)')}
{img('chart_channel_outlook.png', 'Вероятность успеха каналов монетизации')}
{img('chart_ebitda_monthly.png', 'EBITDA: год vs месяц · подпись 8% TAM')}
{img('chart_b2b_split.png', 'B2B: Кравира vs другие клиники РБ')}
{img('chart_b2c_tiers.png', 'Tier-цены B2C по specialty')}
{img('chart_b2c_revshare.png', 'Rev-share 30/70: примеры по tier')}
{img('chart_b2c_growth.png', 'Рост B2C и сценарий upside')}
{img('chart_b2c_sensitivity.png', 'EBITDA vs конверсия B2C (B2B 8% фикс.)')}
{img('chart_b2c_tam_compare.png', 'B2C TAM: частный сектор vs национальный vs экспорт')}
{img('chart_intl_markets.png', 'Международный B2C upside по 9 странам (базовая conv.)')}
{img('chart_intl_compare.png', '9 рынков: осторожный vs базовый B2C upside')}
{img('chart_ecosystem_flywheel.png', 'Flywheel Protocol: B2C → B2B → государство')}
{img('chart_platform_roadmap.png', 'Дорожная карта Protocol 2026-2035')}
{img('chart_future_streams.png', 'Потоки выручки платформенного этапа')}
{img('chart_platform_layers.png', 'Три слоя: ядро · каналы · платформа')}
{img('chart_us_stakeholders.png', 'США: ценность для patient / payer / clinic')}
{img('chart_us_tam.png', 'США: B2C upside Visit Prep + AVS')}
{img('chart_us_revenue_stack.png', 'США: стек выручки 2037+')}
{img('chart_us_insurance_flywheel.png', 'США: payer flywheel Prep → Verified → MLR')}
{_table(['Статья инвестиций', 'BYN', 'Срок'], INVESTMENT_PLAN, caption='Инвестиции 2026-2027')}
{_table(['Драйвер', 'Пояснение'], list(CISZ_DRIVERS))}
{_table(['Альтернатива', 'Охват', 'Слабость', 'Стоимость'], COMPETITOR_MATRIX, caption='Конкуренты')}
{img('chart_market.png', 'Рынок TAM / SAM / SOM')}
{img('chart_market_share.png', 'Рост доли рынка и объёма КЗ')}
{img('chart_revenue.png', 'Выручка B2B / B2C по годам')}
{img('chart_ebitda.png', 'EBITDA по годам')}
{img('chart_opex.png', 'Структура OPEX')}
{img('chart_pricing.png', 'Тарифная лестница B2B')}
{img('chart_margin.png', 'Маржа L0')}
{img('chart_channels.png', 'Структура выручки, год 3')}
{img('chart_roi.png', 'ROI якорного клиента')}
{img('chart_b2c_funnel.png', 'B2C-воронка')}
<div class="highlight">Сертификат ГКНТ: {CERTIFICATE_BYN:,} BYN (571 б.в. × 42 BYN). Запрос направлен на интеграцию МИС,
on-prem L0 и масштабирование на 5-10 частных ОЗ к 2028 г.</div>
</div>"""


def write_business_plan_html(path: Path, assets_rel: str = "_assets") -> None:
    body = (
        _cover("БИЗНЕС-ПЛАН", "инновационного проекта", show_emblem=False, hero_logo=True)
        + _mission_block()
        + f'<div class="toc"><h2>Содержание</h2><pre>{_e(TOC)}</pre></div>'
        + '<div class="section"><h2>3. Резюме</h2></div>'
        + f'<div class="section-body">{_ps(RESUME)}</div>'
        + _ecosystem_html()
        + _market_scope_html()
        + _stakeholders_html()
        + _analogues_html()
        + _sections_html_with_b2c(assets_rel)
        + _continuous_ml_html()
        + _glossary_html()
        + _finance_block(assets_rel)
    )
    path.write_text(_page("Бизнес-план Protocol", body), encoding="utf-8")


def write_zayavka_html(path: Path) -> None:
    rows = [
        ("Номинация", NOMINATION),
        ("Наименование проекта", ORG_PROJECT),
        ("Организация", ORG_NAME),
        ("Руководитель", DIRECTOR),
        ("Адрес", ORG_ADDRESS),
        ("УНП", ORG_UNP),
        ("Контактное лицо", CONTACT_PERSON),
        ("Телефон", PHONE),
        ("E-mail", EMAIL),
        ("Команда проекта", TEAM),
        ("Сайт", WEB),
    ]
    body = (
        _cover("ЗАЯВКА", "форма участника конкурса")
        + '<div class="section"><h2>Данные заявки</h2></div>'
        + f'<div class="section-body">{_table(["Поле", "Значение"], rows, css_class="form-table")}</div>'
        + '<div class="section-body"><p class="muted">Подпись руководителя и печать организации - при подаче бумажного экземпляра.</p></div>'
    )
    path.write_text(_page("Заявка - Protocol", body), encoding="utf-8")


def write_passport_html(path: Path) -> None:
    sections = [
        ("Наименование", ORG_PROJECT),
        ("Описание проекта", PASSPORT_DESC),
        ("Направления", "Медицинские науки и технологии; Информатика и информатизация"),
        ("Новизна", "Нет аналогов в стране, есть за рубежом"),
        ("Стадия", "Работающий прототип"),
        ("Потребители", PASSPORT_CONSUMERS),
        ("Преимущества", PASSPORT_ADVANTAGES),
        ("ИС", PASSPORT_IP),
        ("Сроки", PASSPORT_TIMELINE),
        ("Сертификация", PASSPORT_PRODUCT_CERT),
        ("Достижения", PASSPORT_ACHIEVEMENTS),
        ("Приложения", PASSPORT_EXTRA),
    ]
    parts = [_cover("ПАСПОРТ", "инновационного проекта")]
    for title, text in sections:
        parts.append(f'<div class="section"><h2>{_e(title)}</h2></div>')
        parts.append(f'<div class="section-body">{_ps(str(text))}</div>')
    path.write_text(_page("Паспорт - Protocol", "".join(parts)), encoding="utf-8")


def write_strategy_html(path: Path) -> None:
    body = (
        _cover("СТРАТЕГИЯ", "коммерциализации инновационного проекта")
        + f'<div class="section"><h2>Проект и организация</h2></div>'
        + f'<div class="section-body"><p><strong>{_e(ORG_PROJECT)}</strong></p><p>{_e(ORG_NAME)}</p></div>'
        + f'<div class="section"><h2>Уровень коммерциализации</h2></div>'
        + f'<div class="section-body"><p>{_e(STRATEGY_LEVELS)}</p><p>Опытный образец, B2B-пилот в Кравире, B2C MVP patient.html (ERIP и rev-share - {STRATEGY_YEAR}).</p></div>'
        + f'<div class="section"><h2>Способы коммерциализации</h2></div>'
        + f'<div class="section-body"><p>{_e(STRATEGY_METHODS)}</p></div>'
        + f'<div class="section"><h2>План на ближайший год</h2></div>'
        + f'<div class="section-body">{_ps(STRATEGY_YEAR_PLAN)}</div>'
        + f'<div class="section"><h2>Стратегия на 5 лет</h2></div>'
        + f'<div class="section-body">{_ps(STRATEGY_5Y)}</div>'
    )
    path.write_text(_page("Стратегия - Protocol", body), encoding="utf-8")


def write_roi_html(path: Path) -> None:
    saving = ROI_TOTAL_SAVING
    body = (
        _cover("ROI", "якорного клиента МЦ «Кравира»")
        + f"""
<div class="section"><h2>ROI Protocol L0 для Кравиры</h2></div>
<div class="section-body">
{_table(['Показатель', 'Значение'], [
  ('КЗ/мес', f'{KRAVIRA_KZ_MONTH:,}'.replace(',', ' ')),
  ('Тариф L0', '0,69 BYN/КЗ'),
  ('Стоимость Protocol/мес', f'{ROI_PROTOCOL_COST:,} BYN'.replace(',', ' ')),
  ('Экономия/мес (оценка)', f'{saving:,} BYN'.replace(',', ' ')),
  ('Баланс/мес', f'{ROI_NET:+,} BYN'.replace(',', ' ')),
])}
<div class="highlight">Полная окупаемость достигается при учёте снижения доработок ЦИСЗ (1,5 п.п. × 25 000 × 15 BYN),
повторных визитов и высвобождения 0,35 FTE методиста. Детальный расчёт - в таблицах ниже.</div>
{_table(['Подход', 'Охват', 'Стоимость'], [
  ('Методслужба', '~2%', '~34 BYN/проверенное КЗ'),
  ('Protocol L0', '100%', '0,69 BYN/КЗ'),
])}
{_table(['Статья экономии', 'BYN/мес'], [
  ('Методист (0,35 FTE)', '+1 120'),
  ('Меньше доработок ЦИСЗ', '+5 625'),
  ('Меньше повторных визитов', '+1 000'),
  ('Итого экономия', f'+{saving:,}'.replace(',', ' ')),
  ('Protocol L0', f'−{ROI_PROTOCOL_COST:,}'.replace(',', ' ')),
])}
<p class="muted">Protocol - инвестиция в качество и данные ЦИСЗ; ROI раскрывается при масштабе и снижении скрытых издержек.</p>
</div>"""
    )
    path.write_text(_page("ROI Кравira", body), encoding="utf-8")


def write_all_html(out_dir: Path | None = None) -> list[Path]:
    out = out_dir or KONKURS_DIR
    out.mkdir(parents=True, exist_ok=True)
    files = [
        out / "01_Zayavka_Kravira_Protocol-print.html",
        out / "02_Pasport_Kravira_Protocol-print.html",
        out / "03_Biznes_plan_Kravira_Protocol-print.html",
        out / "04_Strategiya_Kravira_Protocol-print.html",
        out / "06_ROI_Kravira-print.html",
    ]
    write_zayavka_html(files[0])
    write_passport_html(files[1])
    write_business_plan_html(files[2])
    write_strategy_html(files[3])
    write_roi_html(files[4])
    return files
