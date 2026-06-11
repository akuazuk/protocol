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
    B2C_UPSIDE_YEAR3_K,
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
    TAM_REVENUE_YEAR,
    ebitda_k,
    total_rev_k,
)
from konkurs_impact import (  # noqa: E402
    CISZ_CONTEXT,
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@600;700&display=swap');
:root {
  --g700: #0d5c4d; --g600: #126b5c; --g500: #1a8a72; --g100: #e8f5f1;
  --text: #142824; --muted: #4a635e; --amber: #b45309; --border: rgba(13,92,77,.16);
}
* { box-sizing: border-box; }
body { margin: 0; font-family: "DM Sans", system-ui, sans-serif; color: var(--text);
  font-size: 10.5pt; line-height: 1.55; background: #f4faf8; }
.wrap { max-width: 210mm; margin: 0 auto; padding: 0; }
h1,h2,h3 { font-family: Outfit, sans-serif; color: var(--g700); page-break-after: avoid; }
h1 { font-size: 22pt; margin: 0 0 4mm; letter-spacing: .02em; }
h2 { font-size: 14pt; margin: 8mm 0 3mm; padding-bottom: 2mm; border-bottom: 2px solid var(--g500); }
h3 { font-size: 11.5pt; margin: 5mm 0 2mm; color: var(--g600); }
p { margin: 0 0 3mm; text-align: justify; hyphens: auto; }
ul { margin: 2mm 0 4mm; padding-left: 5mm; }
li { margin-bottom: 1.5mm; }
.muted { color: var(--muted); font-size: 9pt; }
.cover {
  min-height: 267mm; display: flex; flex-direction: column; justify-content: center;
  text-align: center; padding: 20mm 18mm; page-break-after: always;
  background: linear-gradient(165deg, #fff 0%, var(--g100) 55%, #fff 100%);
  border-bottom: 4px solid var(--g600);
}
.cover .badge { display: inline-block; font-size: 9pt; color: var(--muted); letter-spacing: .08em;
  text-transform: uppercase; margin-bottom: 6mm; }
.cover .title { font-size: 28pt; font-weight: 700; color: var(--g700); margin: 4mm 0; }
.cover .subtitle { font-size: 13pt; color: var(--muted); margin-bottom: 8mm; }
.cover .project { font-size: 12pt; font-weight: 600; max-width: 140mm; margin: 0 auto 10mm; line-height: 1.45; }
.cover .meta { text-align: left; max-width: 150mm; margin: 10mm auto 0; font-size: 10pt; line-height: 1.65;
  background: #fff; padding: 6mm 8mm; border-radius: 4mm; border: 1px solid var(--border); }
.cover hr { border: none; height: 2px; background: linear-gradient(90deg, transparent, var(--g500), transparent);
  margin: 8mm auto; width: 70%; }
.mission {
  page-break-after: always; padding: 14mm 16mm; background: var(--g700); color: #fff;
}
.mission h2 { color: #fff; border-bottom-color: rgba(255,255,255,.35); font-size: 16pt; }
.mission p { text-align: left; opacity: .95; }
.section { padding: 10mm 16mm 4mm; page-break-inside: avoid; }
.section-body { padding: 0 16mm 6mm; }
.card { background: #fff; border: 1px solid var(--border); border-radius: 3mm; padding: 4mm 5mm;
  margin-bottom: 4mm; box-shadow: 0 1px 4px rgba(13,92,77,.06); page-break-inside: avoid; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
.grid4 { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
.stake { border-left: 4px solid var(--g600); padding-left: 4mm; }
.stake .icon { font-size: 16pt; float: left; margin-right: 3mm; }
.stake h3 { margin-top: 0; }
.stake .tag { font-size: 9pt; color: var(--g500); font-weight: 600; margin-bottom: 2mm; }
table { width: 100%; border-collapse: collapse; font-size: 9pt; margin: 3mm 0 5mm; page-break-inside: avoid; }
th, td { padding: 2.2mm 2.5mm; border: 1px solid #d8ebe6; text-align: left; vertical-align: top; }
th { background: var(--g600); color: #fff; font-weight: 600; }
tr:nth-child(even) td { background: var(--g100); }
.caption { font-size: 9pt; font-weight: 600; color: var(--g700); margin: 4mm 0 2mm; }
.chart { text-align: center; margin: 4mm 0 6mm; page-break-inside: avoid; }
.chart img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 2mm; }
.toc { padding: 12mm 16mm; page-break-after: always; }
.toc pre { font-family: "DM Sans", sans-serif; font-size: 10pt; line-height: 1.7; white-space: pre-wrap;
  background: #fff; padding: 6mm; border-radius: 3mm; border: 1px solid var(--border); }
.highlight { background: #fffbeb; border-left: 4px solid var(--amber); padding: 4mm 5mm; margin: 4mm 0;
  border-radius: 0 2mm 2mm 0; font-size: 10pt; }
.stats { display: flex; flex-wrap: wrap; gap: 3mm; margin: 4mm 0 6mm; }
.stat { flex: 1 1 40mm; background: var(--g100); border: 1px solid var(--border); border-radius: 2mm;
  padding: 3mm 4mm; text-align: center; }
.stat b { display: block; font-size: 13pt; color: var(--g700); }
.form-table td:first-child { width: 38%; font-weight: 600; background: var(--g100); }
@media print {
  body { background: #fff; }
  .section, .section-body { padding-left: 12mm; padding-right: 12mm; }
  .no-print { display: none !important; }
}
@page { size: A4; margin: 12mm 10mm; }
.scenario { border: 1px solid var(--border); border-radius: 3mm; padding: 4mm 5mm; margin-bottom: 5mm;
  page-break-inside: avoid; background: #fff; }
.scenario h3 { margin-top: 0; color: var(--g700); }
.scenario .tag { font-size: 8.5pt; color: var(--g500); font-weight: 600; margin-bottom: 2mm; }
.flow { font-family: monospace; font-size: 8.5pt; background: var(--g100); padding: 3mm 4mm;
  border-radius: 2mm; margin: 2mm 0 3mm; word-break: break-word; }
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


def _cover(title: str, doc_kind: str) -> str:
    return f"""
<div class="cover">
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
  <p>Сделать качество амбулаторной помощи в Беларуси <strong>измеримым и прозрачным</strong> - для каждого
  консультативного заключения, до подписи врачом и до попадания данных в ЦИСЗ. Не заменяя врача и МЭЭ,
  мы даём инструмент соответствия <strong>478 клиническим протоколам Минздрава</strong> на потоке
  <strong>100% КЗ</strong>, а пациенту - право проверить своё заключение по тем же стандартам.</p>
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
}


def _market_scope_html() -> str:
    tier_rows = [(t["name"], t["specialty"], f'{t["price"]:.2f}'.replace(".", ","), f'{t["mix"] * 100:.0f}%') for t in B2C_TIERS]
    return f"""
<div class="section"><h2>Масштаб рынка: все частные ОЗ и все пациенты РБ</h2></div>
<div class="section-body">
<p>{_e(MARKET_SCOPE_NOTE.replace(chr(10), ' '))}</p>
<div class="stats">
  <div class="stat"><b>1%</b><span class="muted">Кравира от TAM</span></div>
  <div class="stat"><b>{MARKET_KZ_MONTH:,}</b><span class="muted">КЗ/мес частный сектор</span></div>
  <div class="stat"><b>8</b><span class="muted">клиник B2B к 2029</span></div>
  <div class="stat"><b>{CLINIC_B2C_REV_Y3_K}</b><span class="muted">тыс. BYN rev-share клиникам</span></div>
</div>
{_table(['Tier B2C', 'Specialty', 'BYN', 'Микс'], tier_rows, caption='Tier-цены B2C (средний чек ~7,47 BYN)')}
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
<div class="section"><h2>8.1. B2C: UX, tier-цены и rev-share (национальный рынок)</h2></div>
<div class="section-body">
<p>{_e(B2C_UX_INTRO.replace(chr(10), ' '))}</p>
{''.join(scenarios_html)}
{_table(['Tier', 'Specialty / приём', 'BYN', 'Микс'], B2C_PRICING_TABLE, caption='Tier-цены B2C по сложности приёма')}
{_table(['Продукт', 'Оплата пациента', 'Клинике 30%', 'Protocol 70%'], B2C_REVSHARE_EXAMPLES, caption='Rev-share при оплате по SMS/QR-ссылке клиники')}
{_table(['Блок отчёта', 'Содержание', 'Зачем'], B2C_REPORT_TABLE, caption='Patient view отчёта L1/L2')}
{_table(['Сценарий', 'Конверсия', 'Сложность'], [
  ('1. QR после визита', 'Средняя', 'Минимальная — любая ОЗ-партнёр'),
  ('2. SMS/email rev-share', 'Высокая', 'Основной канал масштаба'),
  ('3. Национальный SEO', 'Средняя', '2028+, без rev-share'),
], caption='Приоритет запуска B2C')}
<div class="highlight">Пилот Q4 2026: QR + SMS rev-share в Кравире; масштаб 2027+ на сеть частных ОЗ РБ.
Rev-share {int(CLINIC_B2C_REVSHARE * 100)}% мотивирует клиники рассылать ссылку — доп. доход ~{CLINIC_B2C_REV_Y3_K} тыс. BYN/год к 2029 при осторожном сценарии.</div>
<p><strong>Вне scope (не делаем в 2026–2027):</strong></p><ul>{oos}</ul>
</div>"""


def _sections_html_with_b2c() -> str:
    out = []
    for key, title in SECTION_TITLES.items():
        body = SECTIONS.get(key, "")
        out.append(f'<div class="section"><h2>{_e(title)}</h2></div>')
        out.append(f'<div class="section-body">{_ps(body)}</div>')
        if key == "8.":
            out.append(_b2c_ux_html())
    return "".join(out)


def _finance_block(assets_rel: str) -> str:
    def img(name: str, cap: str) -> str:
        return f'<div class="chart"><div class="caption">{_e(cap)}</div><img src="{assets_rel}/{name}" alt=""/></div>'

    stats = f"""
<div class="stats">
  <div class="stat"><b>2,7 млрд</b><span class="muted">BYN рынок платных услуг</span></div>
  <div class="stat"><b>{MARKET_KZ_MONTH:,}</b><span class="muted">КЗ/мес TAM</span></div>
  <div class="stat"><b>{KRAVIRA_B2B_YEAR:,}</b><span class="muted">BYN/год якорь</span></div>
  <div class="stat"><b>+{ebitda_k(FIN_Y3)}</b><span class="muted">тыс. BYN EBITDA 2029</span></div>
</div>""".replace(",", " ")

    tam_rows = [
        ("КЗ/мес, Кравира", f"{KRAVIRA_KZ_MONTH:,}".replace(",", " "), "1% рынка"),
        ("КЗ/мес, частные ОЗ РБ", f"{MARKET_KZ_MONTH:,}".replace(",", " "), "экстраполяция"),
        ("TAM @ 0,99 BYN/год", f"~{TAM_REVENUE_YEAR:,} BYN".replace(",", " "), "30 млн × 0,99"),
        ("SAM 5%", f"{SAM_KZ_YEAR:,} КЗ/год".replace(",", " "), "крупные ОЗ"),
        ("SOM год 3", f"{SOM_Y3_KZ_YEAR:,} КЗ/год".replace(",", " "), "8% TAM"),
    ]
    fin_rows = [
        ("Клиенты (ОЗ)", "1", "3", "8"),
        ("КЗ/мес B2B", f"{FIN_Y1['kz_month']:,}".replace(",", " "), f"{FIN_Y2['kz_month']:,}".replace(",", " "), f"{FIN_Y3['kz_month']:,}".replace(",", " ")),
        ("B2B Кравира, тыс.", str(FIN_Y1["b2b_kravira_k"]), str(FIN_Y2["b2b_kravira_k"]), str(FIN_Y3["b2b_kravira_k"])),
        ("B2B другие ОЗ, тыс.", str(FIN_Y1["b2b_other_k"]), str(FIN_Y2["b2b_other_k"]), str(FIN_Y3["b2b_other_k"])),
        ("Выручка B2C Protocol, тыс.", str(FIN_Y1["b2c_k"]), str(FIN_Y2["b2c_k"]), str(FIN_Y3["b2c_k"])),
        ("Rev-share клиникам, тыс.", str(CLINIC_B2C_REV_Y1_K), str(CLINIC_B2C_REV_Y2_K), str(CLINIC_B2C_REV_Y3_K)),
        ("Выручка API, тыс.", str(FIN_Y1["api_k"]), str(FIN_Y2["api_k"]), str(FIN_Y3["api_k"])),
        ("Итого выручка, тыс.", str(total_rev_k(FIN_Y1)), str(total_rev_k(FIN_Y2)), str(total_rev_k(FIN_Y3))),
        ("OPEX, тыс.", str(FIN_Y1["opex_k"]), str(FIN_Y2["opex_k"]), str(FIN_Y3["opex_k"])),
        ("EBITDA, тыс.", str(ebitda_k(FIN_Y1)), f"+{ebitda_k(FIN_Y2)}", f"+{ebitda_k(FIN_Y3)}"),
    ]
    b2c_scenario_rows = [
        ("Осторожный (модель)", f"{FIN_Y3['b2c_checks']:,}".replace(",", " "), f"{FIN_Y3['b2c_k']} тыс.", "базовый план"),
        ("Upside 0,3% TAM", "~90 000", f"~{B2C_UPSIDE_YEAR3_K} тыс.", "SEO + SMS rev-share"),
        ("Средний чек", "—", f"{B2C_AVG_PRICE} BYN", "микс tier"),
    ]
    return f"""
<div class="section"><h2>Финансовые приложения и графики</h2></div>
<div class="section-body">
{stats}
{_table(['Показатель рынка РБ', 'Значение', 'Источник'], RB_MARKET_TABLE, caption='Контекст рынка платных медуслуг РБ')}
{_table(['Показатель', 'Значение', 'Комментарий'], tam_rows, caption='TAM / SAM / SOM')}
{_table(['Показатель', '2027', '2028', '2029'], fin_rows, caption='Финансовый план (тыс. BYN)')}
{_table(['Сценарий B2C', 'Проверок/год', 'Protocol', 'Комментарий'], b2c_scenario_rows, caption='B2C: осторожный vs upside')}
{img('chart_b2b_split.png', 'B2B: Кравира vs другие клиники РБ')}
{img('chart_b2c_tiers.png', 'Tier-цены B2C по specialty')}
{img('chart_b2c_revshare.png', 'Rev-share 30/70: примеры по tier')}
{img('chart_b2c_growth.png', 'Рост B2C и сценарий upside')}
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
        _cover("БИЗНЕС-ПЛАН", "инновационного проекта")
        + _mission_block()
        + f'<div class="toc"><h2>Содержание</h2><pre>{_e(TOC)}</pre></div>'
        + '<div class="section"><h2>3. Резюме</h2></div>'
        + f'<div class="section-body">{_ps(RESUME)}</div>'
        + _market_scope_html()
        + _stakeholders_html()
        + _analogues_html()
        + _sections_html_with_b2c()
        + _finance_block(assets_rel)
    )
    path.write_text(_page("Бизнес-план Protocol - Кравира", body), encoding="utf-8")


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
        + f'<div class="section-body"><p>{_e(STRATEGY_LEVELS)}</p><p>Опытный образец, B2B-пилот в Кравире, beta B2C ({STRATEGY_YEAR}).</p></div>'
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
