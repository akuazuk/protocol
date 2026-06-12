"""Графики бизнес-плана на Plotly (экспорт PNG через Kaleido)."""
from __future__ import annotations

from pathlib import Path

from konkurs_chart_style import BG_AX, BG_FIG, COLOR_NEGATIVE, COLORS, TEXT
from konkurs_finance import (
    B2C_AVG_PRICE,
    B2C_TIERS,
    FIN_Y1,
    FIN_Y2,
    FIN_Y3,
    KRAVIRA_B2B_YEAR,
    MARKET_KZ_MONTH,
    MARKET_KZ_YEAR,
    ROI_METHODIST_SAVING,
    ROI_NET,
    ROI_PROTOCOL_COST,
    ROI_TOTAL_SAVING,
    SAM_KZ_YEAR,
    SOM_Y3_KZ_YEAR,
    TAM_B2B_CEILING_YEAR_K,
    Y2_MARKET_SHARE,
    Y3_MARKET_SHARE,
    clinic_revshare_byn,
    ebitda_k,
    ebitda_month_k,
    total_rev_k,
)
from konkurs_monetization import (
    ALL_REVENUE_SCENARIOS_Y3,
    BASE_REV_Y3_K,
    EXPANDED_Y3,
    EXTRA_STREAMS_Y3,
)
from konkurs_scenarios import (
    ALL_SCENARIOS_Y3,
    B2C_PROTOCOL_PER_CHECK,
    CHANNEL_OUTLOOK,
    PENETRATION_SENSITIVITY,
    SCENARIO_BASE,
    SCENARIO_CAUTIOUS,
    SCENARIO_OPTIMISTIC,
)

_W = 820
_H = 440
_FONT = "Arial, Helvetica, DejaVu Sans, sans-serif"


def _layout(title: str, *, barmode: str | None = None, legend_y: float = 1.02) -> dict:
    lo = dict(
        title=dict(text=title, font=dict(size=14, color=TEXT), x=0.02, xanchor="left"),
        paper_bgcolor=BG_FIG,
        plot_bgcolor=BG_AX,
        font=dict(family=_FONT, color=TEXT, size=11),
        margin=dict(l=56, r=36, t=62, b=52),
        height=_H,
        width=_W,
        legend=dict(orientation="h", yanchor="bottom", y=legend_y, x=0, font=dict(size=10)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#b8c8c0"),
        yaxis=dict(gridcolor="#b8c8c0", gridwidth=1, zerolinecolor="#a8b8b0", linecolor="#b8c8c0"),
    )
    if barmode:
        lo["barmode"] = barmode
    return lo


def _export(fig, path: Path) -> None:
    """PNG через Kaleido; совместимость Plotly 5.x + Kaleido 1.x."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    try:
        fig.write_image(str(path), scale=2, engine="kaleido")
        if path.is_file() and path.stat().st_size > 100:
            return
    except Exception as e:
        last_err = e
    try:
        import asyncio

        import kaleido

        async def _write() -> None:
            await kaleido.write_fig(fig, str(path))

        asyncio.run(_write())
        if path.is_file() and path.stat().st_size > 100:
            return
    except Exception as e:
        last_err = e
    msg = (
        "Не удалось экспортировать график в PNG. "
        "Варианты: pip install 'kaleido==0.2.1' (Plotly 5) "
        "или pip install -U 'plotly>=6.1.1' 'kaleido>=1.3'."
    )
    if last_err:
        raise RuntimeError(f"{msg} Причина: {last_err}") from last_err
    raise RuntimeError(msg)


def generate_charts(assets_dir: Path) -> dict[str, Path]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    assets_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    c = COLORS
    years = ["2027", "2028", "2029"]

    # --- Выручка B2B/B2C ---
    fig = go.Figure()
    fig.add_trace(go.Bar(name="B2B", x=years, y=[FIN_Y1["b2b_k"], FIN_Y2["b2b_k"], FIN_Y3["b2b_k"]],
                         marker_color=c[0], marker_line=dict(color="#3d6b55", width=0.8), opacity=0.95))
    fig.add_trace(go.Bar(name="B2C", x=years, y=[FIN_Y1["b2c_k"], FIN_Y2["b2c_k"], FIN_Y3["b2c_k"]],
                         marker_color=c[3], marker_line=dict(color="#9a7058", width=0.8), opacity=0.95))
    fig.update_layout(**_layout(f"Выручка по годам · SOM {Y3_MARKET_SHARE:.0%} TAM", barmode="group"))
    fig.update_yaxes(title="тыс. BYN / год")
    p = assets_dir / "chart_revenue.png"
    _export(fig, p)
    paths["revenue"] = p

    # --- Выручка + EBITDA (новый комбинированный) ---
    rev = [total_rev_k(FIN_Y1), total_rev_k(FIN_Y2), total_rev_k(FIN_Y3)]
    ebit = [ebitda_k(FIN_Y1), ebitda_k(FIN_Y2), ebitda_k(FIN_Y3)]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name="Выручка", x=years, y=rev, marker_color=c[0], opacity=0.85), secondary_y=False)
    fig.add_trace(go.Scatter(name="EBITDA", x=years, y=ebit, mode="lines+markers",
                             line=dict(color=c[5], width=3), marker=dict(size=10)), secondary_y=True)
    fig.update_layout(**_layout("Выручка и EBITDA · осторожный сценарий", legend_y=1.08))
    fig.update_yaxes(title_text="Выручка, тыс.", secondary_y=False)
    fig.update_yaxes(title_text="EBITDA, тыс.", secondary_y=True)
    p = assets_dir / "chart_revenue_ebitda.png"
    _export(fig, p)
    paths["revenue_ebitda"] = p

    # --- TAM / SAM / SOM ---
    labels = ["TAM", "SAM 5%", "SOM 8%"]
    vals = [MARKET_KZ_YEAR / 1e6, SAM_KZ_YEAR / 1e6, SOM_Y3_KZ_YEAR / 1e6]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[c[2], c[1], c[0]],
                           text=[f"{v:.1f}" for v in vals], textposition="outside", marker_line_width=0))
    fig.update_layout(**_layout("Рынок КЗ: TAM → SAM → SOM (млн КЗ/год)"))
    fig.update_yaxes(title="млн КЗ / год")
    p = assets_dir / "chart_market.png"
    _export(fig, p)
    paths["market"] = p

    # --- B2B тарифы ---
    fig = go.Figure(go.Bar(
        y=["Сеть 25k+", "Клиника 10k", "Старт до 1k"],
        x=[0.69, 0.79, 0.99], orientation="h",
        marker_color=[c[0], c[1], c[2]], text=["0,69", "0,79", "0,99"], textposition="outside", marker_line_width=0))
    fig.update_layout(**_layout("Тарифная лестница B2B (BYN/КЗ L0)"))
    fig.update_xaxes(title="BYN", range=[0, 1.15])
    p = assets_dir / "chart_pricing.png"
    _export(fig, p)
    paths["pricing"] = p

    # --- Каналы выручки Y3 ---
    rev3 = total_rev_k(FIN_Y3)
    fig = go.Figure(go.Pie(
        labels=["B2B", "B2C", "API"],
        values=[FIN_Y3["b2b_k"], FIN_Y3["b2c_k"], FIN_Y3["api_k"]],
        hole=0.45, marker=dict(colors=[c[0], c[3], c[1]], line=dict(color=BG_FIG, width=2)),
        textinfo="label+percent", textfont_size=11))
    fig.update_layout(**_layout(f"Структура выручки 2029 · {rev3} тыс. BYN"), showlegend=False)
    p = assets_dir / "chart_channels.png"
    _export(fig, p)
    paths["channels"] = p

    # --- EBITDA ---
    ev = [ebitda_k(FIN_Y1), ebitda_k(FIN_Y2), ebitda_k(FIN_Y3)]
    cols = [COLOR_NEGATIVE if v < 0 else c[0] for v in ev]
    fig = go.Figure(go.Bar(x=years, y=ev, marker_color=cols, text=[f"{v:+d}" for v in ev],
                           textposition="outside", marker_line_width=0))
    fig.update_layout(**_layout(f"EBITDA · 2029: {ebitda_month_k(FIN_Y3)} тыс./мес при 8% TAM"))
    fig.update_yaxes(title="тыс. BYN / год")
    p = assets_dir / "chart_ebitda.png"
    _export(fig, p)
    paths["ebitda"] = p

    # --- Доля рынка ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    shares = [1, Y2_MARKET_SHARE * 100, Y3_MARKET_SHARE * 100]
    kz = [FIN_Y1["kz_month"] / 1000, FIN_Y2["kz_month"] / 1000, FIN_Y3["kz_month"] / 1000]
    fig.add_trace(go.Bar(name="Доля TAM, %", x=years, y=shares, marker_color=c[1], opacity=0.8), secondary_y=False)
    fig.add_trace(go.Scatter(name="КЗ/мес, тыс.", x=years, y=kz, mode="lines+markers",
                             line=dict(color=c[3], width=2.5)), secondary_y=True)
    fig.update_layout(**_layout("Проникновение B2B: от TAM 2,5 млн КЗ/мес", legend_y=1.1))
    fig.update_yaxes(title_text="% TAM", secondary_y=False)
    fig.update_yaxes(title_text="тыс. КЗ/мес", secondary_y=True)
    p = assets_dir / "chart_market_share.png"
    _export(fig, p)
    paths["market_share"] = p

    # --- OPEX ---
    cats = ["ФОТ", "Инфра", "Маркетинг", "Прочее"]
    y1, y2, y3 = [180, 35, 35, 30], [270, 55, 55, 40], [420, 85, 85, 60]
    fig = go.Figure()
    for i, (vals, yr, col) in enumerate(zip([y1, y2, y3], years, c[:3])):
        fig.add_trace(go.Bar(name=yr, x=cats, y=vals, marker_color=col, opacity=0.88, marker_line_width=0))
    fig.update_layout(**_layout("Структура OPEX по статьям", barmode="group"))
    fig.update_yaxes(title="тыс. BYN")
    p = assets_dir / "chart_opex.png"
    _export(fig, p)
    paths["opex"] = p

    # --- Маржа B2B ---
    prices = [0.99, 0.79, 0.69]
    margins = [(p - 0.09) / p * 100 for p in prices]
    fig = go.Figure(go.Bar(x=["Старт", "Клиника", "Сеть"], y=margins, marker_color=c[:3],
                           text=[f"{m:.0f}%" for m in margins], textposition="outside", marker_line_width=0))
    fig.update_layout(**_layout("Валовая маржа L0 при себестоимости ~0,09 BYN"))
    fig.update_yaxes(title="%", range=[0, 100])
    p = assets_dir / "chart_margin.png"
    _export(fig, p)
    paths["margin"] = p

    # --- ROI ---
    fig = go.Figure(go.Bar(
        x=["Затраты", "Методист", "ЦИСЗ", "Экономия"],
        y=[ROI_PROTOCOL_COST / 1000, ROI_METHODIST_SAVING / 1000,
           (ROI_TOTAL_SAVING - ROI_METHODIST_SAVING) / 1000, ROI_TOTAL_SAVING / 1000],
        marker_color=[COLOR_NEGATIVE, c[0], c[1], c[2]], marker_line_width=0,
        text=[f"{ROI_NET/1000:+.1f} нетто/мес" if i == 0 else "" for i in range(4)], textposition="outside"))
    fig.update_layout(**_layout(f"ROI якоря Кравира · нетто {ROI_NET/1000:+.1f} тыс. BYN/мес"))
    fig.update_yaxes(title="тыс. BYN / мес")
    p = assets_dir / "chart_roi.png"
    _export(fig, p)
    paths["roi"] = p

    # --- B2C воронка ---
    funnel_y = ["30 млн TAM", "QR/SMS 2%", "Landing 40%", f"Оплата {FIN_Y3['b2c_checks']:,}".replace(",", " ")]
    funnel_v = [100, 2, 0.8, FIN_Y3["b2c_checks"] / 300_000 * 100]
    fig = go.Figure(go.Funnel(y=funnel_y, x=funnel_v, textinfo="value+percent initial",
                              marker=dict(color=[c[2], c[1], c[3], c[0]], line=dict(width=0))))
    fig.update_layout(**_layout("B2C-воронка · 0,23% TAM в осторожном плане"))
    p = assets_dir / "chart_b2c_funnel.png"
    _export(fig, p)
    paths["b2c_funnel"] = p

    # --- B2B split ---
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Кравира", x=years,
                         y=[FIN_Y1["b2b_kravira_k"], FIN_Y2["b2b_kravira_k"], FIN_Y3["b2b_kravira_k"]],
                         marker_color=c[0], marker_line_width=0))
    fig.add_trace(go.Bar(name="Другие ОЗ", x=years,
                         y=[FIN_Y1["b2b_other_k"], FIN_Y2["b2b_other_k"], FIN_Y3["b2b_other_k"]],
                         marker_color=c[3], marker_line_width=0))
    fig.update_layout(**_layout("B2B: якорь vs сеть частных ОЗ", barmode="stack"))
    fig.update_yaxes(title="тыс. BYN / год")
    p = assets_dir / "chart_b2b_split.png"
    _export(fig, p)
    paths["b2b_split"] = p

    # --- B2C tiers ---
    names = [t["name"] for t in B2C_TIERS]
    prices = [t["price"] for t in B2C_TIERS]
    fig = go.Figure(go.Bar(x=names, y=prices, marker_color=c[: len(names)], marker_line_width=0,
                           text=[f"{p:.2f}" for p in prices], textposition="outside"))
    fig.add_hline(y=B2C_AVG_PRICE, line_dash="dash", line_color=c[5],
                  annotation_text=f"средний {B2C_AVG_PRICE} BYN")
    fig.update_layout(**_layout("Tier-цены B2C"))
    fig.update_yaxes(title="BYN")
    p = assets_dir / "chart_b2c_tiers.png"
    _export(fig, p)
    paths["b2c_tiers"] = p

    # --- Rev-share ---
    ex = ["Промо 2,99", "L2 9,99", "Онко 14,99", "Pre-op 12,99"]
    ex_p = [2.99, 9.99, 14.99, 12.99]
    clinic = [clinic_revshare_byn(p)[0] for p in ex_p]
    prot = [clinic_revshare_byn(p)[1] for p in ex_p]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Клинике 30%", x=ex, y=clinic, marker_color=c[3], marker_line_width=0))
    fig.add_trace(go.Bar(name="Protocol 70%", x=ex, y=prot, marker_color=c[0], marker_line_width=0))
    fig.update_layout(**_layout("Rev-share B2B2C", barmode="stack"))
    fig.update_yaxes(title="BYN / проверка")
    p = assets_dir / "chart_b2c_revshare.png"
    _export(fig, p)
    paths["b2c_revshare"] = p

    # --- B2C рост + сценарии ---
    scen_x = ["2027", "2028", "2029 остор.", "2029 баз.", "2029 оптим."]
    scen_y = [FIN_Y1["b2c_k"], FIN_Y2["b2c_k"], SCENARIO_CAUTIOUS["b2c_k"],
              SCENARIO_BASE["b2c_k"], SCENARIO_OPTIMISTIC["b2c_k"]]
    fig = go.Figure(go.Bar(x=scen_x, y=scen_y,
                           marker_color=[c[0], c[0], c[0], c[1], c[5]], marker_line_width=0))
    fig.update_layout(**_layout(f"B2C Protocol · ~{B2C_PROTOCOL_PER_CHECK} BYN/проверка"))
    fig.update_yaxes(title="тыс. BYN / год")
    p = assets_dir / "chart_b2c_growth.png"
    _export(fig, p)
    paths["b2c_growth"] = p

    # --- TAM bridge ---
    bridge_x = ["TAM теор.", "SAM 5%", "SOM B2B", "B2C+API", "Итого"]
    bridge_y = [TAM_B2B_CEILING_YEAR_K, int(SAM_KZ_YEAR * 0.75 / 1000), FIN_Y3["b2b_k"],
                FIN_Y3["b2c_k"] + FIN_Y3["api_k"], total_rev_k(FIN_Y3)]
    fig = go.Figure(go.Bar(x=bridge_x, y=bridge_y, marker_color=[c[2], c[1], c[0], c[3], c[5]],
                           text=[f"{v:,}".replace(",", " ") for v in bridge_y], textposition="outside", marker_line_width=0))
    fig.update_layout(**_layout("От TAM к выручке Protocol 2029 (тыс. BYN/год)"))
    fig.update_yaxes(title="тыс. BYN")
    p = assets_dir / "chart_tam_bridge.png"
    _export(fig, p)
    paths["tam_bridge"] = p

    # --- Сценарии EBITDA ---
    sn = [s["label"].split("(")[0].strip() for s in ALL_SCENARIOS_Y3]
    se = [s["ebitda_k"] for s in ALL_SCENARIOS_Y3]
    fig = go.Figure(go.Bar(x=sn, y=se, marker_color=c[:3], marker_line_width=0,
                           text=[f"+{v}" for v in se], textposition="outside"))
    fig.update_layout(**_layout("EBITDA 2029: три сценария проникновения"))
    fig.update_yaxes(title="тыс. BYN / год")
    p = assets_dir / "chart_scenarios_ebitda.png"
    _export(fig, p)
    paths["scenarios_ebitda"] = p

    # --- Чувствительность ---
    pen_x = [f"{p[0]}%" for p in PENETRATION_SENSITIVITY]
    pen_y = [p[1] for p in PENETRATION_SENSITIVITY]
    fig = go.Figure(go.Bar(x=pen_x, y=pen_y, marker_color=c[0], marker_line_width=0))
    fig.add_hline(y=ebitda_k(FIN_Y3), line_dash="dot", line_color=c[5],
                  annotation_text=f"план 8%: {ebitda_k(FIN_Y3)}")
    fig.update_layout(**_layout("Чувствительность EBITDA к доле TAM B2B"))
    fig.update_yaxes(title="тыс. BYN / год")
    p = assets_dir / "chart_penetration.png"
    _export(fig, p)
    paths["penetration"] = p

    # --- Каналы outlook ---
    ch_y = [x["channel"] for x in CHANNEL_OUTLOOK]
    ch_p = [x["prob"] * 100 for x in CHANNEL_OUTLOOK]
    fig = go.Figure(go.Bar(y=ch_y, x=ch_p, orientation="h", marker_color=c[0], opacity=0.85, marker_line_width=0,
                           text=[f"{x['y3_k']} тыс." for x in CHANNEL_OUTLOOK], textposition="outside"))
    fig.update_layout(**_layout("Вероятность успеха каналов к 2029"))
    fig.update_xaxes(title="%", range=[0, 105])
    p = assets_dir / "chart_channel_outlook.png"
    _export(fig, p)
    paths["channel_outlook"] = p

    # --- Структура по сценариям ---
    labels_s = ["Осторожн.", "Базовый", "Оптимист."]
    b2b_s = [s["b2b_k"] for s in ALL_SCENARIOS_Y3]
    b2c_s = [s["b2c_k"] for s in ALL_SCENARIOS_Y3]
    api_s = [s["api_k"] for s in ALL_SCENARIOS_Y3]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="B2B", x=labels_s, y=b2b_s, marker_color=c[0], marker_line_width=0))
    fig.add_trace(go.Bar(name="B2C", x=labels_s, y=b2c_s, marker_color=c[3], marker_line_width=0))
    fig.add_trace(go.Bar(name="API", x=labels_s, y=api_s, marker_color=c[1], marker_line_width=0))
    fig.update_layout(**_layout("Выручка 2029 по сценариям", barmode="stack"))
    fig.update_yaxes(title="тыс. BYN")
    p = assets_dir / "chart_scenarios_revenue.png"
    _export(fig, p)
    paths["scenarios_revenue"] = p

    # --- EBITDA год vs месяц ---
    fig = go.Figure()
    fig.add_trace(go.Bar(name="год", x=years, y=ebit, marker_color=c[0], opacity=0.8, marker_line_width=0))
    fig.add_trace(go.Bar(name="мес", x=years, y=[ebitda_month_k(FIN_Y1), ebitda_month_k(FIN_Y2), ebitda_month_k(FIN_Y3)],
                         marker_color=c[3], opacity=0.8, marker_line_width=0))
    fig.update_layout(**_layout("EBITDA: год и месяц", barmode="group"))
    p = assets_dir / "chart_ebitda_monthly.png"
    _export(fig, p)
    paths["ebitda_monthly"] = p

    # --- НОВЫЙ: доп. каналы монетизации ---
    stream_names = [s["name"] for s in EXTRA_STREAMS_Y3]
    stream_vals = [s["y3_k"] for s in EXTRA_STREAMS_Y3]
    fig = go.Figure(go.Bar(
        y=stream_names, x=stream_vals, orientation="h",
        marker=dict(color=stream_vals, colorscale=[[0, c[1]], [0.5, c[0]], [1, c[3]]], showscale=False),
        text=[f"{v}" for v in stream_vals], textposition="outside", marker_line_width=0))
    fig.update_layout(**_layout(f"Доп. каналы монетизации 2029 · +{EXPANDED_Y3['extra_rev_k']} тыс. BYN/год"))
    fig.update_xaxes(title="тыс. BYN / год")
    p = assets_dir / "chart_monetization.png"
    _export(fig, p)
    paths["monetization"] = p

    # --- НОВЫЙ: все сценарии выручки включая расширенный ---
    scen_labels = [x[0] for x in ALL_REVENUE_SCENARIOS_Y3]
    scen_rev = [x[1] for x in ALL_REVENUE_SCENARIOS_Y3]
    scen_ebitda = [x[2] for x in ALL_REVENUE_SCENARIOS_Y3]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name="Выручка", x=scen_labels, y=scen_rev, marker_color=c[0], opacity=0.85), secondary_y=False)
    fig.add_trace(go.Scatter(name="EBITDA", x=scen_labels, y=scen_ebitda, mode="markers+lines",
                             marker=dict(size=12, color=c[5]), line=dict(width=2)), secondary_y=True)
    fig.update_layout(**_layout("Сценарии 2029: до 3,5 млн выручки (расширенный)", legend_y=1.12))
    fig.update_yaxes(title_text="Выручка, тыс.", secondary_y=False)
    fig.update_yaxes(title_text="EBITDA, тыс.", secondary_y=True)
    p = assets_dir / "chart_all_scenarios.png"
    _export(fig, p)
    paths["all_scenarios"] = p

    # --- Расширенная выручка Y3 ---
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Базовый", x=["2029"], y=[BASE_REV_Y3_K], marker_color=c[1], marker_line_width=0,
        text=[f"EBITDA +{ebitda_k(FIN_Y3)}"], textposition="outside"))
    fig.add_trace(go.Bar(
        name="Расширенный", x=["2029"], y=[EXPANDED_Y3["total_rev_k"]], marker_color=c[0], marker_line_width=0,
        text=[f"EBITDA +{EXPANDED_Y3['ebitda_k']}"], textposition="outside"))
    fig.update_layout(**_layout(f"Потенциал: +{EXPANDED_Y3['extra_rev_k']} тыс. от 8 доп. каналов", barmode="group"))
    fig.update_yaxes(title="тыс. BYN / год")
    p = assets_dir / "chart_expanded_potential.png"
    _export(fig, p)
    paths["expanded_potential"] = p

    return paths
