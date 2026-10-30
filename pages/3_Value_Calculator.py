"""
3_Value_Calculator.py — Página 4: Valor Económico y Casos de Uso
================================================================
Convierte la calidad del forecast en VALOR ECONÓMICO realista.

Principio de prudencia (aplicado en todos los casos):
  El valor del MODELO no es el valor bruto de tener flexibilidad, sino el
  INCREMENTAL sobre una estrategia naive (programar como ayer), medido contra
  el techo de un ORÁCULO con previsión perfecta. Siempre se muestran las tres
  referencias: oráculo (techo) · modelo · naive (suelo).

Casos de uso (modelos independientes y bien trabajados):
  1. Batería (BESS) — arbitraje con dispatch LP y degradación + P&L completo
  2. Consumidor industrial — desplazamiento de carga flexible
  3. Operador de recarga de VE — carga inteligente
Más una sección con los segmentos de mercado que usan un forecast day-ahead.
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    load_test_predictions, run_battery_economics, load_lear_forecast,
    render_data_source_banner, ZONE_LABELS, render_country_selector,
    load_model_summary,
)
from src.charts import _apply_base, C

# ── Casos de uso y segmentos por país (DK vs FI) ──────────────────────────────
COUNTRY_VALUE = {
    "Denmark": {
        "intro": "Anyone who makes **time-of-use decisions** on the Danish day-ahead "
                 "market (DK1/DK2) and competes on margin.",
        "scale_name": "a 40 MW Danish BESS",
        "scale_mw": 40,
        "vendor_line": "",
        "battery_anchor": None,
        "industrial_anchor": "**Danish context:** the national **Power-to-X / hydrogen** "
            "build-out (electrolysers) plus large cold-storage and food processing are the "
            "classic flexible loads — they shift consumption into cheap wind hours.",
        "ev_anchor": "**Danish context:** CPOs like **Clever** and **Spirii** smart-charge "
            "fleets; with Denmark's high EV penetration, sourcing in the cheapest hours is a "
            "direct margin lever.",
        "segments": [
            ("BESS operators & storage optimisers", "Daily charge/discharge arbitrage; bid into day-ahead",
             "Turns an unprofitable naive battery into a profitable asset (Case 1)"),
            ("Industrial flexible consumers", "Shift production/cooling to cheap hours",
             "Incremental savings over 'same as yesterday' scheduling"),
            ("EV charging networks (CPOs)", "Smart-charge fleets in the cheapest window",
             "Lower €/kWh sourced → better margins or cheaper tariffs"),
            ("Energy retailers / aggregators (VPP)", "Optimise portfolio dispatch & hedge procurement",
             "Fewer imbalance costs, better day-ahead positioning"),
            ("Wind aggregators (DK is wind-heavy)", "Time commitment & hedge around the price curve",
             "Edge comes from the spread/shape, not just the level"),
        ],
    },
    "Finland": {
        "intro": "Finnish market participants making day-ahead (FI) decisions: battery "
                 "operators, utilities and district-heat producers, retailers with "
                 "spot-indexed tariffs, large flexible loads and EV-charging networks.",
        "scale_name": "a 40 MW Finnish BESS",
        "scale_mw": 40,
        "vendor_line": "",
        "battery_anchor": "**Finnish context:** battery-bidding and energy-management "
                          "platforms combine a day-ahead price forecast with a bid "
                          "optimiser. This case covers only the forecast-driven arbitrage "
                          "layer; the **Battery Lab** page stacks it with the FCR-N / FCR-D "
                          "reserve markets for commercial (datasheet) and generic (parametric) batteries.",
        "industrial_anchor": "**Finnish context:** heavy industry (e.g. the Kokkola zinc "
            "smelter), battery-materials sites and data centres are large, partly-flexible "
            "loads, and Finland's cheap, low-carbon electricity keeps attracting new "
            "power-intensive demand. All of them can shift consumption into the cheapest hours.",
        "ev_anchor": "**Finnish context:** charge-point operators smart-charge fleets in the "
            "cheapest window, and published fleet-charging case studies report sizeable "
            "cost reductions from this kind of optimisation. The day-ahead forecast decides "
            "*which* hours are cheapest.",
        "segments": [
            ("BESS operators & storage optimisers", "Stack day-ahead + intraday + FCR/aFRR/mFRR",
             "Day-ahead arbitrage is the base layer of the revenue stack"),
            ("Utilities & district-heat producers", "Power-to-heat (electric boilers, heat storage), fuel-switching by price",
             "Textbook day-ahead arbitrage between electricity and heat"),
            ("Wind producers & aggregators", "Commit wind, hedge around the price curve",
             "Values wind cannibalisation and shapes hedges"),
            ("Retailers with spot-indexed tariffs", "Set exchange-electricity tariffs; spot hedging",
             "Retail price-setting and procurement"),
            ("Large flexible loads (metals, battery materials, data centres)", "Shift flexible loads to cheap hours",
             "Big volumes magnify small per-MWh gains"),
            ("EV charging networks (CPOs)", "Smart-charge fleets; dynamic next-day tariffs",
             "Lower €/kWh sourced → better margins or cheaper tariffs"),
        ],
    },
}

st.set_page_config(page_title="Value & Use Cases · Nordic Forecast",
                   layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { color: #1B2A4A; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Helpers de presentación ───────────────────────────────────────────────────

_BAR_COLORS = {
    "naive":  C["gray"],
    "lear":   C["orange"],
    "model":  C["green"],
    "oracle": C["light_blue"],
}
_BAR_LABELS = {
    "naive":  "Naive<br>(like yesterday)",
    "lear":   "LEAR baseline<br>(linear ARX)",
    "model":  "This model<br>(LightGBM)",
    "oracle": "Oracle<br>(perfect foresight)",
}


def _bars(values: dict, title, ylab="EUR / year"):
    """
    Barras naive→lear→model→oracle (las que existan en `values`),
    de suelo a techo. Patrón reutilizable en los 3 casos.
    """
    order = [k for k in ("naive", "lear", "model", "oracle") if k in values]
    fig = go.Figure(go.Bar(
        x=[_BAR_LABELS[k] for k in order],
        y=[values[k] for k in order],
        marker_color=[_BAR_COLORS[k] for k in order],
        text=[f"€{values[k]:,.0f}" for k in order],
        textposition="outside", textfont=dict(size=13),
    ))
    fig = _apply_base(fig, title)
    fig.update_yaxes(title_text=ylab)
    fig.add_hline(y=0, line_dash="dash", line_color=C["gray"])
    return fig


def _capture_pct(model, floor, oracle):
    """% del headroom (oracle − floor) capturado por el modelo."""
    denom = oracle - floor
    return (model - floor) / denom * 100 if denom > 1e-9 else np.nan


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    country = render_country_selector()
    st.markdown("### Value & Use Cases")
    st.divider()

    # La economía se calcula siempre sobre la zona MODELADA del país
    if len(country["modeled_zones"]) > 1:
        zone = st.selectbox("Zone", options=country["modeled_zones"],
                            format_func=lambda z: ZONE_LABELS[z])
    else:
        zone = country["modeled_zones"][0]
        st.caption(f"Zone: **{ZONE_LABELS[zone]}**")

    profile = st.radio(
        "Use case",
        options=["Battery storage (BESS)",
                 "Industrial load-shifting",
                 "EV charging operator"],
        index=0,
    )

    st.divider()
    st.caption("All figures are back-tested on the **out-of-sample test period** "
               "(Jan–May 2026) and annualised. Settled at **real** prices.")

cval = COUNTRY_VALUE[country["name"]]

# ── Datos base ────────────────────────────────────────────────────────────────

render_data_source_banner(zone)

with st.spinner("Loading test-period data..."):
    df_test = load_test_predictions(zone)

# Serie real + forecast indexada (para los casos industrial / EV)
g = pd.DataFrame()
if df_test is not None and not df_test.empty and "price_actual" in df_test.columns:
    ts_col = "timestamp_cet" if "timestamp_cet" in df_test.columns else "timestamp_utc"
    gi = df_test.copy()
    gi.index = pd.to_datetime(gi[ts_col], utc=True).dt.tz_convert("Europe/Copenhagen")
    g = pd.DataFrame({"a": gi["price_actual"].astype(float),
                      "f": gi["price_pred"].astype(float)}).dropna()
    # Baseline comercial LEAR alineado al mismo índice (puede faltar en algunas zonas)
    _lear = load_lear_forecast(zone)
    if _lear is not None and not _lear.empty:
        _lear = _lear.copy()
        _lear.index = pd.to_datetime(_lear.index, utc=True).tz_convert("Europe/Copenhagen")
        g["l"] = _lear.reindex(g.index)

# ── Header + principio de prudencia ───────────────────────────────────────────

hcol1, hcol2 = st.columns([4, 1])
hcol1.title("Economic Value & Use Cases")
with hcol2.popover("How value is measured"):
    st.markdown(
        "A good forecast is only worth the **extra money it makes over a trivial "
        "baseline** — not the gross value of owning flexible assets. So every case "
        "below shows three bars:\n\n"
        "- **Oracle** — perfect foresight (theoretical ceiling).\n"
        "- **This model** — decisions taken with our forecast.\n"
        "- **Naive** — schedule the same hours as yesterday (the cheap baseline).\n\n"
        "**The model's real value = (model − naive)**, and we report what % of the "
        "oracle's achievable headroom it captures. Decisions use the forecast; "
        "settlement always uses the **real** day-ahead price. Back-tested on the "
        "out-of-sample Jan–May 2026 test window and annualised."
    )

if g.empty:
    st.error("No test-period data available. Train the model and run the pipeline first.")
    st.stop()

st.caption(f"Test window: **{g.index.min():%Y-%m-%d} → {g.index.max():%Y-%m-%d}** "
           f"· {len(g):,} settled hours · zone **{zone}**")

# ── Propuesta de valor (country-aware · euros reales del back-test) ────────────
try:
    _econ = run_battery_economics(zone=zone)
except Exception:
    _econ = None
if _econ and _econ.get("gross_model_annual"):
    _m  = _econ["gross_model_annual"]
    _n  = _econ["gross_naive_annual"]
    _o  = _econ["gross_perfect_annual"]
    _l  = _econ.get("gross_lear_annual")
    _floor, _flbl = (_l, "LEAR") if _l else (_n, "naive")
    _edge = _m - _floor
    _pct  = (_m - _n) / (_o - _n) * 100 if _o > _n else float("nan")
    _scale = cval["scale_mw"]
    _summary = load_model_summary(zone)
    _mae    = _summary.get("MAE_test")
    _vslear = _summary.get("MAE_vs_lear")
    _skill = ""
    if _mae is not None:
        _skill = (f" The model's day-ahead error is **{_mae:.1f} €/MWh MAE**"
                  + (f" — €{abs(_vslear):.1f}/MWh better than the LEAR baseline" if _vslear else "")
                  + ".")
    st.markdown("#### Back-tested value — in euros")
    st.success(
        f"Built on **100% free public data**, backtested out of sample on **Jan–May 2026 "
        f"{country['name']} prices**. A **1 MW / 2 h battery** captures **€{_m:,.0f}/yr** of day-ahead arbitrage "
        f"with our forecast — vs **€{_floor:,.0f}** with a **{_flbl}** baseline and "
        f"**€{_n:,.0f}** scheduling naively: a **€{_edge:,.0f} / MW·yr edge over the "
        f"{_flbl} baseline**, capturing **{_pct:.0f}%** of the headroom a perfect forecast adds over "
        f"naive (gross arbitrage, before fixed costs — full net economics in the Battery case below). "
        f"Scaled to {cval['scale_name']}: **≈ €{_edge * _scale / 1000:,.0f}k/yr over {_flbl}**."
        + _skill
    )
    if cval.get("vendor_line"):
        st.caption(cval["vendor_line"])

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  CASO 1 — BATERÍA (BESS): ARBITRAJE + DEGRADACIÓN + P&L
# ══════════════════════════════════════════════════════════════════════════════

if "Battery" in profile:
    st.markdown("## Battery Storage (BESS) — Day-Ahead Arbitrage")
    st.markdown(
        "A battery charges in cheap hours and discharges in expensive ones. The "
        "**dispatch is optimised by linear programming and is degradation-aware** — "
        "it only cycles when the expected spread covers battery wear, exactly as a "
        "real operator would. Hours are **chosen with each forecast** and **settled "
        "at the real price**. The fair benchmark here is **LEAR** (a respectable "
        "linear forecast), not just the naive floor."
    )
    if cval.get("battery_anchor"):
        st.info(cval["battery_anchor"])

    col_p, col_r = st.columns([1, 2])

    with col_p:
        st.markdown("### Battery & cost assumptions")
        capacity = st.slider("Energy capacity (MWh)", 1.0, 20.0, 2.0, step=0.5)
        power    = st.slider("Power rating (MW)", 0.5, 10.0, 1.0, step=0.5,
                             help="Charge/discharge rate. Capacity ÷ power = hours to "
                                  "fill (the battery's duration).")
        rte      = st.slider("Round-trip efficiency (%)", 70, 95, 88,
                             help="Energy out ÷ energy in. Modern Li-ion ≈ 85–90%.") / 100
        capex_kwh = st.slider("CAPEX (€/kWh)", 100, 500, 300, step=10,
                              help="Installed cost per kWh of energy capacity. "
                                   "Utility Li-ion ≈ 250–350 €/kWh (2024–25).")
        cycle_life = st.slider("Cycle life (equiv. full cycles)", 3000, 10000, 6000, step=500,
                               help="Full charge/discharge cycles until ~80% health "
                                    "(typical LFP warranty: 5,000–8,000).")
        om_kw_yr  = st.slider("Fixed O&M (€/kW-year)", 0, 40, 10,
                              help="Annual operations & maintenance per kW of power.")
        grid_fee  = st.slider("Grid & tax cost (€/MWh cycled)", 0, 30, 6,
                              help="Use-of-system tariffs, taxes and levies paid per MWh "
                                   "moved. A real execution cost ignored by naive models.")
        capture   = st.slider("Spread capture (%)", 50, 100, 85,
                              help="Fraction of the theoretical spread actually captured, "
                                   "after bid/execution slippage and **cannibalisation** as "
                                   "more batteries enter the market and compress spreads.") / 100

    # Degradación marginal = CAPEX amortizado sobre la energía de vida (€/MWh descargado)
    deg_per_mwh = capex_kwh * 1000.0 / cycle_life

    r = run_battery_economics(zone=zone, capacity_mwh=capacity, power_mw=power,
                              efficiency=rte, deg_per_mwh=deg_per_mwh)
    has_lear = "gross_lear_annual" in r

    with col_r:
        st.markdown("### Results — annualised, net of all costs")
        if not r:
            st.info("Not enough data for the battery backtest.")
        else:
            total_capex = capex_kwh * capacity * 1000.0        # €/kWh × kWh
            om_cost     = om_kw_yr * power * 1000.0             # €/kW-yr × kW

            def net_of(strat):
                """Beneficio neto: spread capturado × haircut − desgaste − red − O&M."""
                gross = r[f"gross_{strat}_annual"] * capture
                wear  = r[f"thru_{strat}_annual"] * deg_per_mwh
                grid  = r[f"thru_{strat}_annual"] * grid_fee
                return gross - wear - grid - om_cost

            nets = {"oracle": net_of("perfect"), "model": net_of("model"),
                    "naive":  net_of("naive")}
            if has_lear:
                nets["lear"] = net_of("lear")
            floor_key   = "lear" if has_lear else "naive"
            net_model   = nets["model"]
            net_floor   = nets[floor_key]
            gross_model = r["gross_model_annual"] * capture
            wear_model  = r["thru_model_annual"] * deg_per_mwh
            grid_model  = r["thru_model_annual"] * grid_fee
            op_margin   = gross_model - grid_model - om_cost   # caja antes de desgaste
            cycles_day  = r["cycles_per_day_model"]
            life_years  = (cycle_life / (cycles_day * 365)) if cycles_day > 0 else np.nan
            uplift_real = net_model - net_floor
            cap_pct     = _capture_pct(net_model, net_floor, nets["oracle"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Net profit (model)", f"€{net_model:,.0f}/yr",
                      help="Captured arbitrage − wear − grid/tax − O&M, settled at real "
                           "prices. Economic profit after every running cost.")
            m2.metric(f"Model value vs {floor_key.upper()}", f"€{uplift_real:,.0f}/yr",
                      delta=(f"{cap_pct:.0f}% of oracle headroom"
                             if np.isfinite(cap_pct) else None),
                      delta_color="off",
                      help="The forecast's edge over a **respectable baseline** "
                           "(LEAR). This is the defensible value — much "
                           "smaller than the edge over a naive strawman.")
            m3.metric("Cycling / battery life", f"{cycles_day:.2f} cyc/day",
                      delta=f"≈ {life_years:.0f} yr to 80%" if np.isfinite(life_years) else "—",
                      delta_color="off",
                      help="Degradation-aware dispatch only cycles profitable spreads, "
                           "so the battery lasts longer.")

            st.plotly_chart(
                _bars(nets, f"Net annual profit after all costs — {zone}"),
                width="stretch")

            if has_lear:
                st.info(
                    f"**The number that matters is the edge over LEAR: €{uplift_real:,.0f}/yr.** "
                    f"Against a naive 'like-yesterday' schedule the model looks worth "
                    f"€{net_model - nets['naive']:,.0f}/yr — but a real competitor uses a "
                    f"decent forecast, not persistence. LEAR already captures most of the "
                    f"arbitrage, so the model's *commercial* edge in this single market is "
                    f"modest (though it still wins, and on a better MAE).")
            elif nets["naive"] < 0 < net_model:
                st.success(
                    f"A forecast-driven, degradation-aware dispatch earns "
                    f"**€{net_model:,.0f}/yr**; a naive fixed schedule loses "
                    f"**€{abs(nets['naive']):,.0f}/yr** by over-cycling.")

    # ── P&L completo (investment view) ────────────────────────────────────────
    if r:
        st.markdown("### Full economic simulation (investment view)")
        disc = st.slider("Discount rate / WACC for NPV (%)", 0, 15, 7) / 100
        life_int = int(np.clip(round(life_years if np.isfinite(life_years) else 12), 1, 25))
        # NPV sobre el margen operativo de caja (captured − grid − O&M); el desgaste
        # representa la recuperación del CAPEX, así que NO se descuenta dos veces.
        npv = -total_capex + sum(op_margin / (1 + disc) ** t for t in range(1, life_int + 1))
        payback = total_capex / op_margin if op_margin > 0 else np.inf

        pnl = pd.DataFrame({
            "Line": [
                "Battery CAPEX (one-off)",
                "Captured arbitrage revenue (model)",
                "Grid & tax cost",
                "Fixed O&M",
                "Operating cash margin",
                "Battery wear (degradation)",
                "Economic profit after wear",
                "Simple payback",
                f"NPV over ~{life_int} yr @ {disc:.0%}",
            ],
            "Value": [
                f"−€{total_capex:,.0f}",
                f"€{gross_model:,.0f}/yr",
                f"−€{grid_model:,.0f}/yr",
                f"−€{om_cost:,.0f}/yr",
                f"€{op_margin:,.0f}/yr",
                f"−€{wear_model:,.0f}/yr",
                f"€{net_model:,.0f}/yr",
                (f"{payback:.1f} yr" if np.isfinite(payback) else "never"),
                f"€{npv:,.0f}",
            ],
        })
        cpa, cpb = st.columns([1, 1])
        with cpa:
            st.dataframe(pnl, hide_index=True, width="stretch")
        with cpb:
            st.markdown(
                f"- **Degradation = €{deg_per_mwh:,.0f}/MWh** discharged "
                f"(CAPEX €{capex_kwh}/kWh ÷ {cycle_life:,} cycles). The captured spread "
                "must beat this **plus grid/tax** before a cycle is worth running.\n"
                "- **Payback** uses the operating cash margin; **wear** already "
                "represents CAPEX amortisation, so the two are *two lenses on the same "
                "cost* — don't add them.\n"
                f"- Viable when payback (**{payback:.1f} yr**) < life "
                f"(**≈{life_int} yr**) and NPV > 0."
            )

# ══════════════════════════════════════════════════════════════════════════════
#  CASO 2 — INDUSTRIAL: DESPLAZAMIENTO DE CARGA
# ══════════════════════════════════════════════════════════════════════════════

elif "Industrial" in profile:
    st.markdown("## Industrial Consumer — Flexible Load-Shifting")
    st.markdown(
        "A plant with flexible load moves consumption from the most expensive hours "
        "to the cheapest ones each day, guided by the day-ahead forecast. The naive "
        "baseline simply reuses **yesterday's** cheap/expensive hours."
    )
    if cval.get("industrial_anchor"):
        st.info(cval["industrial_anchor"])

    col_p, col_r = st.columns([1, 2])
    with col_p:
        st.markdown("### Parameters")
        mw_flex    = st.slider("Flexible capacity (MW)", 1, 50, 5)
        hours_flex = st.slider("Flexible hours per day", 1, 8, 3,
                               help="How many hours of load can be moved each day.")
        eff_factor = st.slider("Execution reliability (%)", 50, 100, 80,
                               help="How reliably the plant actually acts on the plan. "
                                    "Applied equally to model and naive, so it cancels "
                                    "out in the model's incremental value.") / 100
        days_year  = st.slider("Operating days per year", 200, 365, 330)
        forecast_cost = st.number_input("Annual cost of the forecast product (€)",
                                        0, 100_000, 12_000, step=1_000,
                                        help="What the plant pays to license/integrate "
                                             "the forecast. The model must beat naive by "
                                             "more than this to be worth buying.")

    def shift_value(rank_series):
        sv = []
        for _, sub in g.groupby(g.index.date):
            if len(sub) < 24:
                continue
            rk = rank_series.reindex(sub.index).dropna()
            if len(rk) < hours_flex * 2:
                continue
            hi = rk.nlargest(hours_flex).index
            lo = rk.nsmallest(hours_flex).index
            sv.append((sub["a"].reindex(hi).mean() - sub["a"].reindex(lo).mean())
                      * mw_flex * hours_flex)
        s = np.array(sv)
        n = len(s)
        return (s.sum() * (days_year / n)) if n else 0.0

    with col_r:
        st.markdown("### Results — annual savings")
        vals = {"model":  shift_value(g["f"]) * eff_factor,
                "naive":  shift_value(g["a"].shift(24)) * eff_factor,
                "oracle": shift_value(g["a"]) * eff_factor}
        has_lear = "l" in g.columns and g["l"].notna().any()
        if has_lear:
            vals["lear"] = shift_value(g["l"]) * eff_factor
        floor_key   = "lear" if has_lear else "naive"
        incremental = vals["model"] - vals[floor_key]
        cap_pct     = _capture_pct(vals["model"], vals[floor_key], vals["oracle"])
        net_of_cost = incremental - forecast_cost

        m1, m2, m3 = st.columns(3)
        m1.metric("Flexibility value (model)", f"€{vals['model']:,.0f}/yr",
                  help="Gross saving from shifting load with the forecast — but most "
                       "of this is the value of *having* flexibility, not of the model.")
        m2.metric(f"Model value vs {floor_key.upper()}", f"€{incremental:,.0f}/yr",
                  delta=f"{cap_pct:.0f}% of oracle headroom" if np.isfinite(cap_pct) else None,
                  delta_color="off",
                  help="The forecast's edge over a respectable baseline (LEAR), not the "
                       "naive strawman. This is the defensible commercial value.")
        m3.metric("Net of forecast cost", f"€{net_of_cost:,.0f}/yr",
                  delta="worth buying" if net_of_cost > 0 else "below break-even",
                  delta_color="normal" if net_of_cost > 0 else "inverse",
                  help="Incremental model value minus what the forecast product costs.")

        st.plotly_chart(
            _bars(vals, f"Annual load-shifting savings — {zone}"),
            width="stretch")
        st.info(
            "The takeaway: a plant already saves a lot just by shifting to "
            "stable cheap hours — even a linear LEAR forecast captures most of it. The "
            "**model's real value is its edge over LEAR**, and it pays off only when that "
            "edge exceeds the product's cost.")

# ══════════════════════════════════════════════════════════════════════════════
#  CASO 3 — OPERADOR DE RECARGA DE VE
# ══════════════════════════════════════════════════════════════════════════════

elif "EV" in profile:
    st.markdown("## EV Charging Operator — Smart Charging")
    st.markdown(
        "An operator with flexible charging sessions schedules them in the cheapest "
        "predicted hours instead of charging uniformly. Naive baseline = cheapest "
        "hours of **yesterday**; oracle = cheapest **actual** hours."
    )
    if cval.get("ev_anchor"):
        st.info(cval["ev_anchor"])

    col_p, col_r = st.columns([1, 2])
    with col_p:
        st.markdown("### Parameters")
        mw_charge   = st.slider("Charging capacity (MW)", 0.5, 20.0, 2.0, step=0.5)
        daily_hours = st.slider("Flexible charging hours / day", 2, 8, 4)
        days_op     = st.slider("Operating days / year", 200, 365, 340)
        forecast_cost = st.number_input("Annual cost of the forecast product (€)",
                                        0, 100_000, 8_000, step=1_000)

    def charge_cost(rank_series):
        """Coste anual cargando en las horas más baratas según rank_series, vs uniforme."""
        sv = []
        for _, sub in g.groupby(g.index.date):
            if len(sub) < 24:
                continue
            rk = rank_series.reindex(sub.index).dropna()
            if len(rk) < daily_hours * 2:
                continue
            cheap = rk.nsmallest(daily_hours).index
            cost_smart   = sub["a"].reindex(cheap).mean() * mw_charge * daily_hours
            cost_uniform = sub["a"].mean() * mw_charge * daily_hours
            sv.append(cost_uniform - cost_smart)        # ahorro vs cargar uniforme
        s = np.array(sv)
        n = len(s)
        return (s.sum() * (days_op / n)) if n else 0.0

    with col_r:
        st.markdown("### Results — annual charging-cost savings")
        vals = {"model":  charge_cost(g["f"]),
                "naive":  charge_cost(g["a"].shift(24)),
                "oracle": charge_cost(g["a"])}
        has_lear = "l" in g.columns and g["l"].notna().any()
        if has_lear:
            vals["lear"] = charge_cost(g["l"])
        floor_key   = "lear" if has_lear else "naive"
        incremental = vals["model"] - vals[floor_key]
        cap_pct     = _capture_pct(vals["model"], vals[floor_key], vals["oracle"])
        net_of_cost = incremental - forecast_cost

        m1, m2, m3 = st.columns(3)
        m1.metric("Smart-charging saving (model)", f"€{vals['model']:,.0f}/yr",
                  help="Saving vs charging uniformly through the day, using the forecast.")
        m2.metric(f"Model value vs {floor_key.upper()}", f"€{incremental:,.0f}/yr",
                  delta=f"{cap_pct:.0f}% of oracle headroom" if np.isfinite(cap_pct) else None,
                  delta_color="off",
                  help="The forecast's edge over a respectable baseline (LEAR).")
        m3.metric("Net of forecast cost", f"€{net_of_cost:,.0f}/yr",
                  delta="worth buying" if net_of_cost > 0 else "below break-even",
                  delta_color="normal" if net_of_cost > 0 else "inverse")

        st.plotly_chart(
            _bars(vals, f"Annual smart-charging savings — {zone}"),
            width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  ¿QUIÉN PUEDE USAR ESTE FORECAST?
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"## Who can use this forecast? — {country['name']}")
st.caption(cval["intro"] + " The forecast's value scales with how much volume they "
           "can move and how volatile their zone is.")

seg = pd.DataFrame(cval["segments"],
                   columns=["Segment", "What they do with it", "Why the forecast matters"])
st.dataframe(seg, hide_index=True, width="stretch")

with st.expander("How to read the incremental value"):
    st.markdown(
        "- The model is worth its **incremental value over a *respectable* baseline "
        "(LEAR)** — not over a naive strawman, and not the gross value of the flexible "
        "asset. That edge is the headline metric in each case.\n"
        "- Value is **zone- and volatility-dependent**: it rises with bigger spreads "
        "(winter, congested zones) and bigger movable volume.\n"
        "- A forecast is only worth paying for while its cost stays **below the incremental "
        "value**; each case above shows that break-even against the cost you enter.\n"
        "- Everything here is back-tested **out-of-sample** on real public data, "
        "with the served day-ahead forecast; the oracle bar makes the remaining "
        "headroom explicit."
    )

st.divider()

# ── Supuestos y limitaciones ────────────────────────────────────────────────────
st.markdown("### Assumptions & limitations (read before quoting any number)")
with st.expander("What this simulation **does** and **does not** capture", expanded=False):
    # Cifras del modelo servido, leídas de served_metrics_<zona>.json
    _parts = []
    for _z in country["modeled_zones"]:
        _m = load_model_summary(_z)
        if isinstance(_m.get("MAE_test"), (int, float)):
            _txt = f"{_z} MAE {_m['MAE_test']:.1f} €/MWh"
            if isinstance(_m.get("MAE_change_vs_lear_pct"), (int, float)):
                _txt += f" ({_m['MAE_change_vs_lear_pct']:+.0f}% vs a zone-aware LEAR)"
            _parts.append(_txt)
    _mae_line = ("; ".join(_parts) + ".") if _parts else "served-model metrics not available."
    st.markdown(
        "**What's solid / real**\n"
        "- The forecast is genuine and **out-of-sample (Jan–May 2026)**, built from "
        "information available before the day-ahead auction (the README lists the "
        f"remaining data-timing limitations); {_mae_line}\n"
        "- Decisions use the forecast; **settlement is always at the real price** — the "
        "standard EPF value methodology.\n"
        "- The benchmark is **LEAR**, a respectable commercial-grade baseline, not a "
        "naive strawman.\n\n"
        "**Modelled here (sliders above)** — grid/tax cost per MWh, spread-capture haircut "
        "(execution slippage + **cannibalisation** as more batteries enter), round-trip "
        "efficiency, degradation-aware dispatch, CAPEX, O&M, WACC/NPV.\n\n"
        "**Simplified — caveats that would move a real number**\n"
        "1. **Day-ahead only.** Real BESS revenue *stacks* FCR-N/D, aFRR, mFRR and "
        "intraday — often **more** than arbitrage. This is a single-market **floor**.\n"
        "2. **Price-taker.** Large fleets compress the very spreads they exploit; the "
        "capture slider approximates this but it isn't a market-equilibrium model.\n"
        "3. **Annualised from ~5 months.** Spreads are seasonal (winter higher), so the "
        "annual figure carries extrapolation risk.\n"
        "4. **Flat degradation.** Real Li-ion ageing is non-linear (depth-of-discharge, "
        "temperature, calendar) — not bankable as-is.\n"
        "5. **No financing structure, insurance, availability/downtime, augmentation, "
        "grid-connection or permitting.**\n\n"
        "**Verdict.** A credible **screening / proof-of-concept** and a strong demonstration "
        "of turning forecast quality into realistic economics — **not** a financeable "
        "investment case. Quote the *edge over LEAR*, and quote these caveats with it."
    )

st.caption("Nordic Energy Forecast · Economic value back-tested on the Jan–May 2026 "
           "out-of-sample period · settled at real prices · benchmarked vs LEAR · "
           "100% public/free data")
