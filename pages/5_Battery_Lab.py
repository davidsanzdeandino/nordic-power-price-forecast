"""
5_Battery_Lab.py — Página 6: Battery Lab (revenue stacking multi-mercado)
=========================================================================
Simula una batería COMERCIAL real (specs de datasheet con fuentes) operando el
stack nórdico completo: FCR-N + FCR-D up/down + arbitraje day-ahead, con las
reglas LER reales de los TSOs nórdicos y precios de reserva realizados
(Fingrid Open Data para FI; Energi Data Service para DK2).

La página cuenta la historia REAL del mercado: el desplome de los precios FCR
por saturación de baterías (2023→2026) y el desplazamiento del valor hacia el
arbitraje — con los mismos números que publican Fingrid, Clean Horizon y los
operadores.
"""

import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import render_country_selector, load_test_predictions
from src.model.battery_specs import BATTERY_SPECS, list_specs
from src.model.revenue_stacking import (
    stacking_backtest, yearly_stack_summary,
    FCRN_POWER_FACTOR, FCRN_ENERGY_H, FCRD_OPP_POWER_FACTOR, FCRD_ENERGY_H,
    FCRN_THROUGHPUT_PER_MWH,
)

st.set_page_config(page_title="Battery Lab · Nordic Forecast",
                   layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { color: #1B2A4A; }
/* Battery names are long: let the sidebar selectbox wrap instead of cutting them */
section[data-testid="stSidebar"] [data-baseweb="select"] div {
  white-space: normal; overflow: visible; text-overflow: clip; height: auto; }
</style>
""", unsafe_allow_html=True)

MASTER_PATH = ROOT / "data" / "master" / "dataset_master.parquet"

# Fuentes públicas de las cifras de terceros citadas en la página
U_CAPALO_FI = "https://capaloai.com/blog/finnish-bess-market/"
U_CAPALO_Q2 = "https://capaloai.com/blog/capalo-zeus-vpp-quarterly-performance-q2-2025/"
U_CH = "https://www.cleanhorizon.com/news/storage-index-update-finland-in-focus/"
U_EBI = "https://europeanbessindex.com/"
U_FG_FCR = "https://www.fingrid.fi/en/news/news/2025/procurement-of-frequency-containment-reserves-from-yearly-market-2026/"
U_FG_1GW = "https://www.fingrid.fi/en/news/news/2026/the-combined-capacity-of-electricity-storages-in-finland-now-exceeds-1000-mw"
U_STATKRAFT = "https://www.statkraft.com/newsroom/news-and-stories/2026/statkraft-signs-contract-with-OX2-to-optimize-large-scale-batteries-in-finland/"
U_LUXCARA = "https://www.energyglobal.com/energy-storage/27022026/luxcara-acquires-construction-ready-battery-storage-project-in-finland/"

# Columnas de precios de reserva por país (realizadas, fuente TSO)
RESERVE_COLS = {
    "Finland": {
        "zone": "FI", "price_col": "price_FI",
        "fcrn": "fingrid_FI_fcrn_price",
        "fcrd_up": "fingrid_FI_fcrd_up_price",
        "fcrd_dn": "fingrid_FI_fcrd_down_price",
        "source": "Fingrid Open Data (datasets 317 / 318 / 283, hourly market, marginal price)",
    },
    "Denmark": {
        "zone": "DK2", "price_col": "price_DK2",
        "fcrn": "energinet_DK2_fcrn_price",
        "fcrd_up": "energinet_DK2_fcrd_up_price",
        "fcrd_dn": "energinet_DK2_fcrd_down_price",
        "source": "Energi Data Service (FcrNdDK2, Nordic D-1/D-2 auction, 'Total')",
    },
}


@st.cache_data(ttl=3600, show_spinner="Loading master dataset (reserve prices)...")
def load_stack_data(country: str) -> pd.DataFrame:
    cfg = RESERVE_COLS[country]
    cols = [cfg["price_col"], cfg["fcrn"], cfg["fcrd_up"], cfg["fcrd_dn"]]
    master = pd.read_parquet(MASTER_PATH)
    have = [c for c in cols if c in master.columns]
    return master[have].copy()


@st.cache_data(ttl=3600, show_spinner="Optimising yearly dispatch (LP, ~10 s)...")
def cached_yearly_summary(country: str, spec_key: str, mode: str) -> pd.DataFrame:
    cfg = RESERVE_COLS[country]
    data = load_stack_data(country)
    if cfg["fcrn"] not in data.columns:
        return pd.DataFrame()
    return yearly_stack_summary(
        data, BATTERY_SPECS[spec_key], price_col=cfg["price_col"],
        fcrn_col=cfg["fcrn"], fcrd_up_col=cfg["fcrd_up"], fcrd_dn_col=cfg["fcrd_dn"],
        mode=mode)


@st.cache_data(ttl=3600, show_spinner="Backtesting strategies on the test window...")
def cached_test_stack(country: str, spec_key: str, mode: str) -> dict:
    """Stack sobre el periodo de test (ene–may 2026) usando NUESTRO forecast real."""
    cfg = RESERVE_COLS[country]
    preds = load_test_predictions(cfg["zone"])
    if preds is None or preds.empty or "price_actual" not in preds.columns:
        return {}
    d = preds.copy()
    ts = "timestamp_utc" if "timestamp_utc" in d.columns else "timestamp_cet"
    d = d.set_index(pd.to_datetime(d[ts], utc=True))
    data = load_stack_data(country)
    if cfg["fcrn"] not in data.columns:
        return {}
    df = pd.DataFrame({
        "price_actual": d["price_actual"],
        "price_pred": d["price_pred"],
        "fcrn_price": data[cfg["fcrn"]].reindex(d.index),
        "fcrd_up_price": data[cfg["fcrd_up"]].reindex(d.index),
        "fcrd_dn_price": data[cfg["fcrd_dn"]].reindex(d.index),
    }, index=d.index)
    return stacking_backtest(df, BATTERY_SPECS[spec_key], mode=mode,
                             country="FI" if country == "Finland" else "DK")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    country_cfg = render_country_selector()
    country = st.session_state.get("country", "Denmark")
    st.markdown("### Battery Lab")
    st.divider()

    spec_key = st.selectbox(
        "Commercial battery", options=list(BATTERY_SPECS.keys()),
        index=list(BATTERY_SPECS.keys()).index("generic_fi_2h"),
        format_func=lambda k: BATTERY_SPECS[k].name)
    spec = BATTERY_SPECS[spec_key]

    mode = st.radio(
        "Reserve bidding strategy", options=["oracle", "causal"], index=1,
        format_func=lambda m: {"oracle": "Oracle (perfect reserve foresight)",
                               "causal": "Causal (naive lag-24h bid — realistic)"}[m],
        help="Oracle decides the reserves knowing the clearing price (theoretical "
             "ceiling). Causal bids with yesterday's price as its forecast and is "
             "only paid when its bid is below the realised marginal price, with "
             "no lookahead.")

if not MASTER_PATH.exists():
    st.error("This page needs the master table produced by run_pipeline.py.")
    st.stop()

cfg = RESERVE_COLS[country]
data = load_stack_data(country)
have_reserves = cfg["fcrn"] in data.columns and data[cfg["fcrn"]].notna().sum() > 1000

st.title("Battery Lab — Multi-market Revenue Stacking")
st.markdown(
    f"A **datasheet-based battery** ({spec.name}) operating the full Nordic "
    f"revenue stack in **{cfg['zone']}**: FCR-N + FCR-D (frequency reserves, "
    f"hourly marginal-price markets) co-optimised with **day-ahead arbitrage** "
    f"under the actual Nordic TSO LER rules. Reserve prices: realised market "
    f"outcomes from **{cfg['source']}**.")

if not have_reserves:
    st.warning(
        f"Reserve-price columns for {cfg['zone']} not found in the master "
        f"dataset. Run `python src/extract/11_energinet_reserves.py` "
        f"(Denmark) or `python src/extract/10_fingrid_balancing.py` (Finland) "
        f"first. Showing the Finnish track is recommended meanwhile.")
    st.stop()

# ── 1) Ficha técnica de la batería ───────────────────────────────────────────
st.markdown("## 1 · The battery (datasheet, not wishful thinking)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("MW / MWh", f"{spec.power_mw:.2f} / {spec.energy_mwh:.2f}",
          help=f"Power {spec.power_mw:.3f} MW, usable energy {spec.energy_mwh:.3f} MWh "
               f"({spec.duration_h:.1f} h)")
c2.metric("Round-trip efficiency", f"{spec.rte:.1%}", help="AC, 25 °C, from the datasheet where published")
c3.metric("Cycle life", f"{spec.cycle_life_efc:,} EFC")
c4.metric("CAPEX (turnkey)", f"€{spec.capex_eur_per_kwh:.0f}/kWh")
c5.metric("Cycle cost", f"€{spec.cycle_cost_eur_mwh():.1f}/MWh",
          help="Cell amortisation per MWh discharged (Xu et al.; literature "
               "range 7-17 €/MWh)")

with st.expander("Full spec, sources, and what is estimated"):
    st.markdown(f"**{spec.name}** — {spec.notes}")
    st.markdown("**Sources:**")
    for s in spec.sources:
        st.markdown(f"- {s}")
    if spec.estimated_fields:
        st.warning(
            "**Estimated fields (manufacturer does not publish them):** "
            + ", ".join(f"`{f}`" for f in spec.estimated_fields)
            + ". Estimates use literature ranges (Preger et al. 2020 for LFP "
              "cycle life; BNEF 2025 for CAPEX; NREL for O&M) and are flagged "
              "so a reviewer can re-parametrise them.")
    st.dataframe(pd.DataFrame(list_specs()).set_index("key"), use_container_width=True)

# ── 2) La historia 2023 → 2026 ───────────────────────────────────────────────
st.markdown("## 2 · What this market actually paid, year by year")
yearly = cached_yearly_summary(country, spec_key, "oracle")
if yearly.empty:
    st.info("Not enough overlapping price/reserve data for the yearly view.")
else:
    fig = go.Figure()
    fig.add_bar(name="FCR-N capacity", x=yearly.index, y=yearly["rev_fcrn_per_mw"] / 1000,
                marker_color="#2E86AB")
    fig.add_bar(name="FCR-D capacity (up+down)", x=yearly.index, y=yearly["rev_fcrd_per_mw"] / 1000,
                marker_color="#6FB1D9")
    fig.add_bar(name="Day-ahead arbitrage", x=yearly.index, y=yearly["rev_da_per_mw"] / 1000,
                marker_color="#F18F01")
    fig.add_scatter(name="Net (after degradation, aux, O&M, grid fees)",
                    x=yearly.index, y=yearly["net_per_mw"] / 1000,
                    mode="lines+markers", line=dict(color="#1B2A4A", width=3))
    fig.update_xaxes(type="category", title_text=None)
    fig.update_layout(barmode="stack", height=420,
                      yaxis_title="€k per MW per year",
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if country == "Finland":
        _res = yearly["rev_fcrn_per_mw"] + yearly["rev_fcrd_per_mw"]
        _drop = _res.loc[2023] / _res.loc[2026] if _res.loc[2026] > 0 else float("nan")
        st.markdown(
            "**Reality check against published figures** *(each one linked to "
            "its source)*: the optimal-dispatch ceiling above can be set "
            "against what was reported for Finland:\n"
            "- 2023: our oracle ≈ "
            f"**€{yearly.loc[2023, 'gross_per_mw']/1000:,.0f}k/MW** gross vs "
            "**€40.7k/MW/month (≈€488k/MW/yr annualised)** reported for "
            "Exilion's 6 MW asset in H2-2023 "
            f"([Capalo AI]({U_CAPALO_FI})). The FCR gold rush.\n"
            "- 2025: ≈ "
            f"**€{yearly.loc[2025, 'gross_per_mw']/1000:,.0f}k/MW** vs the ~€400k/MW/yr "
            f"modelled by the [Clean Horizon Storage Index]({U_CH}) (which adds "
            "aFRR/mFRR energy, not modelled here) and the fall in the FCR-N "
            "hourly price from 46 to 19 €/MW,h reported in the same update.\n"
            "- 2025 monthly cross-check: our FCR+DA ceiling ≈ "
            f"**€{yearly.loc[2025, 'gross_per_mw']/12000:,.1f}k/MW/month** vs the "
            "customer earnings of **€27.6k (Apr) / €21.1k (May) / €18.9k (Jun) "
            "per MW in 2025** that "
            f"[Capalo AI reports for its Zeus VPP]({U_CAPALO_Q2}), trading the "
            "*full* stack (+aFRR/mFRR/intraday). Our ceiling covers fewer "
            "markets, so it is expected to sit below those figures.\n"
            "- 2026 YTD: ≈ "
            f"**€{yearly.loc[2026, 'gross_per_mw']/1000:,.0f}k/MW**, to compare with the "
            f"[European BESS Index]({U_EBI}) range for Finland (€70–130k/MW/yr). "
            "Fingrid procured no FCR-N from its yearly market for 2026 "
            f"([Fingrid, 3 Nov 2025]({U_FG_FCR})), and storage capacity in "
            f"Finland passed 1,000 MW in 2026 ([Fingrid]({U_FG_1GW})).\n\n"
            "*In this ceiling, reserve revenue per MW in 2026 is about "
            f"{_drop:.0f}× lower than in 2023: the investment case now depends on "
            "arbitrage quality (i.e., on price forecasting) and on aFRR/mFRR. "
            "The figures from 2023 no longer apply.*")

    if country == "Denmark":
        st.markdown(
            "**Reality check (DK2):** the 2023 column is dominated by **FCR-D**, "
            "whose DK2 prices were extreme (avg ~70 €/MW,h for FCR-D down in our "
            "data) because the local market was tiny and scarce. Nordic "
            "harmonisation and battery entry collapsed it (FCR-D down avg ~3 "
            "€/MW,h in 2026 YTD). **Caveat:** DK2 procures only tens of MW "
            "locally, so the price-taker assumption is weakest here — read the "
            "2023 bar as the marginal-MW ceiling, not a fleet-scalable revenue. "
            "DK1 is a different system (Continental FCR via regelleistung.net "
            "with Germany — roadmap item; not shown here).")

    da_only = cached_yearly_summary(country, spec_key, "da_only")
    if not da_only.empty:
        merged = pd.DataFrame({
            "Full stack (gross €k/MW)": (yearly["gross_per_mw"] / 1000).round(1),
            "Day-ahead only (gross €k/MW)": (da_only["gross_per_mw"] / 1000).round(1),
        })
        merged["Stacking uplift"] = (merged.iloc[:, 0] / merged.iloc[:, 1]).round(1).astype(str) + "×"
        st.markdown("**Why single-market backtests mislead** — the stack vs day-ahead-only:")
        st.dataframe(merged, use_container_width=True)

# ── 3) Estrategia realista (causal) en el periodo de test, con NUESTRO forecast ──
st.markdown("## 3 · Test window (Jan–May 2026): our forecast inside the stack")
st.markdown(
    "Decisions follow the real information sequence: day-ahead bids use **our "
    "model's forecast** (12:00 D-1 gate); FCR bids close later (18:30 D-1) "
    "and are decided " +
    ("with perfect knowledge of clearing prices (**oracle ceiling**)."
     if mode == "oracle" else
     "with a naive lag-24h reserve-price forecast; acceptance is simulated "
     "against the realised marginal price (**causal — no crystal ball**)."))

res = cached_test_stack(country, spec_key, mode)
res_oracle = cached_test_stack(country, spec_key, "oracle")
if not res:
    st.info("Test predictions not available for this zone.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross revenue (annualised)", f"€{res['per_mw']['gross_annual']/1000:,.0f}k/MW",
              help="FCR-N + FCR-D + day-ahead arbitrage, settled at realised prices")
    c2.metric("Net (all costs)", f"€{res['per_mw']['net_annual']/1000:,.0f}k/MW",
              help="After degradation, regulation losses, auxiliaries, O&M, "
                   "insurance and grid fee")
    if res_oracle and res_oracle.get("gross_annual"):
        c3.metric("vs oracle ceiling", f"{res['gross_annual']/res_oracle['gross_annual']:.0%}",
                  help="Share of the theoretical ceiling captured by the strategy without lookahead")
    c4.metric("DA cycles/day", f"{res['cycles_per_day']:.2f}",
              help=f"Typical warranty limit: {spec.warranty_cycles_per_day}/day")

    breakdown = pd.DataFrame({
        "Revenue stream": ["FCR-N capacity", "FCR-D capacity", "Day-ahead arbitrage"],
        "€k/MW/yr": [res["per_mw"]["rev_fcrn_annual"] / 1000,
                     res["per_mw"]["rev_fcrd_annual"] / 1000,
                     res["per_mw"]["rev_da_annual"] / 1000],
    })
    cost_rows = pd.DataFrame({
        "Cost": ["Degradation", "FCR-N regulation losses", "Auxiliaries (HVAC/standby)",
                 "O&M + insurance + grid fee (fixed)"],
        "€k/yr (asset)": [res["deg_cost_annual"] / 1000, res["reg_losses_annual"] / 1000,
                          res["aux_cost_annual"] / 1000, res["fixed_costs_annual"] / 1000],
    })
    cc1, cc2 = st.columns(2)
    with cc1:
        st.dataframe(breakdown.round(1), use_container_width=True, hide_index=True)
    with cc2:
        st.dataframe(cost_rows.round(1), use_container_width=True, hide_index=True)

    # P&L simple
    capex = spec.total_capex_eur()
    net = res["net_annual"]
    payback = capex / net if net > 0 else np.inf
    st.markdown(
        f"**P&L at 2026 revenue levels:** CAPEX €{capex:,.0f} "
        f"({spec.energy_mwh:.1f} MWh × €{spec.capex_eur_per_kwh:.0f}/kWh) · "
        f"net €{net:,.0f}/yr → **payback "
        + (f"{payback:.1f} years**" if np.isfinite(payback) else "— (negative net)**")
        + (". At 2026 reserve prices, FCR+DA alone gives a much longer payback "
           "than in 2023, which matches the market's shift to aFRR/mFRR stacking, "
           "tolling contracts and revenue floors "
           f"([Statkraft–OX2]({U_STATKRAFT}), [Luxcara]({U_LUXCARA}))."
           if country == "Finland" else "."))

# ── 4) Metodología y límites ─────────────────────────────────────────────────
with st.expander("Methodology, physical rules, and limitations"):
    st.markdown(f"""
**Daily LP co-optimisation** (per local day, `scipy.linprog`/HiGHS). Variables
per hour: charge `c`, discharge `d`, FCR-N `rn`, FCR-D up `ru`, FCR-D down `rd`.

- **Objective:** maximise expected revenue = DA arbitrage (decided on the
  forecast, settled at the realised price) + reserve capacity payments −
  marginal degradation cost (€{spec.cycle_cost_eur_mwh():.1f}/MWh discharged) −
  FCR-N regulation cost.
- **Nordic LER rules** (Nordic FCR technical requirements, 14.5.2025):
  FCR-N needs **±{FCRN_POWER_FACTOR}×C power** (the 0.34 is the SOC-management
  power reservation) and **{FCRN_ENERGY_H:.0f} h × C energy headroom in each
  direction**; FCR-D needs {FCRD_OPP_POWER_FACTOR}×C opposite power and
  ⅓ h × C energy. Same MW never sold twice; reservations are summed.
- **Information sequence:** DA gate 12:00 D-1 (model forecast) → DA prices
  known ~13:00 → FCR hourly gate 18:30 D-1 (reserve prices NOT yet known —
  the causal mode bids with a lag-24h naive forecast and only earns the
  realised marginal price when its bid clears).
- **FCR-N regulation throughput:** estimated at {FCRN_THROUGHPUT_PER_MWH} MWh
  per MW·h sold per direction (documented estimate, not TSO data); it adds
  efficiency losses and wear. FCR-N activation energy is in reality
  compensated at max(spot, imbalance) — we conservatively ignore that revenue.

**What is NOT modeled (and why that matters):** aFRR/mFRR capacity & energy
(the growth market post-FCR-saturation — their realised prices ARE in the
master dataset for analysis), intraday re-optimisation, 15-min MTU products
(post Oct-2025 the day-ahead is 15-min; we aggregate hourly), imbalance/passive
balancing, and portfolio effects. **The stack shown is therefore a
conservative floor for 2025-26** — and still shows the FCR collapse clearly.

**Price-taker assumption:** a 1-5 MW battery does not move these markets, but
the *fleet* does — that cannibalisation is precisely the 2023→2026 story shown
above, and it is endogenous in the realised prices we settle against.
""")

st.caption(
    "Reserve prices: realised hourly marginal prices from the TSO "
    f"({cfg['source']}). Battery specs: vendor datasheets with estimated "
    "fields flagged. All revenue figures are settled against realised prices "
    "— no simulated or mock numbers anywhere on this page.")
