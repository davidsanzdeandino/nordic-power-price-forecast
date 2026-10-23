"""
1_Market_Analysis.py — Página 2: Análisis de Mercado
=====================================================
Exploración interactiva de patrones históricos del mercado nórdico.

Componentes:
  - Selector zona + periodo
  - Heatmap precio hora × mes (la visualización estrella del EDA)
  - Serie temporal multi-zona
  - Distribución de precios
  - Spreads e índices de congestión
"""

import sys
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    load_price_data, render_data_source_banner, ZONE_LABELS, ZONES,
    render_country_selector,
)
from src.charts import (
    chart_heatmap_hour_month, chart_price_distribution,
    chart_spread_timeline, chart_multizone_prices,
)

st.set_page_config(page_title="Market Analysis · Nordic Forecast",
                   layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { color: #1B2A4A; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    country = render_country_selector()
    st.markdown("### Market Analysis")
    st.divider()

    selected_zone = st.selectbox(
        "Primary zone",
        options=country["zones"],
        format_func=lambda z: ZONE_LABELS[z],
    )

    _default_cmp = [z for z in country["zones"] if z != selected_zone][:2]
    compare_zones = st.multiselect(
        "Compare with zones",
        options=[z for z in ZONES if z != selected_zone],
        default=_default_cmp,
        format_func=lambda z: ZONE_LABELS[z],
    )

    all_zones = [selected_zone] + compare_zones

    st.divider()

    year_filter = st.selectbox(
        "Year", options=["All", "2023", "2024", "2025", "2026"], index=0
    )
    season_filter = st.multiselect(
        "Season",
        options=["Winter", "Spring", "Summer", "Autumn"],
        default=["Winter", "Spring", "Summer", "Autumn"],
    )

# ── Carga de datos ────────────────────────────────────────────────────────────

render_data_source_banner(selected_zone)

with st.spinner("Loading market data..."):
    df = load_price_data()

if df.empty:
    st.error("No market data available. Run the ETL pipeline first.")
    st.stop()

# Filtros temporales
if year_filter != "All":
    df = df[df.index.year == int(year_filter)]

season_month_map = {
    "Winter": [12, 1, 2], "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],  "Autumn": [9, 10, 11],
}
if season_filter:
    months_keep = []
    for s in season_filter:
        months_keep += season_month_map.get(s, [])
    df = df[df.index.month.isin(months_keep)]

# ── Header ────────────────────────────────────────────────────────────────────

st.title(f"{country['name']} — Market Analysis")
_yr_span = (f"{df.index.min().year}–{df.index.max().year}"
            if not df.empty else "")
st.markdown(
    f"Exploring historical patterns for **{ZONE_LABELS[selected_zone]}** "
    f"— {len(df):,} hourly observations"
    + (f" · {year_filter}" if year_filter != "All" else f" · {_yr_span}")
)
_market_char = {
    "Denmark": "Wind-driven, German-coupled market: high renewable penetration, frequent "
               "low/negative prices when wind is strong, congestion across Great Belt & Skagerrak.",
    "Finland": "**Europe's most volatile short-term market** and #1 in negative-price hours: "
               "driven by Olkiluoto nuclear (OL3), wind & residual load, and coupling to cheap "
               "Sweden (SE1/SE3) vs expensive Estonia (EstLink).",
}.get(country["name"])
if _market_char:
    st.caption(_market_char)

# ── KPIs rápidos ─────────────────────────────────────────────────────────────

col = f"price_{selected_zone}"
if col in df.columns:
    series = df[col].dropna()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mean price",    f"{series.mean():.1f} €/MWh",
              help="Average day-ahead price across all hours in the current filter.")
    k2.metric("Median price",  f"{series.median():.1f} €/MWh",
              help="The middle price — half the hours are cheaper, half pricier. "
                   "Less sensitive to spikes than the mean.")
    k3.metric("Std deviation", f"{series.std():.1f} €/MWh",
              help="How much prices swing around the mean. Higher = more volatile, "
                   "harder to forecast.")
    k4.metric("Min / Max",     f"{series.min():.0f} / {series.max():.0f} €/MWh",
              help="The cheapest and most expensive single hour in the period — the "
                   "full range the market reached.")
    neg_pct = (series < 0).mean() * 100
    k5.metric("Negative hours", f"{neg_pct:.1f}%",
              delta="High penetration" if neg_pct > 8 else "Normal",
              delta_color="off",
              help="Share of hours with negative prices — producers pay to offload "
                   "power when wind/solar floods the grid. A signature of high "
                   "renewable penetration in the Nordics.")

st.divider()

# ── Heatmap hora × mes ────────────────────────────────────────────────────────

st.markdown("### Average Price by Hour & Month")
st.caption(
    "Each cell shows the average day-ahead price for that hour and month. "
    "Reveals intraday and seasonal patterns at a glance."
)

fig_heatmap = chart_heatmap_hour_month(df, zone=selected_zone)
st.plotly_chart(fig_heatmap, width='stretch')

with st.expander("Key insights from this heatmap"):
    st.markdown("""
    - **Morning peak (7–9h)** and **evening peak (17–20h)** are consistently the most expensive hours
    - **Winter months (Dec–Feb)** show the highest and most volatile prices
    - **Summer nights and weekends** frequently produce the lowest prices — renewable surplus
    - **January–February** combination with evening hours: the highest-risk window for price spikes
    """)

st.divider()

# ── Serie temporal multi-zona ─────────────────────────────────────────────────

st.markdown("### Price Timeline — Multi-Zone Comparison")
st.caption(
    "Overlaying zones shows when they move together (a well-connected grid) and "
    "when they decouple (a congested interconnector forces different local prices). "
    "Switch resolution to trade detail for readability."
)

resample_opt = st.radio(
    "Resolution", ["Daily avg", "Weekly avg", "Hourly"],
    horizontal=True, index=0,
)
resample_map = {"Daily avg": "D", "Weekly avg": "W", "Hourly": "h"}

fig_multi = chart_multizone_prices(
    df, zones=all_zones,
    resample=resample_map[resample_opt],
)
st.plotly_chart(fig_multi, width='stretch')

st.divider()

# ── Distribución de precios ───────────────────────────────────────────────────

st.markdown("### Price Distribution")
st.caption(
    "The shape of the price histogram. Electricity prices are strongly "
    "**right-skewed**: most hours cluster at low-to-moderate prices, with a long "
    "tail of rare but extreme spikes — which is exactly what makes them hard to "
    "forecast and valuable to predict."
)

col_dist, col_stats = st.columns([2, 1])

with col_dist:
    fig_dist = chart_price_distribution(df, zones=all_zones)
    st.plotly_chart(fig_dist, width='stretch')

with col_stats:
    st.markdown("**Distribution statistics**")
    price_col = f"price_{selected_zone}"
    if price_col in df.columns:
        s = df[price_col].dropna()
        stats_table = pd.DataFrame({
            "Statistic": ["Mean", "Median", "Std", "P5", "P25", "P75", "P95",
                          "% Negative", "% > 100 €", "% > 200 €"],
            "Value": [
                f"{s.mean():.1f} €/MWh", f"{s.median():.1f} €/MWh",
                f"{s.std():.1f} €/MWh",
                f"{s.quantile(.05):.1f} €/MWh", f"{s.quantile(.25):.1f} €/MWh",
                f"{s.quantile(.75):.1f} €/MWh", f"{s.quantile(.95):.1f} €/MWh",
                f"{(s < 0).mean()*100:.1f}%", f"{(s > 100).mean()*100:.1f}%",
                f"{(s > 200).mean()*100:.1f}%",
            ]
        })
        st.dataframe(stats_table, hide_index=True, width='stretch')

st.divider()

# ── Spreads y congestión ──────────────────────────────────────────────────────

st.markdown("### Interconnection Spreads & Congestion")
st.caption(
    "A spread > 10 EUR/MWh between neighboring zones indicates the "
    f"interconnector is saturated (congested). {country['congestion_note']}"
)

spread_pairs_available = []
for z1, z2 in country["spread_pairs"]:
    if f"price_{z1}" in df.columns and f"price_{z2}" in df.columns:
        spread_pairs_available.append((z1, z2))

if spread_pairs_available:
    fig_spread = chart_spread_timeline(df, pairs=spread_pairs_available[:3])
    st.plotly_chart(fig_spread, width='stretch')

    # Estadísticas de congestión
    cong_cols = st.columns(len(spread_pairs_available))
    for i, (z1, z2) in enumerate(spread_pairs_available):
        c1, c2 = f"price_{z1}", f"price_{z2}"
        spread = (df[c1] - df[c2]).dropna()
        cong_pct = (spread.abs() > 10).mean() * 100
        with cong_cols[i]:
            st.metric(
                f"{z1}↔{z2} congestion",
                f"{cong_pct:.1f}% of hours",
                delta=f"Max spread: {spread.abs().max():.0f} €/MWh",
                delta_color="off",
            )
else:
    st.info("Spread data not available. Ensure integration.py has run successfully.")