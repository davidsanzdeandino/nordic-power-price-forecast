"""
0_Overview.py — Nordic Energy Price Forecast Dashboard
======================================================
Página 1: Overview / Executive Summary (página de aterrizaje; la registra
app.py con st.navigation como página por defecto)

Estructura:
  - Header con título + descripción
  - 4 KPIs (precio actual, predicción, MAE modelo, ahorro estimado)
  - Gráfico central: predicción vs real (7 días)
  - Fila inferior: feature importance (top 5) + alertas 24h
"""

import sys
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    load_predictions, load_price_data, load_feature_importance,
    load_model_summary, get_market_stats, render_data_source_banner, ZONE_LABELS,
    COUNTRIES, get_country,
)
from src.charts import (
    chart_forecast_vs_actual, chart_feature_importance, chart_price_alerts
)

# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Nordic Energy Forecast",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS mínimo para pulir el aspecto
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.9rem; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
div[data-testid="stSidebarNav"] { font-size: 0.95rem; }
h1 { color: #1B2A4A; }
h2 { color: #1B2A4A; font-size: 1.1rem; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Toggle de país (DK ↔ FI) ──────────────────────────────────────────────
    country_name = st.radio(
        "Market",
        options=list(COUNTRIES.keys()),
        format_func=lambda c: f"{COUNTRIES[c]['modeled_zones'][0][:2]}  {c}",
        horizontal=True,
        key="country",
        help="Switch between the Danish (DK1/DK2) and Finnish (FI) products. "
             "Each is a separate model and market.",
    )
    country = get_country(country_name)

    st.markdown(f"### {country_name} — Energy Forecast")
    st.caption(country["tagline"])
    st.divider()

    zone = st.selectbox(
        "Price zone",
        options=country["zones"],
        format_func=lambda z: ZONE_LABELS[z],
        index=0,
    )
    is_modeled = zone in country["modeled_zones"]
    if not is_modeled:
        st.caption("Context zone (coupling) — forecasts shown for the modeled "
                   f"zone(s): {', '.join(country['modeled_zones'])}.")

    days_to_show = st.slider("Days to display", min_value=3, max_value=30, value=7)

    st.divider()
    st.caption("Data: ENTSO-E · Energinet/Fingrid · Open-Meteo · Yahoo Finance")
    st.caption("Model: LightGBM + AR (DK) · diverse ensemble + AR (FI) | Train: 2023 → Sep-2025")
    st.caption("Test: Jan–May 2026 (out-of-sample)")

    with st.popover("How to read this dashboard"):
        st.markdown(
            "- **Market** (top): switch between Denmark "
            "(DK1/DK2) and Finland (FI) — separate models.\n"
            "- **Price zone**: the modeled zone, or a neighbour shown for coupling context.\n"
            "- **Forecast** = the model's day-ahead price prediction (€/MWh) for each hour.\n"
            "- **Bands (P10–P90)** = an 80% confidence interval — the price is expected to "
            "land inside ~4 times out of 5.\n"
            "- The **green banner** confirms you are seeing *real* pipeline results, not a demo.\n"
            "- Other pages (left): market patterns, hour-by-hour forecast, € value, technical "
            "docs, battery lab and transparency."
        )

# Zona efectiva para forecast: si es de contexto, usar la zona modelada por defecto
zone_fc = zone if is_modeled else country["default_zone"]

# ── Header ────────────────────────────────────────────────────────────────────

st.title(f"{country_name} Electricity Price Forecast")
served_model = ("diverse LightGBM ensemble (L2 + median + Huber) + AR correction"
                if zone_fc == "FI" else "LightGBM + AR correction")
st.markdown(
    f"**Day-ahead price prediction for zone {zone_fc}** — "
    f"{served_model}, trained on 2023 → Sep-2025 and tested out-of-sample on "
    f"Jan–May 2026; key drivers: {country['drivers']}."
)

# Aviso: datos reales vs modo demo
render_data_source_banner(zone_fc)

# ── Cargar datos ──────────────────────────────────────────────────────────────

with st.spinner("Loading data..."):
    df_pred    = load_predictions(zone=zone_fc, days=days_to_show)
    df_prices  = load_price_data()
    df_fi      = load_feature_importance(zone=zone_fc, top_n=5)
    model_meta = load_model_summary(zone=zone_fc)
    stats      = get_market_stats(zone=zone_fc)

# ── KPIs ──────────────────────────────────────────────────────────────────────

st.markdown("#### Market Overview")
k1, k2, k3, k4 = st.columns(4)

with k1:
    current = stats.get("current", "—")
    delta   = stats.get("delta_24h", 0)
    st.metric(
        label=f"Current price {zone_fc}",
        value=f"{current} €/MWh" if current != "—" else "—",
        delta=f"{delta:+.1f} vs 24h ago" if delta != 0 else None,
        delta_color="inverse",   # Rojo si sube (caro), verde si baja (barato)
        help="Latest day-ahead spot price for this zone, and the change vs the same "
             "hour yesterday. Green = cheaper (good for a consumer), red = pricier.",
    )

with k2:
    if not df_pred.empty and "price_pred" in df_pred.columns:
        next_hour_pred = df_pred["price_pred"].iloc[-1]
        p10 = df_pred["P10"].iloc[-1] if "P10" in df_pred.columns else None
        p90 = df_pred["P90"].iloc[-1] if "P90" in df_pred.columns else None
        band_str = f"Band: {p10:.0f}–{p90:.0f}" if p10 and p90 else None
        st.metric(
            label="Next period forecast",
            value=f"{next_hour_pred:.1f} €/MWh",
            delta=band_str,
            delta_color="off",
            help="The model's predicted price for the next hour, with its P10–P90 "
                 "confidence band (80% interval).",
        )
    else:
        st.metric("Next period forecast", "—")

with k3:
    mae = model_meta.get("MAE_test")
    # Baseline de persistencia real si está disponible en los metadatos
    mae_persist = model_meta.get("MAE_persistence")
    if isinstance(mae, (int, float)):
        delta = (f"vs persistence: {mae_persist:.1f}"
                 if isinstance(mae_persist, (int, float)) else None)
        st.metric(
            label="Model MAE (test)",
            value=f"{mae:.1f} €/MWh",
            delta=delta,
            delta_color="off",
            help="Mean Absolute Error on the out-of-sample test set (Jan–May 2026): "
                 "on average the forecast is off by this many €/MWh. Lower is better. "
                 "Beats the LEAR linear benchmark by ~36% in DK1/DK2 and ~27% in FI.",
        )
    else:
        # Sin MAE real: no inventar nada
        st.metric("Model MAE (test)", "—",
                  help="Mean Absolute Error on the test set. Shown once a real model "
                       "is trained (run the pipeline).")

with k4:
    vol = stats.get("vol_7d", "—")
    neg = stats.get("negative_pct", 0)
    st.metric(
        label="7-day volatility",
        value=f"{vol} €/MWh" if vol != "—" else "—",
        delta=f"{neg:.1f}% negative price hours" if neg > 0 else "No negative prices",
        delta_color="off",
        help="Standard deviation of the price over the last 7 days (how 'jumpy' the "
             "market is), plus the share of hours with negative prices — common in the "
             "Nordics when wind/solar floods the grid.",
    )

st.divider()

# ── Gráfico central: Predicción vs Real ──────────────────────────────────────

hc1, hc2 = st.columns([4, 1])
hc1.markdown("#### Forecast vs Actual Price")
with hc2.popover("What am I seeing?"):
    st.markdown(
        "The **line** is the model's day-ahead forecast; the **dots/line of actuals** "
        "is the price that really cleared. The **shaded area** is the P10–P90 band "
        "(80% confidence). When actuals stay inside the band and hug the forecast, "
        "the model is working. Gaps appear at price **spikes** — the hardest moments "
        "to predict."
    )

if not df_pred.empty:
    fig_main = chart_forecast_vs_actual(df_pred, zone=zone, show_bands=True)
    st.plotly_chart(fig_main, width='stretch')
else:
    st.info("No forecast data available. Run `python src/model/train.py` first.")

# ── Fila inferior: Feature Importance + Alertas ───────────────────────────────

mi1, mi2 = st.columns([4, 1])
mi1.markdown("#### Model Intelligence")
with mi2.popover("About these panels"):
    st.markdown(
        "**Feature importance** ranks which inputs the model relies on most "
        "(by total gain across all decision trees). **Price alerts** flag the next "
        "24h where the forecast crosses a threshold you set — the practical output "
        "for shifting load or trading."
    )
col_fi, col_alert = st.columns([1, 1])

with col_fi:
    st.markdown("**Top 5 features driving the forecast**")
    if not df_fi.empty:
        fig_fi = chart_feature_importance(df_fi, top_n=5)
        fig_fi.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_fi, width='stretch')
        st.caption(
            "The model is primarily driven by **residual load** (demand minus renewables), "
            "**wind forecast**, and **price lags** — consistent with the Nordic market literature."
        )
    else:
        st.info("Train the model to see feature importance.")

with col_alert:
    st.markdown("**Next 24h price alerts**")
    if not df_pred.empty:
        # Tomar las próximas 24 horas desde el último disponible
        df_next = df_pred.tail(24).copy()
        high_th = st.number_input(
            "High price alert threshold (€/MWh)",
            min_value=20, max_value=300, value=80, step=5,
            label_visibility="collapsed",
        )
        fig_alert = chart_price_alerts(df_next, high_threshold=float(high_th))
        fig_alert.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_alert, width='stretch')

        n_alerts = int((df_next["price_pred"] >= high_th).sum())
        n_cheap  = int((df_next["price_pred"] <= 5).sum())
        if n_alerts > 0:
            st.warning(f"{n_alerts} hour{'s' if n_alerts != 1 else ''} with forecast ≥ {high_th} €/MWh in the next 24h")
        if n_cheap > 0:
            st.success(f"{n_cheap} hour{'s' if n_cheap != 1 else ''} with forecast ≤ 5 €/MWh — consider shifting load")
    else:
        st.info("No forecast data available.")

st.divider()

# ── Footer con info del modelo ────────────────────────────────────────────────

with st.expander("Model information", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Country:** {country_name}")
        st.markdown(f"**Zone:** {zone_fc}")
        st.markdown(f"**Features:** {model_meta.get('n_features', '—')}")
        st.markdown(f"**Quantiles:** {', '.join(model_meta.get('quantiles', []))}")
    with c2:
        st.markdown(f"**Train end:** {model_meta.get('train_end', '—')}")
        st.markdown(f"**Test start:** {model_meta.get('test_start', '—')}")
        st.markdown(f"**Data until:** {model_meta.get('data_until', '—')}")
    with c3:
        st.markdown("**Top features:**")
        for f in model_meta.get("top5_features", []):
            st.markdown(f"- `{f}`")

st.caption("Nordic Energy Forecast · Built with LightGBM + Optuna, ENTSO-E, Energinet, "
           "Open-Meteo & Yahoo Finance data · 100% public/free sources")