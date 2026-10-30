"""
2_Detailed_Forecast.py — Página 3: Predicción Detallada
========================================================
Predicción hora a hora con bandas de confianza, walk-forward validation,
y explicaciones SHAP interactivas.

Componentes:
  - Selector zona + fecha
  - Tabla de predicciones hora a hora con bandas P10-P90
  - Gráfico predicción detallada
  - Walk-forward MAE mensual
  - SHAP waterfall para hora seleccionada
  - Análisis de calibración
"""

import sys
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    load_predictions, load_walkforward, load_feature_importance,
    render_data_source_banner, ZONE_LABELS, render_country_selector,
)
from src.charts import (
    chart_forecast_vs_actual, chart_walkforward, chart_feature_importance,
)

st.set_page_config(page_title="Forecast · Nordic Energy",
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
    st.markdown("### Detailed Forecast")
    st.divider()

    if len(country["modeled_zones"]) > 1:
        zone = st.selectbox("Zone", options=country["modeled_zones"],
                            format_func=lambda z: ZONE_LABELS[z])
    else:
        zone = country["modeled_zones"][0]
        st.caption(f"Zone: **{ZONE_LABELS[zone]}**")

    forecast_days = st.slider("Days to forecast", 1, 14, 3)

    show_bands = st.checkbox("Show confidence bands (P10-P90)", value=True)
    show_shap  = st.checkbox("Show SHAP explanation", value=True)

    st.divider()
    st.caption("Forecast horizon: day-ahead (24h)")
    st.caption("Confidence bands: P10, P25, P75, P90")

# ── Carga datos ───────────────────────────────────────────────────────────────

render_data_source_banner(zone)

with st.spinner("Generating forecast..."):
    df_pred = load_predictions(zone=zone, days=forecast_days)
    df_wf   = load_walkforward(zone=zone)
    df_fi   = load_feature_importance(zone=zone, top_n=15)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Detailed Price Forecast")
st.markdown(
    f"Hour-by-hour predictions with confidence intervals for **{ZONE_LABELS[zone]}**"
)

# ── Gráfico principal ─────────────────────────────────────────────────────────

fa1, fa2 = st.columns([4, 1])
fa1.markdown("### Forecast vs Actual")
with fa2.popover("About the confidence bands"):
    st.markdown(
        "The solid line is the **point forecast**. The shaded area is the "
        "**P10–P90 band** — a nominal 80% prediction interval estimated by **quantile "
        "regression**, re-calibrated with **conformal prediction** and sorted in each "
        "hour. On the January–May 2026 test it held 62.7% (DK1), 60.8% (DK2) and "
        "81.5% (FI) of hours.\n\n"
        "A **wider band** means the model is less certain (typically around spikes "
        "and high-volatility hours). Toggle the band in the sidebar."
    )

if not df_pred.empty:
    fig = chart_forecast_vs_actual(df_pred, zone=zone, show_bands=show_bands)
    st.plotly_chart(fig, width='stretch')
else:
    st.warning("No forecast data. Train the model with `python src/model/train.py`.")

st.divider()

# ── Tabla de predicciones ─────────────────────────────────────────────────────

st.markdown("### Hourly Forecast Table")
st.caption(
    "Hour-by-hour numbers behind the chart. Rows are **shaded red** when the "
    "forecast is expensive (≥ 100 €/MWh) and **green** when very cheap (≤ 5 €/MWh). "
    "Where the actual price is known, the **Error** column shows forecast − actual."
)

if not df_pred.empty:
    display_cols = ["timestamp_cet", "price_pred"]
    if show_bands:
        for c in ["P10", "P25", "P75", "P90"]:
            if c in df_pred.columns:
                display_cols.append(c)
    if "price_actual" in df_pred.columns:
        display_cols.append("price_actual")
        df_pred["error"] = (df_pred["price_pred"] - df_pred["price_actual"]).round(2)
        display_cols.append("error")

    df_display = df_pred[display_cols].copy()

    # Formatear timestamp para display
    if "timestamp_cet" in df_display.columns:
        df_display["timestamp_cet"] = pd.to_datetime(
            df_display["timestamp_cet"]
        ).dt.strftime("%Y-%m-%d %H:%M")

    # Renombrar columnas
    rename_map = {
        "timestamp_cet": "Hour (CET/CEST)",
        "price_pred":    "Forecast (€/MWh)",
        "price_actual":  "Actual (€/MWh)",
        "error":         "Error (€/MWh)",
        "P10": "P10 (€/MWh)", "P25": "P25 (€/MWh)",
        "P75": "P75 (€/MWh)", "P90": "P90 (€/MWh)",
    }
    df_display = df_display.rename(columns=rename_map)

    # Colorear filas por nivel de precio predicho
    def color_forecast(val):
        try:
            v = float(val)
            if v >= 100:  return "background-color: #FADBD8"
            if v <= 5:    return "background-color: #D5F5E3"
            return ""
        except: return ""

    styled = (df_display.style
              .map(color_forecast, subset=["Forecast (€/MWh)"])
              .format({c: "{:.1f}" for c in df_display.columns
                       if "€" in c or "Error" in c}, na_rep="—"))

    st.dataframe(styled, width='stretch', height=320)

    # Resumen rápido
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg forecast",  f"{df_pred['price_pred'].mean():.1f} €/MWh")
    c2.metric("Min forecast",  f"{df_pred['price_pred'].min():.1f} €/MWh")
    c3.metric("Max forecast",  f"{df_pred['price_pred'].max():.1f} €/MWh")
    if "price_actual" in df_pred.columns:
        mae = (df_pred["price_pred"] - df_pred["price_actual"]).abs().mean()
        c4.metric("MAE (this period)", f"{mae:.2f} €/MWh")
    else:
        c4.metric("Band width (P10-P90)",
                  f"±{((df_pred.get('P90', df_pred['price_pred']) - df_pred.get('P10', df_pred['price_pred'])) / 2).mean():.1f} €/MWh"
                  if "P90" in df_pred.columns else "—")

st.divider()

# ── SHAP Explanation ──────────────────────────────────────────────────────────

if show_shap:
    st.markdown("### Why does the model predict this price?")
    st.caption(
        "SHAP values decompose each prediction into the contribution of each feature. "
        "Red bars push the price UP, green bars push it DOWN."
    )

    if not df_pred.empty and "timestamp_cet" in df_pred.columns:
        ts_options = pd.to_datetime(df_pred["timestamp_cet"]).dt.strftime("%Y-%m-%d %H:%M")
        selected_ts = st.selectbox("Select hour to explain", ts_options.tolist(),
                                    index=min(12, len(ts_options) - 1))

        try:
            from src.model.predict import PricePredictor
            from src.charts import chart_shap_waterfall
            from pathlib import Path as _Path
            import pickle

            model_path = ROOT / "data" / "models" / f"lightgbm_{zone}.pkl"
            if model_path.exists():
                predictor = PricePredictor(zone=zone)
                ts_utc = pd.Timestamp(selected_ts,
                                       tz="Europe/Copenhagen").tz_convert("UTC")
                df_shap = predictor.explain_prediction(ts_utc.isoformat(), top_n=10)

                # Obtener base value y predicción
                pred_row = df_pred[
                    pd.to_datetime(df_pred["timestamp_cet"]).dt.strftime("%Y-%m-%d %H:%M")
                    == selected_ts
                ]
                pred_val   = float(pred_row["price_pred"].iloc[0]) if not pred_row.empty else 0
                base_value = pred_val - df_shap["shap_value"].sum()

                fig_shap = chart_shap_waterfall(df_shap, base_value, pred_val)
                st.plotly_chart(fig_shap, width='stretch')

                # Narrativa en lenguaje natural
                top_up   = df_shap[df_shap["shap_value"] > 0].head(2)
                top_down = df_shap[df_shap["shap_value"] < 0].head(1)

                narrative_parts = [
                    f"The model predicts **{pred_val:.1f} EUR/MWh** for {selected_ts} "
                    f"(base: {base_value:.1f} EUR/MWh)."
                ]
                for _, row in top_up.iterrows():
                    feat = row["feature"].replace("feat_", "").replace("_", " ").title()
                    narrative_parts.append(
                        f"**{feat}** pushes the price **+{row['shap_value']:.1f} EUR/MWh**"
                    )
                for _, row in top_down.iterrows():
                    feat = row["feature"].replace("feat_", "").replace("_", " ").title()
                    narrative_parts.append(
                        f"**{feat}** reduces the price **{row['shap_value']:.1f} EUR/MWh**"
                    )
                st.info(" · ".join(narrative_parts))
            else:
                st.info("Train the model to see SHAP explanations.")
        except Exception as e:
            st.info(f"SHAP explanation not available: {e}")

    st.divider()

# ── Walk-forward validation ───────────────────────────────────────────────────

wf1, wf2 = st.columns([4, 1])
wf1.markdown("### Walk-Forward Validation — Monthly Performance")
with wf2.popover("What is walk-forward?"):
    st.markdown(
        "**Walk-forward validation** is the standard way to test a time-series model: "
        "for each month, the model is trained **only on data up to the previous "
        "month** and then tested on that month — never on the future. It mimics how "
        "the model would have performed in production, month after month.\n\n"
        "The dotted line is **RMSE**, which penalises large misses more than MAE, so "
        "it spikes in volatile months."
    )
st.caption(
    "Each bar is the model MAE for that month (lower = better). "
    "Green = MAE < 13 | Orange = 13–18 | Red = > 18 EUR/MWh — bands calibrated to "
    "this model's real monthly range (~9–23)."
)

fig_wf = chart_walkforward(df_wf, zone=zone)
if fig_wf.data:
    st.plotly_chart(fig_wf, width='stretch')

    if not df_wf.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Average MAE",  f"{df_wf['MAE'].mean():.2f} €/MWh")
        c2.metric("Best month",
                  f"{df_wf.loc[df_wf['MAE'].idxmin(), 'month']} "
                  f"({df_wf['MAE'].min():.2f} €)")
        c3.metric("Worst month",
                  f"{df_wf.loc[df_wf['MAE'].idxmax(), 'month']} "
                  f"({df_wf['MAE'].max():.2f} €)")

st.divider()

# ── Feature importance global ─────────────────────────────────────────────────

st.markdown("### Global Feature Importance")

if not df_fi.empty:
    col_fi, col_desc = st.columns([3, 2])
    with col_fi:
        fig_fi = chart_feature_importance(df_fi, top_n=15)
        st.plotly_chart(fig_fi, width='stretch')
    with col_desc:
        st.markdown("**Feature interpretation:**")
        if country["name"] == "Finland":
            st.markdown("""
| Feature | Why it matters |
|---------|---------------|
| `fingrid_residual_load` | Finnish TSO residual load — the model's **#1** driver |
| `FI_SE3_spread_lag24h` | Coupling to the cheaper Swedish (SE3) market |
| `price_lag24h` | Yesterday's same hour — strong autocorrelation |
| `wx_FI_temp_avg` | Temperature → heating demand (volatile FI winters) |
| `sched_net_FI_EE` | EstLink export balance to Estonia |
| `post_ol3` | Olkiluoto-3 nuclear regime (2023 structural break) |
            """)
        else:
            st.markdown("""
| Feature | Why it matters |
|---------|---------------|
| `residual_load` | Demand minus renewables — drives marginal price |
| `wind_total_fc` | DA wind forecast — most important renewable in DK |
| `price_lag_24h` | Yesterday's same hour — strong autocorrelation |
| `temp_avg` | Temperature → heating demand |
| `hour_sin/cos` | Intraday price cycle |
| `spread_DK1_NO2` | Congestion on Skagerrak (hydro link) |
            """)
        if not df_fi.empty:
            top5_pct = df_fi.head(5)["importance_pct"].sum()
            st.metric("Top 5 features explain", f"{top5_pct:.1f}% of model")