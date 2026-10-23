"""
charts.py
=========
Funciones de visualización Plotly para el dashboard Streamlit.

Cada función devuelve un `plotly.graph_objects.Figure` listo
para `st.plotly_chart(fig, width='stretch')`.

IMPORTADO POR:
  0_Overview.py         → chart_forecast_vs_actual, chart_feature_importance, chart_price_alerts
  1_Market_Analysis.py  → chart_heatmap_hour_month, chart_price_distribution,
                           chart_spread_timeline, chart_multizone_prices
  2_Detailed_Forecast.py→ chart_forecast_vs_actual, chart_walkforward,
                           chart_feature_importance, chart_shap_waterfall
  3_Value_Calculator.py → chart_value_simulation, _apply_base, C
  4_Technical_Docs.py   → chart_walkforward

ESTILO: Paleta sobria profesional — fondo blanco limpio, acentos azul nórdico.
Pensado para lectura en pantalla y en capturas del dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
#  PALETA DE COLORES Y CONSTANTES  (exportada como `C`)
# ══════════════════════════════════════════════════════════════════════════════

# Diccionario C — importado por 3_Value_Calculator.py
C = {
    "dark_blue":  "#1B2A4A",
    "mid_blue":   "#2E75B6",
    "light_blue": "#58A6FF",
    "sky":        "#AED6F1",
    "green":      "#2D7D46",
    "light_green":"#82E0AA",
    "red":        "#C0392B",
    "light_red":  "#F5B7B1",
    "orange":     "#E67E22",
    "gray":       "#95A5A6",
    "light_gray": "#D5D8DC",
    "text":       "#2C3E50",
    "bg":         "#FFFFFF",
    "band_80":    "rgba(46,117,182,0.10)",
    "band_50":    "rgba(46,117,182,0.22)",
}

# Colores por zona — consistentes en todo el dashboard
ZONE_COLORS = {
    "DK1": "#2E75B6",
    "DK2": "#58A6FF",
    "NO2": "#2D7D46",
    "SE3": "#8E44AD",
    "SE4": "#BB8FCE",
    "FI":  "#C0392B",
}

# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT BASE  (exportada como `_apply_base`)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_base(fig: go.Figure, title: str = "",
                yaxis_title: str = "", height: int = 420) -> go.Figure:
    """
    Aplica layout profesional limpio a cualquier figura.
    Importada por 3_Value_Calculator.py como `_apply_base`.
    """
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(family="Inter, -apple-system, sans-serif", size=12,
                  color=C["text"]),
        title=dict(text=title, font=dict(size=14, color=C["dark_blue"]),
                   x=0, xanchor="left"),
        margin=dict(l=55, r=25, t=55, b=45),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor=C["light_gray"], zeroline=False, showgrid=True,
                   gridwidth=1),
        yaxis=dict(gridcolor=C["light_gray"], zeroline=False, showgrid=True,
                   gridwidth=1, title_text=yaxis_title,
                   title_font=dict(size=11)),
        height=height,
    )
    return fig


def _empty_figure(message: str = "No data available") -> go.Figure:
    """Devuelve una figura vacía con un mensaje centrado."""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=message, showarrow=False,
        font=dict(size=14, color=C["gray"]),
    )
    fig.update_layout(
        template="plotly_white", height=250,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  1. FORECAST VS ACTUAL  (0_Overview.py + 2_Detailed_Forecast.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_forecast_vs_actual(df: pd.DataFrame,
                             zone: str = "DK1",
                             show_bands: bool = True,
                             height: int = 420) -> go.Figure:
    """
    Gráfico central: predicción con bandas de confianza vs precio real.

    Args:
        df:          DataFrame con columnas timestamp_cet, price_pred,
                     P10, P25, P75, P90, y opcionalmente price_actual
        zone:        Zona (para el título)
        show_bands:  Mostrar bandas P10-P90 / P25-P75
        height:      Altura en px
    """
    if df.empty:
        return _empty_figure("No forecast data available")

    fig = go.Figure()

    # Eje X
    x = df["timestamp_cet"] if "timestamp_cet" in df.columns else df.index

    # ── Bandas de confianza ──
    if show_bands:
        # Banda P10-P90 (80%)
        if "P10" in df.columns and "P90" in df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([pd.Series(x), pd.Series(x)[::-1]]),
                y=pd.concat([df["P90"], df["P10"][::-1]]),
                fill="toself", fillcolor=C["band_80"],
                line=dict(color="rgba(0,0,0,0)"),
                name="80% CI (P10–P90)", hoverinfo="skip",
            ))
        # Banda P25-P75 (50%)
        if "P25" in df.columns and "P75" in df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([pd.Series(x), pd.Series(x)[::-1]]),
                y=pd.concat([df["P75"], df["P25"][::-1]]),
                fill="toself", fillcolor=C["band_50"],
                line=dict(color="rgba(0,0,0,0)"),
                name="50% CI (P25–P75)", hoverinfo="skip",
            ))

    # ── Predicción puntual ──
    if "price_pred" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["price_pred"],
            mode="lines", line=dict(color=C["orange"], width=2.5),
            name="Forecast",
            hovertemplate="%{y:.1f} €/MWh<extra>Forecast</extra>",
        ))

    # ── Precio real ──
    if "price_actual" in df.columns:
        actual = df["price_actual"].dropna()
        if not actual.empty:
            x_act = (df.loc[actual.index, "timestamp_cet"]
                     if "timestamp_cet" in df.columns else actual.index)
            fig.add_trace(go.Scatter(
                x=x_act, y=actual.values,
                mode="lines", line=dict(color=C["mid_blue"], width=2),
                name="Actual price",
                hovertemplate="%{y:.1f} €/MWh<extra>Actual</extra>",
            ))

    # ── Línea cero ──
    fig.add_hline(y=0, line_dash="dot", line_color=C["light_gray"],
                  annotation_text="0 €", annotation_position="bottom left",
                  annotation_font=dict(size=9, color=C["gray"]))

    _apply_base(fig, title=f"Forecast vs Actual — {zone}",
                yaxis_title="EUR / MWh", height=height)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  2. FEATURE IMPORTANCE  (0_Overview.py + 2_Detailed_Forecast.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_feature_importance(df_fi: pd.DataFrame,
                             zone: str = "DK1",
                             top_n: int = 15,
                             height: int = 420) -> go.Figure:
    """
    Barras horizontales de feature importance.

    Args:
        df_fi:  DataFrame con columnas 'feature', 'importance'
        zone:   Zona (título)
        top_n:  Número de features a mostrar
        height: Altura
    """
    if df_fi.empty:
        return _empty_figure("No feature importance data")

    df_fi = df_fi.sort_values("importance", ascending=True).tail(top_n)
    labels = df_fi["feature"].apply(_clean_feature_name)

    # Degradado de color por importancia
    norm = df_fi["importance"] / df_fi["importance"].max()
    colors = [f"rgba(46,117,182,{0.35 + 0.65*v:.2f})" for v in norm]

    fig = go.Figure(go.Bar(
        x=df_fi["importance"], y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}<br>Importance: %{x:,.0f}<extra></extra>",
    ))

    _apply_base(fig, title=f"Feature Importance — {zone}",
                height=height)
    fig.update_layout(
        xaxis_title="Importance (gain)",
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=180),
    )
    return fig


def _clean_feature_name(name: str) -> str:
    """feat_DK1_residual_load → Residual Load (DK1)"""
    name = str(name)
    zone = ""
    for z in ["DK1", "DK2", "NO2", "SE3", "SE4", "FI"]:
        if z in name:
            zone = z
            break
    clean = name.replace("feat_", "").replace(f"{zone}_", "")
    clean = clean.replace("wx_", "").replace("_", " ").strip().title()
    if zone:
        clean = f"{clean} ({zone})"
    return clean


# ══════════════════════════════════════════════════════════════════════════════
#  3. PRICE ALERTS  (0_Overview.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_price_alerts(df: pd.DataFrame,
                       high_threshold: float = 80.0,
                       low_threshold: float = 5.0,
                       height: int = 350) -> go.Figure:
    """
    Gráfico de las próximas horas con alertas de precio alto / bajo.

    Args:
        df:              DataFrame con timestamp_cet y price_pred
        high_threshold:  Umbral de precio alto (€/MWh)
        low_threshold:   Umbral de precio bajo (€/MWh)
        height:          Altura
    """
    if df.empty or "price_pred" not in df.columns:
        return _empty_figure("No forecast data for alerts")

    x = df["timestamp_cet"] if "timestamp_cet" in df.columns else df.index
    pred = df["price_pred"]

    fig = go.Figure()

    # Línea base
    fig.add_trace(go.Scatter(
        x=x, y=pred, mode="lines",
        line=dict(color=C["mid_blue"], width=2),
        name="Forecast",
        hovertemplate="%{y:.1f} €/MWh<extra></extra>",
    ))

    # Marcadores de spike
    mask_high = pred >= high_threshold
    if mask_high.any():
        fig.add_trace(go.Scatter(
            x=x[mask_high], y=pred[mask_high],
            mode="markers",
            marker=dict(color=C["red"], size=10, symbol="triangle-up"),
            name=f"High (≥{high_threshold:.0f}€)",
            hovertemplate="%{y:.1f} €/MWh<extra>HIGH</extra>",
        ))

    # Marcadores de low
    mask_low = pred <= low_threshold
    if mask_low.any():
        fig.add_trace(go.Scatter(
            x=x[mask_low], y=pred[mask_low],
            mode="markers",
            marker=dict(color=C["green"], size=10, symbol="triangle-down"),
            name=f"Low (≤{low_threshold:.0f}€)",
            hovertemplate="%{y:.1f} €/MWh<extra>LOW</extra>",
        ))

    # Zonas de referencia
    y_max = pred.max() * 1.15 if pred.max() > high_threshold else high_threshold * 1.3
    fig.add_hrect(y0=high_threshold, y1=y_max,
                  fillcolor="rgba(192,57,43,0.06)", line_width=0)

    y_min = min(pred.min() * 0.85, -5)
    fig.add_hrect(y0=y_min, y1=low_threshold,
                  fillcolor="rgba(45,125,70,0.06)", line_width=0)

    _apply_base(fig, title="Price Alerts — Next 24h",
                yaxis_title="EUR / MWh", height=height)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  4. HEATMAP HORA × MES  (1_Market_Analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_heatmap_hour_month(df: pd.DataFrame,
                             zone: str = "DK1",
                             height: int = 450) -> go.Figure:
    """
    Heatmap: precio medio por hora del día (Y) × mes (X).
    Revela patrones estacionales e intradiarios de un vistazo.

    Args:
        df:   DataFrame con índice datetime y columna price_{zone}
        zone: Zona de precio
    """
    col = f"price_{zone}"
    if df.empty or col not in df.columns:
        return _empty_figure(f"No price data for {zone}")

    series = df[col].dropna()

    pivot = pd.DataFrame({
        "month":  series.index.month,
        "hour":   series.index.hour,
        "price":  series.values,
    })
    pivot = pivot.pivot_table(index="hour", columns="month",
                              values="price", aggfunc="mean")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    col_labels = [month_names[m - 1] for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=col_labels,
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale=[
            [0.0, "#EBF5FB"],  [0.25, "#AED6F1"],
            [0.5, "#5DADE2"],  [0.70, "#F0B27A"],
            [0.85, "#E74C3C"], [1.0,  "#922B21"],
        ],
        colorbar=dict(title=dict(text="€/MWh", font=dict(size=10)),
                      tickfont=dict(size=9)),
        hovertemplate=(
            "Month: %{x}<br>Hour: %{y}<br>"
            "Avg price: %{z:.1f} €/MWh<extra></extra>"
        ),
    ))

    _apply_base(fig, title=f"Average Price by Hour & Month — {zone}",
                height=height)
    fig.update_layout(
        xaxis=dict(title="Month", tickfont=dict(size=11)),
        yaxis=dict(title="Hour (CET/CEST)", autorange="reversed"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  5. DISTRIBUCIÓN DE PRECIOS  (1_Market_Analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_price_distribution(df: pd.DataFrame,
                             zones: list[str] | None = None,
                             height: int = 400) -> go.Figure:
    """
    Histograma superpuesto de precios para varias zonas.

    Args:
        df:    DataFrame master con columnas price_{zone}
        zones: Lista de zonas
    """
    zones = zones or ["DK1"]
    fig = go.Figure()

    for zone in zones:
        col = f"price_{zone}"
        if col not in df.columns:
            continue
        values = df[col].dropna().values
        fig.add_trace(go.Histogram(
            x=values, name=zone, nbinsx=80,
            marker_color=ZONE_COLORS.get(zone, C["mid_blue"]),
            opacity=0.55,
            hovertemplate=f"{zone}: " + "%{x:.0f} €/MWh<br>Count: %{y}<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="dot", line_color=C["red"],
                  annotation_text="0 €", annotation_font=dict(size=9))

    _apply_base(fig, title="Price Distribution",
                yaxis_title="Frequency", height=height)
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Price (EUR/MWh)",
        bargap=0.02,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  6. SPREAD TIMELINE  (1_Market_Analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_spread_timeline(df: pd.DataFrame,
                          pairs: list[tuple[str, str]] | None = None,
                          height: int = 400) -> go.Figure:
    """
    Serie temporal de spreads entre pares de zonas con subplots.

    Args:
        df:    DataFrame con columnas price_{zone}
        pairs: Lista de tuplas [(zone_a, zone_b), ...]
    """
    pairs = pairs or [("DK1", "DK2"), ("DK1", "NO2")]
    n = len(pairs)

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"{a} – {b}" for a, b in pairs],
    )

    spread_colors = [C["mid_blue"], C["green"], C["orange"], C["red"]]

    for i, (z1, z2) in enumerate(pairs, 1):
        c1, c2 = f"price_{z1}", f"price_{z2}"
        if c1 not in df.columns or c2 not in df.columns:
            continue

        spread = (df[c1] - df[c2]).dropna()
        # Resample diario para legibilidad
        spread_d = spread.resample("D").mean().dropna()
        color = spread_colors[(i - 1) % len(spread_colors)]

        # Convertir hex a rgba para el fill
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill_color = f"rgba({r},{g},{b},0.12)"

        fig.add_trace(go.Scatter(
            x=spread_d.index, y=spread_d.values,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=fill_color,
            name=f"{z1}–{z2}",
            hovertemplate="%{y:.1f} €/MWh<extra>" + f"{z1}–{z2}" + "</extra>",
            showlegend=False,
        ), row=i, col=1)

        fig.add_hline(y=0, line_dash="dot", line_color=C["light_gray"],
                      row=i, col=1)
        fig.update_yaxes(title_text="€/MWh", title_font=dict(size=9),
                         row=i, col=1)

    _apply_base(fig, title="Interconnection Spreads (daily avg)",
                height=height)
    fig.update_layout(showlegend=False)
    # Estilizar subtítulos
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=11, color=C["dark_blue"]))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  7. MULTI-ZONE PRICES  (1_Market_Analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_multizone_prices(df: pd.DataFrame,
                           zones: list[str] | None = None,
                           resample: str = "D",
                           height: int = 400) -> go.Figure:
    """
    Serie temporal de precios para múltiples zonas con resampling.

    Args:
        df:       DataFrame master
        zones:    Zonas a mostrar
        resample: Frecuencia de resampling ('H', 'D', 'W')
    """
    zones = zones or ["DK1"]
    fig = go.Figure()

    for zone in zones:
        col = f"price_{zone}"
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if resample and resample != "H":
            series = series.resample(resample).mean().dropna()

        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines",
            line=dict(color=ZONE_COLORS.get(zone, C["mid_blue"]), width=1.5),
            name=zone,
            hovertemplate=f"{zone}: " + "%{y:.1f} €/MWh<extra></extra>",
        ))

    freq_label = {"H": "hourly", "D": "daily avg", "W": "weekly avg"}.get(
        resample, resample
    )
    _apply_base(fig, title=f"Price Timeline — {', '.join(zones)} ({freq_label})",
                yaxis_title="EUR / MWh", height=height)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  8. WALK-FORWARD VALIDATION  (2_Detailed_Forecast.py + 4_Technical_Docs.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_walkforward(df_wf: pd.DataFrame,
                      zone: str = "DK1",
                      height: int = 380) -> go.Figure:
    """
    Barras de MAE por mes con colores semáforo + línea RMSE.

    Semáforo: verde < 13, naranja 13–18, rojo > 18 EUR/MWh
    (calibrado al rango real de MAE mensual del modelo: ~9–23).

    Args:
        df_wf: DataFrame con columnas 'month', 'MAE', 'RMSE'
        zone:  Zona (título)
    """
    if df_wf.empty:
        return _empty_figure("No walk-forward data")

    # Colores semáforo según MAE
    colors = [
        C["green"] if v < 13 else C["orange"] if v < 18 else C["red"]
        for v in df_wf["MAE"]
    ]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Barras MAE
    fig.add_trace(
        go.Bar(
            x=df_wf["month"], y=df_wf["MAE"],
            name="MAE",
            marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
            text=[f"{v:.1f}" for v in df_wf["MAE"]],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="MAE: %{y:.2f} €/MWh<extra></extra>",
        ),
        secondary_y=False,
    )

    # Línea RMSE
    if "RMSE" in df_wf.columns:
        fig.add_trace(
            go.Scatter(
                x=df_wf["month"], y=df_wf["RMSE"],
                mode="lines+markers",
                line=dict(color=C["gray"], width=2, dash="dot"),
                marker=dict(size=5, color=C["gray"]),
                name="RMSE",
                hovertemplate="RMSE: %{y:.2f} €/MWh<extra></extra>",
            ),
            secondary_y=True,
        )

    _apply_base(fig, title=f"Walk-Forward Validation — {zone}",
                height=height)
    fig.update_yaxes(title_text="MAE (€/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="RMSE (€/MWh)", secondary_y=True)
    fig.update_layout(barmode="group")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  9. SHAP WATERFALL  (2_Detailed_Forecast.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_shap_waterfall(df_shap: pd.DataFrame,
                         base_value: float = 0,
                         prediction: float = 0,
                         height: int = 420) -> go.Figure:
    """
    Waterfall chart estilo SHAP: contribución de cada feature.
    Rojo = empuja precio arriba, verde = empuja precio abajo.

    Args:
        df_shap:     DataFrame con 'feature' y 'shap_value'
        base_value:  Valor base del modelo (E[f(x)])
        prediction:  Predicción final
        height:      Altura
    """
    if df_shap.empty:
        return _empty_figure("No SHAP data available")

    # Ordenar por valor absoluto (mayor contribución arriba)
    df_s = df_shap.sort_values("shap_value", key=abs, ascending=True).copy()
    labels = df_s["feature"].apply(_clean_feature_name).tolist()
    values = df_s["shap_value"].tolist()

    colors = [C["red"] if v > 0 else C["green"] for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        hovertemplate="%{y}<br>SHAP: %{x:+.2f} €/MWh<extra></extra>",
    ))

    # Anotación base → prediction
    fig.add_annotation(
        x=0.02, y=1.05, xref="paper", yref="paper",
        text=f"Base: {base_value:.1f} €/MWh  →  Prediction: {prediction:.1f} €/MWh",
        showarrow=False,
        font=dict(size=11, color=C["dark_blue"]),
    )

    _apply_base(fig, title="SHAP Feature Contributions", height=height)
    fig.update_layout(
        xaxis_title="SHAP value (€/MWh)",
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=180),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  10. VALUE SIMULATION  (3_Value_Calculator.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_value_simulation(strategies: dict[str, float],
                           title: str = "Annual Savings Comparison",
                           height: int = 380) -> go.Figure:
    """
    Barras comparativas de valor anual por estrategia.

    Args:
        strategies: Dict {nombre_estrategia: valor_anual_eur}
        title:      Título del gráfico
        height:     Altura

    Ejemplo:
        chart_value_simulation({
            "Persistence (naive)": 12_000,
            "LightGBM forecast":   28_000,
        })
    """
    if not strategies:
        return _empty_figure("No simulation data")

    names = list(strategies.keys())
    values = list(strategies.values())

    # Color: azul para el mejor, gris para el resto, rojo si negativo
    max_idx = values.index(max(values))
    colors = []
    for i, v in enumerate(values):
        if i == max_idx:
            colors.append(C["mid_blue"])
        elif v <= 0:
            colors.append(C["light_red"])
        else:
            colors.append(C["gray"])

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors,
        text=[f"€{v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=13, color=C["dark_blue"]),
        hovertemplate="%{x}<br>€%{y:,.0f}/year<extra></extra>",
    ))

    _apply_base(fig, title=title, yaxis_title="EUR / year", height=height)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES ADICIONALES  (no importadas por páginas, pero disponibles)
# ══════════════════════════════════════════════════════════════════════════════

def chart_actual_vs_pred_scatter(df: pd.DataFrame,
                                 zone: str = "DK1",
                                 height: int = 400) -> go.Figure:
    """Scatter plot: precio real (X) vs predicción (Y) con métricas."""
    if df.empty or "price_actual" not in df.columns or "price_pred" not in df.columns:
        return _empty_figure("Need price_actual and price_pred columns")

    data = df[["price_actual", "price_pred"]].dropna()
    if data.empty:
        return _empty_figure("No overlapping actual vs pred data")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=data["price_actual"], y=data["price_pred"],
        mode="markers",
        marker=dict(color=C["mid_blue"], size=4, opacity=0.4),
        name="Observations",
        hovertemplate="Actual: %{x:.1f}<br>Pred: %{y:.1f}<extra></extra>",
    ))

    vmin = min(data["price_actual"].min(), data["price_pred"].min())
    vmax = max(data["price_actual"].max(), data["price_pred"].max())
    fig.add_trace(go.Scatter(
        x=[vmin, vmax], y=[vmin, vmax],
        mode="lines", line=dict(color=C["gray"], dash="dash", width=1),
        name="Perfect forecast",
    ))

    err = data["price_actual"] - data["price_pred"]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | Bias: {err.mean():+.2f}",
        showarrow=False,
        font=dict(size=10, color=C["gray"], family="monospace"),
        bgcolor="rgba(255,255,255,0.8)", borderpad=4,
    )

    _apply_base(fig, title=f"Actual vs Forecast — {zone}", height=height)
    fig.update_layout(
        xaxis_title="Actual price (€/MWh)",
        yaxis_title="Forecast (€/MWh)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return fig


def chart_error_distribution(df: pd.DataFrame,
                             zone: str = "DK1",
                             height: int = 350) -> go.Figure:
    """Histograma de errores de predicción (actual - pred)."""
    if df.empty or "price_actual" not in df.columns or "price_pred" not in df.columns:
        return _empty_figure("Need price_actual and price_pred columns")

    errors = (df["price_actual"] - df["price_pred"]).dropna()
    if errors.empty:
        return _empty_figure("No errors to display")

    fig = go.Figure(go.Histogram(
        x=errors.values, nbinsx=60,
        marker=dict(color=C["mid_blue"], opacity=0.7),
        hovertemplate="Error: %{x:.1f} €/MWh<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color=C["orange"], line_width=2)
    mean_err = errors.mean()
    fig.add_vline(x=mean_err, line_dash="dash", line_color=C["red"],
                  annotation_text=f"Bias: {mean_err:+.2f}",
                  annotation_font=dict(size=10, color=C["red"]))

    _apply_base(fig, title=f"Error Distribution — {zone}",
                yaxis_title="Frequency", height=height)
    fig.update_layout(xaxis_title="Error (Actual − Forecast) €/MWh", bargap=0.02)
    return fig


def chart_ar_improvement(df_wf: pd.DataFrame,
                         height: int = 380) -> go.Figure:
    """Comparativa MAE con y sin corrección AR por mes."""
    needed = {"month", "MAE_raw", "MAE_corrected"}
    if df_wf.empty or not needed.issubset(df_wf.columns):
        return _empty_figure("Need columns: month, MAE_raw, MAE_corrected")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_wf["month"], y=df_wf["MAE_raw"],
        name="Without AR", marker=dict(color=C["gray"], opacity=0.6),
    ))
    fig.add_trace(go.Bar(
        x=df_wf["month"], y=df_wf["MAE_corrected"],
        name="With AR", marker=dict(color=C["green"], opacity=0.85),
    ))
    if "MAE_improvement_pct" in df_wf.columns:
        avg_imp = df_wf["MAE_improvement_pct"].mean()
        fig.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text=f"Avg improvement: {avg_imp:.1f}%",
            showarrow=False,
            font=dict(size=12, color=C["green"]),
            bgcolor="rgba(255,255,255,0.8)", borderpad=6,
        )

    _apply_base(fig, title="AR Correction Impact",
                yaxis_title="MAE (€/MWh)", height=height)
    fig.update_layout(barmode="group")
    return fig


def chart_price_sparkline(series: pd.Series,
                          height: int = 80,
                          color: str | None = None) -> go.Figure:
    """Mini sparkline sin ejes para tarjetas KPI."""
    color = color or C["mid_blue"]
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor="rgba(46,117,182,0.08)",
        hoverinfo="skip",
    ))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig