"""
6_Transparency.py — Página 7: Forecast Transparency Scoreboard
==============================================================
Las cifras de precisión de los forecasts day-ahead nórdicos rara vez se
publican de forma comprobable. Esta página publica las nuestras — en vivo, contra
baselines de referencia, con test de significancia y el contexto de volatilidad
sin el cual un MAE no significa nada.

TODO se calcula en vivo desde artefactos reales del pipeline (predicciones de
test servidas, walk-forward mensual, precios del master). Nada hardcodeado.
"""

import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    render_country_selector, load_test_predictions, load_lear_forecast,
    load_model_summary, ZONE_LABELS,
)
from src.model.dm import diebold_mariano


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 24) -> tuple[float, float]:
    """Diebold-Mariano (HAC h=24 + HLN) de src/model/dm.py. DM < 0 → modelo 1 mejor."""
    return diebold_mariano(e1, e2, loss="mae", h=h, min_n=50)


def _fmt_p(p: float):
    """p-valor legible: '< 1e-6' en vez de 0.00e+00."""
    if p < 1e-6:
        return "< 1e-6"
    return f"{p:.2e}" if p < 0.01 else round(p, 3)



st.set_page_config(page_title="Transparency · Nordic Forecast",
                   layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { color: #1B2A4A; }
</style>
""", unsafe_allow_html=True)

MASTER_PATH = ROOT / "data" / "master" / "dataset_master.parquet"
PRICE_COL = {"DK1": "price_DK1", "DK2": "price_DK2", "FI": "price_FI"}


@st.cache_data(ttl=3600, show_spinner="Computing live scoreboard...")
def build_scoreboard(zone: str) -> dict:
    """Métricas en vivo del test: modelo vs LEAR vs persistencia, regímenes,
    cobertura de bandas y DM tests. Sin números precalculados."""
    preds = load_test_predictions(zone)
    if preds is None or preds.empty or "price_actual" not in preds.columns:
        return {}
    d = preds.copy()
    ts = "timestamp_utc" if "timestamp_utc" in d.columns else "timestamp_cet"
    d = d.set_index(pd.to_datetime(d[ts], utc=True)).sort_index()

    # Persistencia (naive-24h) calculada del propio precio real
    d["price_naive"] = d["price_actual"].shift(24)
    lear = load_lear_forecast(zone)
    if lear is not None and not lear.empty:
        d["price_lear"] = lear.reindex(d.index)

    base = d.dropna(subset=["price_actual", "price_pred", "price_naive"])
    out = {"n": len(base),
           "start": str(base.index.min())[:10], "end": str(base.index.max())[:10],
           "mean_price": float(base["price_actual"].mean()),
           "std_price": float(base["price_actual"].std())}

    e_model = (base["price_pred"] - base["price_actual"]).values
    e_naive = (base["price_naive"] - base["price_actual"]).values
    out["mae_model"] = float(np.abs(e_model).mean())
    out["mae_naive"] = float(np.abs(e_naive).mean())
    out["rmae"] = out["mae_model"] / out["mae_naive"]
    out["dm_vs_naive"] = dm_test(e_model, e_naive)

    if "price_lear" in base.columns and base["price_lear"].notna().sum() > 100:
        bl = base.dropna(subset=["price_lear"])
        e_lear = (bl["price_lear"] - bl["price_actual"]).values
        e_m2 = (bl["price_pred"] - bl["price_actual"]).values
        out["mae_lear"] = float(np.abs(e_lear).mean())
        out["dm_vs_lear"] = dm_test(e_m2, e_lear)

    # Regímenes (sobre el test, calculados del precio real)
    regimes = {
        "Spike (>150 €/MWh)": base["price_actual"] > 150,
        "Negative (<0 €/MWh)": base["price_actual"] < 0,
        "Normal": (base["price_actual"] >= 0) & (base["price_actual"] <= 150),
    }
    reg_rows = []
    for name, mask in regimes.items():
        sub = base[mask]
        if len(sub) < 24:
            continue
        row = {"Regime": name, "hours": int(len(sub)),
               "Model MAE": float((sub["price_pred"] - sub["price_actual"]).abs().mean()),
               "Naive MAE": float((sub["price_naive"] - sub["price_actual"]).abs().mean())}
        if "price_lear" in sub.columns and sub["price_lear"].notna().sum() > 12:
            row["LEAR MAE"] = float((sub["price_lear"] - sub["price_actual"]).abs().mean())
        reg_rows.append(row)
    out["regimes"] = reg_rows

    # Cobertura empírica de bandas en el test (si están las columnas)
    if {"P10", "P90"}.issubset(base.columns):
        cov = ((base["price_actual"] >= base["P10"]) &
               (base["price_actual"] <= base["P90"])).mean()
        out["coverage_p10_p90"] = float(cov)
    return out


@st.cache_data(ttl=3600)
def monthly_walkforward(zone: str) -> pd.DataFrame:
    """Walk-forward mensual del pipeline + persistencia mensual computada
    en vivo del master → rMAE mensual."""
    wf_path = ROOT / "data" / "models" / f"walkforward_{zone}.csv"
    if not wf_path.exists():
        return pd.DataFrame()
    wf = pd.read_csv(wf_path)
    wf["month"] = pd.PeriodIndex(wf["month"], freq="M")

    master = pd.read_parquet(MASTER_PATH, columns=[PRICE_COL[zone]])
    p = master[PRICE_COL[zone]].dropna()
    naive_err = (p - p.shift(24)).abs()
    naive_m = naive_err.groupby(naive_err.index.to_period("M")).mean()
    wf["MAE_naive24"] = wf["month"].map(naive_m).astype(float)
    wf["rMAE"] = wf["MAE"] / wf["MAE_naive24"]
    return wf


# ── Página ────────────────────────────────────────────────────────────────────
with st.sidebar:
    country = render_country_selector()
    st.markdown("### Transparency")
    zone = st.selectbox("Zone", options=country["modeled_zones"],
                        format_func=lambda z: ZONE_LABELS.get(z, z))

st.title("Forecast Transparency Scoreboard")
st.markdown(
    "Accuracy figures for Nordic day-ahead price forecasts are rarely "
    "published in a form a reader can check. This page publishes ours: "
    "computed **live** from "
    "the served model's out-of-sample test predictions against standard "
    "baselines, with Diebold-Mariano significance and the volatility context "
    "without which an MAE means nothing.")

if not MASTER_PATH.exists():
    st.error("This page needs the master table produced by run_pipeline.py.")
    st.stop()

sb = build_scoreboard(zone)
if not sb:
    st.info("Test predictions not available for this zone.")
    st.stop()

# 1) Marcador principal
st.markdown(f"## 1 · Test window {sb['start']} → {sb['end']} ({sb['n']:,} h)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Model MAE (€/MWh)", f"{sb['mae_model']:.2f}")
c2.metric("rMAE vs naive-24h", f"{sb['rmae']:.2f}",
          help="Model MAE ÷ MAE of the 24h persistence forecast on the SAME window, "
               "the metric Lago et al. 2021 recommend instead of MAPE")
c3.metric("Naive-24h MAE (€/MWh)", f"{sb['mae_naive']:.2f}")
if "mae_lear" in sb:
    c4.metric("LEAR MAE (€/MWh)", f"{sb['mae_lear']:.2f}",
              help="Zone-aware LEAR: fitted with each zone's own wind and temperature")
c5.metric("Mean ± σ (€/MWh)", f"{sb['mean_price']:.0f} ± {sb['std_price']:.0f}",
          help="Price mean and standard deviation on the test window. An MAE cannot be "
               "interpreted without the price level and volatility of the period")

dm_rows = []
if sb.get("dm_vs_naive") and not np.isnan(sb["dm_vs_naive"][0]):
    s, p = sb["dm_vs_naive"]
    dm_rows.append({"Comparison": "Model vs naive-24h", "DM stat (HLN)": round(s, 2),
                    "p-value": _fmt_p(p),
                    "Verdict": "model significantly better" if (s < 0 and p < 0.05)
                               else "not significant"})
if sb.get("dm_vs_lear") and not np.isnan(sb["dm_vs_lear"][0]):
    s, p = sb["dm_vs_lear"]
    dm_rows.append({"Comparison": "Model vs LEAR (zone-aware)", "DM stat (HLN)": round(s, 2),
                    "p-value": _fmt_p(p),
                    "Verdict": "model significantly better" if (s < 0 and p < 0.05)
                               else "not significant"})
if dm_rows:
    st.markdown("**Diebold-Mariano significance** (absolute-loss differential, "
                "HAC variance, Harvey-Leybourne-Newbold small-sample correction "
                "— computed live on the test window):")
    st.dataframe(pd.DataFrame(dm_rows), use_container_width=True, hide_index=True)

if "coverage_p10_p90" in sb:
    st.markdown(f"**P10–P90 band coverage on the test window:** "
                f"{sb['coverage_p10_p90']:.1%} (nominal 80%). The split-conformal "
                "calibrator is fitted on the October–December 2025 validation quarter, "
                "so this is out of sample. In Denmark it under-covers; prices there sat "
                "above the validation level, but Finnish prices rose more and its coverage "
                "held, so the cause is not established. In Finland it is close to the target. "
                "The five quantiles are sorted in each hour after the conformal step. "
                "Adaptive conformal inference with a daily update reaches about "
                "79–80% (README and `docs/decisions.md`, DT-24).")

# 2) Walk-forward mensual con contexto
st.markdown("## 2 · Monthly walk-forward of a separate LightGBM")
wf = monthly_walkforward(zone)
if not wf.empty:
    fig = go.Figure()
    x = wf["month"].astype(str)
    fig.add_bar(name="Model MAE", x=x, y=wf["MAE"], marker_color="#2E86AB")
    fig.add_bar(name="Naive-24h MAE (same month)", x=x, y=wf["MAE_naive24"],
                marker_color="#C0C7D1")
    fig.add_scatter(name="Month price σ (context)", x=x, y=wf["price_std"],
                    mode="lines+markers", line=dict(color="#F18F01", dash="dot"),
                    yaxis="y2")
    fig.update_layout(
        barmode="group", height=420,
        yaxis_title="MAE (€/MWh)",
        yaxis2=dict(title="price σ (€/MWh)", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", y=1.1), margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    mean_rmae = wf["rMAE"].mean()
    months_beating = int((wf["rMAE"] < 1).sum())
    st.markdown(
        f"Expanding-window walk-forward of a single LightGBM (not the served "
        f"model), retrained monthly. Mean monthly rMAE "
        f"**{mean_rmae:.2f}**; the model beats the naive in "
        f"**{months_beating}/{len(wf)} months**. The orange line is each "
        f"month's price volatility, shown as context for comparing MAE "
        f"across months.")
else:
    st.info("Walk-forward CSV not found for this zone.")

# 3) Regímenes
st.markdown("## 3 · Where the model earns its keep (and where it struggles)")
if sb.get("regimes"):
    reg = pd.DataFrame(sb["regimes"]).set_index("Regime").round(2)
    st.dataframe(reg, use_container_width=True)
    st.markdown(
        "Spikes dominate the error everywhere (the documented limitation); "
        "the real question is whether the model still beats the baselines "
        "*inside* each regime — that is what this table shows, computed live.")

# 4) Reglas del marcador
with st.expander("Scoreboard rules"):
    st.markdown("""
1. **Out-of-sample.** Everything on this page comes from the served model's
   predictions on the test window, which was not used to fit or choose the
   served models (train ends in September 2025; an earlier version was scored
   on it, see DT-24), or from the expanding-window walk-forward.
2. **Standard baselines.** Naive-24h is computed live from realised
   prices; LEAR is *zone-aware* (uses each zone's own wind/temperature — we
   strengthened our own baseline when an audit found otherwise, see DT-19).
3. **Errors carry context.** Every MAE is shown next to the window's mean
   price and σ; monthly skill is shown as rMAE, not raw MAE.
4. **Significance, not vibes.** Diebold-Mariano with HAC variance and HLN
   correction, computed on this page at render time.
5. **Limitations are printed**: 5-month test window (no summer), hourly
   aggregation of a now-15-min market, split-conformal bands below the 80%
   target in Denmark (daily ACI about 79-80%).
   Full registry: `docs/decisions.md` (DT-13…DT-24).
""")

st.caption(
    "All numbers on this page are recomputed from the pipeline's artifacts "
    "at render time. The window starts on 2 January, one day after the "
    "README tables, so the figures differ slightly.")
