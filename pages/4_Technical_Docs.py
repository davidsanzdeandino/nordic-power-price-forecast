"""
4_Technical_Docs.py — Página 5: Documentación Técnica
======================================================
Referencia técnica del pipeline: fuentes de datos, features, arquitectura del
modelo, evaluación y registro de decisiones.
"""

import sys
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    load_model_summary, load_walkforward, render_data_source_banner, ZONE_LABELS,
    render_country_selector,
)
from src.charts import chart_walkforward

st.set_page_config(page_title="Technical Docs · Nordic Forecast",
                   layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { color: #1B2A4A; }
code { background-color: #EEF2F7; padding: 2px 5px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    country = render_country_selector()
    st.markdown("### Technical Docs")
    st.divider()
    if len(country["modeled_zones"]) > 1:
        zone = st.selectbox("Zone", options=country["modeled_zones"],
                            format_func=lambda z: ZONE_LABELS[z])
    else:
        zone = country["modeled_zones"][0]
        st.caption(f"Zone: **{ZONE_LABELS[zone]}**")
    section = st.radio("Jump to section", [
        "Pipeline Overview", "Data Sources", "Feature Engineering",
        "Model Architecture", "Evaluation", "Decisions Register",
    ])

st.title("Technical Documentation")

render_data_source_banner(zone)

if country["name"] == "Finland":
    st.info(
        "**Finland track.** Same pipeline as Denmark, re-pointed to the FI bidding "
        "zone. Key differences: the dominant market coupling is **Sweden (SE3/SE4)** and "
        "**Estonia (EstLink)** — not Germany; an explicit **Olkiluoto-3 (OL3) nuclear "
        "regime flag** (commercial Apr-2023, a ~75% price drop) and FI nuclear-outage × "
        "low-wind interaction; FI residual load from Fingrid/ENTSO-E wind & load forecasts; "
        "Helsinki weather (Vaasa optional). FI is Europe's most volatile short-term market, "
        "so the achievable MAE is **higher (~21–23 €/MWh)** than Denmark's."
    )

# ── Pipeline Overview ─────────────────────────────────────────────────────────

if section == "Pipeline Overview":
    st.markdown("## Pipeline Overview")
    st.markdown("""
    The project follows a strict **ETL → Feature Engineering → Model → Dashboard** architecture.
    All steps are orchestrated by `run_pipeline.py` and can be run end-to-end with a single command.
    """)

    st.code("""
# Full pipeline (run once; ENTSO-E and Fingrid need free API keys in .env)
python run_pipeline.py                 # Runs all 29 steps (DK1, DK2, FI)
python run_pipeline.py --fast          # Same, but trains without Optuna (faster demo)
python run_pipeline.py --step 3        # Runs a single step
python run_pipeline.py --from 14       # From feature engineering onwards
python run_pipeline.py --from 14 --until 20
python run_pipeline.py --zone FI       # Zone-specific steps for FI only

streamlit run app.py                   # Launch dashboard (DK/FI toggle)
    """, language="bash")

    st.markdown("""
    ### Pipeline steps (`run_pipeline.py`)

    | Step | What it does | Script |
    |------|--------------|--------|
    | 1  | ENTSO-E day-ahead prices | `src/extract/01_entsoe_prices.py` |
    | 2  | ENTSO-E physical data: generation, load, flows, NTC | `src/extract/02_entsoe_physical.py` |
    | 3  | Open-Meteo weather (historical forecast) | `src/extract/03_openmeteo_weather.py` |
    | 4  | Energinet DK balance and settlement | `src/extract/04_energinet_dk.py` |
    | 5  | ENTSO-E REMIT outages | `src/extract/05_remit_unavailabilities.py` |
    | 6  | Gas TTF + CO2 EUA (Yahoo Finance) | `src/extract/06_commodities.py` |
    | 7  | ENTSO-E hydro reservoirs + scheduled day-ahead exchanges | `src/extract/07_entsoe_hydro_exchanges.py` |
    | 8  | Cleaning raw -> clean | `src/transform/cleaning.py` |
    | 9  | Integration -> master dataset | `src/transform/integration.py` |
    | 10 | FI coupling: SE1/EE prices + EstLink + Vaasa weather | `src/extract/08_finland_coupling.py` |
    | 11 | Fingrid FI day-ahead forecasts: load/wind/solar + OL3 | `src/extract/09_fingrid_fi.py` |
    | 12 | Fingrid FI reserve/balancing prices: FCR-N/D, aFRR, mFRR, imbalance | `src/extract/10_fingrid_balancing.py` |
    | 13 | Energinet DK reserve prices: FCR-N/D DK2, aFRR, mFRR, imbalance | `src/extract/11_energinet_reserves.py` |
    | 14 | Feature engineering -> dataset_features | `src/model/features.py` |
    | 15 | Baselines (persistence, weekly mean, LEAR) DK1 | `src/model/baselines.py` |
    | 16 | Baselines (persistence, weekly mean, LEAR) DK2 | `src/model/baselines.py` |
    | 17 | Baselines (persistence, weekly mean, LEAR) FI | `src/model/baselines.py` |
    | 18 | Training DK1 (LightGBM + Optuna + quantiles + conformal + AR + ensemble) | `src/model/train.py` |
    | 19 | Training DK2 (LightGBM + Optuna + quantiles + conformal + AR + ensemble) | `src/model/train.py` |
    | 20 | Training FI (LightGBM + Optuna + quantiles + conformal + AR + ensemble) | `src/model/train.py` |
    | 21 | Full evaluation DK1 | `src/model/evaluate.py` |
    | 22 | Full evaluation DK2 | `src/model/evaluate.py` |
    | 23 | Full evaluation FI | `src/model/evaluate.py` |
    | 24 | Day-ahead prediction check DK1 | `src/model/predict.py` |
    | 25 | Day-ahead prediction check DK2 | `src/model/predict.py` |
    | 26 | Day-ahead prediction check FI | `src/model/predict.py` |
    | 27 | Served-model metrics DK1 | `scripts/gen_served_metrics.py` |
    | 28 | Served-model metrics DK2 | `scripts/gen_served_metrics.py` |
    | 29 | Served-model metrics FI (by regime) | `scripts/evaluate_fi_regimes.py` |

    Steps 27-29 write `data/metadata/served_metrics_<zone>.json`, the figures the
    dashboard shows. They are skipped if the features dataset or the zone's
    trained model is missing.
    """)

    st.markdown("""
    ### Folder structure
    ```
    nordic-power-price-forecast/
    ├── app.py                  # Dashboard entry point (sidebar navigation)
    ├── run_pipeline.py         # End-to-end pipeline orchestrator
    ├── pages/                  # Streamlit multi-page app
    │   ├── 0_Overview.py       # Landing page (Executive Summary)
    │   ├── 1_Market_Analysis.py
    │   ├── 2_Detailed_Forecast.py
    │   ├── 3_Value_Calculator.py
    │   ├── 4_Technical_Docs.py
    │   ├── 5_Battery_Lab.py
    │   └── 6_Transparency.py
    ├── scripts/                # Experiments and served-metric generation
    ├── checks/                 # Manual checks (need a local pipeline run)
    ├── tests/                  # pytest unit tests
    ├── src/
    │   ├── extract/            # Phase 1A: Data extraction (11 scripts, all public sources)
    │   ├── transform/          # Phase 1B-C: Cleaning + Integration
    │   ├── model/              # Phase 2: Feature eng + Training + Prediction
    │   └── utils/              # Logging, metadata
    ├── data/
    │   ├── raw/                # API responses (Parquet)
    │   ├── clean/              # Cleaned data
    │   ├── master/             # Integrated dataset + features
    │   └── models/             # Serialized models (.pkl)
    └── docs/
        ├── decisions.md        # All technical decisions documented
        └── data_dict.md        # Master dataset column dictionary
    ```
    """)

# ── Data Sources ──────────────────────────────────────────────────────────────

elif section == "Data Sources":
    st.markdown("## Data Sources")

    st.markdown("""
    | Source | Data | Access | Script |
    |--------|------|--------|--------|
    | **ENTSO-E Transparency** | Day-ahead prices (DK/NO/SE/FI + **SE1/EE**), generation by type, cross-border flows, NTC, REMIT outages, **EstLink/FI-SE scheduled exchanges** | Free API token | `01_entsoe_prices.py` `02_entsoe_physical.py` `05_remit_unavailabilities.py` `08_finland_coupling.py` |
    | **Energinet DataService** | DK1/DK2 spot prices, physical balance, settlement, FCR-N/D (DK2) and aFRR/mFRR reserve prices (Danish TSO) | Open REST API (no key) | `04_energinet_dk.py` `11_energinet_reserves.py` |
    | **Fingrid Open Data** | Finnish TSO own day-ahead forecasts (load, wind, solar + Olkiluoto-3 signal) and FCR-N/D, aFRR, mFRR and imbalance prices | Free API key | `09_fingrid_fi.py` `10_fingrid_balancing.py` |
    | **Open-Meteo Historical Forecast** | Hourly weather *forecasts* at 10 Nordic locations incl. Vaasa/Olkiluoto/Pori (not ERA5 reanalysis — see note) | Free, no key | `03_openmeteo_weather.py` `08_finland_coupling.py` |
    | **Yahoo Finance** | Gas TTF (EUR/MWh) and CO2 EUA price (EUR/tCO2, via SparkChange Physical Carbon ETC `CO2.L`) | Free (`yfinance`) | `06_commodities.py` |

    ### Coverage
    - **Period:** 2023-01-01 → 2026-05-31 (3.4 years; test = recent 2026)
    - **Zones:** DK1, DK2, FI (target) + NO2, SE1, SE3, SE4, EE + DE/LU (context)
    - **Granularity:** Hourly (~29,900 rows × ~270 columns). ENTSO-E 15-min series
      (prices/forecasts from 2025) auto-aggregated to hourly.

    > **Weather note:** weather inputs come from Open-Meteo's
    > *Historical Forecast* API — the forecast that was actually available the day
    > before — not ERA5 reanalysis (observed weather). This avoids an optimistic
    > bias: in production a day-ahead model only has weather *forecasts*, never the
    > realised weather.

    ### Why this period (2023 → 2026)?
    Post-gas-crisis era (2022 excluded). 2023–2026 represents the new normal:
    high renewable penetration, frequent negative prices, recalibrated interconnector
    flows, and the 2025 migration of ENTSO-E series to 15-min resolution.
    2022 data is downloaded separately (`*_2022_crisis.parquet`) for extreme event analysis.
    """)

    st.markdown("### Weather station locations")
    location_data = {
        "Location": ["Esbjerg", "Aalborg", "Copenhagen", "Kristiansand", "Oslo", "Stockholm",
                     "Helsinki", "Vaasa", "Olkiluoto", "Pori"],
        "Zone": ["DK1", "DK1", "DK2", "NO2", "NO1", "SE3", "FI", "FI", "FI", "FI"],
        "Rationale": [
            "Main offshore wind hub (North Sea)",
            "Onshore wind — North Jutland",
            "Demand centre DK2",
            "Skagerrak interconnector (NO-DK hydro)",
            "Norwegian demand",
            "Swedish demand + SE3 nuclear",
            "Finnish demand",
            "FI onshore wind hub (Ostrobothnia coast)",
            "Olkiluoto nuclear + coastal wind",
            "West-coast wind (FI)",
        ]
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(location_data), hide_index=True, width='stretch')

# ── Feature Engineering ───────────────────────────────────────────────────────

elif section == "Feature Engineering":
    st.markdown("## Feature Engineering")

    _is_fi = country["name"] == "Finland"
    _nfeat = "~215" if _is_fi else "~166"
    _coupling_blurb = (
        "the **coupled Swedish (SE1/SE3/SE4) + Estonian (EstLink)** markets and "
        "**Fingrid** (the Finnish TSO) own day-ahead forecasts — whose residual-load "
        "feature is the single most important in the FI model"
        if _is_fi else
        "the **coupled German (DE/LU) market** (lagged price + renewable/load forecasts) "
        "— which contributed the largest accuracy gain"
    )
    st.markdown(f"""
    All features are generated in `src/model/features.py`. The dataset contains
    **{_nfeat} engineered features**. Beyond the categories below it also includes
    **commodities** (gas TTF, CO₂ EUA), **REMIT** outages, **hydro reservoirs +
    hydro-balance anomaly**, **scheduled exchanges**, and {_coupling_blurb}. Key categories:
    """)

    _mkt_why = ("Lagged spread FI–SE3 / FI–EE captures EstLink & Swedish-link saturation "
                "— a key driver of the Finnish price premium."
                if _is_fi else
                "Spread DK1–NO2 captures Skagerrak saturation — key driver of DK1 price premium.")
    _mkt_key = ("`feat_FI_SE3_spread_lag24h`, `feat_sched_net_FI_EE`, `feat_FI_EE_spread_lag24h`"
                if _is_fi else
                "`feat_spread_DK1_DK2`, `feat_congested_DK1_NO2`, `feat_ntc_utilization_DK1DK2`")
    categories = {
        "Cyclical temporal (8)": {
            "desc": "Sin/cos encoding of hour, day-of-week, month, week-of-year",
            "why":  "Preserves circular continuity (hour 23 ≈ hour 0). More compact than one-hot (8 vs 43 columns).",
            "key":  "`feat_hour_sin`, `feat_hour_cos`, `feat_dow_sin`, `feat_month_sin/cos`",
        },
        "Autoregressive lags (~20)": {
            "desc": "Price lags at 24h, 48h, 168h; rolling mean/std/max/min at 24h, 48h, 168h; price diff",
            "why":  "Electricity prices have strong autocorrelation. Lag-24h captures the same-slot pattern.",
            "key":  f"`feat_{zone}_price_lag24h`, `feat_{zone}_price_roll_mean_24h`",
        },
        "Physical domain (~10)": {
            "desc": "Residual load (demand – wind – solar), wind penetration, total wind forecast, wind change",
            "why":  ("Residual load is the primary driver of the marginal price. For FI the "
                     "**Fingrid** TSO residual load is the model's #1 feature."
                     if _is_fi else
                     "Residual load is the primary driver of the marginal price in Nordic market."),
            "key":  (f"`feat_FI_fingrid_residual_load`, `feat_FI_residual_load`, `feat_FI_wind_penetration`"
                     if _is_fi else
                     f"`feat_{zone}_residual_load`, `feat_{zone}_wind_penetration`, `feat_{zone}_wind_total_fc`"),
        },
        "Weather aggregates (~20)": {
            "desc": "Avg wind speed 100m, avg temperature, temperature², solar radiation per zone",
            "why":  "Aggregated per zone to reduce noise vs individual station columns.",
            "key":  f"`feat_wx_{zone}_wind100m_avg`, `feat_wx_{zone}_temp_avg`, `feat_wx_{zone}_temp_sq`",
        },
        "Market structure (~8)": {
            "desc": "Lagged price spreads, congestion flags, NTC utilisation ratio",
            "why":  _mkt_why,
            "key":  _mkt_key,
        },
        "Interactions (~4)": {
            "desc": "Cold × low-wind index, morning peak × weekday",
            "why":  f"Cold weather AND low wind simultaneously is the worst-case for high prices in {country['name']}.",
            "key":  f"`feat_{zone}_cold_low_wind`",
        },
    }

    for cat, info in categories.items():
        with st.expander(cat, expanded=False):
            st.markdown(f"**{info['desc']}**")
            st.markdown(f"*Why:* {info['why']}")
            st.markdown(f"*Key features:* {info['key']}")

    st.info(
        "**Anti-leakage measures**: price lags/rolling stats use `shift(24)` minimum; "
        "spreads, NTC and commodities are lagged 24h; wind/solar/load **forecasts** and "
        "weather **forecasts** use the day-ahead values available at auction time (12:00 D-1). "
        "Same-timestamp congestion flags (`is_congested_*`), which depend on the realised "
        "price spread, are **excluded** from the feature set — only their lagged version "
        "`feat_congested_*` is used."
    )

# ── Model Architecture ────────────────────────────────────────────────────────

elif section == "Model Architecture":
    st.markdown("## Model Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Primary model: LightGBM")
        st.markdown("""
        **Why LightGBM over XGBoost?**
        - Leaf-wise growth (vs level-wise) → faster convergence
        - Native categorical feature support
        - Benchmarks on Nordic EPF (2024 literature): LightGBM ≥ XGBoost

        **Why gradient boosting over LSTM?**
        - ~24,000 training rows: tabular models typically win over LSTM at this scale
        - Full interpretability via SHAP
        - Lund University (2024): XGBoost with good features beats LSTM on DK demand
        """)

        st.markdown("### Hyperparameter optimisation: Optuna")
        st.code("""
params_to_tune = {
    "learning_rate":      (0.01, 0.1, log=True),
    "num_leaves":         (31, 255),
    "max_depth":          (5, 15),
    "min_child_samples":  (20, 100),
    "subsample":          (0.6, 1.0),
    "colsample_bytree":   (0.6, 1.0),
    "reg_alpha":          (1e-3, 10, log=True),
    "reg_lambda":         (1e-3, 10, log=True),
}
# 100 trials, TPE sampler, MAE objective
        """, language="python")

    with col2:
        st.markdown("### Validation strategy")
        st.markdown("""
        ```
        ──────────────────────────────────────────────────
        Jan 2023 ───────────────── Sep 2025   (Train)
                              Oct ── Dec 2025   (Val → Optuna)
                                      Jan ── May 2026   (Test)
        ──────────────────────────────────────────────────
        ```
        **No random split** — strict temporal order to avoid data leakage.
        Final model trained on Train+Val before the out-of-sample test evaluation.
        A separate **walk-forward** loop (from mid-2024) re-tests month by month.
        """)

        st.markdown("### Probabilistic output")
        st.markdown("""
        5 quantile regression models for uncertainty bands:

        | Model | Quantile | Use |
        |-------|----------|-----|
        | P10   | 10th pct | Lower bound (80% band) |
        | P25   | 25th pct | Lower IQR (50% band) |
        | P50   | Median   | Central prediction |
        | P75   | 75th pct | Upper IQR (50% band) |
        | P90   | 90th pct | Upper bound (80% band) |

        Raw quantile bands are under-confident, so they are **recalibrated with
        Adaptive Conformal Inference (ACI, Gibbs & Candès 2021)**: the band width
        adapts online to recent errors, lifting empirical coverage from ~40% to the
        target ~80%. Verified **out-of-sample** on the test set.
        """)

    st.markdown("### How this maps to a production trading/EMS stack")
    st.caption("Battery-bidding and energy-management platforms are typically built from "
               "the same method families. This project is an open implementation of the "
               "forecasting and back-testing layer for the DK1, DK2 and FI bidding zones.")
    st.markdown("""
| This project | Commercial EMS equivalent |
|--------------|---------------------------|
| LightGBM / diverse-ensemble price forecast | *Electricity-market price forecasting (ML methods)* |
| Conformal **P10–P90** prediction bands | *Probabilistic price model* |
| LP degradation-aware battery dispatch | *Linear/quadratic optimisation · intelligent dispatch* |
| Battery-arbitrage back-test (Value Calculator) | *Back-testing of bid strategies* |
| Day-ahead bid schedule (cheap/expensive hours) | *Automated day-ahead bid decks* |
| Open-Meteo wind/temp features | *Weather-driven generation forecast inputs* |

*Method-family parallel, not a claim about any vendor's internal implementation.*
""")

# ── Evaluation ────────────────────────────────────────────────────────────────

elif section == "Evaluation":
    st.markdown("## Model Evaluation")

    model_meta = load_model_summary(zone=zone)
    df_wf      = load_walkforward(zone=zone)

    st.markdown(f"### Test set metrics — {zone} (Jan–May 2026, out-of-sample)")

    mae = model_meta.get("MAE_test", "—")
    dir_acc = model_meta.get("DirAcc_test", "—")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",          f"{mae:.2f} €/MWh" if isinstance(mae, float) else mae,
              delta="Out-of-sample, no leakage", delta_color="off",
              help="Mean Absolute Error on the unseen 2026 test set. These are real, "
                   "leakage-free results on public data — higher than the optimistic "
                   "figures often quoted in EPF papers, and intentionally so.")
    c2.metric("Direction Acc.", f"{dir_acc:.1f}%" if isinstance(dir_acc, float) else dir_acc,
              delta="Random: 50%", delta_color="off",
              help="Share of hours where the model gets the direction of the price "
                   "change right vs the previous hour. Above 50% beats a coin flip.")
    c3.metric("Features",     model_meta.get("n_features", "—"))
    c4.metric("Quantile models", len(model_meta.get("quantiles", [])))

    st.markdown("### Comparison vs baselines")
    st.caption(
        "The **measured** metrics for this model are the KPIs shown above (loaded from "
        "`data/metadata/`) and the monthly walk-forward chart below. The table here is a "
        "**reference of typical orders of magnitude in the Nord Pool EPF literature**, "
        "for context only — these are *not* this project's results."
    )
    st.markdown("""
    | Model (literature reference) | MAE (€/MWh) | RMSE | sMAPE | Dir. Acc. | R² |
    |-------|------------|------|-------|-----------|-----|
    | Gradient boosting (LightGBM/XGBoost) | 6–9 | 9–13 | 7–10% | 58–63% | 0.70–0.80 |
    | LEAR (Linear ARX) | 9–12 | 13–16 | 10–13% | 55–60% | 0.60–0.70 |
    | Weekly Mean (4w) | 13–15 | 19–22 | 14–17% | ~54% | 0.40–0.45 |
    | Persistence (24h) | 15–18 | 22–25 | 17–20% | ~50% | 0.25–0.35 |

    *Ranges from EPF literature (e.g. Lago et al., 2021) and typical Nord Pool results.
    Run `python src/model/baselines.py` and `evaluate.py` to produce this project's own
    baseline comparison on the downloaded data.*
    """)

    st.markdown("### Walk-forward validation")
    fig_wf = chart_walkforward(df_wf, zone=zone)
    if fig_wf.data:
        st.plotly_chart(fig_wf, width='stretch')

    st.markdown("### Known limitations")
    st.warning("""
    1. **Black swan events**: The model does not predict unprecedented events (grid failures, sudden fuel supply shocks).
    2. **Winter peak accuracy**: MAE is ~25% higher in Dec–Feb due to higher price volatility.
    3. **Monthly retraining required**: Market structure changes (new capacity, new interconnectors) degrade the model over time.
    4. **2022 crisis not in training**: The model may underestimate extreme spikes if a similar gas crisis recurs.
    """)

# ── Decisions Register ────────────────────────────────────────────────────────

elif section == "Decisions Register":
    st.markdown("## Decisions Register")
    st.caption("Every technical decision documented with alternatives considered and rationale.")

    decisions_path = ROOT / "docs" / "decisions.md"
    if decisions_path.exists():
        st.markdown(decisions_path.read_text(encoding="utf-8"))
    else:
        st.info("Decisions register not found at `docs/decisions.md`.")
        st.markdown("""
        Key decisions documented in `docs/decisions.md`:

        | ID | Decision | Alternative | Rationale |
        |----|----------|-------------|-----------|
        | DT-01 | 2023–2026 training period | 2020-2026 | 2022 gas crisis distorts patterns |
        | DT-02 | Hourly granularity | 15-min | ENTSO-E 15-min only from 2025 → aggregated |
        | DT-04 | Keep negative prices (flagged) | Remove outliers | Negative prices are signal, not noise |
        | DT-05 | Interpolate gaps ≤6h | Drop NaN rows | Preserve temporal continuity |
        | DT-06 | Parquet format | CSV | 5-10× compression, preserves types |
        | DT-08 | UTC everywhere | Local timezone | Avoids DST ambiguity in merge |
        | DT-11 | wind_speed_100m | wind_speed_10m | Turbine hub height ~100–150m |
        | DT-12 | Congestion threshold 10 €/MWh | 5 or 20 €/MWh | Nordic market literature standard |
        """)