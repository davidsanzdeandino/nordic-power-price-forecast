"""
data_loader.py
==============
Funciones de carga de datos con caché para el dashboard Streamlit.

Todas las funciones usan @st.cache_data o @st.cache_resource para que
los datos solo se carguen una vez por sesión — crítico para rendimiento.

TTL configurado por tipo de dato:
  - Dataset histórico:  TTL=3600s (1h) — cambia poco
  - Predicciones:       TTL=3600s (1h) — se regeneran cada hora
  - Modelo:             sin TTL  (cache_resource) — vive toda la sesión

Cambios respecto a versión anterior:
  - Corregido manejo de timezone en load_predictions (join UTC/CET robusto)
  - Añadido load_ar_walkforward() para datos de corrección AR
  - Mock data alineada con formato de salida real de predict.py
  - Validación de columnas antes de devolver DataFrames
  - get_market_stats() ahora devuelve también week_avg y month_avg
"""

import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Rutas ─────────────────────────────────────────────────────────────────────

MASTER_PATH   = ROOT / "data" / "master" / "dataset_master.parquet"
FEATURES_PATH = ROOT / "data" / "master" / "dataset_features.parquet"
MODELS_DIR    = ROOT / "data" / "models"
WF_DIR        = MODELS_DIR

DISPLAY_TZ = "Europe/Copenhagen"

# Zonas disponibles (alineado con integration.py y features.py)
ZONES = ["DK1", "DK2", "NO2", "SE3", "SE4", "FI"]
ZONE_LABELS = {
    "DK1": "DK1 — West Denmark",
    "DK2": "DK2 — East Denmark",
    "NO2": "NO2 — South Norway",
    "SE3": "SE3 — Central Sweden",
    "SE4": "SE4 — South Sweden",
    "FI":  "FI  — Finland",
}

# ── Países (toggle DK ↔ FI) ───────────────────────────────────────────────────
# Cada país es un "track" separado del producto: zonas objetivo modeladas, zonas
# de contexto, copy y acoplamiento de mercado propios. DK queda intacto; FI
# cubre el mercado finlandés.
COUNTRIES = {
    "Denmark": {
        "flag": "DK",
        "modeled_zones": ["DK1", "DK2"],          # zonas con modelo entrenado
        "context_zones": ["NO2", "SE3", "SE4"],   # vecinas para contexto
        "default_zone": "DK1",
        "coupling": "the German (DE/LU) market",
        "tagline": "Day-ahead price forecasting for the Danish bidding zones "
                   "DK1/DK2, coupled to the German market.",
        "drivers": "wind generation, German-market coupling, and interconnector "
                   "congestion (Great Belt DK1↔DK2, Skagerrak to Norway)",
        "congestion_note": "The **Great Belt (DK1↔DK2)** and **Skagerrak (DK1↔NO2)** "
                           "are the most frequently congested links.",
        "spread_pairs": [("DK1", "DK2"), ("DK1", "NO2"), ("DK2", "SE4")],
    },
    "Finland": {
        "flag": "FI",
        "modeled_zones": ["FI"],
        "context_zones": ["SE3", "SE4"],          # acoplamiento sueco
        "default_zone": "FI",
        "coupling": "Sweden (SE3/SE4) and Estonia (EstLink)",
        "tagline": "Day-ahead price forecasting for the Finnish bidding zone (FI), "
                   "coupled to Sweden and Estonia.",
        "drivers": "Olkiluoto nuclear availability (OL3), wind generation & residual "
                   "load, Swedish (SE3) coupling, and EstLink exports to Estonia",
        "congestion_note": "**EstLink (FI↔EE)** to Estonia and the **FI↔SE3 / FI↔SE1** "
                           "links to Sweden set the Finnish price premium when congested.",
        "spread_pairs": [("FI", "SE3"), ("FI", "SE1"), ("FI", "EE"), ("SE3", "SE4")],
    },
}
DEFAULT_COUNTRY = "Denmark"


def get_country(country: str = None) -> dict:
    """Config del país (fallback a Denmark). Añade 'name' y la lista de zonas seleccionables."""
    name = country if country in COUNTRIES else DEFAULT_COUNTRY
    cfg = dict(COUNTRIES[name])
    cfg["name"] = name
    cfg["zones"] = cfg["modeled_zones"] + cfg["context_zones"]
    return cfg


def country_of_zone(zone: str) -> str:
    """Devuelve el país cuyo modelo cubre esta zona ('Finland' si zone=='FI', else 'Denmark')."""
    for name, cfg in COUNTRIES.items():
        if zone in cfg["modeled_zones"]:
            return name
    return DEFAULT_COUNTRY


def render_country_selector() -> dict:
    """
    Renderiza el toggle de país (DK ↔ FI) en la sidebar y devuelve su config.
    Usa key='country' → Streamlit sincroniza la selección entre todas las páginas.
    """
    name = st.radio(
        "Market",
        options=list(COUNTRIES.keys()),
        format_func=lambda c: f"{COUNTRIES[c]['flag']}  {c}",
        horizontal=True,
        key="country",
        help="Switch between the Danish (DK1/DK2) and Finnish (FI) products — "
             "separate models and markets.",
    )
    return get_country(name)


# ── Dataset maestro ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def load_master() -> pd.DataFrame:
    """
    Carga el dataset maestro y convierte el índice a hora local CET.
    Devuelve solo las columnas esenciales para el dashboard.
    """
    if not MASTER_PATH.exists():
        st.error(f"Master dataset not found: {MASTER_PATH}")
        return pd.DataFrame()

    df = pd.read_parquet(MASTER_PATH)
    df = df.sort_index()

    # Asegurar que el índice tiene timezone antes de convertir
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(DISPLAY_TZ)
    df.index.name = "timestamp_cet"
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data() -> pd.DataFrame:
    """
    Carga solo las columnas de precio + spreads + flags para gráficos rápidos.
    Subset ligero del master para que charts.py no cargue 100+ columnas.
    """
    df = load_master()
    if df.empty:
        return df

    price_cols  = [c for c in df.columns if c.startswith("price_")]
    spread_cols = [c for c in df.columns if "spread" in c or "congested" in c]
    flag_cols   = [c for c in df.columns if c.startswith("is_")]
    cal_cols    = ["hour_of_day", "day_of_week", "month", "is_weekend"]

    keep = [c for c in price_cols + spread_cols + flag_cols + cal_cols
            if c in df.columns]
    return df[keep].copy()


# ── Predicciones ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Generating forecast...")
def load_predictions(zone: str = "DK1",
                     start: str = None,
                     days: int = 7) -> pd.DataFrame:
    """
    Carga predicciones con bandas de confianza para el rango indicado.

    Args:
        zone:  Zona de precio (DK1, DK2, NO2, SE3, SE4, FI)
        start: Fecha inicio YYYY-MM-DD (None → últimos `days` días disponibles)
        days:  Número de días a incluir

    Returns:
        DataFrame con columnas: timestamp_cet, price_pred, P10, P25, P75, P90,
                                price_actual (si disponible), zone
    """
    try:
        from src.model.predict import PricePredictor
    except ImportError:
        return _mock_predictions(zone, days)

    model_path = MODELS_DIR / f"lightgbm_{zone}.pkl"
    if not model_path.exists():
        return _mock_predictions(zone, days)

    try:
        predictor = PricePredictor(zone=zone)

        if start is None:
            # Últimos `days` días disponibles en el dataset de features
            if FEATURES_PATH.exists():
                df_features = pd.read_parquet(FEATURES_PATH)
                end_ts   = df_features.index.max()
                start_ts = end_ts - pd.Timedelta(days=days - 1)
            else:
                return _mock_predictions(zone, days)
        else:
            start_ts = pd.Timestamp(start, tz="UTC")
            end_ts   = start_ts + pd.Timedelta(days=days) - pd.Timedelta(hours=1)

        df_pred = predictor.predict_with_bands(
            start=start_ts.isoformat(), end=end_ts.isoformat()
        )

        # Adjuntar precio real si disponible — join robusto UTC→UTC
        df_master = load_master()
        target_col = f"price_{zone}"
        if target_col in df_master.columns:
            # Master está en CET → convertir a UTC para el join
            master_utc = df_master[[target_col]].copy()
            master_utc.index = master_utc.index.tz_convert("UTC")

            if "timestamp_utc" in df_pred.columns:
                df_pred = df_pred.set_index("timestamp_utc")
                df_pred["price_actual"] = master_utc[target_col].reindex(
                    df_pred.index
                )
                df_pred = df_pred.reset_index()
            elif "timestamp_cet" in df_pred.columns:
                # Si predict.py ya devolvió en CET, hacer join directo
                df_pred_indexed = df_pred.set_index("timestamp_cet")
                df_pred_indexed["price_actual"] = df_master[target_col].reindex(
                    df_pred_indexed.index
                )
                df_pred = df_pred_indexed.reset_index()

        return df_pred

    except Exception as e:
        st.warning(f"Could not load the model: {e}. Showing demo data.")
        return _mock_predictions(zone, days)


def _mock_predictions(zone: str, days: int = 7) -> pd.DataFrame:
    """
    Genera predicciones simuladas realistas cuando el modelo no está disponible.
    Útil para desarrollo del dashboard antes de tener datos reales.

    Los patrones simulan:
      - Ciclo diario (pico mañana/tarde, valle noche)
      - Ciclo semanal (fin de semana más barato)
      - Volatilidad proporcional al nivel de precio
      - Bandas más anchas en horas pico (mayor incertidumbre)
    """
    np.random.seed(42)
    n_hours = days * 24
    now = pd.Timestamp("2024-11-01", tz="Europe/Copenhagen")
    idx = pd.date_range(start=now, periods=n_hours, freq="h",
                        tz="Europe/Copenhagen")

    # Precio base con ciclo diario y semanal realista
    hours = np.arange(n_hours)
    # Ciclo diario: pico a las 8-9 y 17-18, valle a las 3-4
    base  = 55 + 20 * np.sin(2 * np.pi * (hours % 24) / 24 - np.pi / 2)
    # Ciclo semanal: más barato el fin de semana
    base += 8 * np.sin(2 * np.pi * (hours % 168) / 168)
    noise = np.random.normal(0, 8, n_hours)
    price = (base + noise).clip(-10, 250)

    # Predicción con un pequeño offset (simula error del modelo)
    pred_offset = np.random.normal(0, 5, n_hours)
    pred  = (price + pred_offset).clip(-15, 260)

    # Bandas proporcionales al precio (más incertidumbre en picos)
    band_base = np.abs(np.random.normal(12, 4, n_hours))
    hour_of_day = hours % 24
    # Bandas más anchas en horas pico
    peak_factor = 1 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2).clip(0)
    band = band_base * peak_factor

    df = pd.DataFrame({
        "timestamp_cet": idx,
        "price_actual":  np.round(price, 2),
        "price_pred":    np.round(pred, 2),
        "P10":           np.round((pred - band * 1.5).clip(-15), 2),
        "P25":           np.round((pred - band * 0.8).clip(-15), 2),
        "P75":           np.round(pred + band * 0.8, 2),
        "P90":           np.round(pred + band * 1.5, 2),
        "zone":          zone,
        "generated_at":  "Demo data",
    })
    return df


# ── Walk-forward results ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_walkforward(zone: str = "DK1") -> pd.DataFrame:
    """Carga los resultados de walk-forward validation (LightGBM puro)."""
    path = WF_DIR / f"walkforward_{zone}.csv"
    if path.exists():
        return pd.read_csv(path)
    # Demo data si no existe
    return _mock_walkforward()


@st.cache_data(ttl=3600, show_spinner=False)
def load_ar_walkforward(zone: str = "DK1") -> pd.DataFrame:
    """
    Carga resultados de walk-forward con corrección AR.
    Incluye columnas MAE_raw y MAE_corrected para comparar
    el impacto del post-procesamiento AR.
    """
    path = WF_DIR / f"walkforward_ar_{zone}.csv"
    if path.exists():
        return pd.read_csv(path)
    # Demo data si no existe
    return _mock_ar_walkforward()


def _mock_walkforward() -> pd.DataFrame:
    """Mock de walk-forward estándar."""
    months = pd.date_range("2024-04", periods=9, freq="MS").strftime("%Y-%m")
    np.random.seed(42)
    mae_vals  = np.random.uniform(6.5, 12.0, 9).round(2)
    rmse_vals = mae_vals * np.random.uniform(1.3, 1.6, 9)
    return pd.DataFrame({
        "month":   months,
        "MAE":     mae_vals,
        "RMSE":    rmse_vals.round(2),
        "n_train": range(8736, 8736 + 9 * 744, 744),
    })


def _mock_ar_walkforward() -> pd.DataFrame:
    """Mock de walk-forward con corrección AR."""
    months = pd.date_range("2024-04", periods=9, freq="MS").strftime("%Y-%m")
    np.random.seed(42)
    mae_raw = np.random.uniform(7.0, 12.5, 9).round(2)
    # AR mejora típicamente 5-15%
    improvement = np.random.uniform(0.04, 0.14, 9)
    mae_corrected = (mae_raw * (1 - improvement)).round(2)
    return pd.DataFrame({
        "month":               months,
        "MAE_raw":             mae_raw,
        "MAE_corrected":       mae_corrected,
        "MAE_improvement_pct": ((mae_raw - mae_corrected) / mae_raw * 100).round(1),
        "RMSE_raw":            (mae_raw * np.random.uniform(1.3, 1.6, 9)).round(2),
        "RMSE_corrected":      (mae_corrected * np.random.uniform(1.3, 1.6, 9)).round(2),
        "AR_order":            np.random.choice([2, 3, 4], 9),
    })


# ── Feature importance ────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def load_feature_importance(zone: str = "DK1", top_n: int = 15) -> pd.DataFrame:
    """Carga la feature importance del modelo entrenado."""
    try:
        from src.model.predict import PricePredictor
        model_path = MODELS_DIR / f"lightgbm_{zone}.pkl"
        if not model_path.exists():
            return _mock_feature_importance(zone)
        predictor = PricePredictor(zone=zone)
        df_imp = predictor.get_feature_importance(top_n=top_n)
        df_imp.attrs["is_demo"] = False
        return df_imp
    except Exception as e:
        st.warning(f"Real feature importance not available ({e}). Showing an example (demo).")
        return _mock_feature_importance(zone)


def _mock_feature_importance(zone: str = "DK1") -> pd.DataFrame:
    """Feature importance demo — nombres alineados con features.py."""
    features = [
        f"feat_{zone}_residual_load",
        f"feat_{zone}_wind_total_fc",
        f"feat_{zone}_price_lag24h",
        f"feat_wx_{zone}_temp_avg",
        "feat_hour_sin",
        f"feat_{zone}_price_roll_mean_24h",
        f"feat_{zone}_price_lag168h",
        f"feat_wx_{zone}_wind100m_avg",
        f"feat_spread_{zone}_NO2" if zone != "NO2" else f"feat_spread_{zone}_DK1",
        "feat_hour_cos",
        "feat_dow_sin",
        f"feat_{zone}_wind_fc_lag24h",
        f"feat_{zone}_cold_low_wind",
        "feat_month_sin",
        f"feat_ntc_utilization_{zone}DK2" if zone != "DK2" else f"feat_ntc_utilization_{zone}DK1",
    ]
    importances = np.array([850, 720, 610, 480, 390, 350, 310, 280,
                             240, 220, 190, 175, 160, 140, 120], dtype=float)
    total = importances.sum()
    return pd.DataFrame({
        "feature":        features,
        "importance":     importances,
        "importance_pct": (importances / total * 100).round(2),
    })


# ── Modelo summary ────────────────────────────────────────────────────────────

def _load_real_metrics(zone: str) -> dict:
    """
    Lee las métricas reales de test del último metadato de entrenamiento
    (data/metadata/train_<zone>_*.json) para mostrarlas en el dashboard.
    summary() del predictor no las incluye, así que las añadimos aquí.
    Devuelve {} si no hay metadatos.
    """
    import json, glob
    meta_dir = ROOT / "data" / "metadata"
    files = sorted(glob.glob(str(meta_dir / f"train_{zone}_*.json")))
    if not files:
        return {}
    try:
        with open(files[-1], encoding="utf-8") as f:
            stats = json.load(f).get("stats", {})
    except Exception:
        return {}
    keep = ["MAE_test", "MAE_test_ar", "MAE_test_ens_ar", "RMSE_test",
            "DirAcc_test", "R2_test", "MAE_wf_mean", "coverage_80",
            "coverage_80_conformal"]
    out = {k: stats[k] for k in keep if k in stats and stats[k] is not None}
    out["production_model"] = ("Diverse ensemble (L2+median+Huber)"
                               if zone in {"FI"} else "LightGBM") + " + AR"
    # Mostrar el MAE/DirAcc del modelo REALMENTE servido por predict.py
    # (medido sobre su salida real), no el de un sub-modelo interno. Lo persiste
    # scripts/evaluate_fi_regimes.py en served_metrics_<zone>.json.
    served_path = meta_dir / f"served_metrics_{zone}.json"
    if served_path.exists():
        try:
            served = json.load(open(served_path, encoding="utf-8"))
            for k in ("MAE_test", "DirAcc_test", "MAE_vs_lear", "MAE_lear"):
                if served.get(k) is not None:
                    out[k] = served[k]
        except Exception:
            pass
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def load_model_summary(zone: str = "DK1") -> dict:
    """Metadatos del modelo para mostrar en el dashboard."""
    try:
        from src.model.predict import PricePredictor
        model_path = MODELS_DIR / f"lightgbm_{zone}.pkl"
        if not model_path.exists():
            return _mock_model_summary(zone)
        predictor = PricePredictor(zone=zone)
        summary = predictor.summary()
        summary["is_demo"] = False
        # Enriquecer con métricas reales de test/walk-forward desde metadatos
        summary.update(_load_real_metrics(zone))
        return summary
    except Exception as e:
        st.warning(f"Real model summary not available ({e}). Showing an example (demo).")
        return _mock_model_summary(zone)


def _mock_model_summary(zone: str) -> dict:
    return {
        "is_demo":       True,
        "zone":          zone,
        "n_features":    87,
        "trained_at":    "20241201_1430",
        "train_end":     "2024-06-30 23:00",
        "test_start":    "2024-10-01",
        "quantiles":     ["P10", "P25", "P50", "P75", "P90"],
        "data_from":     "2023-01-01",
        "data_until":    "2024-12-31",
        "top5_features": [
            f"feat_{zone}_residual_load",
            f"feat_{zone}_wind_total_fc",
            f"feat_{zone}_price_lag24h",
            f"feat_wx_{zone}_temp_avg",
            "feat_hour_sin",
        ],
        "MAE_test":      7.34,
        "RMSE_test":     11.2,
        "DirAcc_test":   61.2,
    }


# ── Predicciones del periodo de test completo (para simulación económica) ─────

@st.cache_data(ttl=3600, show_spinner="Loading test-period forecast...")
def load_test_predictions(zone: str = "DK1") -> pd.DataFrame:
    """
    Carga predicciones + precio real de **todo el periodo de test out-of-sample**
    (no solo los últimos días), para estimaciones económicas anuales robustas.
    """
    meta = load_model_summary(zone)
    test_start = str(meta.get("test_start", "2026-01-01"))[:10]
    data_until = str(meta.get("data_until", "2026-05-31"))[:10]
    try:
        start = pd.Timestamp(test_start)
        end   = pd.Timestamp(data_until)
        days  = max((end - start).days + 1, 7)
    except Exception:
        days = 150
    return load_predictions(zone=zone, start=test_start, days=days)


@st.cache_data(ttl=3600, show_spinner="Running battery dispatch backtest...")
def run_battery_dispatch(zone: str = "DK1",
                         capacity_mwh: float = 2.0,
                         power_mw: float = 1.0,
                         efficiency: float = 0.88,
                         deg_per_mwh: float = 0.0) -> dict:
    """
    Corre el backtest LP de arbitraje (oracle/forecast/naive) sobre el periodo de
    test y devuelve beneficios brutos y throughput anualizados. El dispatch es
    **consciente de la degradación** (`deg_per_mwh`): solo cicla spreads que
    cubren el desgaste. Cacheado por parámetros que afectan al despacho; el resto
    de la economía (O&M, capex, payback) se aplica aguas arriba.
    """
    from src.model.battery_backtest import battery_economics_backtest
    df = load_test_predictions(zone)
    if df is None or df.empty or "price_actual" not in df.columns:
        return {}
    d = df.copy()
    ts_col = "timestamp_utc" if "timestamp_utc" in d.columns else "timestamp_cet"
    if ts_col in d.columns:
        d = d.set_index(pd.to_datetime(d[ts_col], utc=True))
    return battery_economics_backtest(
        d, capacity_mwh=capacity_mwh, power_mw=power_mw,
        efficiency=efficiency, deg_per_mwh=deg_per_mwh
    )


@st.cache_data(ttl=86400, show_spinner="Fitting LEAR baseline...")
def load_lear_forecast(zone: str = "DK1") -> pd.Series:
    """
    Genera al vuelo la previsión horaria del **baseline LEAR** (Linear ARX, el
    benchmark estándar de la literatura Nord Pool) sobre el periodo de test.
    Sirve para medir la ventaja *comercial* del modelo frente a un forecast
    respetable — no solo frente a la persistencia ingenua.
    """
    try:
        from src.model.baselines import LEARBaseline
        if not FEATURES_PATH.exists():
            return pd.Series(dtype=float)
        df = pd.read_parquet(FEATURES_PATH)
        col = f"price_{zone}"
        if col not in df.columns:
            return pd.Series(dtype=float)
        series = df[col].dropna()
        meta = load_model_summary(zone)
        test_start = str(meta.get("test_start", "2026-01-01"))[:10]
        lear = LEARBaseline(alpha=0.1, zone=zone)   # exógenas de la propia zona
        lear.fit(series, test_start, df_extra=df)
        return lear.predict(series, test_start, df_extra=df)   # índice UTC
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner="Running battery dispatch backtest...")
def run_battery_economics(zone: str = "DK1",
                          capacity_mwh: float = 2.0,
                          power_mw: float = 1.0,
                          efficiency: float = 0.88,
                          deg_per_mwh: float = 0.0) -> dict:
    """
    Backtest de arbitraje con batería devolviendo las 4 referencias honestas:
    **oracle** (techo) · **model** (nuestro forecast) · **lear** (baseline
    comercial) · **naive** (suelo). Dispatch LP consciente de la degradación.
    """
    from src.model.battery_backtest import battery_economics_backtest
    df = load_test_predictions(zone)
    if df is None or df.empty or "price_actual" not in df.columns:
        return {}
    d = df.copy()
    ts_col = "timestamp_utc" if "timestamp_utc" in d.columns else "timestamp_cet"
    d = d.set_index(pd.to_datetime(d[ts_col], utc=True))

    base = battery_economics_backtest(
        d, forecast_col="price_pred",
        capacity_mwh=capacity_mwh, power_mw=power_mw,
        efficiency=efficiency, deg_per_mwh=deg_per_mwh)
    if not base:
        return {}
    out = {"n_days": base["n_days"], "cycles_per_day_model": base["cycles_per_day_forecast"]}
    for k in ("perfect", "naive"):
        out[f"gross_{k}_annual"] = base[f"gross_{k}_annual"]
        out[f"thru_{k}_annual"]  = base[f"thru_{k}_annual"]
    out["gross_model_annual"] = base["gross_forecast_annual"]
    out["thru_model_annual"]  = base["thru_forecast_annual"]

    # LEAR como tercera estrategia (decide con LEAR, liquida a real)
    lear = load_lear_forecast(zone)
    if lear is not None and not lear.empty:
        d2 = d.copy()
        d2["price_lear"] = lear.reindex(d2.index)
        d2 = d2.dropna(subset=["price_lear", "price_actual"])
        rl = battery_economics_backtest(
            d2, forecast_col="price_lear",
            capacity_mwh=capacity_mwh, power_mw=power_mw,
            efficiency=efficiency, deg_per_mwh=deg_per_mwh)
        if rl:
            out["gross_lear_annual"] = rl["gross_forecast_annual"]
            out["thru_lear_annual"]  = rl["thru_forecast_annual"]
    return out


# ── Estadísticas de mercado ───────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_stats(zone: str = "DK1") -> dict:
    """
    Calcula KPIs de mercado para el header del Resumen Ejecutivo.
    Devuelve precio actual, cambio vs ayer, volatilidad, y stats adicionales.
    """
    df = load_price_data()
    col = f"price_{zone}"
    if df.empty or col not in df.columns:
        # Sin datos reales: NO fabricar KPIs. La UI mostrará "—".
        return {}

    series = df[col].dropna()
    if len(series) < 48:
        return {}

    current  = float(series.iloc[-1])
    prev_24h = float(series.iloc[-25]) if len(series) > 25 else current
    delta    = current - prev_24h

    # Ventanas recientes (sustituye a Series.last(), deprecado en pandas 2.x)
    last_ts  = series.index[-1]
    last_7d  = series[series.index >= last_ts - pd.Timedelta(days=7)]
    last_30d = series[series.index >= last_ts - pd.Timedelta(days=30)]
    vol_7d   = float(last_7d.std())

    neg_pct   = float((series < 0).mean() * 100)
    spike_pct = float((series > 200).mean() * 100)

    week_avg  = float(last_7d.mean()) if not last_7d.empty else current
    month_avg = float(last_30d.mean()) if not last_30d.empty else current

    return {
        "current":      round(current, 1),
        "delta_24h":    round(delta, 1),
        "vol_7d":       round(vol_7d, 1),
        "negative_pct": round(neg_pct, 1),
        "spike_pct":    round(spike_pct, 1),
        "week_avg":     round(week_avg, 1),
        "month_avg":    round(month_avg, 1),
    }


# ── Utilidades de zona ────────────────────────────────────────────────────────

def get_available_zones() -> list[str]:
    """
    Devuelve las zonas que realmente tienen datos de precio en el master.
    Fallback a ZONES si el master no está disponible.
    """
    df = load_price_data()
    if df.empty:
        return ZONES
    return [z for z in ZONES if f"price_{z}" in df.columns]


def get_zone_label(zone: str) -> str:
    """Devuelve el label legible de una zona."""
    return ZONE_LABELS.get(zone, zone)


# ── Estado de los datos: REAL vs DEMO ──────────────────────────────────────────

def get_data_status(zone: str = "DK1") -> dict:
    """
    Comprueba en disco qué artefactos REALES del pipeline existen para la zona.

    Sirve para que el dashboard avise con honestidad cuándo está mostrando
    resultados reales del modelo y cuándo está cayendo a datos de ejemplo
    (demo) por falta de artefactos. Crítico para no enseñar métricas falsas.
    """
    return {
        "master":      MASTER_PATH.exists(),
        "features":    FEATURES_PATH.exists(),
        "model":       (MODELS_DIR / f"lightgbm_{zone}.pkl").exists(),
        "walkforward": (WF_DIR / f"walkforward_{zone}.csv").exists(),
    }


def is_demo_mode(zone: str = "DK1") -> bool:
    """True si faltan artefactos esenciales (master + modelo) → modo demo."""
    s = get_data_status(zone)
    return not (s["master"] and s["model"])


def render_data_source_banner(zone: str = "DK1") -> bool:
    """
    Pinta un aviso claro en la parte superior de cada página indicando si
    los datos mostrados son REALES (generados por el pipeline) o SIMULADOS.

    Devuelve True si está en modo demo. Llamar al principio de cada página,
    justo después de seleccionar la zona.
    """
    status = get_data_status(zone)
    demo = not (status["master"] and status["model"])

    if demo:
        missing = [k for k, v in status.items() if not v]
        st.error(
            f"**DEMO MODE: simulated data (not real model results).** "
            f"The pipeline artifacts for **{zone}** were not found "
            f"(missing: `{'`, `'.join(missing)}`). The figures and charts are "
            f"synthetic example data. To see real results, run "
            f"`python run_pipeline.py` (or `--fast` to train without Optuna)."
        )
    else:
        st.caption(
            f"Real pipeline data · zone {zone} · "
            f"model `lightgbm_{zone}.pkl` loaded from `data/models/`"
        )
    return demo