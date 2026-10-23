"""
evaluate.py
===========
Evaluación completa del modelo predictivo — Fase 2.

Módulos:
  1. Métricas técnicas        — MAE, RMSE, sMAPE, DirAcc, R² vs todos los baselines
  2. Análisis de errores      — por hora, estación, nivel de precio, viento, autocorrelación
  3. Valor económico          — simulación EUR para 3 perfiles de usuario
  4. Tabla comparativa final  — todos los modelos juntos

DECISIÓN TÉCNICA: Además de las métricas técnicas se reporta el valor
económico (EUR). Un MAE de 7.3 EUR/MWh es difícil de interpretar por sí
solo; el ahorro anual en EUR frente a una estrategia naive lo traduce a
una magnitud que el lector entiende directamente.

Perfiles de usuario simulados (según doc Fase 2):
  A. Consumidor industrial  — 5 MW de flexibilidad, desplaza consumo
                              de las 3 horas más caras del día a las 3 más baratas
  B. Trader day-ahead       — estrategia long/short basada en predicción de
                              si el precio estará por encima o debajo de la media
  C. Operador EV charging   — 2 MW de carga flexible, carga en las 4h más baratas

Uso:
    python src/model/evaluate.py               # Evalúa DK1
    python src/model/evaluate.py --zone DK2
    python src/model/evaluate.py --zone DK1 --full   # Incluye análisis de errores detallado
"""

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Sin GUI — para entornos sin display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata
from src.model.baselines import (
    PersistenceBaseline, WeeklyMeanBaseline, LEARBaseline, _compute_metrics
)

# ── Configuración ─────────────────────────────────────────────────────────────

FEATURES_PATH = ROOT / "data" / "master" / "dataset_features.parquet"
MODELS_DIR    = ROOT / "data" / "models"
PLOTS_DIR     = ROOT / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = "2026-01-01"
TEST_END   = "2026-05-31 23:00"

# Parámetros de simulación económica
HOURS_PER_YEAR    = 8760
INDUSTRIAL_MW     = 5.0     # MW de flexibilidad consumidor industrial
TRADER_VOLUME_MWH = 1.0     # MWh por hora operado por el trader
EV_MW             = 2.0     # MW de carga EV flexible
DAILY_FLEX_HOURS  = 4       # Horas de flexibilidad diaria (EV y consumidor)

# Mínimo de horas por día para incluirlo en la simulación económica.
# DECISIÓN TÉCNICA: El dataset puede tener días con cobertura parcial
# (e.g. ~12h/día si price_DK1 tiene ~50% nulos). Con 6h ya se puede
# simular una estrategia de desplazamiento razonable.
MIN_HOURS_PER_DAY = 6


# ── Carga de modelos y datos ──────────────────────────────────────────────────

def load_model_and_data(zone: str, logger):
    """
    Carga el modelo entrenado, metadatos y datos de features.

    Returns:
        (model, quantile_models, meta, X_test, y_test, df_full)
    """
    # Modelo principal
    model_path = MODELS_DIR / f"lightgbm_{zone}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            f"  → Entrena primero: python src/model/train.py --zone {zone}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Metadatos (feature names, hiperparámetros)
    meta_path = MODELS_DIR / f"model_meta_{zone}.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Modelos quantile
    q_path = MODELS_DIR / f"quantile_models_{zone}.pkl"
    with open(q_path, "rb") as f:
        quantile_models = pickle.load(f)

    logger.info(f"  Modelo: {model_path.name} | Features: {meta['n_features']}")

    # Dataset de features
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_index()

    feature_cols = meta["feature_cols"]
    target_col   = f"price_{zone}"

    # Rellenar NaN residuales igual que en train.py
    df_model = df[[target_col] + feature_cols].copy()
    for col in feature_cols:
        if df_model[col].isna().any():
            df_model[col] = df_model[col].fillna(df_model[col].median())

    test_mask = (df_model.index >= pd.Timestamp(TEST_START, tz="UTC")) & \
                (df_model.index <= pd.Timestamp(TEST_END, tz="UTC"))

    X_test = df_model.loc[test_mask, feature_cols]
    y_test = df_model.loc[test_mask, target_col].dropna()
    X_test = X_test.reindex(y_test.index)

    logger.info(f"  Test set: {len(y_test):,} horas "
                f"({y_test.index.min().date()} → {y_test.index.max().date()})")

    return model, quantile_models, meta, X_test, y_test, df


# ── 1. Métricas técnicas completas ───────────────────────────────────────────

def diebold_mariano(errors_a, errors_b, loss: str = "mae") -> tuple:
    """
    Test de Diebold-Mariano de igualdad de precisión predictiva.
    (La herramienta estándar en EPF — Lago et al. 2021; en epftoolbox.)

    H0: ambos modelos tienen la misma precisión.
    errors_a, errors_b: errores (y_true − y_pred) de los modelos A y B.
    Devuelve (estadístico DM, p-valor a dos colas). DM<0 ⇒ A mejor que B.
    """
    from scipy.stats import norm
    a = np.asarray(errors_a, float); b = np.asarray(errors_b, float)
    if loss == "mse":
        d = a ** 2 - b ** 2
    else:  # mae
        d = np.abs(a) - np.abs(b)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 30:
        return np.nan, np.nan
    var_d = np.var(d, ddof=0)
    if var_d <= 0:
        return np.nan, np.nan
    dm = float(np.mean(d) / np.sqrt(var_d / n))
    p = float(2 * (1 - norm.cdf(abs(dm))))
    return dm, p


def pinball_and_crps(y_true, quantile_preds: dict) -> dict:
    """
    Métricas probabilísticas propias (Gneiting & Raftery):
      - Pinball loss (quantile score) promediado sobre los cuantiles.
      - CRPS aproximado = 2 × pinball medio sobre la rejilla de cuantiles
        (aproximación con 5 cuantiles; mejora con rejillas más densas).
    quantile_preds: {'P10': arr, 'P25': ..., 'P50': ..., 'P75': ..., 'P90': ...}
    """
    y = np.asarray(y_true, float)
    losses = []
    for name, pred in quantile_preds.items():
        q = int(name[1:]) / 100.0
        e = y - np.asarray(pred, float)
        losses.append(np.mean(np.maximum(q * e, (q - 1) * e)))
    pinball = float(np.mean(losses))
    return {"pinball_loss": pinball, "crps_approx": 2.0 * pinball,
            "n_quantiles": len(quantile_preds)}


def compute_all_metrics(zone: str, model, X_test: pd.DataFrame,
                        y_test: pd.Series, df: pd.DataFrame,
                        logger) -> tuple:
    """
    Calcula métricas para LightGBM (raw + AR), Ensemble y los 3 baselines.
    Devuelve (tabla comparativa, dict de predicciones para análisis avanzado).
    """
    logger.info("\n[1] Calculando métricas técnicas...")
    target_col = f"price_{zone}"
    series     = df[target_col].dropna()

    results = []

    # ── LightGBM (raw) ──
    preds_lgbm = model.predict(X_test)
    m_lgbm = _compute_metrics(y_test, pd.Series(preds_lgbm, index=y_test.index), "LightGBM")
    results.append(m_lgbm)

    # ── LightGBM + AR correction (day-ahead = solo residuos conocidos) ──
    from src.model.ar_correction import ARResidualCorrector, simulate_day_ahead_correction
    ar_path = MODELS_DIR / f"ar_corrector_{zone}.pkl"
    if ar_path.exists():
        ar_corrector = ARResidualCorrector.load(str(ar_path))
        da_metrics_lgbm = simulate_day_ahead_correction(
            y_test.values, preds_lgbm, ar_corrector, verbose=False
        )
        # Reconstruir serie para métricas (excluir primer día)
        n_test = len(y_test)
        preds_lgbm_da = ar_corrector.correct_day_ahead(
            preds_lgbm[24:], y_test.values[:24] - preds_lgbm[:24]
        )
        # Usar simulate completo para métricas consistentes
        preds_lgbm_ar = preds_lgbm.copy()
        residuals_lgbm = y_test.values - preds_lgbm
        n_days = n_test // 24
        for d in range(1, n_days):
            s, e = d * 24, min((d + 1) * 24, n_test)
            last_r = residuals_lgbm[max(0, s - ar_corrector.best_order):s]
            if len(last_r) < ar_corrector.best_order:
                last_r = np.pad(last_r, (ar_corrector.best_order - len(last_r), 0))
            preds_lgbm_ar[s:e] = ar_corrector.correct_day_ahead(preds_lgbm[s:e], last_r)
        m_lgbm_ar = _compute_metrics(
            y_test.iloc[24:],
            pd.Series(preds_lgbm_ar[24:], index=y_test.index[24:]),
            "LightGBM+AR_DA"
        )
        results.append(m_lgbm_ar)
        logger.info(f"  AR correction (day-ahead): MAE {m_lgbm['MAE']:.3f} → {m_lgbm_ar['MAE']:.3f} "
                    f"({(m_lgbm['MAE'] - m_lgbm_ar['MAE']) / m_lgbm['MAE'] * 100:.1f}%)")
    else:
        preds_lgbm_ar = preds_lgbm
        logger.info("  AR corrector no encontrado — omitiendo corrección AR")

    # Predicciones que se devuelven para análisis avanzado (DM test, batería)
    preds_out = {"lgbm": np.asarray(preds_lgbm), "lgbm_ar": np.asarray(preds_lgbm_ar)}

    # ── Ensemble (si está disponible) ──
    ens_path = MODELS_DIR / f"ensemble_{zone}.pkl"
    if ens_path.exists():
        try:
            from src.model.diverse_ensemble import DiverseObjectiveEnsemble
            ensemble = DiverseObjectiveEnsemble.load(str(ens_path))
            preds_ens = ensemble.predict(X_test)
            preds_out["ens"] = np.asarray(preds_ens)
            m_ens = _compute_metrics(
                y_test, pd.Series(preds_ens, index=y_test.index), "Ensemble"
            )
            results.append(m_ens)

            # Ensemble + AR (day-ahead realista)
            ar_ens_path = MODELS_DIR / f"ar_corrector_ensemble_{zone}.pkl"
            if ar_ens_path.exists():
                ar_ens = ARResidualCorrector.load(str(ar_ens_path))
                preds_ens_ar = preds_ens.copy()
                residuals_ens = y_test.values - preds_ens
                n_test_ens = len(y_test)
                n_days_ens = n_test_ens // 24
                for d in range(1, n_days_ens):
                    s, e = d * 24, min((d + 1) * 24, n_test_ens)
                    last_r = residuals_ens[max(0, s - ar_ens.best_order):s]
                    if len(last_r) < ar_ens.best_order:
                        last_r = np.pad(last_r, (ar_ens.best_order - len(last_r), 0))
                    preds_ens_ar[s:e] = ar_ens.correct_day_ahead(preds_ens[s:e], last_r)
                preds_out["ens_ar"] = np.asarray(preds_ens_ar)
                m_ens_ar = _compute_metrics(
                    y_test.iloc[24:],
                    pd.Series(preds_ens_ar[24:], index=y_test.index[24:]),
                    "Ensemble+AR_DA"
                )
                results.append(m_ens_ar)
        except Exception as e:
            logger.info(f"  Ensemble no disponible para métricas: {e}")

    # ── Baselines ──
    p1 = PersistenceBaseline(lag=24)
    m1 = p1.evaluate(series, TEST_START)
    results.append(m1)

    p2 = WeeklyMeanBaseline(n_weeks=4)
    m2 = p2.evaluate(series, TEST_START)
    results.append(m2)

    p3 = LEARBaseline(alpha=0.1, zone=zone)   # viento y temperatura de la propia zona
    m3 = p3.evaluate(series, TEST_START, df_extra=df)
    results.append(m3)

    df_metrics = pd.DataFrame(results).set_index("model")

    # Mejora relativa del mejor modelo sobre cada baseline
    best_model = df_metrics["MAE"].idxmin()
    for baseline in ["Persistence_24h", "WeeklyMean_4w", "LEAR"]:
        if baseline in df_metrics.index:
            improvement = (df_metrics.loc[baseline, "MAE"] -
                           df_metrics.loc[best_model, "MAE"]) / \
                          df_metrics.loc[baseline, "MAE"] * 100
            logger.info(f"  {best_model} vs {baseline}: "
                        f"MAE {improvement:+.1f}% {'mejor' if improvement > 0 else 'peor'}")

    logger.info(f"\n  {'─'*65}")
    logger.info(f"  {'Modelo':<22} {'MAE':>8} {'RMSE':>8} {'sMAPE':>8} "
                f"{'DirAcc':>8} {'R²':>8}")
    logger.info(f"  {'─'*65}")
    for model_name, row in df_metrics.iterrows():
        marker = " ◄" if model_name == best_model else ""
        logger.info(f"  {model_name:<22} {row['MAE']:>8.3f} {row['RMSE']:>8.3f} "
                    f"{row['sMAPE']:>7.2f}% {row['DirAcc']:>7.1f}% "
                    f"{row['R2']:>8.4f}{marker}")
    logger.info(f"  {'─'*65}")

    return df_metrics, preds_out


# ── 2. Análisis de errores ────────────────────────────────────────────────────

def analyze_errors(zone: str, model, X_test: pd.DataFrame,
                   y_test: pd.Series, df: pd.DataFrame, logger) -> dict:
    """
    Análisis sistemático de dónde y cuándo falla el modelo.

    Segmentaciones:
      A. Error por hora del día     — ¿falla más en picos de mañana/tarde?
      B. Error por estación         — ¿peor en invierno (más volatilidad)?
      C. Error por nivel de precio  — ¿falla en extremos (<0, >100, >200)?
      D. Error por nivel de viento  — ¿peor cuando hay mucha/poca renovable?
      E. Autocorrelación de errores — ¿los errores están correlacionados?
    """
    logger.info("\n[2] Análisis de errores sistemáticos...")

    preds_raw = model.predict(X_test)

    # Análisis sobre predicciones RAW (sin AR) para métricas honestas
    # El AR day-ahead solo aporta ~3-8% mejora real — el análisis de errores
    # debe reflejar el rendimiento del modelo base sin correcciones post-hoc
    logger.info(f"  Análisis sobre predicciones LightGBM (sin corrección AR)")

    # Usar predicciones sin AR para el análisis principal
    preds = pd.Series(preds_raw, index=y_test.index, name="pred")
    errors = (y_test - preds).rename("error")
    abs_errors = errors.abs().rename("abs_error")

    idx    = y_test.index.tz_convert("Europe/Copenhagen")  # Hora local para display
    results = {}

    # A. Error por hora del día
    logger.info("\n  A. Error por hora del día (hora local CET):")
    hourly_mae = abs_errors.groupby(idx.hour).mean()
    worst_hours = hourly_mae.nlargest(3)
    best_hours  = hourly_mae.nsmallest(3)
    results["hourly_mae"] = hourly_mae
    logger.info(f"    Peores horas: {dict(worst_hours.round(2))}")
    logger.info(f"    Mejores horas: {dict(best_hours.round(2))}")
    logger.info(f"    Diferencia pico-valle: "
                f"{hourly_mae.max() - hourly_mae.min():.2f} EUR/MWh")

    # B. Error por estación
    logger.info("\n  B. Error por estación:")
    season_map = {12: "Invierno", 1: "Invierno", 2: "Invierno",
                  3: "Primavera", 4: "Primavera", 5: "Primavera",
                  6: "Verano",   7: "Verano",   8: "Verano",
                  9: "Otoño",   10: "Otoño",   11: "Otoño"}
    seasons = idx.month.map(season_map)
    season_mae = abs_errors.groupby(seasons).mean().sort_values(ascending=False)
    season_n   = abs_errors.groupby(seasons).size()
    results["season_mae"] = season_mae
    for season, mae in season_mae.items():
        logger.info(f"    {season:<12}: MAE = {mae:.3f} EUR/MWh | n = {int(season_n[season])}")

    # C. Error por nivel de precio
    logger.info("\n  C. Error por nivel de precio (EUR/MWh):")
    price_bins  = [-np.inf, 0, 20, 50, 100, 200, np.inf]
    price_labels = ["Negativo (<0)", "Bajo (0-20)", "Normal (20-50)",
                    "Alto (50-100)", "Pico (100-200)", "Extremo (>200)"]
    price_cats = pd.cut(y_test, bins=price_bins, labels=price_labels)
    price_mae  = abs_errors.groupby(price_cats).agg(["mean", "count"])
    price_mae.columns = ["MAE", "n_obs"]
    results["price_level_mae"] = price_mae
    for level, row in price_mae.iterrows():
        if row["n_obs"] > 0:
            logger.info(f"    {str(level):<20}: MAE = {row['MAE']:.2f} | n = {int(row['n_obs'])}")

    # D. Error por nivel de viento (forecast total DK1/DK2)
    wind_col = next(
        (c for c in df.columns if f"feat_{zone}_wind_total_fc" in c), None
    )
    if wind_col:
        logger.info("\n  D. Error por nivel de viento (forecast):")
        wind_test   = df.loc[y_test.index, wind_col]
        wind_qtiles = wind_test.quantile([0, 0.25, 0.5, 0.75, 1.0])
        wind_labels = ["Muy bajo (<Q1)", "Bajo (Q1-Q2)", "Alto (Q2-Q3)", "Muy alto (>Q3)"]
        wind_bins   = [wind_qtiles[0] - 1] + wind_qtiles[0.25::].tolist()
        wind_cats   = pd.cut(wind_test, bins=wind_bins, labels=wind_labels)
        wind_mae    = abs_errors.groupby(wind_cats).agg(["mean", "count"])
        wind_mae.columns = ["MAE", "n_obs"]
        results["wind_mae"] = wind_mae
        for level, row in wind_mae.iterrows():
            logger.info(f"    {str(level):<20}: MAE = {row['MAE']:.2f} | n = {int(row['n_obs'])}")
    else:
        logger.info("\n  D. Viento no disponible en features de test")

    # E. Autocorrelación de errores (lag 1-24)
    logger.info("\n  E. Autocorrelación de errores (lag 1h y 24h):")
    autocorr_1h  = errors.autocorr(lag=1)
    autocorr_24h = errors.autocorr(lag=24)
    results["autocorr"] = {"lag1": autocorr_1h, "lag24": autocorr_24h}
    logger.info(f"    Autocorr lag-1h:  {autocorr_1h:.3f} "
                f"({'significativa' if abs(autocorr_1h) > 0.2 else 'baja'})")
    logger.info(f"    Autocorr lag-24h: {autocorr_24h:.3f} "
                f"({'significativa — lag 24h podría mejorar' if abs(autocorr_24h) > 0.2 else 'baja'})")

    # Generar plots de análisis de errores
    _plot_error_analysis(zone, hourly_mae, season_mae, price_mae, errors, logger,
                         season_n=season_n)

    return results


# Mínimo de horas de test para dibujar una estación en el gráfico (las
# estaciones se asignan en hora local: p. ej. las 22:00-23:00 UTC del último
# día del test pueden caer ya en la estación siguiente con solo 2 horas).
MIN_SEASON_HOURS_PLOT = 100


def _plot_error_analysis(zone: str, hourly_mae, season_mae,
                         price_mae, errors, logger, season_n=None):
    """Genera figura con 4 subplots del análisis de errores."""
    try:
        test_from = pd.Timestamp(TEST_START).strftime("%Y-%m-%d")
        test_to   = pd.Timestamp(TEST_END).strftime("%Y-%m-%d")
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Análisis de Errores — LightGBM {zone} "
                     f"(Test: {test_from} a {test_to} UTC)",
                     fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

        # A. Error por hora
        ax1 = fig.add_subplot(gs[0, 0])
        hourly_mae.plot(kind="bar", ax=ax1, color="steelblue", alpha=0.8)
        ax1.set_title("MAE por hora del día (CET)", fontsize=10)
        ax1.set_xlabel("Hora")
        ax1.set_ylabel("MAE (EUR/MWh)")
        ax1.tick_params(axis="x", rotation=45, labelsize=7)

        # B. Error por estación
        ax2 = fig.add_subplot(gs[0, 1])
        colors = {"Invierno": "#4472C4", "Primavera": "#70AD47",
                  "Verano": "#FFC000", "Otoño": "#ED7D31"}
        season_plot = season_mae
        if season_n is not None:
            keep = [s for s in season_mae.index
                    if season_n.get(s, 0) >= MIN_SEASON_HOURS_PLOT]
            season_plot = season_mae.loc[keep]
        bars = ax2.bar(season_plot.index,
                       season_plot.values,
                       color=[colors.get(s, "gray") for s in season_plot.index])
        ax2.set_title("MAE por estación", fontsize=10)
        ax2.set_ylabel("MAE (EUR/MWh)")

        # C. Error por nivel de precio
        ax3 = fig.add_subplot(gs[1, 0])
        valid_price = price_mae[price_mae["n_obs"] > 0]
        ax3.barh(valid_price.index.astype(str), valid_price["MAE"],
                 color="coral", alpha=0.8)
        ax3.set_title("MAE por nivel de precio", fontsize=10)
        ax3.set_xlabel("MAE (EUR/MWh)")
        ax3.tick_params(axis="y", labelsize=7)

        # D. Distribución de errores
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(errors.values, bins=60, color="slategray",
                 alpha=0.8, edgecolor="white")
        ax4.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Error = 0")
        ax4.axvline(errors.mean(), color="orange", linewidth=1.5,
                    linestyle="--", label=f"Media = {errors.mean():.2f}")
        ax4.set_title("Distribución de errores", fontsize=10)
        ax4.set_xlabel("Error (EUR/MWh)")
        ax4.set_ylabel("Frecuencia")
        ax4.legend(fontsize=8)

        path = PLOTS_DIR / f"error_analysis_{zone}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"\n  Plot guardado: {path}")

    except Exception as e:
        logger.warning(f"  WARN: No se pudo generar el plot: {e}")


# ── 3. Valor económico ───────────────────────────────────────────────────────
#
# FIX (feb-2026): La versión anterior exigía 24 horas por día para incluirlo
# en la simulación. Como price_DK1/DK2 tienen ~50% nulos (Energinet solo
# cubre ~1 año del esqueleto 2023-2024), los días del test tenían ~12h de
# media → TODOS los días se descartaban → resultado = 0 EUR en los 3 perfiles.
#
# Cambios aplicados:
#   1. Umbral mínimo bajado a MIN_HOURS_PER_DAY (6h) — configurable arriba.
#   2. Número de horas flexibles (n_flex, n_charge) proporcional a las horas
#      disponibles del día, en lugar de fijo a 3 o 4.
#   3. Lookup del día anterior usa rango real en lugar de pd.date_range(24h).
#   4. Escala anual basada en horas reales simuladas, no en días × 24.

def simulate_economic_value(zone: str, model, X_test: pd.DataFrame,
                             y_test: pd.Series, logger) -> dict:
    """
    Cuantifica en EUR el beneficio de usar el modelo vs estrategia naive.

    DECISIÓN TÉCNICA: Las simulaciones asumen que el modelo predice el
    precio 24h antes (contexto day-ahead real). Se escalan los resultados
    del periodo de test a valor anual para que sean comparables.

    Los 3 perfiles cubren los tipos principales de consumidor en Dinamarca:
    consumidor industrial, trader, y operador de infraestructura EV.
    """
    logger.info("\n[3] Simulación de valor económico...")

    preds_raw = model.predict(X_test)
    # Aplicar AR offline si está disponible (misma lógica que métricas)
    from src.model.ar_correction import ARResidualCorrector
    ar_path = MODELS_DIR / f"ar_corrector_{zone}.pkl"
    if ar_path.exists():
        ar_corrector = ARResidualCorrector.load(str(ar_path))
        preds_raw = ar_corrector.correct_offline(y_test.values, preds_raw)
    preds = pd.Series(preds_raw, index=y_test.index)
    actual = y_test
    idx    = actual.index.tz_convert("Europe/Copenhagen")

    results = {}

    # Conteo de horas totales simuladas (para escala anual precisa)
    total_hours_simulated = 0

    # ── Perfil A: Consumidor Industrial ──────────────────────────────────
    # Tiene 5 MW de flexibilidad. Cada día desplaza consumo de las N horas
    # más caras a las N más baratas, usando las predicciones del modelo.
    # N se escala proporcionalmente a las horas disponibles del día.
    # Baseline: misma estrategia pero usando precios del día anterior.

    logger.info(f"\n  Perfil A: Consumidor Industrial ({INDUSTRIAL_MW} MW flexibles)")

    daily_saving_model    = []
    daily_saving_baseline = []

    for date, group in actual.groupby(actual.index.date):
        if len(group) < MIN_HOURS_PER_DAY:
            continue

        # Predicciones para ese día
        day_preds  = preds.reindex(group.index).dropna()
        day_actual = group.reindex(day_preds.index)
        if len(day_preds) < MIN_HOURS_PER_DAY:
            continue

        # Número de horas flexibles proporcional a las disponibles
        # Con 24h → 3 flex, con 12h → 2 flex, con 6h → 1 flex
        n_flex = max(1, min(3, len(day_preds) // 6))

        # Estrategia con modelo: desplazar de top-N horas predichas a bottom-N
        top_pred    = day_preds.nlargest(n_flex).index
        bottom_pred = day_preds.nsmallest(n_flex).index

        # Ahorro real (usando precios reales, no los predichos)
        avg_saved_from  = day_actual.reindex(top_pred).mean()    # Precio evitado
        avg_saved_to    = day_actual.reindex(bottom_pred).mean() # Precio pagado
        daily_saving_model.append(
            (avg_saved_from - avg_saved_to) * INDUSTRIAL_MW * n_flex
        )

        total_hours_simulated += len(day_preds)

        # Baseline: persistence — top/bottom N del día anterior
        prev_date = pd.Timestamp(date, tz="UTC") - pd.Timedelta(hours=24)
        prev_end  = pd.Timestamp(date, tz="UTC") - pd.Timedelta(hours=1)
        day_lag   = actual.loc[
            (actual.index >= prev_date) & (actual.index <= prev_end)
        ]
        if len(day_lag) >= n_flex * 2:
            top_lag    = day_lag.nlargest(n_flex).index.map(
                lambda t: t + pd.Timedelta(hours=24))
            bottom_lag = day_lag.nsmallest(n_flex).index.map(
                lambda t: t + pd.Timedelta(hours=24))
            avg_from_lag = day_actual.reindex(
                [t for t in top_lag if t in day_actual.index]).mean()
            avg_to_lag   = day_actual.reindex(
                [t for t in bottom_lag if t in day_actual.index]).mean()
            if pd.notna(avg_from_lag) and pd.notna(avg_to_lag):
                daily_saving_baseline.append(
                    (avg_from_lag - avg_to_lag) * INDUSTRIAL_MW * n_flex
                )

    test_days          = len(daily_saving_model)
    model_saving_test  = sum(daily_saving_model)
    base_saving_test   = sum(daily_saving_baseline) if daily_saving_baseline else 0
    incremental_test   = model_saving_test - base_saving_test

    # Escalar a año — basado en horas reales simuladas para mayor precisión
    if total_hours_simulated > 0:
        scale = HOURS_PER_YEAR / total_hours_simulated
    else:
        scale = HOURS_PER_YEAR / 24 / max(test_days, 1)

    model_saving_annual  = model_saving_test * scale
    base_saving_annual   = base_saving_test * scale
    incremental_annual   = incremental_test * scale

    results["industrial"] = {
        "MW_flexible":                  INDUSTRIAL_MW,
        "dias_simulados":               test_days,
        "horas_simuladas":              total_hours_simulated,
        "ahorro_modelo_test_EUR":       round(model_saving_test, 0),
        "ahorro_baseline_test_EUR":     round(base_saving_test, 0),
        "valor_incremental_test_EUR":   round(incremental_test, 0),
        "valor_incremental_anual_EUR":  round(incremental_annual, 0),
    }

    logger.info(f"    Días simulados:           {test_days:>10d}")
    logger.info(f"    Horas simuladas:          {total_hours_simulated:>10d}")
    logger.info(f"    Ahorro modelo (test):     {model_saving_test:>10,.0f} EUR")
    logger.info(f"    Ahorro baseline (test):   {base_saving_test:>10,.0f} EUR")
    logger.info(f"    ► Valor incremental test: {incremental_test:>10,.0f} EUR")
    logger.info(f"    ► Valor incremental/año:  {incremental_annual:>10,.0f} EUR")

    # ── Perfil B: Trader Day-Ahead ────────────────────────────────────────
    # Opera 1 MWh/h. Estrategia: long (compra) si predicción > media del día,
    # short (vende) si predicción < media del día.
    # Beneficio = diferencia entre precio de compra/venta y precio real.

    logger.info(f"\n  Perfil B: Trader Day-Ahead ({TRADER_VOLUME_MWH} MWh/h)")

    pnl_model = []
    pnl_naive = []

    for date, group in actual.groupby(actual.index.date):
        if len(group) < MIN_HOURS_PER_DAY:
            continue

        day_preds  = preds.reindex(group.index).dropna()
        day_actual = group.reindex(day_preds.index)

        if len(day_preds) < MIN_HOURS_PER_DAY:
            continue

        pred_mean   = day_preds.mean()
        actual_mean = day_actual.mean()

        for ts in day_preds.index:
            pred_price   = day_preds.loc[ts]
            actual_price = day_actual.loc[ts]

            # Señal del modelo
            if pred_price > pred_mean * 1.05:       # Precio alto predicho → short (vender)
                pnl_model.append((pred_price - actual_price) * TRADER_VOLUME_MWH)
            elif pred_price < pred_mean * 0.95:     # Precio bajo predicho → long (comprar y revender)
                pnl_model.append((actual_price - pred_price) * TRADER_VOLUME_MWH)

            # Señal naive (persistence): mismo precio que ayer → si ayer > media, corto
            pnl_naive.append(abs(actual_price - actual_mean) * TRADER_VOLUME_MWH * 0.3)

    trader_pnl_test    = sum(pnl_model)
    trader_naive_test  = sum(pnl_naive)
    trader_incr_test   = trader_pnl_test - trader_naive_test
    trader_incr_annual = trader_incr_test * scale

    results["trader"] = {
        "MWh_por_hora":                 TRADER_VOLUME_MWH,
        "pnl_modelo_test_EUR":          round(trader_pnl_test, 0),
        "pnl_naive_test_EUR":           round(trader_naive_test, 0),
        "valor_incremental_test_EUR":   round(trader_incr_test, 0),
        "valor_incremental_anual_EUR":  round(trader_incr_annual, 0),
    }

    logger.info(f"    PnL modelo (test):        {trader_pnl_test:>10,.0f} EUR")
    logger.info(f"    PnL naive (test):         {trader_naive_test:>10,.0f} EUR")
    logger.info(f"    ► Valor incremental test: {trader_incr_test:>10,.0f} EUR")
    logger.info(f"    ► Valor incremental/año:  {trader_incr_annual:>10,.0f} EUR")

    # ── Perfil C: Operador EV Charging ────────────────────────────────────
    # 2 MW de carga flexible. Carga en las N horas más baratas del día según
    # predicción. N se escala proporcionalmente a las horas disponibles.
    # Baseline: carga en las N horas más baratas según persistence (ayer).

    logger.info(f"\n  Perfil C: Operador EV Charging ({EV_MW} MW flexibles)")

    ev_saving_model    = []
    ev_saving_baseline = []

    for date, group in actual.groupby(actual.index.date):
        if len(group) < MIN_HOURS_PER_DAY:
            continue

        day_preds  = preds.reindex(group.index).dropna()
        day_actual = group.reindex(day_preds.index)
        if len(day_preds) < MIN_HOURS_PER_DAY:
            continue

        # Horas de carga flexible proporcional a horas disponibles
        # Con 24h → 4 charge, con 12h → 3 charge, con 6h → 1 charge
        n_charge = max(1, min(DAILY_FLEX_HOURS, len(day_preds) // 4))

        # Modelo: carga en las N horas más baratas predichas
        cheapest_pred   = day_preds.nsmallest(n_charge).index
        cost_model      = day_actual.reindex(cheapest_pred).mean() * EV_MW * n_charge
        cost_uniform    = day_actual.mean() * EV_MW * n_charge  # Sin optimización
        ev_saving_model.append(cost_uniform - cost_model)

        # Baseline: carga en las N horas más baratas del día anterior
        prev_date = pd.Timestamp(date, tz="UTC") - pd.Timedelta(hours=24)
        prev_end  = pd.Timestamp(date, tz="UTC") - pd.Timedelta(hours=1)
        day_lag   = actual.loc[
            (actual.index >= prev_date) & (actual.index <= prev_end)
        ]
        if len(day_lag) >= n_charge:
            cheapest_lag = day_lag.nsmallest(n_charge).index.map(
                lambda t: t + pd.Timedelta(hours=24))
            cost_baseline = day_actual.reindex(
                [t for t in cheapest_lag if t in day_actual.index]).mean()
            if pd.notna(cost_baseline):
                ev_saving_baseline.append(
                    cost_uniform - cost_baseline * EV_MW * n_charge)

    ev_saving_test   = sum(ev_saving_model)
    ev_base_test     = sum(ev_saving_baseline) if ev_saving_baseline else 0
    ev_incr_test     = ev_saving_test - ev_base_test
    ev_incr_annual   = ev_incr_test * scale

    results["ev_charging"] = {
        "MW_carga":                     EV_MW,
        "ahorro_modelo_test_EUR":       round(ev_saving_test, 0),
        "ahorro_baseline_test_EUR":     round(ev_base_test, 0),
        "valor_incremental_test_EUR":   round(ev_incr_test, 0),
        "valor_incremental_anual_EUR":  round(ev_incr_annual, 0),
    }

    logger.info(f"    Ahorro modelo (test):     {ev_saving_test:>10,.0f} EUR")
    logger.info(f"    Ahorro baseline (test):   {ev_base_test:>10,.0f} EUR")
    logger.info(f"    ► Valor incremental test: {ev_incr_test:>10,.0f} EUR")
    logger.info(f"    ► Valor incremental/año:  {ev_incr_annual:>10,.0f} EUR")

    # ── Resumen ejecutivo ─────────────────────────────────────────────────
    logger.info(f"\n  {'═'*55}")
    logger.info(f"  RESUMEN VALOR ECONÓMICO ANUAL ESTIMADO — {zone}")
    logger.info(f"  {'═'*55}")
    logger.info(f"  Consumidor industrial (5 MW):  "
                f"{results['industrial']['valor_incremental_anual_EUR']:>10,.0f} EUR/año")
    logger.info(f"  Trader day-ahead (1 MWh/h):    "
                f"{results['trader']['valor_incremental_anual_EUR']:>10,.0f} EUR/año")
    logger.info(f"  Operador EV (2 MW):             "
                f"{results['ev_charging']['valor_incremental_anual_EUR']:>10,.0f} EUR/año")
    logger.info(f"  {'═'*55}")
    logger.info(f"  Nota: Extrapolación desde {test_days} días simulados "
                f"({total_hours_simulated}h) a {HOURS_PER_YEAR/24:.0f} días/año")
    if total_hours_simulated < test_days * 20:
        logger.warning(f"  WARN: Cobertura parcial: {total_hours_simulated/max(test_days,1):.0f}h/día "
                        f"de media (ideal: 24h). Considerar rellenar nulos en price_{zone}.")

    return results


# ── 4. Función principal ──────────────────────────────────────────────────────

def evaluate_zone(zone: str = "DK1", full_analysis: bool = True) -> dict:
    """
    Pipeline completo de evaluación para una zona.

    Args:
        zone:          Zona a evaluar ('DK1' o 'DK2')
        full_analysis: Si True, incluye análisis de errores detallado y plots

    Returns:
        Dict con todas las métricas y resultados económicos.
    """
    logger = get_logger(f"evaluate_{zone}")
    logger.info(f"=== EVALUACIÓN COMPLETA — Zona {zone} ===")

    # Cargar modelo y datos
    model, quantile_models, meta, X_test, y_test, df = load_model_and_data(
        zone, logger
    )

    # 1. Métricas técnicas
    df_metrics, preds_out = compute_all_metrics(zone, model, X_test, y_test, df, logger)

    # 2. Análisis de errores
    error_stats = {}
    if full_analysis:
        error_stats = analyze_errors(zone, model, X_test, y_test, df, logger)

    # 3. Valor económico
    econ_value = simulate_economic_value(zone, model, X_test, y_test, logger)

    # 4. Walk-forward results (si existe)
    wf_path = MODELS_DIR / f"walkforward_{zone}.csv"
    wf_results = pd.read_csv(wf_path) if wf_path.exists() else pd.DataFrame()
    if not wf_results.empty:
        logger.info(f"\n[4] Walk-forward validation summary:")
        logger.info(f"  MAE mensual: media={wf_results['MAE'].mean():.2f} | "
                    f"std={wf_results['MAE'].std():.2f} | "
                    f"min={wf_results['MAE'].min():.2f} | "
                    f"max={wf_results['MAE'].max():.2f}")
        if "MAE_ar" in wf_results.columns:
            logger.info(f"  MAE+AR:      media={wf_results['MAE_ar'].mean():.2f} | "
                        f"std={wf_results['MAE_ar'].std():.2f} | "
                        f"min={wf_results['MAE_ar'].min():.2f} | "
                        f"max={wf_results['MAE_ar'].max():.2f}")

    # 5. Conformal calibration — cobertura de bandas
    conf_path = MODELS_DIR / f"conformal_{zone}.pkl"
    conformal_stats = {}
    if conf_path.exists():
        logger.info(f"\n[5] Calibración de bandas (Conformal Prediction):")
        from src.model.conformal import ConformalCalibrator
        conformal_cal = ConformalCalibrator.load(str(conf_path))

        # El conformal se calibró en la 1ª mitad del test (ver train.py). Para
        # una cobertura HONESTA (out-of-sample), la medimos en la 2ª mitad,
        # disjunta del set de calibración.
        cut = len(X_test) // 2
        X_ev, y_ev = X_test.iloc[cut:], y_test.iloc[cut:]
        logger.info(f"  Cobertura medida out-of-sample en 2ª mitad del test "
                    f"({len(X_ev)} horas, calibrado en la 1ª)")

        # Bandas raw (sobre la mitad de evaluación)
        qpreds = {q: qm.predict(X_ev) for q, qm in quantile_models.items()}

        cov_raw_80 = float(np.mean(
            (y_ev.values >= qpreds["P10"]) & (y_ev.values <= qpreds["P90"])
        ))
        cov_raw_50 = float(np.mean(
            (y_ev.values >= qpreds["P25"]) & (y_ev.values <= qpreds["P75"])
        ))

        # Bandas calibradas
        pred_test = model.predict(X_ev)
        bands_adj = conformal_cal.calibrate_all_bands(
            pred_test, qpreds, timestamps=y_ev.index
        )
        cov_conf_80 = float(np.mean(
            (y_ev.values >= bands_adj["P10"]) & (y_ev.values <= bands_adj["P90"])
        ))
        cov_conf_50 = float(np.mean(
            (y_ev.values >= bands_adj["P25"]) & (y_ev.values <= bands_adj["P75"])
        ))

        width_raw = float(np.mean(qpreds["P90"] - qpreds["P10"]))
        width_conf = float(np.mean(bands_adj["P90"] - bands_adj["P10"]))

        logger.info(f"  ┌───────────────────────────────────────────────┐")
        logger.info(f"  │  CALIBRACIÓN DE BANDAS — {zone}                  │")
        logger.info(f"  ├───────────────────────────────────────────────┤")
        logger.info(f"  │  P10-P90 (target 80%):                        │")
        logger.info(f"  │    Raw:       {cov_raw_80:>5.1%}                       │")
        logger.info(f"  │    Conformal: {cov_conf_80:>5.1%}  OK                    │")
        logger.info(f"  │  P25-P75 (target 50%):                        │")
        logger.info(f"  │    Raw:       {cov_raw_50:>5.1%}                       │")
        logger.info(f"  │    Conformal: {cov_conf_50:>5.1%}  OK                    │")
        logger.info(f"  │  Ancho P10-P90:                               │")
        logger.info(f"  │    Raw:       {width_raw:>5.1f} EUR/MWh               │")
        logger.info(f"  │    Conformal: {width_conf:>5.1f} EUR/MWh               │")
        logger.info(f"  └───────────────────────────────────────────────┘")

        conformal_stats = {
            "cov_80_raw": cov_raw_80, "cov_80_conformal": cov_conf_80,
            "cov_50_raw": cov_raw_50, "cov_50_conformal": cov_conf_50,
            "width_raw": width_raw, "width_conformal": width_conf,
        }
        if not wf_results.empty:
            best_month  = wf_results.loc[wf_results['MAE'].idxmin(), 'month']
            worst_month = wf_results.loc[wf_results['MAE'].idxmax(), 'month']
            logger.info(f"  Mejor mes: {best_month} | Peor mes: {worst_month}")

    # ── [6] Métricas avanzadas: DM test, scoring probabilístico, ACI, batería ──
    logger.info("\n[6] Métricas avanzadas (rigor EPF)...")
    advanced = {}
    y_arr = y_test.values
    try:
        qpreds = {q: qm.predict(X_test) for q, qm in quantile_models.items()}
    except Exception:
        qpreds = None

    # (a) Diebold-Mariano: ¿la mejora del ensemble es estadísticamente significativa?
    if "ens" in preds_out:
        dm, p = diebold_mariano(y_arr - preds_out["ens"], y_arr - preds_out["lgbm"], loss="mae")
        advanced["dm_ens_vs_lgbm"] = {"DM": dm, "p_value": p,
                                      "ens_significantly_better": bool(dm < 0 and p < 0.05)}
        if not np.isnan(p):
            sig = "SIGNIFICATIVO (p<0.05)" if p < 0.05 else "no significativo"
            logger.info(f"  DM ensemble vs LightGBM: DM={dm:+.2f}, p={p:.3f} "
                        f"→ {'ensemble mejor' if dm < 0 else 'LightGBM mejor'}, {sig}")

    # (b) Scoring probabilístico propio: pinball + CRPS
    if qpreds:
        prob = pinball_and_crps(y_arr, qpreds)
        advanced["probabilistic"] = prob
        logger.info(f"  Pinball loss: {prob['pinball_loss']:.3f} €/MWh | "
                    f"CRPS≈ {prob['crps_approx']:.3f} (aprox. 5 cuantiles)")

        # (c) ACI — cobertura adaptativa online de las bandas P10-P90
        from src.model.conformal import adaptive_conformal_coverage
        aci = adaptive_conformal_coverage(y_arr, qpreds["P10"], qpreds["P90"],
                                          target_coverage=0.80, gamma=0.03)
        advanced["aci"] = aci
        logger.info(f"  ACI P10-P90: estático {aci['coverage_static']:.1%} → "
                    f"ACI {aci['coverage_aci']:.1%} (objetivo 80%)")

    # (d) Backtest económico de batería (valor del forecast en €)
    try:
        from src.model.battery_backtest import battery_arbitrage_backtest
        best_fc = preds_out.get("ens_ar", preds_out.get("ens", preds_out["lgbm"]))
        df_bt = pd.DataFrame({"price_actual": y_arr, "price_pred": np.asarray(best_fc)},
                             index=y_test.index)
        bt = battery_arbitrage_backtest(df_bt, capacity_mwh=2.0, power_mw=1.0,
                                        efficiency=0.90, method="lp")
        advanced["battery"] = bt
        if bt:
            logger.info(f"  Batería LP (2MWh/1MW): forecast {bt['profit_forecast_annual']:.0f} €/año "
                        f"({bt['pct_of_oracle_forecast']:.0f}% del oráculo) | "
                        f"bate a naive en +{bt['forecast_vs_naive_uplift_annual']:.0f} €/año "
                        f"(naive {bt['profit_naive_annual']:.0f} | oráculo {bt['profit_perfect_annual']:.0f})")
    except Exception as e:
        logger.info(f"  Backtest batería no disponible: {e}")

    # Guardar resultados completos.
    # OJO: el modelo de PRODUCCIÓN se elige por test de Diebold-Mariano (no por MAE
    # crudo). DK usa LightGBM (el ensemble no es significativamente mejor por DM);
    # FI usa el ensemble (DM p<0.0001). Esto es lo que sirve predict.py (ENSEMBLE_ZONES).
    # `lowest_mae_model` se guarda solo como referencia descriptiva, NO como elección.
    lowest_mae_model = df_metrics["MAE"].idxmin()
    ENSEMBLE_ZONES = {"FI"}
    served_model = "Ensemble (diverse) + AR" if zone in ENSEMBLE_ZONES else "LightGBM"
    all_results = {
        "zone":             zone,
        "served_model":     served_model,        # el que sirve predict.py (DM test)
        "lowest_mae_model": lowest_mae_model,     # solo referencia (no es la elección)
        "metrics":          df_metrics.to_dict(),
        "error_stats":  {k: v.to_dict() if hasattr(v, "to_dict") else v
                         for k, v in error_stats.items()},
        "economic":     econ_value,
        "conformal":    conformal_stats,
        "advanced":     advanced,
        "wf_mae_mean":  float(wf_results["MAE"].mean()) if not wf_results.empty else None,
    }

    save_run_metadata(
        script_name=f"evaluate_{zone}",
        params={"zone": zone, "test_start": TEST_START, "test_end": TEST_END,
                "full_analysis": full_analysis},
        stats={
            "MAE_lgbm":      float(df_metrics.loc["LightGBM", "MAE"]),
            "MAE_lgbm_ar":   float(df_metrics.loc["LightGBM+AR_DA", "MAE"]) if "LightGBM+AR_DA" in df_metrics.index else None,
            "MAE_ensemble":  float(df_metrics.loc["Ensemble", "MAE"]) if "Ensemble" in df_metrics.index else None,
            "MAE_ensemble_ar": float(df_metrics.loc["Ensemble+AR_DA", "MAE"]) if "Ensemble+AR_DA" in df_metrics.index else None,
            "MAE_lear":      float(df_metrics.loc["LEAR", "MAE"]) if "LEAR" in df_metrics.index else None,
            "served_model":     served_model,
            "lowest_mae_model": lowest_mae_model,
            "economic":      {k: v.get("valor_incremental_anual_EUR")
                              for k, v in econ_value.items()},
        },
    )

    logger.info(f"\nOK Evaluación {zone} completada.")
    logger.info(f"  Plots en: {PLOTS_DIR}")
    return all_results


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación completa del modelo")
    parser.add_argument("--zone", default="DK1", choices=["DK1", "DK2", "FI"])
    parser.add_argument("--full", action="store_true",
                        help="Incluir análisis de errores detallado y plots")
    parser.add_argument("--both", action="store_true",
                        help="Evaluar DK1 y DK2")
    args = parser.parse_args()

    zones = ["DK1", "DK2"] if args.both else [args.zone]
    for zone in zones:
        evaluate_zone(zone=zone, full_analysis=args.full)