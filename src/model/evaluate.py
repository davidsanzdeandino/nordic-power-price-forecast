"""
evaluate.py
===========
Evaluación del modelo servido en el test (ene-may 2026) — Fase 2.

Módulos:
  1. Métricas técnicas        — fila «Served» (la ruta de predict_with_bands,
                                la misma del dashboard y de las métricas
                                servidas), comparadores en bruto (LightGBM,
                                ensemble) y los 3 baselines
  2. Análisis de errores      — por hora, estación, nivel de precio y viento
  3. Valor económico          — simulación EUR para 3 perfiles de usuario
  4. Bandas                   — cobertura del conformal ajustado en
                                validación, ACI con actualización diaria,
                                pinball y CRPS sobre las bandas servidas
  5. Diebold-Mariano (HAC)    — descriptivos: el test no decide nada
  6. Batería                  — LP diario con la previsión servida

El test no se usa para elegir, ajustar ni calibrar: el modelo servido, el AR y
el conformal salen de train.py (validación oct-dic 2025). Escribe
results/evaluation_test2026_<zona>.json.

Perfiles de usuario simulados:
  A. Consumidor industrial  — 5 MW de flexibilidad, desplaza consumo
                              de las 3 horas más caras del día a las 3 más baratas
  B. Trader day-ahead       — estrategia long/short basada en predicción de
                              si el precio estará por encima o debajo de la media
  C. Operador EV charging   — 2 MW de carga flexible, carga en las 4h más baratas

Uso:
    python src/model/evaluate.py --zone DK1
    python src/model/evaluate.py --all --full
"""

import sys
import pickle
import argparse
from datetime import datetime
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
from src.model.config import (
    TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END, DELIVERY_TZ, RESULTS_DIR,
)
from src.model import dm as _dm

# ── Configuración ─────────────────────────────────────────────────────────────

FEATURES_PATH = ROOT / "data" / "master" / "dataset_features.parquet"
MODELS_DIR    = ROOT / "data" / "models"
PLOTS_DIR     = ROOT / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros de simulación económica
HOURS_PER_YEAR    = 8760
INDUSTRIAL_MW     = 5.0     # MW de flexibilidad consumidor industrial
TRADER_VOLUME_MWH = 1.0     # MWh por hora operado por el trader
EV_MW             = 2.0     # MW de carga EV flexible
DAILY_FLEX_HOURS  = 4       # Horas de flexibilidad diaria (EV y consumidor)

# ACI: tasa de aprendizaje fija (no se ajustó) y objetivo de cobertura
ACI_GAMMA  = 0.03
ACI_TARGET = 0.80

# Mínimo de horas por día para incluirlo en la simulación económica.
MIN_HOURS_PER_DAY = 6


# ── Carga de modelos y datos ──────────────────────────────────────────────────

def load_model_and_data(zone: str, logger):
    """
    Carga el LightGBM de referencia, metadatos, cuantiles y datos de test.

    Returns:
        (model, quantile_models, meta, X_test, y_test, df_full)
    """
    model_path = MODELS_DIR / f"lightgbm_{zone}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            f"  → Entrena primero: python src/model/train.py --zone {zone}"
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / f"model_meta_{zone}.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(MODELS_DIR / f"quantile_models_{zone}.pkl", "rb") as f:
        quantile_models = pickle.load(f)

    logger.info(f"  Modelo: {model_path.name} | Features: {meta['n_features']}")

    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_index()

    feature_cols = meta["feature_cols"]
    target_col   = f"price_{zone}"

    # Rellenar NaN con las medianas de train guardadas por train.py
    fill = meta.get("fill_values")
    if fill is None:
        logger.warning("  model_meta sin fill_values: medianas de las filas <= TRAIN_END")
        train_rows = df.index <= pd.Timestamp(TRAIN_END, tz="UTC")
        fill = df.loc[train_rows, feature_cols].median().to_dict()
    df_model = df[[target_col] + feature_cols].copy()
    df_model[feature_cols] = df_model[feature_cols].fillna(pd.Series(fill))

    test_mask = (df_model.index >= pd.Timestamp(TEST_START, tz="UTC")) & \
                (df_model.index <= pd.Timestamp(TEST_END, tz="UTC"))

    X_test = df_model.loc[test_mask, feature_cols]
    y_test = df_model.loc[test_mask, target_col].dropna()
    X_test = X_test.reindex(y_test.index)

    logger.info(f"  Test set: {len(y_test):,} horas "
                f"({y_test.index.min().date()} → {y_test.index.max().date()})")

    return model, quantile_models, meta, X_test, y_test, df


# ── 1. Métricas técnicas completas ───────────────────────────────────────────

def diebold_mariano(errors_a, errors_b, loss: str = "mae", h: int = 24) -> tuple:
    """
    Diebold-Mariano con varianza HAC (Newey-West, h) y corrección HLN.
    Reexporta src.model.dm.diebold_mariano. DM < 0 ⇒ A mejor que B.
    """
    return _dm.diebold_mariano(errors_a, errors_b, loss=loss, h=h)


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
                        logger, served: pd.DataFrame = None) -> tuple:
    """
    Métricas en el test: fila «Served» (predict_with_bands, la ruta del
    dashboard), comparadores en bruto (LightGBM train+val, LightGBM train,
    ensemble) y baselines. El test no decide nada: todo es descriptivo.

    Args:
        served: salida de predict.served_predictions(zone) (si es None se
                calcula aquí).

    Returns:
        (tabla comparativa, dict de predicciones alineadas con y_test)
    """
    from src.model.predict import served_predictions

    logger.info("\n[1] Calculando métricas técnicas...")
    target_col = f"price_{zone}"
    series     = df[target_col].dropna()

    if served is None:
        served = served_predictions(zone, TEST_START, TEST_END, include_pre_conformal=True)
    common = y_test.index.intersection(served.index)
    if len(common) < len(y_test):
        logger.warning(f"  {len(y_test) - len(common)} horas del test sin predicción servida")
    y_c = y_test.loc[common]
    X_c = X_test.loc[common]
    served = served.loc[common]

    results = []
    preds_out = {"index": common, "y": y_c.to_numpy(dtype=float)}

    # ── Modelo servido (la cifra de cabecera) ──
    preds_out["served"] = served["price_pred"].to_numpy(dtype=float)
    results.append(_compute_metrics(y_c, served["price_pred"], "Served"))

    # ── Comparadores en bruto ──
    preds_lgbm = model.predict(X_c)
    preds_out["lgbm"] = np.asarray(preds_lgbm, dtype=float)
    results.append(_compute_metrics(y_c, pd.Series(preds_lgbm, index=common), "LightGBM"))

    lgb_train_path = MODELS_DIR / f"lightgbm_train_{zone}.pkl"
    if lgb_train_path.exists():
        with open(lgb_train_path, "rb") as f:
            lgbm_train = pickle.load(f)
        p = lgbm_train.predict(X_c)
        preds_out["lgbm_train"] = np.asarray(p, dtype=float)
        results.append(_compute_metrics(y_c, pd.Series(p, index=common), "LightGBM_train"))

    ens_path = MODELS_DIR / f"ensemble_{zone}.pkl"
    if ens_path.exists():
        from src.model.diverse_ensemble import DiverseObjectiveEnsemble
        ensemble = DiverseObjectiveEnsemble.load(str(ens_path))
        p = ensemble.predict(X_c)
        preds_out["ens"] = np.asarray(p, dtype=float)
        results.append(_compute_metrics(y_c, pd.Series(p, index=common), "Ensemble"))

    # ── Baselines (mismo test; persistencia y LEAR son los de results/baselines) ──
    p1 = PersistenceBaseline(lag=24)
    results.append(p1.evaluate(series, TEST_START))
    preds_out["persistence"] = series.shift(24).reindex(common).to_numpy(dtype=float)

    p2 = WeeklyMeanBaseline(n_weeks=4)
    results.append(p2.evaluate(series, TEST_START))

    p3 = LEARBaseline(alpha=0.1, zone=zone)   # viento y temperatura de la propia zona
    results.append(p3.evaluate(series, TEST_START, df_extra=df))
    lear_pred = p3.predict(series, TEST_START, df_extra=df)
    preds_out["lear"] = lear_pred.reindex(common).to_numpy(dtype=float)

    df_metrics = pd.DataFrame(results).set_index("model")

    for baseline in ["Persistence_24h", "WeeklyMean_4w", "LEAR"]:
        if baseline in df_metrics.index:
            improvement = (df_metrics.loc[baseline, "MAE"] -
                           df_metrics.loc["Served", "MAE"]) / \
                          df_metrics.loc[baseline, "MAE"] * 100
            logger.info(f"  Served vs {baseline}: MAE {improvement:+.1f}%")

    logger.info(f"\n  {'─'*65}")
    logger.info(f"  {'Modelo':<22} {'MAE':>8} {'RMSE':>8} {'sMAPE':>8} "
                f"{'DirAcc':>8} {'R²':>8}")
    logger.info(f"  {'─'*65}")
    for model_name, row in df_metrics.iterrows():
        marker = " ◄ servido" if model_name == "Served" else ""
        logger.info(f"  {model_name:<22} {row['MAE']:>8.3f} {row['RMSE']:>8.3f} "
                    f"{row['sMAPE']:>7.2f}% {row['DirAcc']:>7.1f}% "
                    f"{row['R2']:>8.4f}{marker}")
    logger.info(f"  {'─'*65}")

    preds_out["served_frame"] = served
    return df_metrics, preds_out


# ── 2. Análisis de errores ────────────────────────────────────────────────────

def analyze_errors(zone: str, preds: pd.Series,
                   y_test: pd.Series, df: pd.DataFrame, logger) -> dict:
    """
    Análisis sistemático de dónde y cuándo falla el modelo servido.

    Args:
        preds: previsión servida alineada con y_test.

    Segmentaciones:
      A. Error por hora del día     — ¿falla más en picos de mañana/tarde?
      B. Error por estación         — ¿peor en invierno (más volatilidad)?
      C. Error por nivel de precio  — ¿falla en extremos (<0, >100, >200)?
      D. Error por nivel de viento  — ¿peor cuando hay mucha/poca renovable?
      E. Autocorrelación de errores — ¿los errores están correlacionados?
    """
    logger.info("\n[2] Análisis de errores sistemáticos...")

    logger.info("  Análisis sobre la previsión servida (predict_with_bands)")
    preds = pd.Series(np.asarray(preds, dtype=float), index=y_test.index, name="pred")
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
        fig.suptitle(f"Análisis de Errores — modelo servido {zone} "
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

def simulate_economic_value(zone: str, preds: pd.Series,
                             y_test: pd.Series, logger) -> dict:
    """
    Cuantifica en EUR el beneficio de usar la previsión servida vs estrategia naive.

    DECISIÓN TÉCNICA: Las simulaciones asumen que el modelo predice el
    precio 24h antes (contexto day-ahead real). Se escalan los resultados
    del periodo de test a valor anual para que sean comparables.

    Los 3 perfiles cubren los tipos principales de consumidor en Dinamarca:
    consumidor industrial, trader, y operador de infraestructura EV.
    """
    logger.info("\n[3] Simulación de valor económico...")

    # Previsión servida (day-ahead): nada de correct_offline, que usa
    # residuos intradía que no se conocen en la subasta.
    preds = pd.Series(np.asarray(preds, dtype=float), index=y_test.index)
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

def _coverage(y, lo, hi) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def evaluate_zone(zone: str = "DK1", full_analysis: bool = True) -> dict:
    """
    Evaluación completa de una zona en el test. Todo lo que se mide aquí es
    descriptivo: el modelo servido, el AR y el conformal vienen de train.py.

    Escribe results/evaluation_test2026_<zona>.json.
    """
    from src.model.predict import served_predictions, served_model_label
    from src.model.conformal import adaptive_conformal_coverage, count_crossed_hours
    from src.model.results_io import write_json

    logger = get_logger(f"evaluate_{zone}")
    logger.info(f"=== EVALUACIÓN COMPLETA — Zona {zone} ===")

    model, quantile_models, meta, X_test, y_test, df = load_model_and_data(zone, logger)
    served_model = meta.get("served_model", "lightgbm")
    use_ar = bool(meta.get("use_ar", False))
    label = served_model_label(served_model, use_ar)
    logger.info(f"  Modelo servido (elegido en validación): {label}")

    # 1. Métricas técnicas (fila Served = predict_with_bands)
    served = served_predictions(zone, TEST_START, TEST_END, include_pre_conformal=True)
    df_metrics, preds = compute_all_metrics(zone, model, X_test, y_test, df, logger,
                                            served=served)
    idx = preds["index"]
    y_arr = preds["y"]
    served = preds["served_frame"]
    served_series = pd.Series(preds["served"], index=idx)
    y_c = pd.Series(y_arr, index=idx)

    # 2. Análisis de errores (sobre la previsión servida)
    error_stats = {}
    if full_analysis:
        error_stats = analyze_errors(zone, served_series, y_c, df, logger)

    # 3. Valor económico (previsión servida)
    econ_value = simulate_economic_value(zone, served_series, y_c, logger)

    # 4. Walk-forward (LightGBM, descriptivo)
    wf_path = MODELS_DIR / f"walkforward_{zone}.csv"
    wf_results = pd.read_csv(wf_path) if wf_path.exists() else pd.DataFrame()
    if not wf_results.empty:
        logger.info(f"\n[4] Walk-forward (LightGBM, {len(wf_results)} pliegues): "
                    f"MAE media={wf_results['MAE'].mean():.2f} | "
                    f"std={wf_results['MAE'].std():.2f} | "
                    f"min={wf_results['MAE'].min():.2f} | max={wf_results['MAE'].max():.2f}")

    # 5. Bandas servidas: conformal ajustado en validación, medido en TODO el test
    logger.info("\n[5] Bandas servidas (conformal ajustado en validación):")
    bands_pre = {q: served[f"{q}_preconf"].to_numpy(float)
                 for q in ("P10", "P25", "P50", "P75", "P90") if f"{q}_preconf" in served}
    bands_srv = {q: served[q].to_numpy(float)
                 for q in ("P10", "P25", "P50", "P75", "P90") if q in served}
    bands_unsorted = {q: served[f"{q}_presort"].to_numpy(float)
                      for q in ("P10", "P25", "P50", "P75", "P90") if f"{q}_presort" in served}
    conformal_stats = {
        "calibrated_on": f"validation quarter {VAL_START} to {VAL_END} (train.py)",
        "measured_on": "whole test window, served bands",
        "n_hours": int(len(y_arr)),
        "P10_P90_before_conformal_pct": 100 * _coverage(y_arr, bands_pre["P10"], bands_pre["P90"]),
        "P10_P90_served_pct": 100 * _coverage(y_arr, bands_srv["P10"], bands_srv["P90"]),
        "P25_P75_before_conformal_pct": 100 * _coverage(y_arr, bands_pre["P25"], bands_pre["P75"]),
        "P25_P75_served_pct": 100 * _coverage(y_arr, bands_srv["P25"], bands_srv["P75"]),
        "P10_P90_width_before_conformal": float(np.mean(bands_pre["P90"] - bands_pre["P10"])),
        "P10_P90_width_served": float(np.mean(bands_srv["P90"] - bands_srv["P10"])),
        "quantiles_sorted_per_hour": True,
        "hours_with_crossed_quantiles_before_conformal": count_crossed_hours(bands_pre),
        "hours_with_crossed_quantiles_before_sorting": count_crossed_hours(bands_unsorted),
        "hours_with_crossed_quantiles_served": count_crossed_hours(bands_srv),
    }
    if bands_unsorted:
        prob_unsorted = pinball_and_crps(y_arr, bands_unsorted)
        conformal_stats.update({
            "P10_P90_before_sorting_pct": 100 * _coverage(y_arr, bands_unsorted["P10"],
                                                          bands_unsorted["P90"]),
            "P25_P75_before_sorting_pct": 100 * _coverage(y_arr, bands_unsorted["P25"],
                                                          bands_unsorted["P75"]),
            "P10_P90_width_before_sorting": float(np.mean(bands_unsorted["P90"]
                                                          - bands_unsorted["P10"])),
            "pinball_loss_before_sorting": prob_unsorted["pinball_loss"],
        })
    logger.info(f"  Horas con cuantiles cruzados: antes de ordenar "
                f"{conformal_stats['hours_with_crossed_quantiles_before_sorting']} → servido "
                f"{conformal_stats['hours_with_crossed_quantiles_served']}")
    logger.info(f"  P10-P90: antes del conformal {conformal_stats['P10_P90_before_conformal_pct']:.1f}% "
                f"→ servido {conformal_stats['P10_P90_served_pct']:.1f}% (objetivo 80%)")
    logger.info(f"  P25-P75: antes del conformal {conformal_stats['P25_P75_before_conformal_pct']:.1f}% "
                f"→ servido {conformal_stats['P25_P75_served_pct']:.1f}% (objetivo 50%)")

    # 6. Métricas avanzadas (descriptivas en el test)
    logger.info("\n[6] Métricas avanzadas (descriptive, test)...")
    advanced = {}

    # (a) Diebold-Mariano HAC (h=24) + HLN
    dm_rows = {}
    pairs = [("served_vs_lightgbm", "served", "lgbm"),
             ("served_vs_lear", "served", "lear"),
             ("ensemble_vs_lightgbm", "ens", "lgbm")]
    for name, a, b in pairs:
        if a in preds and b in preds:
            ea, eb = y_arr - preds[a], y_arr - preds[b]
            ok = ~(np.isnan(ea) | np.isnan(eb))
            stat, p = diebold_mariano(ea[ok], eb[ok], loss="mae", h=24)
            dm_rows[name] = {"DM": stat, "p_value": p, "n": int(ok.sum()),
                             "better": (a if stat < 0 else b) if np.isfinite(stat) else None}
            logger.info(f"  DM {name}: {stat:+.2f} (p={p:.4f})")
    advanced["diebold_mariano"] = {
        "label": "descriptive, test",
        "method": "absolute loss, HAC variance (Newey-West, Bartlett weights, h=24), "
                  "Harvey-Leybourne-Newbold correction, Student t(n-1)",
        "convention": "DM < 0 means the first model is more accurate",
        **dm_rows,
    }

    # (b) Pinball y CRPS sobre las bandas servidas
    prob = pinball_and_crps(y_arr, bands_srv)
    advanced["probabilistic"] = {**prob, "bands": ("served (AR-shifted, conformal, sorted per hour)"
                                                   if use_ar
                                                   else "served (conformal, sorted per hour)")}
    logger.info(f"  Pinball: {prob['pinball_loss']:.3f} €/MWh | CRPS≈ {prob['crps_approx']:.3f}")

    # (c) ACI con actualización diaria sobre las bandas servidas antes del conformal
    day_labels = pd.DatetimeIndex(idx).tz_convert(DELIVERY_TZ).normalize()
    aci = adaptive_conformal_coverage(y_arr, bands_pre["P10"], bands_pre["P90"],
                                      target_coverage=ACI_TARGET, gamma=ACI_GAMMA,
                                      day_labels=day_labels)
    advanced["aci"] = {**aci, "bands": "served P10-P90 before the conformal step",
                       "k0": 1.0, "gamma_note": "fixed, not tuned"}
    logger.info(f"  ACI diario P10-P90: estático {aci['coverage_static']:.1%} → "
                f"ACI {aci['coverage_aci']:.1%} (objetivo 80%)")

    # (d) Backtest de batería con la previsión servida (y con la persistencia)
    try:
        from src.model.battery_backtest import battery_arbitrage_backtest
        df_bt = pd.DataFrame({"price_actual": y_arr, "price_pred": preds["served"],
                              "price_persistence": preds["persistence"]}, index=idx)
        bt = battery_arbitrage_backtest(df_bt, capacity_mwh=2.0, power_mw=1.0,
                                        efficiency=0.90, method="lp")
        bt_p = battery_arbitrage_backtest(df_bt, forecast_col="price_persistence",
                                          capacity_mwh=2.0, power_mw=1.0,
                                          efficiency=0.90, method="lp")
        advanced["battery"] = {**bt, "forecast_used": "served"}
        if bt_p:
            advanced["battery_persistence_forecast"] = {
                "profit_forecast_annual": bt_p["profit_forecast_annual"],
                "pct_of_oracle_forecast": bt_p["pct_of_oracle_forecast"],
                "n_days": bt_p["n_days"],
            }
        if bt:
            logger.info(f"  Batería LP (2MWh/1MW): servido {bt['profit_forecast_annual']:.0f} €/año "
                        f"({bt['pct_of_oracle_forecast']:.0f}% del oráculo) | "
                        f"horario fijo {bt['profit_naive_annual']:.0f} | "
                        f"oráculo {bt['profit_perfect_annual']:.0f}")
    except Exception as e:
        logger.info(f"  Backtest batería no disponible: {e}")

    test_prices = {"mean": float(np.mean(y_arr)), "std": float(np.std(y_arr, ddof=1)),
                   "n": int(len(y_arr))}
    bias = {name: float(np.mean(preds[name] - y_arr))
            for name in ("served", "lgbm", "lgbm_train", "ens") if name in preds}

    metrics_table = {m: {k: row[k] for k in ("n_obs", "MAE", "RMSE", "sMAPE", "DirAcc", "R2")}
                     for m, row in df_metrics.iterrows()}
    all_results = {
        "zone":          zone,
        "generated_by":  "src/model/evaluate.py (evaluate_zone)",
        "generated_on":  datetime.now().strftime("%Y-%m-%d"),
        "method":        "single run of the served forecast (predict_with_bands: model chosen on "
                         "validation, day-ahead AR per delivery day if selected, conformal bands "
                         "fitted on validation) over the whole test window; comparators, DM, "
                         "coverage, pinball/CRPS and the battery backtest come from this same run",
        "test_window_utc": {"start": TEST_START, "end": TEST_END},
        "served_model":  served_model,
        "use_ar":        use_ar,
        "served_label":  label,
        "selection_note": "served model, AR term and conformal calibration were fixed "
                          "on the validation quarter by src/model/train.py",
        "metrics":       metrics_table,
        "mean_forecast_minus_actual": bias,
        "test_prices":   test_prices,
        "error_stats":   error_stats if full_analysis else {},
        "economic":      econ_value,
        "conformal":     conformal_stats,
        "advanced":      advanced,
        "walk_forward_mae_mean": float(wf_results["MAE"].mean()) if not wf_results.empty else None,
    }
    out = write_json(RESULTS_DIR / f"evaluation_test2026_{zone}.json", all_results)

    save_run_metadata(
        script_name=f"evaluate_{zone}",
        params={"zone": zone, "test_start": TEST_START, "test_end": TEST_END,
                "full_analysis": full_analysis},
        stats={"MAE_served": float(df_metrics.loc["Served", "MAE"]),
               "served_label": label,
               "economic": {k: v.get("valor_incremental_anual_EUR")
                            for k, v in econ_value.items()}},
    )

    logger.info(f"\nOK Evaluación {zone} completada → results/{out.name}")
    return all_results


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación completa del modelo")
    parser.add_argument("--zone", default="DK1", choices=["DK1", "DK2", "FI"])
    parser.add_argument("--full", action="store_true",
                        help="Incluir análisis de errores detallado y plots")
    parser.add_argument("--all", action="store_true", help="Evaluar DK1, DK2 y FI")
    args = parser.parse_args()

    zones = ["DK1", "DK2", "FI"] if args.all else [args.zone]
    for zone in zones:
        evaluate_zone(zone=zone, full_analysis=args.full)
