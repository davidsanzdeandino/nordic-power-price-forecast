"""
baselines.py
============
Tres modelos baseline para comparación con LightGBM — Fase 2.

DECISIÓN TÉCNICA: Los baselines son imprescindibles en cualquier proyecto
de forecasting serio. Sin ellos no puedes saber si tu modelo "bueno"
realmente aporta valor o si una estrategia simple lo iguala.
Un modelo que no bate la persistence naive no justifica su complejidad.

Baselines implementados:
  1. Persistence Naive    — predice que mañana = hoy (mismo slot)
  2. Weekly Seasonal Mean — promedio del mismo slot de las últimas N semanas
  3. LEAR (Linear Estimated ARX) — regresión lineal con lags + features básicas
                                   Es el baseline "serio" de la literatura académica
                                   de predicción de precios eléctricos nórdicos.

Referencias:
  - Lago et al. (2021) "Forecasting day-ahead electricity prices: A review"
    → LEAR es el benchmark estándar en papers de Nord Pool
  - Uniejewski et al. (2019) "Variance Stabilizing Transformations for EPF"
    → Transformación log para estabilizar varianza en precios volátiles

Uso:
    from src.model.baselines import PersistenceBaseline, WeeklyMeanBaseline, LEARBaseline
    # O ejecutar directamente para ver métricas sobre los datos disponibles:
    python src/model/baselines.py
"""

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger

# ── Configuración ─────────────────────────────────────────────────────────────

FEATURES_PATH = ROOT / "data" / "master" / "dataset_features.parquet"
MASTER_PATH   = ROOT / "data" / "master" / "dataset_master.parquet"

TARGET_COLS = {
    "DK1": "price_DK1",
    "DK2": "price_DK2",
    "FI":  "price_FI",
}

# Inicio del test (enero-mayo de 2026). Los baselines se ajustan con todo lo
# anterior (train + validación) y se puntúan desde aquí.
TEST_START = "2026-01-01"

# Fichero publicado que escriben los pasos 15-17 (una zona por paso).
RESULTS_PATH = ROOT / "results" / "baselines_test2026.json"

# Nombre de cada baseline en el JSON publicado, en el orden de la tabla.
RESULT_KEYS = {"Persistence_24h": "persistence_24h",
               "WeeklyMean_4w": "weekly_mean_4w",
               "LEAR": "lear"}


# ── 1. Persistence Naive ──────────────────────────────────────────────────────

class PersistenceBaseline:
    """
    Predicción = precio del mismo slot de hace 24 horas (lag-24h).

    Es el baseline más simple posible y sorprendentemente competitivo
    en mercados eléctricos con alta autocorrelación diaria.
    MAE típico en Nord Pool: 12-20 EUR/MWh.

    DECISIÓN: lag=24h en lugar de lag=168h (semanal) porque la
    autocorrelación a 24h es sistemáticamente más alta en Nord Pool.
    """

    name = "Persistence_24h"

    def __init__(self, lag: int = 24):
        self.lag = lag

    def predict(self, series: pd.Series) -> pd.Series:
        """
        Args:
            series: Serie temporal de precios con DatetimeIndex UTC.
        Returns:
            Serie de predicciones (mismo índice, NaN en las primeras `lag` filas).
        """
        return series.shift(self.lag).rename(f"pred_{self.name}")

    def evaluate(self, series: pd.Series, test_start: str) -> dict:
        preds  = self.predict(series)
        actual = series

        mask = actual.index >= pd.Timestamp(test_start, tz="UTC")
        y_true = actual[mask].dropna()
        y_pred = preds[mask].reindex(y_true.index).dropna()
        y_true = y_true.reindex(y_pred.index)

        return _compute_metrics(y_true, y_pred, self.name)


# ── 2. Weekly Seasonal Mean ───────────────────────────────────────────────────

class WeeklyMeanBaseline:
    """
    Predicción = promedio del mismo slot horario de las últimas N semanas.

    Captura la estacionalidad semanal mejor que la persistence pura.
    Suaviza los outliers al promediar varias semanas.
    MAE típico en Nord Pool: 10-16 EUR/MWh.

    DECISIÓN: n_weeks=4 (últimas 4 semanas) como balance entre
    reactividad a cambios recientes y suavizado de ruido.
    """

    name = "WeeklyMean_4w"

    def __init__(self, n_weeks: int = 4):
        self.n_weeks = n_weeks

    def predict(self, series: pd.Series) -> pd.Series:
        """
        Para cada timestamp, predice como la media del mismo slot
        en las n_weeks semanas anteriores.
        """
        preds = pd.Series(np.nan, index=series.index, name=f"pred_{self.name}")

        # Offset de 1 semana = 168 horas
        week_hours = 168

        for i in range(self.n_weeks * week_hours, len(series)):
            past_values = []
            for w in range(1, self.n_weeks + 1):
                idx = i - w * week_hours
                if idx >= 0:
                    past_values.append(series.iloc[idx])
            if past_values:
                preds.iloc[i] = np.mean(past_values)

        return preds

    def evaluate(self, series: pd.Series, test_start: str) -> dict:
        preds  = self.predict(series)
        mask   = series.index >= pd.Timestamp(test_start, tz="UTC")
        y_true = series[mask].dropna()
        y_pred = preds[mask].reindex(y_true.index).dropna()
        y_true = y_true.reindex(y_pred.index)

        return _compute_metrics(y_true, y_pred, self.name)


# ── 3. LEAR (Linear Estimated ARX) ───────────────────────────────────────────

class LEARBaseline:
    """
    Linear Estimated ARX — regresión lineal Ridge con:
      - Lags de precio: 24h, 48h, 168h, 169h, 170h, 171h, 172h, 173h, 174h
      - Lags del mismo slot de las últimas 2 semanas
      - Features básicas de calendario: hora, día semana, mes (sin/cos)
      - Temperatura media de la zona (si disponible)
      - Wind forecast total (si disponible)

    Es el benchmark estándar de la literatura académica de EPF (Electricity
    Price Forecasting) para mercados nórdicos. Papers de Lago (2021) y
    Uniejewski (2019) lo usan como referencia obligatoria.

    MAE típico en Nord Pool: 7-12 EUR/MWh con buen feature set.
    Si LightGBM no supera LEAR de forma consistente, hay un problema
    en el feature engineering o en el entrenamiento del modelo.

    DECISIÓN: Ridge (L2) en lugar de OLS para manejar multicolinealidad
    entre los múltiples lags de precio. Alpha=0.1 es un punto de partida
    razonable — se puede ajustar si el modelo sobreajusta.
    """

    name = "LEAR"

    # Lags estándar de LEAR según Lago et al. (2021)
    # Incluye 24h + 7 horas adicionales alrededor del día anterior
    # + 168h (semana anterior) + 169-174h (semana anterior ± rango)
    PRICE_LAGS = [24, 48, 168, 169, 170, 171, 172, 173, 174]

    def __init__(self, alpha: float = 0.1, zone: str = "DK1"):
        self.alpha    = alpha
        self.zone     = zone   # exógenas (viento/temp) de la PROPIA zona, no hardcode DK1
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=self.alpha)),
        ])
        self._is_fitted = False

    def _build_features(self, series: pd.Series,
                        df_extra: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Construye la matriz de features X a partir de la serie de precios
        y opcionalmente de columnas extra del dataset maestro.
        """
        X = pd.DataFrame(index=series.index)

        # Lags de precio
        for lag in self.PRICE_LAGS:
            X[f"price_lag{lag}"] = series.shift(lag)

        # Features temporales cíclicas
        idx = pd.to_datetime(series.index, utc=True)
        X["hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
        X["hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
        X["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
        X["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
        X["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        X["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        X["is_weekend"] = (idx.dayofweek >= 5).astype(int)

        # Features extra del dataset maestro (meteo, viento) si están disponibles.
        # IMPORTANTE: usar las exógenas de la PROPIA zona (self.zone), no DK1 fijo —
        # de lo contrario el LEAR de FI/DK2 usaría viento/temperatura de DK1 (zona
        # equivocada), distorsionando la comparación honesta vs el modelo.
        if df_extra is not None:
            z = self.zone
            # Temperatura media de la zona (proxy demanda calefacción)
            temp_cols = [c for c in df_extra.columns if "wx_" in c and "temp_avg" in c
                         and f"_{z}_" in c]
            if temp_cols:
                X["temperature"] = df_extra[temp_cols[0]].reindex(X.index)

            # Viento forecast total de la zona (el predictor exógeno más importante)
            wind_cols = [c for c in df_extra.columns if f"feat_{z}_wind_total_fc" in c]
            if not wind_cols:
                # Fallback a columnas de forecast ENTSO-E de la zona
                wind_cols = [c for c in df_extra.columns
                             if f"forecast_{z}_" in c and "Wind" in c]
            if wind_cols:
                wind_total = df_extra[wind_cols].sum(axis=1).reindex(X.index)
                X["wind_forecast"] = wind_total
            else:
                import logging
                logging.getLogger("baselines").warning(
                    f"LEAR {z}: sin columna de viento forecast — el baseline corre sin "
                    f"viento y puede quedar artificialmente débil. Verifica nombres de columna.")

        return X

    def fit(self, series: pd.Series, test_start: str,
            df_extra: pd.DataFrame | None = None) -> "LEARBaseline":
        """
        Entrena el modelo LEAR con datos anteriores a test_start.

        Args:
            series:     Serie temporal de precios con DatetimeIndex UTC.
            test_start: Fecha de inicio del periodo de test (string).
            df_extra:   DataFrame maestro con features adicionales (opcional).
        """
        X = self._build_features(series, df_extra)
        y = series.copy()

        # Split temporal estricto
        cutoff = pd.Timestamp(test_start, tz="UTC")
        mask_train = series.index < cutoff

        X_train = X[mask_train].dropna()
        y_train = y[mask_train].reindex(X_train.index)

        self.pipeline.fit(X_train, y_train)
        self._is_fitted = True
        self._feature_names = X_train.columns.tolist()

        return self

    def predict(self, series: pd.Series, test_start: str,
                df_extra: pd.DataFrame | None = None) -> pd.Series:
        """
        Genera predicciones para el periodo de test.

        NOTA: En una implementación productiva, el modelo se reentrenaría
        rolling cada día. Aquí usamos el modelo entrenado en train para
        simplicidad — es suficiente para comparación con LightGBM.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo no entrenado. Llama a fit() primero.")

        X = self._build_features(series, df_extra)
        cutoff = pd.Timestamp(test_start, tz="UTC")
        mask_test = series.index >= cutoff

        X_test = X[mask_test].dropna()
        preds  = self.pipeline.predict(X_test)

        return pd.Series(preds, index=X_test.index, name=f"pred_{self.name}")

    def evaluate(self, series: pd.Series, test_start: str,
                 df_extra: pd.DataFrame | None = None) -> dict:
        self.fit(series, test_start, df_extra)
        preds  = self.predict(series, test_start, df_extra)
        y_true = series.reindex(preds.index)

        return _compute_metrics(y_true, preds, self.name)

    def get_coefficients(self) -> pd.Series | None:
        """Devuelve los coeficientes del modelo (útil para interpretabilidad)."""
        if not self._is_fitted:
            return None
        coefs = self.pipeline.named_steps["ridge"].coef_
        return pd.Series(coefs, index=self._feature_names).sort_values(key=abs, ascending=False)


# ── Métricas compartidas ──────────────────────────────────────────────────────

def _compute_metrics(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> dict:
    """
    Calcula el conjunto completo de métricas técnicas.

    Métricas incluidas:
      - MAE   → métrica principal (EUR/MWh, intuitiva para el usuario)
      - RMSE  → penaliza más los errores grandes (sensible a picos)
      - sMAPE → error porcentual simétrico (evita división por cero en precios bajos)
      - Direction Accuracy → % de horas donde se acierta la dirección del movimiento
      - R²    → proporción de varianza explicada
    """
    errors = y_true.values - y_pred.values

    mae   = np.mean(np.abs(errors))
    rmse  = np.sqrt(np.mean(errors ** 2))

    # sMAPE: versión simétrica de MAPE, más estable con precios cercanos a 0
    denom  = (np.abs(y_true.values) + np.abs(y_pred.values)) / 2
    smape  = np.mean(np.where(denom > 0.1, np.abs(errors) / denom * 100, 0))

    # Direction Accuracy: ¿el modelo acierta si el precio sube o baja?
    actual_dir = np.sign(np.diff(y_true.values))
    pred_dir   = np.sign(np.diff(y_pred.values))
    dir_acc    = np.mean(actual_dir == pred_dir) * 100

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true.values - np.mean(y_true.values)) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "model":       model_name,
        "n_obs":       len(y_true),
        "MAE":         round(mae,   3),
        "RMSE":        round(rmse,  3),
        "sMAPE":       round(smape, 3),
        "DirAcc":      round(dir_acc, 2),
        "R2":          round(r2,    4),
    }


def run_all_baselines(zone: str = "DK1", write: bool = True,
                      out_path: Path | None = None) -> pd.DataFrame:
    """
    Ejecuta los 3 baselines para una zona y devuelve tabla comparativa.

    Args:
        zone:     Zona de precio a evaluar ('DK1', 'DK2' o 'FI')
        write:    Si True, escribe la zona en results/baselines_test2026.json
                  (o en out_path) sin tocar las demás zonas.
        out_path: Ruta alternativa del JSON (tests).

    Returns:
        DataFrame con métricas de los 3 baselines.
    """
    logger = get_logger("baselines")
    logger.info(f"=== BASELINES — Zona {zone} | Test desde {TEST_START} ===")

    # Cargar datos
    path = FEATURES_PATH if FEATURES_PATH.exists() else MASTER_PATH
    if not path.exists():
        logger.error(f"No se encontró dataset en {path}")
        raise FileNotFoundError(str(path))

    df = pd.read_parquet(path)
    target_col = TARGET_COLS.get(zone)
    if target_col not in df.columns:
        raise ValueError(f"Columna {target_col} no encontrada. Columnas disponibles: "
                         f"{[c for c in df.columns if 'price' in c.lower()]}")

    series = df[target_col].dropna()
    logger.info(f"Serie: {len(series):,} observaciones | "
                f"{series.index.min().date()} → {series.index.max().date()}")

    results = []

    # 1. Persistence
    logger.info("\n[1] Persistence Naive (lag=24h)...")
    p1 = PersistenceBaseline(lag=24)
    m1 = p1.evaluate(series, TEST_START)
    results.append(m1)
    logger.info(f"    MAE={m1['MAE']:.2f} | RMSE={m1['RMSE']:.2f} | "
                f"sMAPE={m1['sMAPE']:.1f}% | DirAcc={m1['DirAcc']:.1f}%")

    # 2. Weekly Mean
    logger.info("\n[2] Weekly Seasonal Mean (4 semanas)...")
    p2 = WeeklyMeanBaseline(n_weeks=4)
    m2 = p2.evaluate(series, TEST_START)
    results.append(m2)
    logger.info(f"    MAE={m2['MAE']:.2f} | RMSE={m2['RMSE']:.2f} | "
                f"sMAPE={m2['sMAPE']:.1f}% | DirAcc={m2['DirAcc']:.1f}%")

    # 3. LEAR
    logger.info("\n[3] LEAR (Linear ARX + Ridge)...")
    p3 = LEARBaseline(alpha=0.1, zone=zone)
    m3 = p3.evaluate(series, TEST_START, df_extra=df)
    results.append(m3)
    logger.info(f"    MAE={m3['MAE']:.2f} | RMSE={m3['RMSE']:.2f} | "
                f"sMAPE={m3['sMAPE']:.1f}% | DirAcc={m3['DirAcc']:.1f}%")

    # Mostrar coeficientes más importantes del LEAR
    coefs = p3.get_coefficients()
    if coefs is not None:
        logger.info(f"\n  Top 5 coeficientes LEAR:")
        for feat, val in coefs.head(5).items():
            logger.info(f"    {feat:35s}: {val:+.4f}")

    # Tabla resumen
    df_results = pd.DataFrame(results).set_index("model")
    logger.info(f"\n{'='*60}")
    logger.info(f"TABLA COMPARATIVA BASELINES — {zone}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{df_results.to_string()}")

    if write:
        path = write_baselines_json(zone, results, series, out_path)
        logger.info(f"\n  Escrito: {path.name} (zona {zone})")

    return df_results


def _zone_block(results: list, series: pd.Series) -> dict:
    """Bloque de una zona del JSON publicado a partir de las métricas calculadas."""
    test = series[series.index >= pd.Timestamp(TEST_START, tz="UTC")]
    by_model = {m["model"]: m for m in results}
    mae_persistence = by_model["Persistence_24h"]["MAE"]
    baselines = {}
    for model, key in RESULT_KEYS.items():
        m = by_model[model]
        baselines[key] = {
            "model": model,
            "n_obs": int(m["n_obs"]),
            "MAE": float(m["MAE"]),
            "RMSE": float(m["RMSE"]),
            "sMAPE_pct": float(m["sMAPE"]),
            "DirAcc_pct": float(m["DirAcc"]),
            "R2": float(m["R2"]),
            "rMAE_vs_persistence_24h": round(float(m["MAE"]) / float(mae_persistence), 3),
        }
    return {
        "test_period": {
            "start_utc": test.index.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_utc": test.index.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours": int(len(test)),
        },
        "series_period": {
            "start": series.index.min().strftime("%Y-%m-%d"),
            "end": series.index.max().strftime("%Y-%m-%d"),
            "hourly_observations": int(len(series)),
        },
        "baselines": baselines,
    }


def _zone_order(zone: str):
    """Orden de las zonas en el JSON: DK1, DK2, FI y después el resto por nombre."""
    order = list(TARGET_COLS)
    return (order.index(zone), "") if zone in order else (len(order), zone)


def write_baselines_json(zone: str, results: list, series: pd.Series,
                         out_path: Path | None = None) -> Path:
    """Escribe (o actualiza) results/baselines_test2026.json con la zona dada.

    Las demás zonas del fichero se conservan: cada paso 15-17 reescribe solo la
    suya, junto con la cabecera común.
    """
    from src.model.results_io import write_json

    path = Path(out_path) if out_path is not None else RESULTS_PATH
    old = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    zones = dict(old.get("zones", {}))
    zones[zone] = _zone_block(results, series)
    out = {
        "description": "Day-ahead price baselines on the out-of-sample test set "
                       "(January to May 2026, hourly, UTC).",
        "source": "Written by src/model/baselines.py (run_all_baselines), pipeline "
                  "steps 15-17, one zone per step.",
        "generated_on": datetime.now().strftime("%Y-%m-%d"),
        "method": "Persistence and the weekly mean use only past prices. LEAR is fitted "
                  f"once on the hours before the test start ({TEST_START}, train plus "
                  "validation) and uses the wind forecast and mean temperature of its own "
                  "zone from the feature table of step 14. Every test hour with a "
                  "prediction is scored.",
        "units": {
            "MAE": "EUR/MWh",
            "RMSE": "EUR/MWh",
            "sMAPE_pct": "%",
            "DirAcc_pct": "% of hours where the sign of the hour-on-hour change is right",
            "R2": "coefficient of determination",
        },
        "models": {
            "persistence_24h": "Persistence naive: the price of the same hour on the "
                               "previous day (lag 24 h).",
            "weekly_mean_4w": "Weekly seasonal mean: average of the same hour and weekday "
                              "over the previous 4 weeks.",
            "lear": "LEAR: linear ARX model with Ridge regularisation, using the wind and "
                    "temperature of its own zone (zone-aware).",
        },
        "notes": "rMAE_vs_persistence_24h is MAE / MAE of persistence_24h for the same "
                 "zone, rounded to 3 decimals.",
        "zones": {z: zones[z] for z in sorted(zones, key=_zone_order)},
    }
    return write_json(path, out, ndigits=None)


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluar baselines de forecasting")
    parser.add_argument("--zone", default="DK1", choices=["DK1", "DK2", "FI"],
                        help="Zona de precio a evaluar")
    args = parser.parse_args()

    results = run_all_baselines(zone=args.zone)
    print(f"\nResultados guardados en log. Zona evaluada: {args.zone}")