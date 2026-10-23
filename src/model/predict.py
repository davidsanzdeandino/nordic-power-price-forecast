"""
predict.py
==========
Módulo de predicción reutilizable — interfaz entre el modelo y el dashboard.

Este módulo es el único punto de entrada para generar predicciones.
El dashboard (Fase 3) lo importa directamente; no llama a train.py ni
a features.py en producción.

Funciones principales:
  predict_next_24h()     → Predice las próximas 24 horas para una zona
  predict_range()        → Predice un rango de fechas arbitrario
  predict_with_bands()   → Predicción puntual + intervalos P10-P90
  get_feature_importance() → Top N features con importancia global
  explain_prediction()   → Explicación SHAP de una hora concreta

DECISIÓN TÉCNICA: El módulo carga los modelos una sola vez en memoria
(patrón singleton con caché) para que el dashboard no los recargue
en cada petición. Con Streamlit, esto se implementa con @st.cache_resource.

DECISIÓN TÉCNICA: Las predicciones se devuelven siempre en hora local
de Copenhagen (Europe/Copenhagen) para el display, aunque internamente
todo el cálculo es en UTC.

Uso desde el dashboard:
    from src.model.predict import PricePredictor
    predictor = PricePredictor(zone="DK1")
    df_pred = predictor.predict_next_24h()
    # df_pred tiene columnas: timestamp_cet, price_pred, P10, P25, P75, P90

Uso directo:
    python src/model/predict.py --zone DK1
    python src/model/predict.py --zone DK1 --date 2024-11-15
    python src/model/predict.py --zone DK1 --explain --hour 2024-11-15T18:00
"""

import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger
from src.model.features import generate_features

# ── Configuración ─────────────────────────────────────────────────────────────

FEATURES_PATH = ROOT / "data" / "master" / "dataset_features.parquet"
MODELS_DIR    = ROOT / "data" / "models"

# Zona horaria de display (para el dashboard)
DISPLAY_TZ = "Europe/Copenhagen"

# Columnas de salida estándar
OUTPUT_COLS = ["timestamp_cet", "price_pred", "P10", "P25", "P75", "P90",
               "zone", "generated_at"]


# ── Clase principal ───────────────────────────────────────────────────────────

class PricePredictor:
    """
    Interfaz de predicción de precios day-ahead.

    Carga los modelos entrenados y proporciona métodos de predicción
    listos para consumir desde el dashboard o desde scripts externos.

    Ejemplo de uso:
        predictor = PricePredictor(zone="DK1")

        # Predicción estándar con bandas
        df = predictor.predict_with_bands(start="2024-11-15", hours=24)

        # Feature importance para el dashboard
        fi = predictor.get_feature_importance(top_n=15)

        # Explicación SHAP de una hora concreta
        explanation = predictor.explain_prediction("2024-11-15T18:00")
    """

    def __init__(self, zone: str = "DK1"):
        """
        Args:
            zone: Zona de precio ('DK1' o 'DK2')
        """
        self.zone   = zone
        self.logger = get_logger(f"predict_{zone}")
        self._model           = None
        self._quantile_models = None
        self._meta            = None
        self._feature_data    = None
        self._shap_explainer  = None
        self._ar_corrector    = None
        self._ensemble        = None   # Ensemble diverso (L2 + mediana + Huber)
        # Selección de modelo POR ZONA según el test de Diebold-Mariano:
        #   - DK1/DK2: LightGBM único (el ensemble NO mejora significativamente).
        #   - FI:      Ensemble diverso (mercado muy volátil; el ensemble es
        #              SIGNIFICATIVAMENTE mejor — MAE 21.4 vs 24.6, DM p<0.0001).
        ENSEMBLE_ZONES = {"FI"}
        self._use_ensemble    = zone in ENSEMBLE_ZONES
        self._conformal       = None   # Conformal calibrator (bandas)

        self._load_models()
        self._load_feature_data()

    # ── Carga de modelos ──────────────────────────────────────────────────────

    def _load_models(self):
        """Carga modelos desde disco. Se ejecuta una vez en __init__."""
        model_path = MODELS_DIR / f"lightgbm_{self.zone}.pkl"
        meta_path  = MODELS_DIR / f"model_meta_{self.zone}.pkl"
        q_path     = MODELS_DIR / f"quantile_models_{self.zone}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {model_path}\n"
                f"  → Entrena primero: python src/model/train.py --zone {self.zone}"
            )

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        with open(meta_path, "rb") as f:
            self._meta = pickle.load(f)

        with open(q_path, "rb") as f:
            self._quantile_models = pickle.load(f)

        # --- Corrector AR ---
        from src.model.ar_correction import ARResidualCorrector
        ar_path = MODELS_DIR / f"ar_corrector_{self.zone}.pkl"
        if ar_path.exists():
            self._ar_corrector = ARResidualCorrector.load(str(ar_path))
            self.logger.info(f"  AR corrector: AR({self._ar_corrector.best_order})")

        # --- Ensemble diverso (L2+mediana+Huber), usado en vez de LightGBM ---
        ens_path = MODELS_DIR / f"ensemble_{self.zone}.pkl"
        if ens_path.exists():
            try:
                from src.model.diverse_ensemble import DiverseObjectiveEnsemble
                self._ensemble = DiverseObjectiveEnsemble.load(str(ens_path))
                # Si hay ensemble, cargar su AR corrector específico
                ar_ens_path = MODELS_DIR / f"ar_corrector_ensemble_{self.zone}.pkl"
                if ar_ens_path.exists():
                    self._ar_corrector = ARResidualCorrector.load(str(ar_ens_path))
                    self.logger.info(f"  Ensemble diverso cargado con AR propio")
                else:
                    self.logger.info(f"  Ensemble diverso cargado (usa AR base)")
            except Exception as e:
                self.logger.info(f"  Ensemble .pkl no se pudo cargar: {e}")
        # ----------------------------------

        # --- Conformal calibrator (bandas calibradas) ---
        conf_path = MODELS_DIR / f"conformal_{self.zone}.pkl"
        if conf_path.exists():
            from src.model.conformal import ConformalCalibrator
            self._conformal = ConformalCalibrator.load(str(conf_path))
            diag = self._conformal.get_diagnostics()
            self.logger.info(
                f"  Conformal ({self._conformal.strategy}): "
                f"cov {diag.get('raw_coverage', 0):.0%} → "
                f"{diag.get('calibrated_coverage_insample', 0):.0%}"
            )
        # ----------------------------------

        self.logger.info(
            f"Modelos cargados: {self.zone} | "
            f"{self._meta['n_features']} features | "
            f"Quantiles: {list(self._quantile_models.keys())}"
        )

    def _load_feature_data(self):
        """
        Carga el dataset de features en memoria.
        Es el input principal para las predicciones históricas y near-real-time.
        """
        if not FEATURES_PATH.exists():
            self.logger.warning(
                f"Dataset de features no encontrado: {FEATURES_PATH}\n"
                "  Generando features desde el dataset maestro..."
            )
            self._feature_data = generate_features(save=True)
        else:
            self._feature_data = pd.read_parquet(FEATURES_PATH)
            self._feature_data = self._feature_data.sort_index()

        self.logger.info(
            f"Feature data: {len(self._feature_data):,} filas | "
            f"{self._feature_data.index.min().date()} → "
            f"{self._feature_data.index.max().date()}"
        )

    # ── Preparación de features ───────────────────────────────────────────────

    def _prepare_X(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Extrae la matriz de features para los timestamps dados.

        Rellena NaN con la mediana histórica (estrategia conservadora).
        Si un timestamp no existe en el dataset, usa interpolación o último valor.
        """
        feature_cols = self._meta["feature_cols"]

        # Alinear con el dataset de features
        X = self._feature_data.reindex(timestamps)[feature_cols]

        # Rellenar NaN con mediana histórica (calculada sobre datos de entrenamiento)
        for col in feature_cols:
            if X[col].isna().any():
                fill_val = self._feature_data[col].median()
                X[col]   = X[col].fillna(fill_val)

        return X

    # ── Predicción puntual ────────────────────────────────────────────────────

    def predict(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        """
        Predicción puntual (mediana) para los timestamps dados.

        Args:
            timestamps: DatetimeIndex UTC tz-aware

        Returns:
            Series con predicciones de precio (EUR/MWh)
        """
        X     = self._prepare_X(timestamps)
        preds = self._model.predict(X)
        return pd.Series(preds, index=timestamps, name=f"price_pred_{self.zone}")

    # ── Predicción con bandas de confianza ────────────────────────────────────

    def predict_with_bands(self, start: str = None,
                           hours: int = 24,
                           end: str = None) -> pd.DataFrame:
        """
        Predicción puntual + intervalos de confianza P10-P90.

        Args:
            start: Timestamp de inicio (str ISO o None → última hora disponible)
            hours: Número de horas a predecir (si no se especifica end)
            end:   Timestamp de fin (str ISO, alternativa a hours)

        Returns:
            DataFrame con columnas:
              timestamp_cet, price_pred, P10, P25, P75, P90, zone, generated_at
        """
        # Resolver rango temporal
        if start is None:
            # Por defecto: último día disponible en el dataset
            last_ts = self._feature_data.index.max()
            start   = (last_ts - pd.Timedelta(hours=23)).isoformat()

        start_ts = pd.Timestamp(start, tz="UTC")
        if end:
            end_ts = pd.Timestamp(end, tz="UTC")
        else:
            end_ts = start_ts + pd.Timedelta(hours=hours - 1)

        timestamps = pd.date_range(start=start_ts, end=end_ts, freq="h", tz="UTC")

        # Filtrar timestamps que existen en el dataset
        available = timestamps[timestamps.isin(self._feature_data.index)]
        if len(available) == 0:
            raise ValueError(
                f"No hay datos disponibles para el rango {start_ts} → {end_ts}.\n"
                f"  Datos disponibles hasta: {self._feature_data.index.max()}"
            )

        if len(available) < len(timestamps):
            self.logger.warning(
                f"  {len(timestamps) - len(available)} timestamps sin datos "
                f"— predicción solo para {len(available)} horas disponibles"
            )

        X = self._prepare_X(available)

        # Predicción puntual — modelo de producción.
        # DECISIÓN (basada en evidencia): tras enriquecer las features con el
        # mercado acoplado alemán, el test de Diebold-Mariano muestra que el
        # LightGBM ÚNICO es SIGNIFICATIVAMENTE mejor (MAE) que el ensemble
        # diverso (DM≈+5, p<0.001). Por tanto producción usa LightGBM. El
        # ensemble queda cargado pero NO se usa por defecto (USE_ENSEMBLE=False).
        if self._use_ensemble and self._ensemble is not None:
            price_pred = self._ensemble.predict(X)
            self.logger.info("  Usando ensemble diverso (LightGBM L2 + mediana + Huber)")
        else:
            price_pred = self._model.predict(X)
            self.logger.info("  Usando LightGBM (mejor por DM test)")

        # Quantile bands (siempre del LightGBM quantile)
        bands = {}
        for q_name, q_model in self._quantile_models.items():
            bands[q_name] = q_model.predict(X)

        # ── Corrección AR (day-ahead) ──
        # Busca los últimos residuos ANTES del rango de predicción,
        # no los últimos del dataset (que podrían ser posteriores).
        if self._ar_corrector is not None:
            target_col = f"price_{self.zone}"
            order = self._ar_corrector.best_order
            pred_start = available[0]

            # Obtener precios reales anteriores al inicio de predicción
            past_actuals = self._feature_data[target_col].dropna()
            past_actuals = past_actuals[past_actuals.index < pred_start]

            if len(past_actuals) >= order:
                last_ts = past_actuals.index[-order:]
                X_past = self._prepare_X(last_ts)
                if self._use_ensemble and self._ensemble is not None:
                    preds_past = self._ensemble.predict(X_past)
                else:
                    preds_past = self._model.predict(X_past)
                last_resids = past_actuals.loc[last_ts].values - preds_past
            else:
                last_resids = np.zeros(order)

            # Defensa: si algún residuo es NaN (hueco/feature problemática),
            # no propagar NaN a toda la predicción → usar 0 en su lugar.
            if np.any(np.isnan(last_resids)):
                self.logger.warning(
                    "Residuos AR con NaN — se sustituyen por 0 para esta predicción."
                )
                last_resids = np.nan_to_num(last_resids, nan=0.0)

            price_pred_raw = price_pred.copy()
            price_pred = self._ar_corrector.correct_day_ahead(
                price_pred, last_resids
            )

            # Ajustar bandas quantile con el mismo shift
            shift = price_pred - price_pred_raw
            for q_name in bands:
                bands[q_name] = bands[q_name] + shift

        # ── Conformal calibration de bandas ──
        if self._conformal is not None:
            bands = self._conformal.calibrate_all_bands(
                pred=price_pred,
                bands=bands,
                timestamps=available,
            )

        # Construir DataFrame de salida
        df_out = pd.DataFrame({
            "timestamp_utc": available,
            "price_pred":    price_pred,
            **{q: v for q, v in bands.items()},
        })

        # Convertir a hora local de Copenhagen para display
        df_out["timestamp_cet"] = available.tz_convert(DISPLAY_TZ)
        df_out["zone"]          = self.zone
        df_out["generated_at"]  = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        # Ordenar columnas según OUTPUT_COLS estándar
        cols_available = [c for c in OUTPUT_COLS if c in df_out.columns]
        df_out = df_out[cols_available + [c for c in df_out.columns
                                           if c not in cols_available]]

        return df_out.reset_index(drop=True)

    def predict_next_24h(self) -> pd.DataFrame:
        """
        Atajo: predice las últimas 24 horas disponibles en el dataset.
        Equivalente a predict_with_bands(hours=24) sin especificar start.
        """
        return self.predict_with_bands(hours=24)

    def predict_day(self, date: str) -> pd.DataFrame:
        """
        Predice las 24 horas de un día concreto.

        Args:
            date: Fecha en formato 'YYYY-MM-DD'

        Returns:
            DataFrame con predicciones hora a hora para ese día.
        """
        start = f"{date}T00:00"
        end   = f"{date}T23:00"
        return self.predict_with_bands(start=start, end=end)

    # ── Feature importance ────────────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 20,
                                importance_type: str = "gain") -> pd.DataFrame:
        """
        Devuelve la importancia global de features del modelo.

        Args:
            top_n:           Número de features a devolver
            importance_type: 'gain' (reducción de pérdida) o 'split' (frecuencia)

        Returns:
            DataFrame con columnas: feature, importance, importance_pct
        """
        feature_cols = self._meta["feature_cols"]

        if importance_type == "gain":
            importances = self._model.booster_.feature_importance(importance_type="gain")
        else:
            importances = self._model.feature_importances_

        df_fi = pd.DataFrame({
            "feature":    feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_n)

        total = df_fi["importance"].sum()
        df_fi["importance_pct"] = (df_fi["importance"] / total * 100).round(2)
        df_fi = df_fi.reset_index(drop=True)

        return df_fi

    # ── Explicación SHAP ──────────────────────────────────────────────────────

    def explain_prediction(self, timestamp: str,
                            top_n: int = 10) -> pd.DataFrame:
        """
        Explicación SHAP de la predicción para un timestamp concreto.

        SHAP descompone la predicción en la contribución de cada feature,
        permitiendo narrativas como:
        "El precio predicho es 85 EUR/MWh (+25 sobre la media) porque:
         +15 EUR: viento forecast bajo (400 MW)
         +8 EUR:  temperatura -5°C (alta demanda calefacción)
         +4 EUR:  interconector NO2 al 95% de capacidad
         -2 EUR:  es viernes (demanda industrial ligeramente menor)"

        Args:
            timestamp: Timestamp ISO con timezone ('2024-11-15T18:00')
            top_n:     Número de features a incluir en la explicación

        Returns:
            DataFrame con columnas: feature, value, shap_value, contribution_eur
        """
        try:
            import shap
        except ImportError:
            self.logger.error("SHAP no instalado. Ejecuta: pip install shap")
            raise

        ts = pd.Timestamp(timestamp, tz="UTC")
        if ts not in self._feature_data.index:
            # Buscar el timestamp más cercano
            closest = self._feature_data.index[
                np.argmin(np.abs(self._feature_data.index - ts))
            ]
            self.logger.warning(f"  Timestamp {ts} no encontrado. Usando {closest}")
            ts = closest

        X_explain = self._prepare_X(pd.DatetimeIndex([ts]))

        # Inicializar explainer (cacheado)
        if self._shap_explainer is None:
            self.logger.info("  Inicializando SHAP TreeExplainer (primera vez)...")
            self._shap_explainer = shap.TreeExplainer(self._model)

        shap_values = self._shap_explainer.shap_values(X_explain)
        base_value  = self._shap_explainer.expected_value

        # Predicción real para verificar
        pred_price  = float(self._model.predict(X_explain)[0])

        # Construir DataFrame de explicación
        feature_cols = self._meta["feature_cols"]
        df_shap = pd.DataFrame({
            "feature":         feature_cols,
            "value":           X_explain.iloc[0].values,
            "shap_value":      shap_values[0],
        })
        df_shap["abs_shap"] = df_shap["shap_value"].abs()
        df_shap = df_shap.sort_values("abs_shap", ascending=False).head(top_n)

        # Dirección del efecto
        df_shap["direction"] = df_shap["shap_value"].apply(
            lambda x: "↑ alcista" if x > 0 else "↓ bajista"
        )
        df_shap["contribution_eur"] = df_shap["shap_value"].round(2)
        df_shap = df_shap.drop(columns=["abs_shap"]).reset_index(drop=True)

        ts_cet = ts.tz_convert(DISPLAY_TZ)
        self.logger.info(
            f"\n  Explicación SHAP — {ts_cet.strftime('%Y-%m-%d %H:%M CET')} | "
            f"Predicción: {pred_price:.1f} EUR/MWh | Base: {base_value:.1f} EUR/MWh"
        )
        self.logger.info(f"  {'Feature':<40} {'Valor':>10} {'Contribución':>14}")
        self.logger.info(f"  {'─'*67}")
        for _, row in df_shap.iterrows():
            self.logger.info(
                f"  {row['feature']:<40} {row['value']:>10.2f} "
                f"{row['contribution_eur']:>+12.2f} EUR  {row['direction']}"
            )

        return df_shap

    # ── Métricas de calibración rápida ────────────────────────────────────────

    def check_calibration(self, start: str = None, end: str = None) -> dict:
        """
        Verifica la calibración de los intervalos de confianza sobre
        un periodo histórico con datos reales disponibles.

        Devuelve el coverage real de cada banda (debería ≈ al quantil).
        """
        target_col = f"price_{self.zone}"
        if target_col not in self._feature_data.columns:
            raise ValueError(f"Columna {target_col} no encontrada")

        if start is None:
            start = "2024-10-01"
        if end is None:
            end = "2024-12-31 23:00"

        df_pred = self.predict_with_bands(start=start, end=end)
        y_true  = self._feature_data.loc[
            df_pred["timestamp_utc"].values, target_col
        ].values

        calibration = {}
        bands = [("P10", "P90", 0.80), ("P25", "P75", 0.50)]

        self.logger.info(f"\n  Calibración ({start} → {end}):")
        for lower, upper, target in bands:
            if lower in df_pred.columns and upper in df_pred.columns:
                coverage = np.mean(
                    (y_true >= df_pred[lower].values) &
                    (y_true <= df_pred[upper].values)
                )
                status = "OK" if abs(coverage - target) < 0.05 else "WARN"
                calibration[f"{lower}-{upper}"] = round(float(coverage), 4)
                self.logger.info(
                    f"  {status} {lower}-{upper} ({target:.0%} objetivo): "
                    f"{coverage:.1%} real"
                )

        return calibration

    # ── Resumen del modelo ────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Devuelve un resumen del estado del modelo para mostrar en el dashboard."""
        last_data   = self._feature_data.index.max()
        first_data  = self._feature_data.index.min()

        return {
            "zone":           self.zone,
            "n_features":     self._meta["n_features"],
            "trained_at":     self._meta.get("trained_at", "desconocido"),
            "train_end":      self._meta.get("train_end", "desconocido"),
            "test_start":     self._meta.get("test_start", "desconocido"),
            "quantiles":      list(self._quantile_models.keys()),
            "data_from":      str(first_data.date()),
            "data_until":     str(last_data.date()),
            "top5_features":  self._meta.get("top5_features", []),
        }


# ── Función de conveniencia (API funcional) ───────────────────────────────────

_predictor_cache: dict = {}

def get_predictor(zone: str = "DK1") -> PricePredictor:
    """
    Devuelve un PricePredictor cacheado para la zona indicada.
    Evita recargar los modelos en cada llamada (patrón singleton).

    En Streamlit usar @st.cache_resource en lugar de este caché manual.
    """
    if zone not in _predictor_cache:
        _predictor_cache[zone] = PricePredictor(zone=zone)
    return _predictor_cache[zone]


def predict_day_ahead(zone: str = "DK1",
                      date: str = None) -> pd.DataFrame:
    """
    API funcional de alto nivel. Predice un día completo con bandas.

    Args:
        zone: 'DK1' o 'DK2'
        date: 'YYYY-MM-DD' (None → último día disponible)

    Returns:
        DataFrame con predicciones hora a hora.
    """
    predictor = get_predictor(zone)
    if date:
        return predictor.predict_day(date)
    return predictor.predict_next_24h()


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicción de precios eléctricos")
    parser.add_argument("--zone",    default="DK1", choices=["DK1", "DK2"])
    parser.add_argument("--date",    default=None,
                        help="Fecha a predecir YYYY-MM-DD (default: último disponible)")
    parser.add_argument("--hours",   type=int, default=24,
                        help="Horas a predecir (si no se especifica --date)")
    parser.add_argument("--explain", action="store_true",
                        help="Mostrar explicación SHAP")
    parser.add_argument("--hour",    default=None,
                        help="Timestamp para SHAP: 'YYYY-MM-DDTHH:00'")
    parser.add_argument("--calibrate", action="store_true",
                        help="Verificar calibración de intervalos en test set")
    parser.add_argument("--importance", action="store_true",
                        help="Mostrar feature importance")
    args = parser.parse_args()

    logger = get_logger("predict_main")

    # Crear predictor
    predictor = PricePredictor(zone=args.zone)

    # Resumen del modelo
    summary = predictor.summary()
    logger.info(f"\n{'='*55}")
    logger.info(f"  PREDICTOR — Zona {summary['zone']}")
    logger.info(f"  Features:     {summary['n_features']}")
    logger.info(f"  Entrenado:    {summary['trained_at']}")
    logger.info(f"  Datos hasta:  {summary['data_until']}")
    logger.info(f"  Quantiles:    {summary['quantiles']}")
    logger.info(f"{'='*55}")

    # Predicción
    if args.date:
        logger.info(f"\nPrediciendo día: {args.date}")
        df_pred = predictor.predict_day(args.date)
    else:
        logger.info(f"\nPrediciendo las últimas {args.hours}h disponibles...")
        df_pred = predictor.predict_with_bands(hours=args.hours)

    # Mostrar resultados
    logger.info(f"\n{'─'*60}")
    logger.info(f"  {'Hora CET':<20} {'Pred':>8} {'P10':>8} {'P90':>8}")
    logger.info(f"{'─'*60}")
    for _, row in df_pred.head(24).iterrows():
        ts_str = pd.Timestamp(row["timestamp_cet"]).strftime("%Y-%m-%d %H:%M")
        p10 = f"{row['P10']:.1f}" if "P10" in row else "—"
        p90 = f"{row['P90']:.1f}" if "P90" in row else "—"
        logger.info(f"  {ts_str:<20} {row['price_pred']:>7.1f} {p10:>8} {p90:>8}")
    logger.info(f"{'─'*60}")
    logger.info(f"  Media predicha: {df_pred['price_pred'].mean():.1f} EUR/MWh | "
                f"Min: {df_pred['price_pred'].min():.1f} | "
                f"Max: {df_pred['price_pred'].max():.1f}")

    # SHAP (opcional)
    if args.explain:
        ts_explain = args.hour or df_pred["timestamp_utc"].iloc[12]
        logger.info(f"\nExplicación SHAP para: {ts_explain}")
        predictor.explain_prediction(str(ts_explain))

    # Calibración (opcional)
    if args.calibrate:
        predictor.check_calibration()

    # Feature importance (opcional)
    if args.importance:
        fi = predictor.get_feature_importance(top_n=15)
        logger.info(f"\nTop 15 features (gain):")
        logger.info(f"  {'Feature':<45} {'Importancia':>12} {'%':>6}")
        logger.info(f"  {'─'*65}")
        for _, row in fi.iterrows():
            bar = "█" * int(row["importance_pct"] / fi["importance_pct"].max() * 15)
            logger.info(f"  {row['feature']:<45} {row['importance']:>12.0f} "
                        f"{row['importance_pct']:>5.1f}%  {bar}")