"""
train.py
========
Entrenamiento del modelo predictivo — Fase 2.

Pipeline completo:
  1. Carga dataset de features (data/master/dataset_features.parquet)
  2. Split temporal estricto (train / val / test)
  3. Entrena LightGBM con hiperparámetros por defecto → benchmark inicial
  4. Optimiza hiperparámetros con Optuna (100 trials, validación bayesiana)
  5. Entrena modelo final con mejores hiperparámetros
  6. Walk-forward validation (expanding window, evaluación mensual)
  7. Quantile regression: P10, P25, P50, P75, P90 (bandas de confianza)
  8. Serializa modelos en data/models/

DECISIÓN TÉCNICA: Split temporal estricto, nunca aleatorio.
Un random split filtraría información futura al modelo (data leakage) y
daría métricas falsamente optimistas. El split temporal simula la realidad:
entrenas con datos históricos y predices el futuro.

  Train:      2023-01-01 → 2024-06-30  (~18 meses)
  Validation: 2024-07-01 → 2024-09-30  (3 meses — para Optuna)
  Test:       2024-10-01 → 2024-12-31  (3 meses — evaluación final)

Uso:
    python src/model/train.py               # Entrena DK1 (por defecto)
    python src/model/train.py --zone DK2    # Entrena DK2
    python src/model/train.py --zone DK1 --no-optuna  # Rápido, sin tuning
"""

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata, robust_pickle_dump

# ── Configuración ─────────────────────────────────────────────────────────────

FEATURES_PATH = ROOT / "data" / "master" / "dataset_features.parquet"
MODELS_DIR    = ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Splits temporales
TRAIN_END  = "2025-09-30 23:00"
VAL_START  = "2025-10-01"
VAL_END    = "2025-12-31 23:00"
TEST_START = "2026-01-01"

# Walk-forward: a partir de qué mes comienza la validación rolling
WF_START = "2024-06-01"

# Quantiles para predicción probabilística
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# Optuna
N_TRIALS        = 100
OPTUNA_TIMEOUT  = 3600   # segundos máximos (1h) — por si va lento

# LightGBM early stopping
EARLY_STOPPING_ROUNDS = 50


# ── Preparación de datos ──────────────────────────────────────────────────────

def load_and_prepare(zone: str, logger) -> tuple:
    """
    Carga el dataset de features y prepara X, y para la zona indicada.

    Returns:
        (df_full, X, y, feature_cols, splits)
        donde splits = dict con índices de train/val/test
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {FEATURES_PATH}\n"
            "  → Ejecuta primero: python src/model/features.py"
        )

    target_col = f"price_{zone}"
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_index()

    if target_col not in df.columns:
        raise ValueError(
            f"Columna objetivo '{target_col}' no encontrada.\n"
            f"  Columnas de precio disponibles: "
            f"{[c for c in df.columns if c.startswith('price_')]}"
        )

    # Feature columns: solo las que empiezan por 'feat_' + flags de calendario.
    # DECISIÓN: Excluir otros precios como features directas para evitar
    # data leakage entre zonas (el modelo no conocería price_NO2 en producción
    # porque se predice simultáneamente). Se incluyen los lags ya calculados.
    #
    # ANTI-LEAKAGE: los flags `is_congested_*` del master se calculan con el
    # spread de precio del MISMO timestamp (integration.py:step_derived_features),
    # que NO se conoce a las 12:00 del día anterior en la subasta day-ahead.
    # Se excluyen. Sus equivalentes causales ya existen como `feat_congested_*`
    # (spread con shift(24)). Solo se admiten flags de calendario deterministas.
    SAFE_IS_FLAGS = ("is_weekend", "is_holiday")
    leaky_flags = [c for c in df.columns
                   if c.startswith("is_") and not c.startswith(SAFE_IS_FLAGS)]
    if leaky_flags:
        logger.info(f"  Excluidas {len(leaky_flags)} flags con leakage temporal: {leaky_flags}")
    feature_cols = (
        [c for c in df.columns if c.startswith("feat_")] +
        [c for c in df.columns if c.startswith("is_") and c.startswith(SAFE_IS_FLAGS)]
    )

    # Eliminar filas donde el target es NaN
    df_model = df[[target_col] + feature_cols].dropna(subset=[target_col])

    # Eliminar filas donde TODAS las features son NaN (primeras horas por lags)
    min_feat_required = int(len(feature_cols) * 0.5)
    df_model = df_model.dropna(thresh=min_feat_required + 1)

    # Rellenar NaN residuales en features con mediana (conservador)
    for col in feature_cols:
        if df_model[col].isna().any():
            df_model[col] = df_model[col].fillna(df_model[col].median())

    y = df_model[target_col]
    X = df_model[feature_cols]

    # Splits temporales
    train_mask = X.index <= pd.Timestamp(TRAIN_END, tz="UTC")
    val_mask   = (X.index >= pd.Timestamp(VAL_START, tz="UTC")) & \
                 (X.index <= pd.Timestamp(VAL_END, tz="UTC"))
    test_mask  = X.index >= pd.Timestamp(TEST_START, tz="UTC")

    splits = {
        "train": train_mask,
        "val":   val_mask,
        "test":  test_mask,
    }

    logger.info(f"  Dataset:    {len(df_model):,} filas × {len(feature_cols)} features")
    logger.info(f"  Train:      {train_mask.sum():,} horas  "
                f"({X[train_mask].index.min().date()} → {X[train_mask].index.max().date()})")
    logger.info(f"  Validation: {val_mask.sum():,} horas  "
                f"({X[val_mask].index.min().date()} → {X[val_mask].index.max().date()})")
    logger.info(f"  Test:       {test_mask.sum():,} horas  "
                f"({X[test_mask].index.min().date()} → {X[test_mask].index.max().date()})")

    return df, X, y, feature_cols, splits


# ── Entrenamiento LightGBM base ───────────────────────────────────────────────

def train_lgbm_default(X: pd.DataFrame, y: pd.Series,
                       splits: dict, logger) -> object:
    """
    Entrena LightGBM con hiperparámetros por defecto.
    Sirve como benchmark antes del tuning con Optuna.
    """
    import lightgbm as lgb

    X_train, y_train = X[splits["train"]], y[splits["train"]]
    X_val,   y_val   = X[splits["val"]],   y[splits["val"]]

    params_default = {
        "objective":        "regression",
        "metric":           "mae",
        "n_estimators":     2000,
        "learning_rate":    0.05,
        "num_leaves":       63,
        "min_child_samples": 30,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }

    model = lgb.LGBMRegressor(**params_default)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    preds_val = model.predict(X_val)
    mae_val   = np.mean(np.abs(y_val.values - preds_val))
    logger.info(f"  LightGBM default → MAE val: {mae_val:.3f} EUR/MWh "
                f"| n_estimators efectivos: {model.best_iteration_}")

    return model, mae_val


# ── Optimización Optuna ───────────────────────────────────────────────────────

def optimize_with_optuna(X: pd.DataFrame, y: pd.Series,
                         splits: dict, n_trials: int, logger) -> dict:
    """
    Optimización bayesiana de hiperparámetros con Optuna.

    DECISIÓN TÉCNICA: Optuna usa TPE (Tree-structured Parzen Estimator)
    que es más eficiente que grid search o random search. Con 100 trials
    y early stopping por árbol, el tiempo total es ~30-60 minutos.

    DECISIÓN: Optimizar sobre MAE en el set de validación (2024-Q3).
    MAE es la métrica principal porque es intuitiva para el usuario
    (error en EUR/MWh) y no está sesgada por outliers como RMSE.
    """
    try:
        import optuna
        import lightgbm as lgb
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("Optuna no instalado. Ejecuta: pip install optuna")
        raise

    X_train, y_train = X[splits["train"]], y[splits["train"]]
    X_val,   y_val   = X[splits["val"]],   y[splits["val"]]

    def objective(trial):
        params = {
            "objective":         "regression",
            "metric":            "mae",
            "n_estimators":      2000,            # Con early stopping
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "max_depth":         trial.suggest_int("max_depth", 5, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        preds = model.predict(X_val)
        return np.mean(np.abs(y_val.values - preds))

    logger.info(f"  Ejecutando {n_trials} trials Optuna (timeout: {OPTUNA_TIMEOUT}s)...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=OPTUNA_TIMEOUT,
                   show_progress_bar=False)

    best_params = study.best_params
    best_mae    = study.best_value

    logger.info(f"  OK Optuna completado: MAE óptimo = {best_mae:.3f} EUR/MWh")
    logger.info(f"  Mejores hiperparámetros:")
    for k, v in best_params.items():
        logger.info(f"    {k:25s}: {v}")

    return best_params, best_mae


def train_lgbm_tuned(X: pd.DataFrame, y: pd.Series,
                     splits: dict, best_params: dict, logger) -> object:
    """
    Entrena el modelo final con los hiperparámetros optimizados.
    Entrena sobre train + val (todo lo anterior al test).
    """
    import lightgbm as lgb

    # Para el modelo final: entrenar con train+val juntos
    trainval_mask = splits["train"] | splits["val"]
    X_tv, y_tv    = X[trainval_mask], y[trainval_mask]
    X_val, y_val  = X[splits["val"]],  y[splits["val"]]

    final_params = {
        "objective":    "regression",
        "metric":       "mae",
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs":       -1,
        "verbose":      -1,
        **best_params,
    }

    model = lgb.LGBMRegressor(**final_params)
    model.fit(
        X_tv, y_tv,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    logger.info(f"  Modelo tuned: n_estimators efectivos = {model.best_iteration_}")
    return model


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_validation(X: pd.DataFrame, y: pd.Series,
                            best_params: dict, logger) -> pd.DataFrame:
    """
    Walk-forward validation con ventana expandible — evaluación mensual.
    Incluye corrección AR para mostrar el beneficio combinado.

    DECISIÓN TÉCNICA: Es más informativa que un único split train/test.
    Muestra cómo evoluciona el rendimiento del modelo a medida que
    tiene más datos de entrenamiento, y detecta si hay degradación
    temporal (el modelo empeora en periodos específicos del año).

    Esquema:
      Ene-Jun 2023 → predice Jul 2023
      Ene-Jul 2023 → predice Ago 2023
      ...
      Ene-Sep 2024 → predice Oct 2024
      ...
    """
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error
    from src.model.ar_correction import ARResidualCorrector

    logger.info("\n  Walk-forward validation (expanding window)...")

    wf_start_ts = pd.Timestamp(WF_START, tz="UTC")
    months      = pd.date_range(start=wf_start_ts, end=X.index.max(), freq="MS")

    results = []
    for predict_month_start in months:
        predict_month_end = predict_month_start + pd.offsets.MonthEnd(1)
        predict_month_end = predict_month_end.replace(hour=23, tzinfo=predict_month_start.tzinfo)

        # Train: todo antes del mes a predecir
        train_mask = X.index < predict_month_start
        pred_mask  = (X.index >= predict_month_start) & (X.index <= predict_month_end)

        if train_mask.sum() < 1000 or pred_mask.sum() == 0:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_pr, y_pr = X[pred_mask],  y[pred_mask]

        # Entrenamiento rápido (sin early stopping para speed)
        params = {
            "objective":    "regression",
            "metric":       "mae",
            "n_estimators": min(500, best_params.get("n_estimators", 500)),
            "random_state": 42,
            "n_jobs":       -1,
            "verbose":      -1,
            **{k: v for k, v in best_params.items() if k != "n_estimators"},
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_pr)
        mae   = mean_absolute_error(y_pr, preds)
        rmse  = np.sqrt(np.mean((y_pr.values - preds) ** 2))

        # ── AR correction sobre residuos del train ──
        preds_train = model.predict(X_tr)
        ar = ARResidualCorrector(max_order=3)
        ar.fit(y_tr.values, preds_train, verbose=False)
        # Day-ahead: solo usa residuos conocidos (fin del train), NO residuos futuros
        from src.model.ar_correction import simulate_day_ahead_correction
        da_metrics = simulate_day_ahead_correction(y_pr.values, preds, ar, verbose=False)
        mae_ar   = da_metrics["MAE_day_ahead_corrected"]
        rmse_ar  = np.sqrt(np.mean((y_pr.values[24:] - preds[24:]) ** 2))  # excluir 1er día sin resid

        results.append({
            "month":       predict_month_start.strftime("%Y-%m"),
            "n_train":     len(X_tr),
            "n_test":      len(X_pr),
            "MAE":         round(mae, 3),
            "MAE_ar":      round(mae_ar, 3),
            "RMSE":        round(rmse, 3),
            "RMSE_ar":     round(rmse_ar, 3),
            "AR_improve":  round((mae - mae_ar) / mae * 100, 1),
            "price_mean":  round(y_pr.mean(), 2),
            "price_std":   round(y_pr.std(), 2),
        })
        logger.info(f"    {predict_month_start.strftime('%Y-%m')}: "
                    f"MAE={mae:.2f} → AR={mae_ar:.2f} "
                    f"({(mae - mae_ar) / mae * 100:+.1f}%) | "
                    f"train={len(X_tr):,}h")

    df_wf = pd.DataFrame(results)
    if not df_wf.empty:
        logger.info(f"\n  Walk-forward MAE (raw): "
                    f"media={df_wf['MAE'].mean():.2f} | "
                    f"std={df_wf['MAE'].std():.2f} | "
                    f"max={df_wf['MAE'].max():.2f}")
        logger.info(f"  Walk-forward MAE (AR):  "
                    f"media={df_wf['MAE_ar'].mean():.2f} | "
                    f"std={df_wf['MAE_ar'].std():.2f} | "
                    f"max={df_wf['MAE_ar'].max():.2f}")
        logger.info(f"  Mejora AR media: {df_wf['AR_improve'].mean():.1f}%")
    return df_wf


# ── Quantile Regression ───────────────────────────────────────────────────────

def train_quantile_models(X: pd.DataFrame, y: pd.Series,
                          splits: dict, best_params: dict, logger) -> dict:
    """
    Entrena modelos de quantile regression para predicción probabilística.

    DECISIÓN TÉCNICA: Quantile regression en lugar de intervalos basados
    en residuos porque los residuos del modelo no son gaussianos — en
    precios eléctricos la distribución de errores tiene colas pesadas
    (fat tails) por los picos de precio. Los quantiles capturan esto
    correctamente.

    Modelos: P10, P25, P50, P75, P90
    → P10-P90: intervalo de confianza del 80%
    → P25-P75: intervalo de confianza del 50% (IQR)
    → P50: predicción central (mediana, más robusta que media en precios)
    """
    import lightgbm as lgb

    logger.info("\n  Entrenando modelos de quantile regression...")

    trainval_mask = splits["train"] | splits["val"]
    X_tv, y_tv    = X[trainval_mask], y[trainval_mask]
    X_val, y_val  = X[splits["val"]],  y[splits["val"]]

    quantile_models = {}

    for q in QUANTILES:
        params_q = {
            "objective":    "quantile",
            "alpha":        q,           # alpha = quantile a predecir
            "metric":       "quantile",
            "n_estimators": 1000,
            "random_state": 42,
            "n_jobs":       -1,
            "verbose":      -1,
            **{k: v for k, v in best_params.items()
               if k not in ["n_estimators"]},
        }

        model_q = lgb.LGBMRegressor(**params_q)
        model_q.fit(
            X_tv, y_tv,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        quantile_models[f"P{int(q*100):02d}"] = model_q
        logger.info(f"    P{int(q*100):02d} entrenado OK")

    # Verificar calibración sobre el set de test
    logger.info("\n  Verificando calibración de intervalos (test set)...")
    X_test = X[splits["test"]]
    y_test = y[splits["test"]]

    coverage_80 = _check_coverage(
        y_test.values,
        quantile_models["P10"].predict(X_test),
        quantile_models["P90"].predict(X_test),
        target_coverage=0.80,
        logger=logger,
        label="80% (P10-P90)"
    )
    coverage_50 = _check_coverage(
        y_test.values,
        quantile_models["P25"].predict(X_test),
        quantile_models["P75"].predict(X_test),
        target_coverage=0.50,
        logger=logger,
        label="50% (P25-P75)"
    )

    return quantile_models, {"coverage_80": coverage_80, "coverage_50": coverage_50}


def _check_coverage(y_true, lower, upper, target_coverage, logger, label) -> float:
    """Verifica que el intervalo contiene el valor real el % esperado de veces."""
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    status   = "OK" if abs(coverage - target_coverage) < 0.05 else "WARN"
    logger.info(f"    {status} Cobertura {label}: "
                f"{coverage:.1%} (objetivo: {target_coverage:.0%})")
    return float(coverage)


# ── Serialización ─────────────────────────────────────────────────────────────

def save_models(zone: str, model_main, quantile_models: dict,
                feature_cols: list, best_params: dict,
                wf_results: pd.DataFrame, logger):
    """
    Guarda todos los modelos y metadatos asociados en data/models/.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # Modelo principal (guardado robusto ante OneDrive/sync)
    main_path = MODELS_DIR / f"lightgbm_{zone}.pkl"
    robust_pickle_dump(model_main, main_path)
    logger.info(f"  Modelo principal: {main_path}")

    # Modelos quantile
    quantile_path = MODELS_DIR / f"quantile_models_{zone}.pkl"
    robust_pickle_dump(quantile_models, quantile_path)
    logger.info(f"  Modelos quantile: {quantile_path}")

    # Metadatos del modelo (feature names, hiperparámetros, splits)
    meta = {
        "zone":          zone,
        "trained_at":    ts,
        "feature_cols":  feature_cols,
        "best_params":   best_params,
        "train_end":     TRAIN_END,
        "val_start":     VAL_START,
        "val_end":       VAL_END,
        "test_start":    TEST_START,
        "quantiles":     QUANTILES,
        "n_features":    len(feature_cols),
        "has_ar":        True,   # Flag: corrección AR disponible
    }
    meta_path = MODELS_DIR / f"model_meta_{zone}.pkl"
    robust_pickle_dump(meta, meta_path)

    # Walk-forward results como CSV
    if not wf_results.empty:
        wf_path = MODELS_DIR / f"walkforward_{zone}.csv"
        wf_results.to_csv(wf_path, index=False)
        logger.info(f"  Walk-forward results: {wf_path}")

    logger.info(f"  Metadatos: {meta_path}")


# ── Función principal ─────────────────────────────────────────────────────────

def train_zone(zone: str = "DK1", run_optuna: bool = True,
               n_trials: int = N_TRIALS) -> dict:
    """
    Pipeline completo de entrenamiento para una zona.

    Args:
        zone:       Zona de precio ('DK1' o 'DK2')
        run_optuna: Si False, usa hiperparámetros razonables por defecto (más rápido)
        n_trials:   Número de trials de Optuna

    Returns:
        Dict con métricas finales del modelo.
    """
    logger = get_logger(f"train_{zone}")
    logger.info(f"=== ENTRENAMIENTO MODELO — Zona {zone} ===")
    logger.info(f"  Optuna: {'SÍ' if run_optuna else 'NO'} | Trials: {n_trials}")

    # 1. Cargar datos
    logger.info("\n[1] Cargando dataset de features...")
    df, X, y, feature_cols, splits = load_and_prepare(zone, logger)

    # 2. LightGBM con defaults (benchmark)
    logger.info("\n[2] LightGBM con hiperparámetros por defecto...")
    model_default, mae_default = train_lgbm_default(X, y, splits, logger)

    # 3. Optimización Optuna
    if run_optuna:
        logger.info("\n[3] Optimización de hiperparámetros con Optuna...")
        best_params, mae_optuna = optimize_with_optuna(
            X, y, splits, n_trials, logger
        )
        logger.info(f"\n  Mejora Optuna vs default: "
                    f"{mae_default:.3f} → {mae_optuna:.3f} EUR/MWh "
                    f"({(mae_default - mae_optuna) / mae_default * 100:.1f}% reducción)")
    else:
        logger.info("\n[3] Optuna omitido — usando hiperparámetros razonables...")
        best_params = {
            "learning_rate":     0.03,
            "num_leaves":        95,
            "max_depth":         9,
            "min_child_samples": 40,
            "subsample":         0.8,
            "colsample_bytree":  0.75,
            "reg_alpha":         0.1,
            "reg_lambda":        1.0,
        }

    # 4. Modelo final con mejores hiperparámetros
    logger.info("\n[4] Entrenando modelo final (train+val)...")
    model_final = train_lgbm_tuned(X, y, splits, best_params, logger)

    # 5. Evaluación en test
    logger.info("\n[5] Evaluación en test set...")
    X_test = X[splits["test"]]
    y_test = y[splits["test"]]
    preds_test = model_final.predict(X_test)
    errors     = y_test.values - preds_test

    mae_test  = float(np.mean(np.abs(errors)))
    rmse_test = float(np.sqrt(np.mean(errors ** 2)))
    denom     = (np.abs(y_test.values) + np.abs(preds_test)) / 2
    smape_test = float(np.mean(np.where(denom > 0.1, np.abs(errors) / denom * 100, 0)))
    dir_acc   = float(np.mean(np.sign(np.diff(y_test.values)) ==
                               np.sign(np.diff(preds_test))) * 100)
    ss_res    = np.sum(errors ** 2)
    ss_tot    = np.sum((y_test.values - np.mean(y_test.values)) ** 2)
    r2_test   = float(1 - ss_res / ss_tot)

    logger.info(f"\n  ┌─────────────────────────────────────────┐")
    logger.info(f"  │  MÉTRICAS TEST — {zone}                   │")
    logger.info(f"  ├─────────────────────────────────────────┤")
    logger.info(f"  │  MAE:          {mae_test:7.3f} EUR/MWh       │")
    logger.info(f"  │  RMSE:         {rmse_test:7.3f} EUR/MWh       │")
    logger.info(f"  │  sMAPE:        {smape_test:7.2f} %              │")
    logger.info(f"  │  Dir Accuracy: {dir_acc:7.2f} %              │")
    logger.info(f"  │  R²:           {r2_test:7.4f}               │")
    logger.info(f"  └─────────────────────────────────────────┘")
    logger.info(f"\n  Benchmark Nord Pool: MAE 5-15 EUR/MWh | DirAcc 55-65%")

    # 6. Walk-forward validation
    logger.info("\n[6] Walk-forward validation (expanding window)...")
    wf_results = walk_forward_validation(X, y, best_params, logger)

    # 7. Quantile regression
    logger.info("\n[7] Quantile regression (P10-P90)...")
    quantile_models, coverage_stats = train_quantile_models(
        X, y, splits, best_params, logger
    )

    # 7b. Conformal calibration — ajustar bandas para cobertura garantizada
    from src.model.conformal import ConformalCalibrator
    logger.info("\n[7b] Conformal calibration de bandas...")

    # DECISIÓN TÉCNICA (split-conformal honesto): el set de calibración y el
    # de medición de cobertura deben ser DISJUNTOS para que la cobertura sea
    # out-of-sample (no in-sample, que sería optimista por construcción).
    # Los quantile models entrenaron con train+val, así que solo el test está
    # sin ver. Partimos el test en dos mitades temporales:
    #   - 1ª mitad  → calibración del conformal (se guarda este calibrador)
    #   - 2ª mitad  → medición de cobertura (out-of-sample, honesta)
    test_index = X[splits["test"]].index
    n_test = len(test_index)

    if n_test >= 200:
        cut       = n_test // 2
        cal_idx   = test_index[:cut]
        eval_idx  = test_index[cut:]
        X_cal, y_cal = X.loc[cal_idx], y.loc[cal_idx]
        X_ev,  y_ev  = X.loc[eval_idx], y.loc[eval_idx]

        pred_cal = model_final.predict(X_cal)
        pred_ev  = model_final.predict(X_ev)
        qpreds_cal = {q: m.predict(X_cal) for q, m in quantile_models.items()}
        qpreds_ev  = {q: m.predict(X_ev)  for q, m in quantile_models.items()}

        conformal_cal = ConformalCalibrator(strategy="adaptive", target_coverage=0.80)
        conformal_cal.fit(
            y_cal=y_cal.values,
            pred_cal=pred_cal,
            q_lower_cal=qpreds_cal["P10"],
            q_upper_cal=qpreds_cal["P90"],
            timestamps_cal=y_cal.index,
            verbose=True,
        )

        # Guardar calibrador (ajustado en la 1ª mitad del test)
        conf_path = MODELS_DIR / f"conformal_{zone}.pkl"
        conformal_cal.save(str(conf_path))
        logger.info(f"  Conformal calibrator guardado: {conf_path.name}")

        # Cobertura OUT-OF-SAMPLE en la 2ª mitad del test (honesta)
        bands_adj = conformal_cal.calibrate_all_bands(
            pred_ev, qpreds_ev, timestamps=y_ev.index
        )
        cov_raw = np.mean(
            (y_ev.values >= qpreds_ev["P10"]) & (y_ev.values <= qpreds_ev["P90"])
        )
        cov_conf = np.mean(
            (y_ev.values >= bands_adj["P10"]) & (y_ev.values <= bands_adj["P90"])
        )
        cov_50_raw = np.mean(
            (y_ev.values >= qpreds_ev["P25"]) & (y_ev.values <= qpreds_ev["P75"])
        )
        cov_50_conf = np.mean(
            (y_ev.values >= bands_adj["P25"]) & (y_ev.values <= bands_adj["P75"])
        )
        coverage_stats["coverage_80_conformal"] = float(cov_conf)
        coverage_stats["coverage_50_conformal"] = float(cov_50_conf)
        coverage_stats["conformal_eval"] = "out-of-sample (2ª mitad del test)"
        logger.info(f"  [out-of-sample] P10-P90: {cov_raw:.1%} → {cov_conf:.1%} (target 80%)")
        logger.info(f"  [out-of-sample] P25-P75: {cov_50_raw:.1%} → {cov_50_conf:.1%} (target 50%)")
    else:
        logger.warning("  WARN: Test insuficiente para split-conformal — omitiendo conformal")

    # 8. Feature importance
    logger.info("\n[8] Top 15 features por importancia (gain)...")
    feat_imp = pd.Series(
        model_final.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    for feat, imp in feat_imp.head(15).items():
        bar = "█" * int(imp / feat_imp.max() * 20)
        logger.info(f"  {feat:45s} {bar}")

    # ── [AR] Corrección AR de residuos ──
    from src.model.ar_correction import ARResidualCorrector
    logger.info("\n[AR] Ajustando Corrector Autorregresivo de Residuos...")
    trainval_mask = splits["train"] | splits["val"]
    preds_tv = model_final.predict(X[trainval_mask])
    ar_corrector = ARResidualCorrector(max_order=3)
    ar_corrector.fit(y[trainval_mask].values, preds_tv, verbose=False)
    ar_path = MODELS_DIR / f"ar_corrector_{zone}.pkl"
    ar_corrector.save(str(ar_path))
    logger.info(f"  AR({ar_corrector.best_order}) → "
                f"φ₁={ar_corrector.ar_coefs[0]:.4f} | "
                f"Guardado: {ar_path.name}")

    # Métricas test CON AR (day-ahead = solo residuos conocidos, SIN ver futuro)
    from src.model.ar_correction import simulate_day_ahead_correction
    da_test_metrics = simulate_day_ahead_correction(
        y_test.values, preds_test, ar_corrector, verbose=False
    )
    mae_test_ar  = da_test_metrics["MAE_day_ahead_corrected"]
    rmse_test_ar = float(np.sqrt(np.mean((y_test.values[24:] - preds_test[24:]) ** 2)))
    logger.info(f"  Test MAE: {mae_test:.3f} → {mae_test_ar:.3f} con AR day-ahead "
                f"({(mae_test - mae_test_ar) / mae_test * 100:.1f}% mejora)")

    # ── [ENS] Ensemble robusto por DIVERSIDAD DE OBJETIVO (L2 + mediana + Huber) ──
    # Sustituye al antiguo LGBM+XGB+Ridge (XGB≈LGBM, corr 0.99 → redundante).
    # Validado en walk-forward 2024: +2.0 pp DirAcc (mejor en 12/12 meses) y
    # ~+1.5% MAE vs LightGBM L2. Ver scripts/experiment_*.py y decisions.md DT-14.
    logger.info("\n[ENS] Entrenando Ensemble diverso (LightGBM L2 + mediana + Huber)...")
    try:
        from src.model.diverse_ensemble import DiverseObjectiveEnsemble
        ensemble = DiverseObjectiveEnsemble(
            zone=zone,
            base_params={**{"n_estimators": 2000}, **best_params},
        )
        # Fit en TRAIN con early stopping en VAL (disjuntos → sin leakage)
        ensemble.fit(X[splits["train"]], y[splits["train"]],
                     X[splits["val"]], y[splits["val"]], verbose=True)

        # Guardar ensemble
        ens_path = MODELS_DIR / f"ensemble_{zone}.pkl"
        ensemble.save(str(ens_path))
        logger.info(f"  Ensemble guardado: {ens_path.name}")

        # Métricas ensemble en test
        preds_ens = ensemble.predict(X_test)
        mae_ens = float(np.mean(np.abs(y_test.values - preds_ens)))
        dir_ens = float(np.mean(np.sign(np.diff(y_test.values)) ==
                                np.sign(np.diff(preds_ens))) * 100)

        # Ensemble + AR (day-ahead realista). AR ajustado sobre residuos train+val.
        preds_ens_tv = ensemble.predict(X[trainval_mask])
        ar_ens = ARResidualCorrector(max_order=3)
        ar_ens.fit(y[trainval_mask].values, preds_ens_tv, verbose=False)
        ar_ens_path = MODELS_DIR / f"ar_corrector_ensemble_{zone}.pkl"
        ar_ens.save(str(ar_ens_path))
        da_ens_metrics = simulate_day_ahead_correction(
            y_test.values, preds_ens, ar_ens, verbose=False
        )
        mae_ens_ar = da_ens_metrics["MAE_day_ahead_corrected"]

        logger.info(f"\n  ┌───────────────────────────────────────────────┐")
        logger.info(f"  │  COMPARATIVA TEST — {zone}                       │")
        logger.info(f"  ├───────────────────────────────────────────────┤")
        logger.info(f"  │  LightGBM solo:    MAE {mae_test:6.2f} | DirAcc {dir_acc:4.1f}% │")
        logger.info(f"  │  Ensemble diverso: MAE {mae_ens:6.2f} | DirAcc {dir_ens:4.1f}% ◄ │")
        logger.info(f"  │  Ensemble + AR:    MAE {mae_ens_ar:6.2f}              │")
        logger.info(f"  └───────────────────────────────────────────────┘")
        has_ensemble = True
    except Exception as e:
        import traceback
        logger.warning(f"  WARN: Ensemble diverso no disponible: {e}")
        logger.debug(traceback.format_exc())
        has_ensemble = False
        mae_ens = None
        mae_ens_ar = None

    # 9. Guardar modelos
    logger.info("\n[9] Guardando modelos...")
    save_models(zone, model_final, quantile_models, feature_cols,
                best_params, wf_results, logger)

    # Métricas finales
    final_metrics = {
        "zone":          zone,
        "MAE_test":      round(mae_test, 3),
        "MAE_test_ar":   round(mae_test_ar, 3),
        "MAE_test_ens":  round(mae_ens, 3) if mae_ens else None,
        "MAE_test_ens_ar": round(mae_ens_ar, 3) if mae_ens_ar else None,
        "RMSE_test":     round(rmse_test, 3),
        "sMAPE_test":    round(smape_test, 3),
        "DirAcc_test":   round(dir_acc, 2),
        "R2_test":       round(r2_test, 4),
        "MAE_wf_mean":   round(wf_results["MAE"].mean(), 3) if not wf_results.empty else None,
        "MAE_wf_ar_mean": round(wf_results["MAE_ar"].mean(), 3) if "MAE_ar" in wf_results.columns else None,
        "coverage_80":   coverage_stats["coverage_80"],
        "coverage_50":   coverage_stats["coverage_50"],
        "n_features":    len(feature_cols),
        "best_params":   best_params,
        "top5_features": feat_imp.head(5).index.tolist(),
        "ar_order":      ar_corrector.best_order,
        "ar_phi1":       round(ar_corrector.ar_coefs[0], 4),
        "has_ensemble":  has_ensemble,
    }

    save_run_metadata(
        script_name=f"train_{zone}",
        params={
            "zone": zone, "n_trials": n_trials,
            "run_optuna": run_optuna,
            "train_end": TRAIN_END, "test_start": TEST_START,
        },
        stats=final_metrics,
    )

    logger.info(f"\nOK Entrenamiento {zone} completado.")
    return final_metrics


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento modelo predictivo")
    parser.add_argument("--zone",      default="DK1", choices=["DK1", "DK2", "FI"])
    parser.add_argument("--no-optuna", action="store_true",
                        help="Omitir optimización Optuna (más rápido)")
    parser.add_argument("--trials",    type=int, default=N_TRIALS,
                        help=f"Número de trials Optuna (default: {N_TRIALS})")
    parser.add_argument("--both",      action="store_true",
                        help="Entrenar DK1 y DK2 secuencialmente")
    args = parser.parse_args()

    zones = ["DK1", "DK2"] if args.both else [args.zone]

    all_results = {}
    for zone in zones:
        result = train_zone(
            zone=zone,
            run_optuna=not args.no_optuna,
            n_trials=args.trials,
        )
        all_results[zone] = result

    # Tabla resumen comparativa si se entrenaron ambas zonas
    if len(all_results) > 1:
        print("\n=== RESUMEN COMPARATIVO ===")
        summary = pd.DataFrame(all_results).T[
            ["MAE_test", "RMSE_test", "sMAPE_test", "DirAcc_test", "R2_test"]
        ]
        print(summary.to_string())