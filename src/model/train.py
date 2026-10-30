"""
train.py
========
Entrenamiento de los modelos de una zona — Fase 2.

Pipeline (train_zone):
  1. Carga la tabla de features y fija el conjunto de features de la zona
     (regla en src/model/config.py; DK 166 columnas, FI 224).
  2. Split temporal estricto (config.py): train hasta sep-2025, validación
     oct-dic 2025, test ene-may 2026.
  3. Hiperparámetros: los congelados en src/model/best_params.json, o Optuna
     (opcional, --optuna) sobre el MAE de validación.
  4. Elección del modelo servido EN VALIDACIÓN: LightGBM de 2000 árboles
     entrenado con train frente al ensemble diverso (train, early stopping en
     val). Regla fijada antes de puntuar: gana el MAE de validación más bajo;
     si la diferencia no es significativa (Diebold-Mariano HAC h=24 + HLN,
     p >= 0.05) se sirve el ensemble.
  5. Corrección AR day-ahead decidida FUERA DE MUESTRA en validación: dos
     pliegues expansivos por mes de entrega (ajusta oct y puntúa nov; ajusta
     oct-nov y puntúa dic), con offsets por día de entrega local. Se usa si
     baja el MAE conjunto; después se ajusta con toda la validación.
  6. Cuantiles P10-P90: entrenados con train y early stopping en validación.
  7. Calibración conformal (adaptive, 80 %) ajustada en validación.
  8. LightGBM de 2000 árboles con train+val: solo comparador y SHAP.
  9. Walk-forward mensual de un LightGBM (500 árboles), descriptivo.
 10. Guarda modelos, metadatos y results/training_<zona>.json.

Este script no puntúa nada en el test: la evaluación del test es
src/model/evaluate.py, con la misma ruta de predicción que el dashboard.

DECISIÓN TÉCNICA: split temporal estricto, nunca aleatorio. Ningún modelo
servido ve la validación ni el test al entrenar; la validación solo se usa
para early stopping, elección, AR y calibración.

Uso:
    python src/model/train.py --zone DK1          # parámetros congelados
    python src/model/train.py --zone FI --optuna  # vuelve a buscar con Optuna
    python src/model/train.py --all
"""

import sys
import json
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
from src.model.config import (
    TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END, DELIVERY_TZ,
    LGB_COMMON, FINAL_N_ESTIMATORS, BEST_PARAMS_PATH, RESULTS_DIR,
    FEATURE_RULE, EXPECTED_N_FEATURES, RENAMED_FEATURES, SAFE_IS_FLAGS,
    select_feature_cols,
)

# ── Configuración ─────────────────────────────────────────────────────────────

FEATURES_PATH = ROOT / "data" / "master" / "dataset_features.parquet"
MODELS_DIR    = ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Walk-forward: a partir de qué mes comienza la validación rolling
WF_START = "2024-06-01"
WF_N_ESTIMATORS = 500

# Quantiles para predicción probabilística
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_MAX_ESTIMATORS = 1000

# Optuna (opcional)
N_TRIALS        = 100
OPTUNA_TIMEOUT  = 3600   # segundos máximos (1h)

# LightGBM early stopping (solo con un conjunto de parada disjunto del de ajuste)
EARLY_STOPPING_ROUNDS = 50

# Elección de modelo y AR en validación
DM_HORIZON      = 24
SELECTION_ALPHA = 0.05
AR_MAX_ORDER    = 3
AR_MIN_SCORE_HOURS = 24 * 7   # meses de entrega con menos horas no se puntúan
CONFORMAL_TARGET = 0.80


# ── Preparación de datos ──────────────────────────────────────────────────────

def _split_masks(index: pd.DatetimeIndex) -> dict:
    train = index <= pd.Timestamp(TRAIN_END, tz="UTC")
    val = ((index >= pd.Timestamp(VAL_START, tz="UTC"))
           & (index <= pd.Timestamp(VAL_END, tz="UTC")))
    test = ((index >= pd.Timestamp(TEST_START, tz="UTC"))
            & (index <= pd.Timestamp(TEST_END, tz="UTC")))
    return {"train": train, "val": val, "test": test}


def load_and_prepare(zone: str, logger, strict: bool = False) -> tuple:
    """
    Carga el dataset de features y prepara X, y para la zona indicada.

    Conjunto de features (config.select_feature_cols): columnas feat_* (en DK,
    sin el bloque de acoplamiento finlandés) más is_weekend / is_holiday*.
    Los NaN se rellenan con la mediana del tramo de train, en todas las
    particiones.

    Args:
        strict: exige el número de features publicado (EXPECTED_N_FEATURES).

    Returns:
        (df_full, X, y, feature_cols, splits, fill_values)
        splits = dict de máscaras booleanas train / val / test;
        fill_values = dict {columna: mediana de train}.
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

    # ANTI-LEAKAGE: los flags `is_congested_*` del master se calculan con el
    # spread de precio del MISMO timestamp (integration.py:step_derived_features),
    # que NO se conoce a las 12:00 del día anterior. Se excluyen; sus equivalentes
    # causales son `feat_congested_*` (spread con shift(24)). Solo se admiten
    # flags de calendario deterministas.
    leaky_flags = [c for c in df.columns
                   if c.startswith("is_") and not c.startswith(SAFE_IS_FLAGS)]
    if leaky_flags:
        logger.info(f"  Excluidas {len(leaky_flags)} flags con leakage temporal: {leaky_flags}")
    feature_cols = select_feature_cols(df.columns, zone)
    logger.info(f"  Features {zone}: {len(feature_cols)} ({FEATURE_RULE[zone]})")
    if strict and len(feature_cols) != EXPECTED_N_FEATURES[zone]:
        raise ValueError(
            f"{zone}: la regla de features da {len(feature_cols)} columnas, "
            f"se esperaban {EXPECTED_N_FEATURES[zone]}"
        )

    # Eliminar filas donde el target es NaN
    df_model = df[[target_col] + feature_cols].dropna(subset=[target_col])

    # Eliminar filas donde casi todas las features son NaN (primeras horas por lags)
    min_feat_required = int(len(feature_cols) * 0.5)
    df_model = df_model.dropna(thresh=min_feat_required + 1)

    splits = _split_masks(df_model.index)

    # Rellenar NaN residuales con la mediana del tramo de TRAIN (nunca de val/test)
    medians = df_model.loc[splits["train"], feature_cols].median()
    fill_values = {c: float(v) for c, v in medians.items()}
    X = df_model[feature_cols].fillna(medians)
    y = df_model[target_col]

    logger.info(f"  Dataset:    {len(df_model):,} filas × {len(feature_cols)} features")
    for name in ("train", "val", "test"):
        m = splits[name]
        if m.any():
            logger.info(f"  {name:<10s}: {m.sum():,} horas  "
                        f"({X[m].index.min().date()} → {X[m].index.max().date()})")

    return df, X, y, feature_cols, splits, fill_values


def check_feature_set_against_saved_meta(zone: str, feature_cols: list, logger,
                                         models_dir: Path = None) -> None:
    """
    Antes de sobrescribir los modelos: el conjunto de features (y su orden)
    tiene que coincidir con el del modelo guardado, salvo los renombrados de
    config.RENAMED_FEATURES. Lanza RuntimeError si no coincide.
    """
    meta_path = (models_dir or MODELS_DIR) / f"model_meta_{zone}.pkl"
    if not meta_path.exists():
        logger.info(f"  (sin {meta_path.name}: no hay conjunto previo que comprobar)")
        return
    with open(meta_path, "rb") as f:
        saved = pickle.load(f)["feature_cols"]
    saved = [RENAMED_FEATURES.get(c, c) for c in saved]
    if saved != list(feature_cols):
        missing = [c for c in saved if c not in feature_cols]
        extra = [c for c in feature_cols if c not in saved]
        raise RuntimeError(
            f"{zone}: el conjunto de features no coincide con {meta_path.name} "
            f"(faltan {missing[:5]}, sobran {extra[:5]}, "
            f"mismo conjunto en otro orden: {not missing and not extra})"
        )
    logger.info(f"  Conjunto de features idéntico al de {meta_path.name} "
                f"({len(saved)} columnas, mismo orden)")


def load_best_params(zone: str, path: Path = BEST_PARAMS_PATH) -> dict:
    """Hiperparámetros congelados de Optuna para la zona (best_params.json)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return dict(data["zones"][zone]["params"])


# ── Modelos LightGBM ──────────────────────────────────────────────────────────

def _point_params(best_params: dict, n_estimators: int) -> dict:
    params = {"objective": "regression", "metric": "mae", **LGB_COMMON}
    params.update({k: v for k, v in best_params.items() if k != "n_estimators"})
    params["n_estimators"] = int(n_estimators)
    return params


def fit_lgbm_fixed(X: pd.DataFrame, y: pd.Series, best_params: dict,
                   n_estimators: int = FINAL_N_ESTIMATORS):
    """LightGBM puntual con un número de árboles fijo (sin eval_set ni early stopping)."""
    import lightgbm as lgb
    model = lgb.LGBMRegressor(**_point_params(best_params, n_estimators))
    model.fit(X, y)
    return model


def train_lgbm_default(X: pd.DataFrame, y: pd.Series,
                       splits: dict, logger) -> tuple:
    """
    LightGBM con hiperparámetros por defecto, train con early stopping en
    validación (disjuntos). Solo referencia en el log y en training_<zona>.json.
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
        **LGB_COMMON,
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
    mae_val   = float(np.mean(np.abs(y_val.values - preds_val)))
    logger.info(f"  LightGBM default → MAE val: {mae_val:.3f} EUR/MWh "
                f"| árboles (early stopping en val): {model.best_iteration_}")
    return model, mae_val


def optimize_with_optuna(X: pd.DataFrame, y: pd.Series,
                         splits: dict, n_trials: int, logger) -> tuple:
    """
    Optimización bayesiana (TPE) de hiperparámetros sobre el MAE de validación
    (oct-dic 2025), con LightGBM entrenado en train y early stopping en val.
    Opcional: por defecto train_zone usa best_params.json.
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
            **LGB_COMMON,
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
    logger.info(f"  OK Optuna completado: MAE val óptimo = {best_mae:.3f} EUR/MWh")
    for k, v in best_params.items():
        logger.info(f"    {k:25s}: {v}")
    return best_params, best_mae


def train_lgbm_tuned(X: pd.DataFrame, y: pd.Series, splits: dict,
                     best_params: dict, logger,
                     n_estimators: int = FINAL_N_ESTIMATORS):
    """
    LightGBM de referencia con train+val y un número de árboles FIJO
    (declarado como hiperparámetro). No hay eval_set: el trimestre de
    validación forma parte de los datos de ajuste y no puede parar nada.
    Es comparador y base de SHAP / importancia, no el modelo servido.
    """
    trainval_mask = splits["train"] | splits["val"]
    model = fit_lgbm_fixed(X[trainval_mask], y[trainval_mask], best_params, n_estimators)
    logger.info(f"  LightGBM train+val: {model.booster_.num_trees()} árboles (fijo, sin early stopping)")
    return model


# ── Elección del modelo servido en validación ─────────────────────────────────

def choose_served_model(y_val, pred_ens, pred_lgbm,
                        alpha: float = SELECTION_ALPHA, h: int = DM_HORIZON) -> dict:
    """
    Regla fijada antes de puntuar: gana el MAE de validación más bajo; si el
    LightGBM es mejor pero la diferencia no es significativa (DM HAC + HLN,
    p >= alpha), se sirve el ensemble.

    DM < 0 significa que el ensemble es mejor que el LightGBM.
    """
    from src.model.dm import diebold_mariano
    y = np.asarray(y_val, float)
    e_ens = y - np.asarray(pred_ens, float)
    e_lgb = y - np.asarray(pred_lgbm, float)
    mae_ens = float(np.mean(np.abs(e_ens)))
    mae_lgb = float(np.mean(np.abs(e_lgb)))
    dm, p = diebold_mariano(e_ens, e_lgb, loss="mae", h=h)
    if mae_ens <= mae_lgb:
        served, reason = "ensemble", "lower validation MAE"
    elif p < alpha:
        served, reason = "lightgbm", f"lower validation MAE and DM p < {alpha}"
    else:
        served, reason = "ensemble", (f"LightGBM has the lower validation MAE but "
                                      f"DM p >= {alpha}, so the ensemble is kept")
    return {
        "rule": ("lowest validation MAE wins; if LightGBM wins without DM "
                 f"significance (HAC h={h}, HLN, p >= {alpha}) the ensemble is served"),
        "MAE_val_ensemble": mae_ens,
        "MAE_val_lightgbm_train": mae_lgb,
        "DM_ensemble_vs_lightgbm": dm,
        "DM_p_value": p,
        "DM_convention": "DM < 0 means the ensemble is more accurate",
        "n_val_hours": int(len(y)),
        "served_model": served,
        "reason": reason,
    }


def decide_ar_on_val(y_val: pd.Series, pred_val, logger=None,
                     max_order: int = AR_MAX_ORDER,
                     min_score_hours: int = AR_MIN_SCORE_HOURS) -> dict:
    """
    Decide la corrección AR day-ahead con residuos de validación FUERA DE
    MUESTRA: pliegues expansivos por mes de entrega local. Para cada mes m
    (salvo el primero) se ajusta el AR con los meses anteriores y se puntúa m
    con day_ahead_offsets (solo residuos de días de entrega anteriores).
    use_ar = el MAE conjunto de los meses puntuados baja con el AR.
    Después se ajusta el AR con toda la validación (corrector final).

    Returns:
        dict con los pliegues, la decisión y el corrector final ('corrector').
    """
    from src.model.ar_correction import ARResidualCorrector, day_ahead_offsets

    idx = pd.DatetimeIndex(y_val.index)
    y = y_val.to_numpy(dtype=float)
    pred = np.asarray(pred_val, dtype=float)
    resid = pd.Series(y - pred, index=idx)
    local = idx.tz_convert(DELIVERY_TZ)
    month = np.asarray(local.year * 12 + local.month - 1)
    months = sorted(set(month.tolist()))

    folds, err_raw, err_ar = [], [], []
    for i, m in enumerate(months):
        if i == 0:
            continue
        score = month == m
        if score.sum() < min_score_hours:
            continue
        fit = month < m
        ar = ARResidualCorrector(max_order=max_order).fit(y[fit], pred[fit], verbose=False)
        off = day_ahead_offsets(idx[score], resid, ar, warn=False)
        e_raw = np.abs(y[score] - pred[score])
        e_ar = np.abs(y[score] - (pred[score] + off))
        err_raw.append(e_raw)
        err_ar.append(e_ar)
        folds.append({
            "fit_months": [f"{mm // 12}-{mm % 12 + 1:02d}" for mm in months if mm < m],
            "score_month": f"{m // 12}-{m % 12 + 1:02d}",
            "n_fit_hours": int(fit.sum()),
            "n_scored_hours": int(score.sum()),
            "ar_order": int(ar.best_order),
            "MAE_raw": float(e_raw.mean()),
            "MAE_ar": float(e_ar.mean()),
        })

    if not folds:
        raise ValueError("validación demasiado corta para decidir el AR fuera de muestra")
    mae_raw = float(np.concatenate(err_raw).mean())
    mae_ar = float(np.concatenate(err_ar).mean())
    use_ar = mae_ar < mae_raw

    final = ARResidualCorrector(max_order=max_order).fit(y, pred, verbose=False)
    if logger is not None:
        for f in folds:
            logger.info(f"    AR pliegue {f['score_month']}: MAE {f['MAE_raw']:.3f} → "
                        f"{f['MAE_ar']:.3f} (AR({f['ar_order']}))")
        logger.info(f"  AR fuera de muestra (val): MAE {mae_raw:.3f} → {mae_ar:.3f} "
                    f"→ use_ar={use_ar}")
    return {
        "method": ("expanding folds by local delivery month inside the validation "
                   "quarter; offsets per delivery day from residuals up to 23:00 "
                   "of the previous day (Europe/Copenhagen)"),
        "folds": folds,
        "MAE_raw_scored": mae_raw,
        "MAE_ar_scored": mae_ar,
        "use_ar": bool(use_ar),
        "final_fit": {"n_hours": int(len(y)), **final.get_params()},
        "corrector": final,
    }


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_validation(X: pd.DataFrame, y: pd.Series,
                            best_params: dict, logger) -> pd.DataFrame:
    """
    Walk-forward mensual con ventana expandible de un LightGBM de
    WF_N_ESTIMATORS árboles. Descriptivo: mide la estabilidad del LightGBM a
    lo largo del tiempo; no es el modelo servido y no decide nada.
    """
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error

    logger.info("\n  Walk-forward validation (expanding window)...")

    wf_start_ts = pd.Timestamp(WF_START, tz="UTC")
    months      = pd.date_range(start=wf_start_ts, end=X.index.max(), freq="MS")

    results = []
    for predict_month_start in months:
        predict_month_end = predict_month_start + pd.offsets.MonthEnd(1)
        predict_month_end = predict_month_end.replace(hour=23, tzinfo=predict_month_start.tzinfo)

        train_mask = X.index < predict_month_start
        pred_mask  = (X.index >= predict_month_start) & (X.index <= predict_month_end)
        if train_mask.sum() < 1000 or pred_mask.sum() == 0:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_pr, y_pr = X[pred_mask],  y[pred_mask]

        model = lgb.LGBMRegressor(**_point_params(best_params, WF_N_ESTIMATORS))
        model.fit(X_tr, y_tr)

        preds = model.predict(X_pr)
        mae   = mean_absolute_error(y_pr, preds)
        rmse  = np.sqrt(np.mean((y_pr.values - preds) ** 2))

        results.append({
            "month":       predict_month_start.strftime("%Y-%m"),
            "n_train":     len(X_tr),
            "n_test":      len(X_pr),
            "MAE":         round(mae, 3),
            "RMSE":        round(rmse, 3),
            "price_mean":  round(y_pr.mean(), 2),
            "price_std":   round(y_pr.std(), 2),
        })
        logger.info(f"    {predict_month_start.strftime('%Y-%m')}: "
                    f"MAE={mae:.2f} | train={len(X_tr):,}h")

    df_wf = pd.DataFrame(results)
    if not df_wf.empty:
        logger.info(f"\n  Walk-forward MAE: media={df_wf['MAE'].mean():.2f} | "
                    f"std={df_wf['MAE'].std():.2f} | max={df_wf['MAE'].max():.2f}")
    return df_wf


def _wf_summary(df_wf: pd.DataFrame) -> dict:
    if df_wf.empty:
        return {}
    out = {
        "model": f"LightGBM, {WF_N_ESTIMATORS} trees, expanding window, retrained monthly",
        "note": "descriptive; not the served model and not used for any choice",
        "folds": int(len(df_wf)),
        "first_month": str(df_wf["month"].iloc[0]),
        "last_month": str(df_wf["month"].iloc[-1]),
        "MAE_monthly": {"mean": float(df_wf["MAE"].mean()), "std": float(df_wf["MAE"].std()),
                        "min": float(df_wf["MAE"].min()), "max": float(df_wf["MAE"].max())},
    }
    return out


# ── Quantile Regression ───────────────────────────────────────────────────────

def train_quantile_models(X: pd.DataFrame, y: pd.Series,
                          splits: dict, best_params: dict, logger) -> tuple:
    """
    Modelos de quantile regression P10, P25, P50, P75, P90, entrenados con
    train y early stopping en validación (disjuntos). Sin reajuste con
    train+val. La cobertura en bruto se mide en validación.
    """
    import lightgbm as lgb

    logger.info("\n  Entrenando modelos de quantile regression (train, ES en val)...")
    X_tr, y_tr   = X[splits["train"]], y[splits["train"]]
    X_val, y_val = X[splits["val"]],   y[splits["val"]]

    quantile_models, n_trees = {}, {}
    for q in QUANTILES:
        params_q = {
            "objective":    "quantile",
            "alpha":        q,
            "metric":       "quantile",
            **LGB_COMMON,
            **{k: v for k, v in best_params.items() if k != "n_estimators"},
            "n_estimators": QUANTILE_MAX_ESTIMATORS,
        }
        model_q = lgb.LGBMRegressor(**params_q)
        model_q.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        name = f"P{int(q*100):02d}"
        quantile_models[name] = model_q
        n_trees[name] = int(model_q.best_iteration_ or QUANTILE_MAX_ESTIMATORS)
        logger.info(f"    {name} entrenado ({n_trees[name]} árboles)")

    logger.info("\n  Cobertura en bruto en validación (in-sample para el early stopping):")
    cov80 = _check_coverage(y_val.values, quantile_models["P10"].predict(X_val),
                            quantile_models["P90"].predict(X_val), 0.80, logger, "80% (P10-P90)")
    cov50 = _check_coverage(y_val.values, quantile_models["P25"].predict(X_val),
                            quantile_models["P75"].predict(X_val), 0.50, logger, "50% (P25-P75)")
    return quantile_models, {"coverage_80_val_raw": cov80, "coverage_50_val_raw": cov50,
                             "n_trees": n_trees}


def _check_coverage(y_true, lower, upper, target_coverage, logger, label) -> float:
    """Proporción de valores reales dentro del intervalo."""
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    status   = "OK" if abs(coverage - target_coverage) < 0.05 else "WARN"
    logger.info(f"    {status} Cobertura {label}: "
                f"{coverage:.1%} (objetivo: {target_coverage:.0%})")
    return float(coverage)


# ── Serialización ─────────────────────────────────────────────────────────────

def save_models(zone: str, model_final, model_train, ensemble, quantile_models: dict,
                conformal, ar_corrector, feature_cols: list, best_params: dict,
                fill_values: dict, selection: dict, ar_block: dict,
                top5: list, wf_results: pd.DataFrame, logger) -> dict:
    """Guarda modelos y metadatos en data/models/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    robust_pickle_dump(model_final, MODELS_DIR / f"lightgbm_{zone}.pkl")
    robust_pickle_dump(model_train, MODELS_DIR / f"lightgbm_train_{zone}.pkl")
    ensemble.save(str(MODELS_DIR / f"ensemble_{zone}.pkl"))
    robust_pickle_dump(quantile_models, MODELS_DIR / f"quantile_models_{zone}.pkl")
    conformal.save(str(MODELS_DIR / f"conformal_{zone}.pkl"))
    ar_corrector.save(str(MODELS_DIR / f"ar_corrector_served_{zone}.pkl"))

    meta = {
        "zone":          zone,
        "trained_at":    ts,
        "feature_cols":  feature_cols,
        "feature_rule":  FEATURE_RULE[zone],
        "n_features":    len(feature_cols),
        "fill_values":   fill_values,
        "best_params":   best_params,
        "train_end":     TRAIN_END,
        "val_start":     VAL_START,
        "val_end":       VAL_END,
        "test_start":    TEST_START,
        "test_end":      TEST_END,
        "quantiles":     QUANTILES,
        "served_model":  selection["served_model"],
        "use_ar":        bool(ar_block["use_ar"]),
        "n_estimators_final": FINAL_N_ESTIMATORS,
        "selection":     {k: v for k, v in selection.items()},
        "ar":            {k: v for k, v in ar_block.items() if k != "corrector"},
        "top5_features": top5,
    }
    robust_pickle_dump(meta, MODELS_DIR / f"model_meta_{zone}.pkl")

    if not wf_results.empty:
        wf_results.to_csv(MODELS_DIR / f"walkforward_{zone}.csv", index=False)
    logger.info(f"  Modelos y metadatos guardados en data/models/ ({zone})")
    return meta


# ── Función principal ─────────────────────────────────────────────────────────

def train_zone(zone: str = "DK1", run_optuna: bool = False,
               n_trials: int = N_TRIALS, params_json: Path = BEST_PARAMS_PATH,
               check_saved_features: bool = True) -> dict:
    """
    Pipeline completo de entrenamiento para una zona (ver docstring del módulo).

    Args:
        zone:        'DK1', 'DK2' o 'FI'
        run_optuna:  True vuelve a buscar hiperparámetros con Optuna; False
                     (defecto) usa los congelados de best_params.json.
        n_trials:    trials de Optuna.
        params_json: fichero de hiperparámetros congelados.
        check_saved_features: exigir que el conjunto de features coincida con
                     el del modelo guardado (salvo renombrados).

    Returns:
        Dict con las métricas de validación, la elección y la decisión AR.
    """
    from src.model.diverse_ensemble import DiverseObjectiveEnsemble
    from src.model.conformal import ConformalCalibrator
    from src.model.results_io import write_json

    logger = get_logger(f"train_{zone}")
    logger.info(f"=== ENTRENAMIENTO MODELO — Zona {zone} ===")
    logger.info(f"  Hiperparámetros: {'Optuna (' + str(n_trials) + ' trials)' if run_optuna else 'best_params.json'}"
                f" | LightGBM n_jobs={LGB_COMMON['n_jobs']}")

    # 1. Datos
    logger.info("\n[1] Cargando dataset de features...")
    df, X, y, feature_cols, splits, fill_values = load_and_prepare(zone, logger, strict=True)
    if check_saved_features:
        check_feature_set_against_saved_meta(zone, feature_cols, logger)
    tr, va = splits["train"], splits["val"]

    # 2. Referencia: LightGBM por defecto (train, early stopping en val)
    logger.info("\n[2] LightGBM con hiperparámetros por defecto (referencia)...")
    model_default, mae_default = train_lgbm_default(X, y, splits, logger)

    # 3. Hiperparámetros
    if run_optuna:
        logger.info("\n[3] Optimización de hiperparámetros con Optuna...")
        best_params, mae_optuna = optimize_with_optuna(X, y, splits, n_trials, logger)
        params_source = f"Optuna, {n_trials} trials (this run)"
    else:
        best_params = load_best_params(zone, params_json)
        mae_optuna = None
        params_source = "src/model/best_params.json"
        logger.info(f"\n[3] Hiperparámetros congelados ({params_source}): {best_params}")

    # 4. Elección del modelo servido en validación
    logger.info("\n[4] Elección del modelo servido en validación...")
    model_train = fit_lgbm_fixed(X[tr], y[tr], best_params, FINAL_N_ESTIMATORS)
    ensemble = DiverseObjectiveEnsemble(
        zone=zone, base_params={"n_estimators": FINAL_N_ESTIMATORS, **best_params})
    ensemble.fit(X[tr], y[tr], X[va], y[va], verbose=True)
    pred_val_lgbm = model_train.predict(X[va])
    pred_val_ens = ensemble.predict(X[va])
    selection = choose_served_model(y[va].values, pred_val_ens, pred_val_lgbm)
    selection["ensemble_n_trees"] = {name: int(m.best_iteration_ or m.n_estimators)
                                     for name, m in ensemble.models.items()}
    selection["lightgbm_train_n_trees"] = int(model_train.booster_.num_trees())
    logger.info(f"  MAE val: ensemble {selection['MAE_val_ensemble']:.3f} | "
                f"LightGBM(train) {selection['MAE_val_lightgbm_train']:.3f} | "
                f"DM {selection['DM_ensemble_vs_lightgbm']:+.2f} "
                f"(p={selection['DM_p_value']:.4f}) → {selection['served_model']} "
                f"({selection['reason']})")
    pred_val_served = pred_val_ens if selection["served_model"] == "ensemble" else pred_val_lgbm

    # 5. Corrección AR decidida fuera de muestra en validación
    logger.info("\n[5] Corrección AR day-ahead (decisión en validación)...")
    ar_block = decide_ar_on_val(y[va], pred_val_served, logger)

    # 6. Cuantiles (train, early stopping en val)
    logger.info("\n[6] Quantile regression (P10-P90)...")
    quantile_models, q_stats = train_quantile_models(X, y, splits, best_params, logger)

    # 7. Conformal ajustado en validación (bandas en bruto, sin AR)
    logger.info("\n[7] Conformal calibration en validación...")
    q_val = {q: m.predict(X[va]) for q, m in quantile_models.items()}
    conformal = ConformalCalibrator(strategy="adaptive", target_coverage=CONFORMAL_TARGET)
    conformal.fit(y_cal=y[va].values, pred_cal=pred_val_served,
                  q_lower_cal=q_val["P10"], q_upper_cal=q_val["P90"],
                  timestamps_cal=y[va].index, verbose=False)
    conf_diag = conformal.get_diagnostics()
    logger.info(f"  Conformal (val, in-sample): {conf_diag['raw_coverage']:.1%} → "
                f"{conf_diag['calibrated_coverage_insample']:.1%}")

    # 8. LightGBM train+val (comparador y SHAP)
    logger.info("\n[8] LightGBM de referencia con train+val (2000 árboles fijos)...")
    model_final = train_lgbm_tuned(X, y, splits, best_params, logger,
                                   n_estimators=FINAL_N_ESTIMATORS)

    # 9. Walk-forward (descriptivo)
    logger.info("\n[9] Walk-forward validation (expanding window)...")
    wf_results = walk_forward_validation(X, y, best_params, logger)

    # 10. Importancia de features (LightGBM train+val)
    feat_imp = pd.Series(model_final.feature_importances_,
                         index=feature_cols).sort_values(ascending=False)
    logger.info("\n[10] Top 15 features por importancia (split):")
    for feat, imp in feat_imp.head(15).items():
        logger.info(f"  {feat:45s} {'█' * int(imp / feat_imp.max() * 20)}")
    top5 = feat_imp.head(5).index.tolist()

    # 11. Guardar
    logger.info("\n[11] Guardando modelos...")
    save_models(zone, model_final, model_train, ensemble, quantile_models, conformal,
                ar_block["corrector"], feature_cols, best_params, fill_values,
                selection, ar_block, top5, wf_results, logger)

    training = {
        "zone": zone,
        "generated_by": "src/model/train.py (train_zone)",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "splits_utc": {"train_end": TRAIN_END, "val_start": VAL_START, "val_end": VAL_END,
                       "test_start": TEST_START, "test_end": TEST_END},
        "rows": {k: int(v.sum()) for k, v in splits.items()},
        "features": {"n": len(feature_cols), "rule": FEATURE_RULE[zone],
                     "nan_fill": "median of the training rows"},
        "hyperparameters": {
            "source": params_source,
            "params": best_params,
            "point_models_n_estimators": FINAL_N_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "lightgbm_common": LGB_COMMON,
        },
        "reference_default_lightgbm": {
            "MAE_val": mae_default,
            "trees_early_stopping_val": int(model_default.best_iteration_ or 0),
        },
        "optuna_MAE_val": mae_optuna,
        "served": {
            "model": selection["served_model"],
            "use_ar": bool(ar_block["use_ar"]),
            "trained_on": "train only (up to TRAIN_END); the validation quarter is "
                          "used for early stopping, model choice, AR and calibration",
        },
        "selection_val": {k: v for k, v in selection.items()},
        "ar_val": {k: v for k, v in ar_block.items() if k != "corrector"},
        "quantiles_val": q_stats,
        "conformal_val": {"strategy": "adaptive", "target": CONFORMAL_TARGET,
                          "fitted_on": "validation quarter, raw quantile bands",
                          **conf_diag},
        "walk_forward": _wf_summary(wf_results),
        "top5_features": top5,
        "note": "No test-window metric is computed here; see results/evaluation_test2026_<zone>.json.",
    }
    write_json(RESULTS_DIR / f"training_{zone}.json", training)
    save_run_metadata(
        script_name=f"train_{zone}",
        params={"zone": zone, "run_optuna": run_optuna, "n_trials": n_trials,
                "train_end": TRAIN_END, "val_end": VAL_END},
        stats={k: v for k, v in training.items() if k not in ("generated_by",)},
    )
    logger.info(f"\nOK Entrenamiento {zone} completado → results/training_{zone}.json")
    return training


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento modelo predictivo")
    parser.add_argument("--zone",   default="DK1", choices=["DK1", "DK2", "FI"])
    parser.add_argument("--all",    action="store_true", help="Entrenar DK1, DK2 y FI")
    parser.add_argument("--optuna", action="store_true",
                        help="Volver a buscar hiperparámetros con Optuna (lento)")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help=f"Número de trials Optuna (default: {N_TRIALS})")
    args = parser.parse_args()

    zones = ["DK1", "DK2", "FI"] if args.all else [args.zone]
    for zone in zones:
        train_zone(zone=zone, run_optuna=args.optuna, n_trials=args.trials)
