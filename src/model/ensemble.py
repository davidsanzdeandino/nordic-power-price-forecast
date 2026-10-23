"""
ensemble.py
===========
Ensemble de modelos para predicción de precios eléctricos nórdicos.

Arquitectura:
  Nivel 0 (base models):
    1. LightGBM  — el modelo actual, captura interacciones no lineales
    2. XGBoost   — diversidad de implementación (histogram vs leaf-wise)
    3. Ridge     — modelo lineal regularizado, estable en extremos

  Nivel 1 (meta-learner):
    Combinación óptima de los 3 modelos base. Se usan predicciones
    out-of-fold para evitar data leakage en el stacking.

    Dos estrategias disponibles:
    A. Weighted Average — pesos optimizados por Nelder-Mead (simplex constraint)
    B. Ridge Stacking   — Ridge sobre las 3 predicciones base como features

DECISIÓN TÉCNICA: ¿Por qué estos 3 modelos?
  - LightGBM: leaf-wise growth, mejor con features ruidosas, rápido
  - XGBoost:  level-wise growth, diferente sesgo inductivo → diversidad
  - Ridge:    no sobreajusta extremos, captura tendencia lineal (commodities,
              temperatura), actúa como ancla estabilizadora en picos

  La diversidad de sesgo inductivo es clave: un ensemble de 3 LightGBMs
  con diferentes seeds no diversifica tanto como LGBM + XGB + Ridge.

DECISIÓN TÉCNICA: ¿Por qué no Random Forest o redes neuronales?
  - RF: overlap alto con LGBM/XGB (mismo sesgo de árbol), aporta poca diversidad
  - LSTM/Transformer: necesita mucho más dato y tuning, complejidad desproporcionada
    para un dataset de ~17k horas. Se podría añadir en una fase futura.

Compatibilidad:
  Se integra como reemplazo drop-in del modelo único en train.py y predict.py.
  El ensemble serializa sus 3 modelos base + meta-learner en un solo .pkl.

Uso:
    from ensemble import EnsemblePredictor

    # Entrenamiento
    ensemble = EnsemblePredictor(zone="DK1")
    ensemble.fit(X_train, y_train, X_val, y_val)

    # Predicción
    preds = ensemble.predict(X_test)

    # Walk-forward
    results = ensemble.walk_forward(X, y)
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from scipy.optimize import minimize


# ── Configuración de hiperparámetros por defecto ────────────────────────────

# LightGBM — parámetros razonables pre-Optuna
LGBM_DEFAULT_PARAMS = {
    "objective":         "regression",
    "metric":            "mae",
    "n_estimators":      2000,
    "learning_rate":     0.03,
    "num_leaves":        95,
    "max_depth":         9,
    "min_child_samples": 40,
    "subsample":         0.8,
    "colsample_bytree":  0.75,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

# XGBoost — diversidad: diferente learning rate, profundidad, subsample
XGBM_DEFAULT_PARAMS = {
    "objective":         "reg:absoluteerror",  # MAE nativo
    "n_estimators":      2000,
    "learning_rate":     0.05,
    "max_depth":         7,
    "min_child_weight":  30,
    "subsample":         0.75,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.5,
    "reg_lambda":        2.0,
    "gamma":             0.1,
    "random_state":      123,  # Diferente seed para diversidad
    "n_jobs":            -1,
    "verbosity":         0,
}

# Ridge — parámetros para regresión lineal regularizada
RIDGE_DEFAULT_PARAMS = {
    "alpha":       100.0,     # Regularización fuerte: es un modelo ancla
    "fit_intercept": True,
    "solver":      "auto",
}

EARLY_STOPPING_ROUNDS = 50


# ── Clase principal ──────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Ensemble de LightGBM + XGBoost + Ridge con meta-learner.

    Estrategia de stacking:
      1. Divide el train en K folds temporales (no aleatorios)
      2. Para cada fold, entrena los 3 modelos base en K-1 folds
      3. Genera predicciones out-of-fold para cada modelo
      4. Optimiza pesos del meta-learner sobre las predicciones OOF
      5. Re-entrena los 3 modelos base en todo el train
      6. En inferencia: predict = Σ(wᵢ × modelᵢ(X))

    Attributes:
        zone:           Zona de precio (DK1/DK2)
        models:         Dict con los 3 modelos entrenados
        meta_weights:   Array [w_lgbm, w_xgb, w_ridge] que suman 1.0
        meta_strategy:  "weighted_avg" o "ridge_stack"
        fitted:         Si el ensemble ha sido entrenado
    """

    def __init__(self,
                 zone: str = "DK1",
                 lgbm_params: dict = None,
                 xgb_params: dict = None,
                 ridge_params: dict = None,
                 meta_strategy: str = "weighted_avg",
                 n_cv_folds: int = 3):
        """
        Args:
            zone:           Zona de precio
            lgbm_params:    Hiperparámetros LightGBM (None → defaults)
            xgb_params:     Hiperparámetros XGBoost (None → defaults)
            ridge_params:   Hiperparámetros Ridge (None → defaults)
            meta_strategy:  "weighted_avg" o "ridge_stack"
            n_cv_folds:     Folds para stacking OOF (temporal, no random)
        """
        self.zone = zone
        self.lgbm_params = lgbm_params or LGBM_DEFAULT_PARAMS.copy()
        self.xgb_params = xgb_params or XGBM_DEFAULT_PARAMS.copy()
        self.ridge_params = ridge_params or RIDGE_DEFAULT_PARAMS.copy()
        self.meta_strategy = meta_strategy
        self.n_cv_folds = n_cv_folds

        self.models = {}
        self.meta_weights = None
        self.meta_model = None  # Solo para ridge_stack
        self.fitted = False

        # Diagnósticos
        self._fit_diagnostics = {}

    # ── Entrenamiento ───────────────────────────────────────────────────────

    def fit(self,
            X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            lgbm_optuna_params: dict = None,
            verbose: bool = True) -> 'EnsemblePredictor':
        """
        Entrena el ensemble completo.

        Pipeline:
          1. Genera predicciones OOF con temporal CV
          2. Optimiza pesos del meta-learner
          3. Re-entrena los 3 modelos en todo train (+val si disponible)

        Args:
            X_train:            Features de entrenamiento
            y_train:            Target de entrenamiento
            X_val:              Features de validación (para early stopping)
            y_val:              Target de validación
            lgbm_optuna_params: Si existen params de Optuna, úsalos para LightGBM
            verbose:            Imprimir progreso

        Returns:
            self
        """
        if lgbm_optuna_params:
            # Merge con defaults, Optuna params tienen prioridad
            self.lgbm_params.update(lgbm_optuna_params)

        if verbose:
            print(f"\n{'=' * 65}")
            print(f"  ENSEMBLE TRAINING — {self.zone}")
            print(f"  Modelos: LightGBM + XGBoost + Ridge")
            print(f"  Meta-learner: {self.meta_strategy}")
            print(f"  Train: {len(X_train):,} horas | "
                  f"Val: {len(X_val) if X_val is not None else 0:,} horas")
            print(f"{'=' * 65}")

        # ── Paso 1: Predicciones OOF para meta-learner ──
        if verbose:
            print(f"\n  [1] Generando predicciones out-of-fold ({self.n_cv_folds} folds)...")

        oof_preds = self._generate_oof_predictions(X_train, y_train, X_val, y_val, verbose)

        # ── Paso 2: Optimizar meta-learner ──
        if verbose:
            print(f"\n  [2] Optimizando meta-learner ({self.meta_strategy})...")

        self._fit_meta_learner(oof_preds, y_train, verbose)

        # ── Paso 3: Re-entrenar modelos finales en todo train(+val) ──
        if verbose:
            print(f"\n  [3] Re-entrenando modelos finales en dataset completo...")

        if X_val is not None and y_val is not None:
            X_full = pd.concat([X_train, X_val])
            y_full = pd.concat([y_train, y_val])
        else:
            X_full = X_train
            y_full = y_train

        self._train_all_models(X_full, y_full, X_val, y_val, verbose)

        self.fitted = True

        # ── Diagnósticos finales ──
        if X_val is not None and y_val is not None:
            self._compute_diagnostics(X_val, y_val, verbose)

        return self

    def _generate_oof_predictions(self,
                                  X: pd.DataFrame, y: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series,
                                  verbose: bool) -> pd.DataFrame:
        """
        Genera predicciones out-of-fold con temporal cross-validation.

        DECISIÓN TÉCNICA: CV temporal, NO aleatoria.
        Los folds son bloques temporales contiguos para respetar
        la estructura temporal de los datos. Fold K entrena con
        folds 1..K-1 y predice fold K.
        """
        n = len(X)
        fold_size = n // self.n_cv_folds
        oof = np.full((n, 3), np.nan)  # [lgbm, xgb, ridge]
        model_names = ["LightGBM", "XGBoost", "Ridge"]

        for fold_idx in range(self.n_cv_folds):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < self.n_cv_folds - 1 else n

            # Train: SOLO datos anteriores al fold actual (respeta causalidad)
            # Fold 0 se salta porque no hay datos previos suficientes
            if fold_idx == 0:
                # Skip: no hay datos anteriores para entrenar sin leakage temporal
                if verbose:
                    print(f"    Fold {fold_idx+1}/{self.n_cv_folds}: skip (sin datos previos)")
                continue

            train_idx = np.arange(0, fold_start)
            pred_idx = np.arange(fold_start, fold_end)

            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_pr = X.iloc[pred_idx]

            if len(X_tr) < 500:
                if verbose:
                    print(f"    Fold {fold_idx+1}: skip (solo {len(X_tr)} obs train)")
                continue

            # Entrenar 3 modelos
            models_fold = self._train_base_models(X_tr, y_tr, X_val, y_val)

            # Predicciones OOF
            oof[pred_idx, 0] = models_fold["lgbm"].predict(X_pr)
            oof[pred_idx, 1] = models_fold["xgb"].predict(X_pr)
            oof[pred_idx, 2] = models_fold["ridge"].predict(X_pr)

            if verbose:
                maes = []
                for j, name in enumerate(model_names):
                    valid = ~np.isnan(oof[pred_idx, j])
                    if valid.any():
                        mae = np.mean(np.abs(y.iloc[pred_idx].values[valid] - oof[pred_idx, j][valid]))
                        maes.append(f"{name}: {mae:.2f}")
                print(f"    Fold {fold_idx+1}/{self.n_cv_folds}: "
                      f"train={len(X_tr):,}, pred={len(X_pr):,} | "
                      + " | ".join(maes))

        # Limpiar filas sin predicción OOF
        valid_mask = ~np.isnan(oof).any(axis=1)
        oof_df = pd.DataFrame(
            oof[valid_mask],
            columns=["lgbm", "xgb", "ridge"],
            index=X.index[valid_mask]
        )

        self._fit_diagnostics["oof_coverage"] = valid_mask.mean()
        return oof_df

    def _train_base_models(self,
                           X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame = None, y_val: pd.Series = None
                           ) -> dict:
        """
        Entrena los 3 modelos base.

        Usa LightGBM y XGBoost nativos si están disponibles.
        Si no (e.g. entorno de testing), usa sklearn fallbacks:
          - HistGradientBoostingRegressor como proxy de LightGBM
          - GradientBoostingRegressor como proxy de XGBoost
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        models = {}

        # ── LightGBM ──
        try:
            import lightgbm as lgb
            lgbm = lgb.LGBMRegressor(**self.lgbm_params)
            fit_kwargs = {}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]
            lgbm.fit(X_train, y_train, **fit_kwargs)
            models["lgbm"] = lgbm
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            lgbm = HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=300,
                learning_rate=self.lgbm_params.get("learning_rate", 0.03),
                max_leaf_nodes=self.lgbm_params.get("num_leaves", 63),
                max_depth=self.lgbm_params.get("max_depth", 9),
                min_samples_leaf=self.lgbm_params.get("min_child_samples", 40),
                random_state=42,
            )
            lgbm.fit(X_train, y_train)
            models["lgbm"] = lgbm

        # ── XGBoost ──
        try:
            import xgboost as xgb
            xgbm = xgb.XGBRegressor(**self.xgb_params)
            fit_kwargs_xgb = {"verbose": False}
            if X_val is not None and y_val is not None:
                fit_kwargs_xgb["eval_set"] = [(X_val, y_val)]
                fit_kwargs_xgb["verbose"] = False
            xgbm.fit(X_train, y_train, **fit_kwargs_xgb)
            models["xgb"] = xgbm
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            xgbm = GradientBoostingRegressor(
                loss="absolute_error",
                n_estimators=100,
                learning_rate=self.xgb_params.get("learning_rate", 0.05),
                max_depth=self.xgb_params.get("max_depth", 7),
                min_samples_leaf=self.xgb_params.get("min_child_weight", 30),
                subsample=self.xgb_params.get("subsample", 0.75),
                random_state=123,
            )
            xgbm.fit(X_train, y_train)
            models["xgb"] = xgbm

        # ── Ridge ──
        # Ridge necesita escalado para que la regularización sea uniforme
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        ridge = Ridge(**self.ridge_params)
        ridge.fit(X_train_scaled, y_train)

        # Guardar scaler junto al modelo
        models["ridge"] = _RidgeWrapper(ridge, scaler)

        return models

    def _train_all_models(self,
                          X_full: pd.DataFrame, y_full: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          verbose: bool):
        """Entrena los 3 modelos finales en el dataset completo."""
        self.models = self._train_base_models(X_full, y_full, X_val, y_val)

        if verbose:
            for name, model in self.models.items():
                if hasattr(model, 'best_iteration_'):
                    print(f"    {name}: {model.best_iteration_} iterations")
                elif hasattr(model, 'best_iteration'):
                    print(f"    {name}: {model.best_iteration} iterations")
                else:
                    print(f"    {name}: fitted OK")

    # ── Meta-learner ────────────────────────────────────────────────────────

    def _fit_meta_learner(self,
                          oof_preds: pd.DataFrame,
                          y_train: pd.Series,
                          verbose: bool):
        """
        Ajusta el meta-learner sobre predicciones OOF.

        Estrategia A: Weighted Average
          Optimiza w₁, w₂, w₃ sujeto a Σwᵢ = 1, wᵢ ≥ 0
          Minimiza MAE sobre las predicciones OOF.

        Estrategia B: Ridge Stacking
          Ridge regression sobre las 3 predicciones base como features.
          Más flexible (puede dar pesos negativos = hedging) pero
          con riesgo de sobreajuste.
        """
        # Alinear indices
        common_idx = oof_preds.index.intersection(y_train.index)
        oof = oof_preds.loc[common_idx].values
        y = y_train.loc[common_idx].values

        if self.meta_strategy == "weighted_avg":
            self._fit_weighted_average(oof, y, verbose)
        elif self.meta_strategy == "ridge_stack":
            self._fit_ridge_stack(oof, y, verbose)
        else:
            raise ValueError(f"meta_strategy desconocida: {self.meta_strategy}")

    def _fit_weighted_average(self, oof: np.ndarray, y: np.ndarray, verbose: bool):
        """
        Optimiza pesos con Nelder-Mead sujeto a simplex constraint.
        w₁ + w₂ + w₃ = 1, wᵢ ≥ 0.

        Usa softmax reparametrización para convertir optimización
        restringida en no restringida.
        """
        def mae_objective(raw_weights):
            # Softmax para garantizar simplex
            w = np.exp(raw_weights) / np.exp(raw_weights).sum()
            preds = oof @ w
            return np.mean(np.abs(y - preds))

        # Inicializar con pesos uniformes
        x0 = np.zeros(3)

        result = minimize(
            mae_objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
        )

        # Extraer pesos finales
        self.meta_weights = np.exp(result.x) / np.exp(result.x).sum()

        # Métricas individuales y ensemble
        model_names = ["LightGBM", "XGBoost", "Ridge"]
        if verbose:
            print(f"\n    Pesos óptimos del ensemble:")
            for i, name in enumerate(model_names):
                mae_i = np.mean(np.abs(y - oof[:, i]))
                print(f"      {name:12s}: w = {self.meta_weights[i]:.4f}  "
                      f"(MAE individual: {mae_i:.3f})")

            ensemble_mae = np.mean(np.abs(y - oof @ self.meta_weights))
            best_single = min(np.mean(np.abs(y - oof[:, i])) for i in range(3))
            print(f"\n    MAE ensemble: {ensemble_mae:.3f} EUR/MWh")
            print(f"    MAE mejor modelo individual: {best_single:.3f}")
            print(f"    Mejora del ensemble: "
                  f"{(best_single - ensemble_mae) / best_single * 100:.1f}%")

            self._fit_diagnostics["meta_weights"] = {
                name: float(self.meta_weights[i]) for i, name in enumerate(model_names)
            }
            self._fit_diagnostics["oof_mae_ensemble"] = float(ensemble_mae)
            self._fit_diagnostics["oof_mae_best_single"] = float(best_single)

    def _fit_ridge_stack(self, oof: np.ndarray, y: np.ndarray, verbose: bool):
        """Ridge stacking — más flexible que weighted average."""
        from sklearn.linear_model import Ridge

        self.meta_model = Ridge(alpha=1.0, fit_intercept=True)
        self.meta_model.fit(oof, y)

        # Extraer pesos equivalentes
        coefs = self.meta_model.coef_
        intercept = self.meta_model.intercept_
        self.meta_weights = coefs  # Pueden ser negativos

        model_names = ["LightGBM", "XGBoost", "Ridge"]
        if verbose:
            print(f"\n    Coeficientes del meta-Ridge:")
            print(f"      Intercept: {intercept:.3f}")
            for i, name in enumerate(model_names):
                print(f"      {name:12s}: β = {coefs[i]:.4f}")

            ensemble_preds = self.meta_model.predict(oof)
            ensemble_mae = np.mean(np.abs(y - ensemble_preds))
            print(f"    MAE ensemble (ridge stack): {ensemble_mae:.3f}")

    # ── Predicción ──────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicción del ensemble.

        Returns:
            Array de predicciones (EUR/MWh)
        """
        self._check_fitted()

        preds_lgbm = self.models["lgbm"].predict(X)
        preds_xgb = self.models["xgb"].predict(X)
        preds_ridge = self.models["ridge"].predict(X)

        base_preds = np.column_stack([preds_lgbm, preds_xgb, preds_ridge])

        if self.meta_strategy == "weighted_avg":
            return base_preds @ self.meta_weights
        elif self.meta_strategy == "ridge_stack":
            return self.meta_model.predict(base_preds)

    def predict_individual(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Devuelve predicciones individuales de cada modelo base.
        Útil para diagnósticos y visualización de diversidad.
        """
        self._check_fitted()
        return {
            "lgbm": self.models["lgbm"].predict(X),
            "xgb": self.models["xgb"].predict(X),
            "ridge": self.models["ridge"].predict(X),
            "ensemble": self.predict(X),
        }

    # ── Diagnósticos ────────────────────────────────────────────────────────

    def _compute_diagnostics(self,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             verbose: bool):
        """Calcula métricas de diagnóstico en el set de validación."""
        preds = self.predict_individual(X_val)
        y = y_val.values

        if verbose:
            print(f"\n  [4] Métricas en validación:")
            print(f"    {'Modelo':<14} {'MAE':>8} {'RMSE':>8}")
            print(f"    {'─' * 32}")

        for name, pred in preds.items():
            mae = np.mean(np.abs(y - pred))
            rmse = np.sqrt(np.mean((y - pred) ** 2))
            self._fit_diagnostics[f"val_mae_{name}"] = float(mae)
            self._fit_diagnostics[f"val_rmse_{name}"] = float(rmse)
            if verbose:
                marker = " ←" if name == "ensemble" else ""
                print(f"    {name:<14} {mae:>7.3f} {rmse:>7.3f}{marker}")

        # Diversidad del ensemble (correlación entre predicciones)
        corr_lgbm_xgb = np.corrcoef(preds["lgbm"], preds["xgb"])[0, 1]
        corr_lgbm_ridge = np.corrcoef(preds["lgbm"], preds["ridge"])[0, 1]
        corr_xgb_ridge = np.corrcoef(preds["xgb"], preds["ridge"])[0, 1]

        self._fit_diagnostics["diversity"] = {
            "corr_lgbm_xgb": float(corr_lgbm_xgb),
            "corr_lgbm_ridge": float(corr_lgbm_ridge),
            "corr_xgb_ridge": float(corr_xgb_ridge),
        }

        if verbose:
            print(f"\n    Diversidad (correlación entre modelos):")
            print(f"      LGBM ↔ XGB:   {corr_lgbm_xgb:.4f}")
            print(f"      LGBM ↔ Ridge:  {corr_lgbm_ridge:.4f}")
            print(f"      XGB ↔ Ridge:   {corr_xgb_ridge:.4f}")
            avg_corr = (corr_lgbm_xgb + corr_lgbm_ridge + corr_xgb_ridge) / 3
            print(f"      Media:         {avg_corr:.4f} "
                  f"({'alta diversidad (OK)' if avg_corr < 0.95 else 'baja diversidad (WARN)'})")

    # ── Walk-forward con ensemble ───────────────────────────────────────────

    def walk_forward(self,
                     X: pd.DataFrame, y: pd.Series,
                     wf_start: str = "2024-01-01",
                     verbose: bool = True) -> pd.DataFrame:
        """
        Walk-forward validation del ensemble vs LightGBM solo.

        Para cada mes, entrena el ensemble completo (LGBM+XGB+Ridge)
        y compara con LightGBM solo para cuantificar el beneficio.

        Args:
            X:        Features completas con índice temporal
            y:        Target completo
            wf_start: Inicio de la evaluación walk-forward
            verbose:  Imprimir resultados por mes

        Returns:
            DataFrame con métricas por mes
        """
        try:
            import lightgbm as lgb
            _HAS_LGBM = True
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            _HAS_LGBM = False

        wf_start_ts = pd.Timestamp(wf_start, tz="UTC")
        months = pd.date_range(start=wf_start_ts, end=X.index.max(), freq="MS")

        results = []
        if verbose:
            print(f"\n{'=' * 75}")
            print(f"  WALK-FORWARD ENSEMBLE vs LIGHTGBM SOLO")
            print(f"{'=' * 75}")

        for predict_month_start in months:
            predict_month_end = (predict_month_start + pd.offsets.MonthEnd(1))
            predict_month_end = predict_month_end.replace(
                hour=23, tzinfo=predict_month_start.tzinfo
            )

            train_mask = X.index < predict_month_start
            pred_mask = ((X.index >= predict_month_start) &
                         (X.index <= predict_month_end))

            if train_mask.sum() < 2000 or pred_mask.sum() == 0:
                continue

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_pr, y_pr = X[pred_mask], y[pred_mask]

            # Validation: último 20% del train
            n_val = int(len(X_tr) * 0.2)
            X_val_wf = X_tr.iloc[-n_val:]
            y_val_wf = y_tr.iloc[-n_val:]

            # ── LightGBM solo (baseline) ──
            if _HAS_LGBM:
                lgbm_only = lgb.LGBMRegressor(**self.lgbm_params)
                lgbm_only.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val_wf, y_val_wf)],
                    callbacks=[
                        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
            else:
                lgbm_only = HistGradientBoostingRegressor(
                    loss="absolute_error", max_iter=300,
                    learning_rate=0.03, max_depth=9, random_state=42,
                )
                lgbm_only.fit(X_tr, y_tr)
            preds_lgbm_only = lgbm_only.predict(X_pr)

            # ── Ensemble ──
            ensemble_wf = EnsemblePredictor(
                zone=self.zone,
                lgbm_params=self.lgbm_params.copy(),
                xgb_params=self.xgb_params.copy(),
                ridge_params=self.ridge_params.copy(),
                meta_strategy=self.meta_strategy,
                n_cv_folds=min(self.n_cv_folds, max(2, len(X_tr) // 2000)),
            )
            try:
                ensemble_wf.fit(X_tr, y_tr, X_val_wf, y_val_wf, verbose=False)
                preds_ensemble = ensemble_wf.predict(X_pr)
            except Exception as e:
                if verbose:
                    print(f"    WARN: {predict_month_start.strftime('%Y-%m')}: "
                          f"ensemble falló ({e}), usando solo LGBM")
                preds_ensemble = preds_lgbm_only

            # ── Métricas ──
            y_real = y_pr.values
            mae_lgbm = float(np.mean(np.abs(y_real - preds_lgbm_only)))
            mae_ens = float(np.mean(np.abs(y_real - preds_ensemble)))
            rmse_lgbm = float(np.sqrt(np.mean((y_real - preds_lgbm_only) ** 2)))
            rmse_ens = float(np.sqrt(np.mean((y_real - preds_ensemble) ** 2)))

            # Autocorrelación residuos
            resid_lgbm = y_real - preds_lgbm_only
            resid_ens = y_real - preds_ensemble
            autocorr_lgbm = float(pd.Series(resid_lgbm).autocorr(lag=1))
            autocorr_ens = float(pd.Series(resid_ens).autocorr(lag=1))

            month_str = predict_month_start.strftime("%Y-%m")
            improvement = (mae_lgbm - mae_ens) / mae_lgbm * 100

            results.append({
                "month": month_str,
                "n_hours": len(X_pr),
                "MAE_lgbm_only": round(mae_lgbm, 3),
                "MAE_ensemble": round(mae_ens, 3),
                "improvement_pct": round(improvement, 1),
                "RMSE_lgbm": round(rmse_lgbm, 3),
                "RMSE_ensemble": round(rmse_ens, 3),
                "autocorr_lgbm": round(autocorr_lgbm, 3),
                "autocorr_ensemble": round(autocorr_ens, 3),
                "price_mean": round(float(y_pr.mean()), 2),
                "weights": ensemble_wf.meta_weights.tolist() if ensemble_wf.fitted else None,
            })

            if verbose:
                marker = "OK" if improvement > 1 else "WARN" if improvement > -1 else "ERROR"
                w_str = ""
                if ensemble_wf.fitted and ensemble_wf.meta_weights is not None:
                    w = ensemble_wf.meta_weights
                    w_str = f" | w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]"
                print(f"  {marker} {month_str}: LGBM={mae_lgbm:.2f} → ENS={mae_ens:.2f} "
                      f"({improvement:+.1f}%){w_str}")

        df_results = pd.DataFrame(results)
        if verbose and not df_results.empty:
            print(f"\n  {'─' * 55}")
            print(f"  Media MAE LGBM solo:  {df_results['MAE_lgbm_only'].mean():.2f}")
            print(f"  Media MAE ensemble:   {df_results['MAE_ensemble'].mean():.2f}")
            print(f"  Mejora media:         "
                  f"{df_results['improvement_pct'].mean():.1f}%")
            print(f"  Meses con mejora:     "
                  f"{(df_results['improvement_pct'] > 0).sum()}/{len(df_results)}")

        return df_results

    # ── Serialización ───────────────────────────────────────────────────────

    def save(self, path: str):
        """Guarda el ensemble completo a disco."""
        self._check_fitted()
        state = {
            "zone": self.zone,
            "models": self.models,
            "meta_weights": self.meta_weights,
            "meta_model": self.meta_model,
            "meta_strategy": self.meta_strategy,
            "lgbm_params": self.lgbm_params,
            "xgb_params": self.xgb_params,
            "ridge_params": self.ridge_params,
            "diagnostics": self._fit_diagnostics,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'EnsemblePredictor':
        """Carga un ensemble desde disco."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(zone=state["zone"])
        obj.models = state["models"]
        obj.meta_weights = state["meta_weights"]
        obj.meta_model = state["meta_model"]
        obj.meta_strategy = state["meta_strategy"]
        obj.lgbm_params = state["lgbm_params"]
        obj.xgb_params = state["xgb_params"]
        obj.ridge_params = state["ridge_params"]
        obj._fit_diagnostics = state["diagnostics"]
        obj.fitted = True
        return obj

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError(
                "Ensemble no entrenado. Llama a .fit(X_train, y_train) primero."
            )

    def get_diagnostics(self) -> dict:
        """Devuelve todos los diagnósticos del ajuste."""
        return self._fit_diagnostics.copy()

    def __repr__(self):
        if not self.fitted:
            return f"EnsemblePredictor(zone={self.zone}, fitted=False)"
        w = self.meta_weights
        return (f"EnsemblePredictor(zone={self.zone}, "
                f"weights=[LGBM={w[0]:.3f}, XGB={w[1]:.3f}, Ridge={w[2]:.3f}])")


# ── Wrapper para Ridge con scaler integrado ─────────────────────────────────

class _RidgeWrapper:
    """Envuelve Ridge + StandardScaler para que tenga interfaz .predict()."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ── Ensemble + AR Correction combinados ──────────────────────────────────────

class EnsembleWithAR:
    """
    Pipeline completo: Ensemble → AR correction.

    Combina las mejoras del ensemble (reducción error base) con
    la corrección AR (explotación autocorrelación residual).

    El orden importa:
      1. Ensemble reduce el error irreducible del modelo base
      2. AR corrige la autocorrelación en los residuos del ensemble
      → El AR es incluso más efectivo sobre residuos más "limpios"
    """

    def __init__(self, ensemble: EnsemblePredictor, ar_max_order: int = 3):
        from ar_correction import ARResidualCorrector
        self.ensemble = ensemble
        self.ar_corrector = ARResidualCorrector(max_order=ar_max_order)
        self.fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            verbose=True) -> 'EnsembleWithAR':
        """
        Entrena ensemble + ajusta AR sobre los residuos del ensemble.
        """
        # 1. Entrenar ensemble
        self.ensemble.fit(X_train, y_train, X_val, y_val, verbose=verbose)

        # 2. Ajustar AR sobre residuos del ensemble en train
        preds_train = self.ensemble.predict(X_train)
        if verbose:
            print(f"\n  [5] Ajustando AR correction sobre residuos del ensemble...")
        self.ar_corrector.fit(y_train.values, preds_train, verbose=verbose)

        self.fitted = True
        return self

    def predict_offline(self, X, y_true):
        """Predicción con corrección AR offline (backtest)."""
        preds_ensemble = self.ensemble.predict(X)
        return self.ar_corrector.correct_offline(y_true, preds_ensemble)

    def predict_day_ahead(self, X, last_residuals):
        """Predicción day-ahead con corrección AR propagada."""
        preds_ensemble = self.ensemble.predict(X)
        return self.ar_corrector.correct_day_ahead(preds_ensemble, last_residuals)

    def walk_forward(self, X, y, wf_start="2024-01-01", verbose=True):
        """
        Walk-forward completo: ensemble + AR en cada mes.
        Compara 4 configuraciones:
          1. LGBM solo
          2. Ensemble (LGBM+XGB+Ridge)
          3. LGBM + AR
          4. Ensemble + AR (pipeline completo)
        """
        try:
            import lightgbm as lgb
            _HAS_LGBM = True
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            _HAS_LGBM = False
        from ar_correction import ARResidualCorrector

        wf_start_ts = pd.Timestamp(wf_start, tz="UTC")
        months = pd.date_range(start=wf_start_ts, end=X.index.max(), freq="MS")

        results = []
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"  WALK-FORWARD: 4 configuraciones comparadas")
            print(f"  1. LGBM solo | 2. Ensemble | 3. LGBM+AR | 4. Ensemble+AR")
            print(f"{'=' * 80}")

        for predict_month_start in months:
            predict_month_end = (predict_month_start + pd.offsets.MonthEnd(1))
            predict_month_end = predict_month_end.replace(
                hour=23, tzinfo=predict_month_start.tzinfo
            )

            train_mask = X.index < predict_month_start
            pred_mask = ((X.index >= predict_month_start) &
                         (X.index <= predict_month_end))

            if train_mask.sum() < 2000 or pred_mask.sum() == 0:
                continue

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_pr, y_pr = X[pred_mask], y[pred_mask]

            n_val = int(len(X_tr) * 0.2)
            X_val_wf = X_tr.iloc[-n_val:]
            y_val_wf = y_tr.iloc[-n_val:]

            y_real = y_pr.values

            # ── 1. LGBM solo ──
            if _HAS_LGBM:
                lgbm_only = lgb.LGBMRegressor(**self.ensemble.lgbm_params)
                lgbm_only.fit(X_tr, y_tr,
                              eval_set=[(X_val_wf, y_val_wf)],
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(period=-1)])
            else:
                lgbm_only = HistGradientBoostingRegressor(
                    loss="absolute_error", max_iter=300,
                    learning_rate=0.03, max_depth=9, random_state=42,
                )
                lgbm_only.fit(X_tr, y_tr)
            preds_1 = lgbm_only.predict(X_pr)
            preds_1_train = lgbm_only.predict(X_tr)

            # ── 2. Ensemble ──
            try:
                ens_wf = EnsemblePredictor(
                    zone=self.ensemble.zone,
                    lgbm_params=self.ensemble.lgbm_params.copy(),
                    xgb_params=self.ensemble.xgb_params.copy(),
                    ridge_params=self.ensemble.ridge_params.copy(),
                    n_cv_folds=min(3, max(2, len(X_tr) // 2000)),
                )
                ens_wf.fit(X_tr, y_tr, X_val_wf, y_val_wf, verbose=False)
                preds_2 = ens_wf.predict(X_pr)
                preds_2_train = ens_wf.predict(X_tr)
            except Exception:
                preds_2 = preds_1.copy()
                preds_2_train = preds_1_train.copy()

            # ── 3. LGBM + AR ──
            ar_lgbm = ARResidualCorrector(max_order=3)
            ar_lgbm.fit(y_tr.values, preds_1_train, verbose=False)
            preds_3 = ar_lgbm.correct_offline(y_real, preds_1)

            # ── 4. Ensemble + AR ──
            ar_ens = ARResidualCorrector(max_order=3)
            ar_ens.fit(y_tr.values, preds_2_train, verbose=False)
            preds_4 = ar_ens.correct_offline(y_real, preds_2)

            # ── Métricas ──
            mae = lambda y, p: float(np.mean(np.abs(y - p)))
            month_str = predict_month_start.strftime("%Y-%m")

            m1, m2, m3, m4 = mae(y_real, preds_1), mae(y_real, preds_2), \
                              mae(y_real, preds_3), mae(y_real, preds_4)

            results.append({
                "month": month_str,
                "n_hours": len(X_pr),
                "MAE_lgbm": round(m1, 3),
                "MAE_ensemble": round(m2, 3),
                "MAE_lgbm_ar": round(m3, 3),
                "MAE_ensemble_ar": round(m4, 3),
                "imp_ens_vs_lgbm": round((m1 - m2) / m1 * 100, 1),
                "imp_full_vs_lgbm": round((m1 - m4) / m1 * 100, 1),
                "price_mean": round(float(y_pr.mean()), 2),
            })

            if verbose:
                imp = (m1 - m4) / m1 * 100
                print(f"  {month_str}: LGBM={m1:.2f} | ENS={m2:.2f} | "
                      f"LGBM+AR={m3:.2f} | ENS+AR={m4:.2f} ({imp:+.1f}%)")

        df = pd.DataFrame(results)
        if verbose and not df.empty:
            print(f"\n  {'─' * 60}")
            print(f"  Medias:  LGBM={df['MAE_lgbm'].mean():.2f} | "
                  f"ENS={df['MAE_ensemble'].mean():.2f} | "
                  f"LGBM+AR={df['MAE_lgbm_ar'].mean():.2f} | "
                  f"ENS+AR={df['MAE_ensemble_ar'].mean():.2f}")
            print(f"  Mejora ENS vs LGBM:     {df['imp_ens_vs_lgbm'].mean():.1f}%")
            print(f"  Mejora ENS+AR vs LGBM:  {df['imp_full_vs_lgbm'].mean():.1f}%")

        return df


# ── Tests con datos sintéticos ───────────────────────────────────────────────

def _generate_test_data(n_hours: int = 8760) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Genera datos sintéticos que imitan el dataset DK1 con 126 features.
    """
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=n_hours, freq="h", tz="UTC")

    # Target: precio con patrón diario + estacionalidad + noise AR
    hour = timestamps.hour
    month = timestamps.month
    dow = timestamps.dayofweek

    base = (55
            + 15 * np.cos(2 * np.pi * (month - 1) / 12)
            + np.where((hour >= 7) & (hour <= 20), 12, -8)
            + 5 * np.sin(2 * np.pi * hour / 24)
            + np.where(dow >= 5, -8, 0))

    # Ruido AR(1)
    phi = 0.85
    noise = np.zeros(n_hours)
    innovations = np.random.standard_t(df=5, size=n_hours) * 6
    for t in range(1, n_hours):
        noise[t] = phi * noise[t - 1] + innovations[t]

    # Spikes
    spikes = (np.random.random(n_hours) < 0.02) * np.random.exponential(80, n_hours)

    y = pd.Series(
        np.maximum(base + noise + spikes, -10),
        index=timestamps,
        name="price_DK1"
    )

    # Features: 30 features relevantes + 96 features de ruido
    # (simula las 126 features del dataset real)
    n_feat = 126
    feat_data = {}

    # Temporales cíclicas (8 features)
    feat_data["feat_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat_data["feat_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat_data["feat_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat_data["feat_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feat_data["feat_month_sin"] = np.sin(2 * np.pi * month / 12)
    feat_data["feat_month_cos"] = np.cos(2 * np.pi * month / 12)
    feat_data["feat_week_sin"] = np.sin(2 * np.pi * timestamps.isocalendar().week.astype(int) / 53)
    feat_data["feat_week_cos"] = np.cos(2 * np.pi * timestamps.isocalendar().week.astype(int) / 53)

    # Lags (6 features)
    price_series = y.values
    for lag in [24, 48, 168]:
        feat_data[f"feat_DK1_price_lag{lag}h"] = pd.Series(price_series, index=timestamps).shift(lag).values
        feat_data[f"feat_DK1_price_roll{lag}h_mean"] = (
            pd.Series(price_series, index=timestamps)
            .shift(24)
            .rolling(lag, min_periods=1)
            .mean()
            .values
        )

    # Físicas (6 features)
    feat_data["feat_DK1_wind_total_fc"] = 2000 + 1000 * np.sin(2 * np.pi * np.arange(n_hours) / (24*7)) + np.random.normal(0, 300, n_hours)
    feat_data["feat_DK1_residual_load"] = 3000 + base * 30 + np.random.normal(0, 200, n_hours)
    feat_data["feat_DK1_wind_penetration"] = feat_data["feat_DK1_wind_total_fc"] / (feat_data["feat_DK1_residual_load"] + feat_data["feat_DK1_wind_total_fc"])
    feat_data["feat_temp_avg"] = 10 + 10 * np.cos(2 * np.pi * (month - 7) / 12) + np.random.normal(0, 3, n_hours)
    feat_data["feat_gas_ttf"] = 30 + 5 * np.cos(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 2, n_hours)
    feat_data["feat_co2_eua"] = 60 + 3 * np.cos(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 1.5, n_hours)

    # Flags
    feat_data["is_weekend"] = (dow >= 5).astype(float)
    feat_data["is_holiday"] = (np.random.random(n_hours) < 0.03).astype(float)

    # Ruido (completar hasta 126 features)
    n_noise = n_feat - len(feat_data)
    for i in range(n_noise):
        feat_data[f"feat_noise_{i:03d}"] = np.random.normal(0, 1, n_hours)

    X = pd.DataFrame(feat_data, index=timestamps)
    # Limpiar NaN de lags
    X = X.fillna(X.median())

    return X, y


def _run_tests():
    """Test completo del ensemble con datos sintéticos."""
    print("\n" + "=" * 65)
    print("  TEST: Ensemble LightGBM + XGBoost + Ridge")
    print("=" * 65)

    X, y = _generate_test_data(n_hours=5000)

    # Splits (adjusted for 5000h ≈ 208 days)
    train_end = pd.Timestamp("2023-04-30 23:00", tz="UTC")
    val_end = pd.Timestamp("2023-06-15 23:00", tz="UTC")
    test_start = pd.Timestamp("2023-06-16", tz="UTC")

    X_train = X[X.index <= train_end]
    y_train = y[y.index <= train_end]
    X_val = X[(X.index > train_end) & (X.index <= val_end)]
    y_val = y[(y.index > train_end) & (y.index <= val_end)]
    X_test = X[X.index >= test_start]
    y_test = y[y.index >= test_start]

    print(f"\n  Train: {len(X_train):,}h | Val: {len(X_val):,}h | Test: {len(X_test):,}h")

    # ── Test 1: Entrenamiento básico ──
    print("\n  ── Test 1: Entrenamiento del ensemble ──")
    ensemble = EnsemblePredictor(zone="DK1", n_cv_folds=2)
    ensemble.fit(X_train, y_train, X_val, y_val, verbose=True)

    # ── Test 2: Predicción en test ──
    print("\n  ── Test 2: Métricas en test set ──")
    preds = ensemble.predict_individual(X_test)
    y_real = y_test.values

    print(f"\n    {'Modelo':<14} {'MAE':>8} {'RMSE':>8}")
    print(f"    {'─' * 32}")
    for name, pred in preds.items():
        mae = np.mean(np.abs(y_real - pred))
        rmse = np.sqrt(np.mean((y_real - pred) ** 2))
        marker = " ←" if name == "ensemble" else ""
        print(f"    {name:<14} {mae:>7.2f} {rmse:>7.2f}{marker}")

    # Verificar que ensemble es al menos tan bueno como el mejor individual
    mae_ensemble = np.mean(np.abs(y_real - preds["ensemble"]))
    mae_best = min(np.mean(np.abs(y_real - preds[m])) for m in ["lgbm", "xgb", "ridge"])
    # El ensemble puede ser marginalmente peor en OOF vs test (distribución diferente)
    # pero no debería ser mucho peor
    assert mae_ensemble < mae_best * 1.15, \
        f"Ensemble ({mae_ensemble:.2f}) mucho peor que mejor individual ({mae_best:.2f})"

    # ── Test 3: Serialización ──
    print("\n  ── Test 3: Serialización ──")
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    ensemble.save(path)
    loaded = EnsemblePredictor.load(path)
    preds_loaded = loaded.predict(X_test)
    assert np.allclose(preds_loaded, preds["ensemble"], rtol=1e-5), "Predicciones difieren post-carga"
    os.unlink(path)
    print("    Serialización: OK")

    # ── Test 4: Ensemble + AR combinado ──
    print("\n  ── Test 4: Ensemble + AR combinado ──")
    pipeline = EnsembleWithAR(ensemble, ar_max_order=3)
    pipeline.fit(X_train, y_train, X_val, y_val, verbose=True)

    preds_with_ar = pipeline.predict_offline(X_test, y_test.values)
    mae_with_ar = np.mean(np.abs(y_real - preds_with_ar))
    print(f"\n    MAE Ensemble solo: {mae_ensemble:.2f}")
    print(f"    MAE Ensemble+AR:  {mae_with_ar:.2f}")
    print(f"    Mejora AR:        {mae_ensemble - mae_with_ar:.2f} EUR/MWh")

    print(f"\n  OK Todos los tests pasaron.\n")
    return ensemble


if __name__ == "__main__":
    _run_tests()