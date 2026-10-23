"""
ar_correction.py
================
Corrección AR(1-3) de residuos post-LightGBM.

Motivación:
  Los residuos del LightGBM tienen autocorrelación lag-1h = 0.898.
  Esto significa que el error en hora t predice ~90% del error en t+1.
  Un modelo AR sobre los residuos captura esta señal "gratis" y corrige
  las predicciones sin reentrenar el modelo base.

Estrategia:
  1. Fase de ajuste: calcula residuos en train, ajusta AR(p) con p ∈ {1,2,3}
  2. Fase de corrección: pred_corregida = pred_lgbm + AR_forecast(residuos_recientes)

Modos de operación:
  A. OFFLINE (evaluación / backtest):
     Recorrido secuencial hora a hora. En cada hora t, el residuo real
     de t-1 ya es conocido → se usa directamente para corregir t.
     Esto es lo que da la mejora máxima y es el escenario realista
     para near-real-time o intraday corrections.

  B. DAY-AHEAD (producción):
     A las 12:00 del día D se predicen las 24h del día D+1.
     No se conocen los residuos de D+1. Pero sí se conocen hasta la
     hora actual del día D. El último residuo conocido se propaga
     con decaimiento AR a las 24-36h futuras.

  C. ONLINE ROLLING (intraday updates):
     Conforme avanza el día D+1 y se conocen precios reales,
     se actualiza la corrección para las horas restantes.
     Es la combinación más potente: day-ahead + actualización horaria.

Compatibilidad:
  Se integra con train.py (entrenamiento), predict.py (inferencia)
  y evaluate.py (evaluación). Los parámetros AR se serializan junto
  al modelo LightGBM.

Uso:
    from ar_correction import ARResidualCorrector

    # Ajuste (después de entrenar LightGBM)
    corrector = ARResidualCorrector(max_order=3)
    corrector.fit(y_train, preds_train)

    # Corrección offline (evaluación)
    preds_corrected = corrector.correct_offline(y_test, preds_test)

    # Corrección day-ahead (producción)
    preds_corrected = corrector.correct_day_ahead(
        preds_tomorrow, last_known_residuals
    )
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List


class ARResidualCorrector:
    """
    Corrector de residuos basado en modelos AR(p).

    Ajusta un modelo autorregresivo sobre los residuos (y_true - y_pred)
    del modelo base (LightGBM) y lo usa para corregir predicciones futuras.

    Attributes:
        max_order:    Orden máximo del AR a probar (1, 2, o 3)
        best_order:   Orden seleccionado por AIC
        ar_coefs:     Coeficientes AR [φ₁, φ₂, ..., φₚ]
        intercept:    Intercepto del modelo AR (típicamente ~0)
        residual_std: Desviación estándar de los residuos del AR
        fitted:       Si el modelo ha sido ajustado
    """

    def __init__(self, max_order: int = 3):
        """
        Args:
            max_order: Orden máximo del AR a considerar (1-3).
                       Se selecciona el mejor por AIC.
        """
        assert 1 <= max_order <= 5, "max_order debe estar entre 1 y 5"
        self.max_order = max_order
        self.best_order = None
        self.ar_coefs = None
        self.intercept = 0.0
        self.residual_std = None
        self.fitted = False

        # Diagnósticos del ajuste
        self._fit_diagnostics = {}

    # ── Ajuste ──────────────────────────────────────────────────────────────

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray,
            verbose: bool = True) -> 'ARResidualCorrector':
        """
        Ajusta el modelo AR sobre los residuos del modelo base.

        Args:
            y_true: Valores reales (array 1D, ordenado temporalmente)
            y_pred: Predicciones del modelo base (mismo tamaño que y_true)
            verbose: Imprimir diagnósticos del ajuste

        Returns:
            self (para encadenamiento)
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        assert len(y_true) == len(y_pred), "y_true y y_pred deben tener el mismo tamaño"

        residuals = y_true - y_pred

        # Eliminar NaN (pueden venir de horas faltantes)
        valid = ~np.isnan(residuals)
        residuals_clean = residuals[valid]

        if len(residuals_clean) < 100:
            raise ValueError(
                f"Solo {len(residuals_clean)} residuos válidos. "
                "Se necesitan al menos 100 para un ajuste AR robusto."
            )

        # Probar órdenes AR 1..max_order, seleccionar por AIC
        best_aic = np.inf
        results_by_order = {}

        for order in range(1, self.max_order + 1):
            coefs, intercept, aic, residual_std = self._fit_ar_ols(
                residuals_clean, order
            )
            results_by_order[order] = {
                "coefs": coefs,
                "intercept": intercept,
                "aic": aic,
                "residual_std": residual_std,
            }
            if aic < best_aic:
                best_aic = aic
                self.best_order = order
                self.ar_coefs = coefs
                self.intercept = intercept
                self.residual_std = residual_std

        self.fitted = True
        self._fit_diagnostics = {
            "n_residuals": len(residuals_clean),
            "residual_mean": float(np.mean(residuals_clean)),
            "residual_std_raw": float(np.std(residuals_clean)),
            "results_by_order": results_by_order,
            "best_order": self.best_order,
            "best_aic": best_aic,
            "autocorr_before": {
                lag: float(pd.Series(residuals_clean).autocorr(lag=lag))
                for lag in [1, 2, 3, 24]
            },
        }

        # Calcular autocorrelación de los residuos del AR (post-corrección)
        ar_residuals = self._compute_ar_residuals(residuals_clean)
        self._fit_diagnostics["autocorr_after"] = {
            lag: float(pd.Series(ar_residuals).autocorr(lag=lag))
            for lag in [1, 2, 3, 24]
        }
        self._fit_diagnostics["std_reduction_pct"] = (
            (1 - self.residual_std / self._fit_diagnostics["residual_std_raw"]) * 100
        )

        if verbose:
            self._print_diagnostics()

        return self

    def _fit_ar_ols(self, residuals: np.ndarray, order: int
                    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Ajusta un AR(p) por OLS (Ordinary Least Squares).

        Más robusto y rápido que statsmodels para órdenes bajos.
        Equivalente a una regresión lineal con lags como predictores.

        Returns:
            (coefs, intercept, aic, residual_std)
        """
        n = len(residuals)

        # Construir matriz de diseño con lags
        # y = residuals[order:], X = [residuals[order-1:-1], ..., residuals[0:n-order]]
        y = residuals[order:]
        X = np.column_stack([
            residuals[order - lag: n - lag] for lag in range(1, order + 1)
        ])

        # Añadir columna de intercepto
        X_with_intercept = np.column_stack([np.ones(len(y)), X])

        # OLS: β = (X'X)^(-1) X'y
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback: regularización mínima
            XtX = X_with_intercept.T @ X_with_intercept
            XtX += 1e-8 * np.eye(XtX.shape[0])
            beta = np.linalg.solve(XtX, X_with_intercept.T @ y)

        intercept = beta[0]
        coefs = beta[1:]

        # Residuos del AR
        y_hat = X_with_intercept @ beta
        ar_resid = y - y_hat
        residual_std = float(np.std(ar_resid))

        # AIC = n * ln(σ²) + 2k
        n_obs = len(y)
        k = order + 1  # coefs + intercept
        sigma2 = np.mean(ar_resid ** 2)
        aic = n_obs * np.log(sigma2 + 1e-12) + 2 * k

        return coefs, intercept, aic, residual_std

    def _compute_ar_residuals(self, residuals: np.ndarray) -> np.ndarray:
        """Calcula los residuos del modelo AR (innovaciones)."""
        n = len(residuals)
        order = self.best_order
        ar_resid = np.full(n, np.nan)

        for t in range(order, n):
            pred = self.intercept + sum(
                self.ar_coefs[lag] * residuals[t - lag - 1]
                for lag in range(order)
            )
            ar_resid[t] = residuals[t] - pred

        return ar_resid[order:]

    def _print_diagnostics(self):
        """Imprime un resumen completo del ajuste."""
        d = self._fit_diagnostics
        print("\n" + "=" * 65)
        print("  AR RESIDUAL CORRECTOR — Diagnósticos del ajuste")
        print("=" * 65)
        print(f"  Residuos analizados:     {d['n_residuals']:,}")
        print(f"  Media residuos (bias):   {d['residual_mean']:+.3f} EUR/MWh")
        print(f"  Std residuos (antes AR): {d['residual_std_raw']:.3f} EUR/MWh")
        print()

        # Tabla comparativa de órdenes
        print("  Comparación de órdenes AR:")
        print(f"  {'Orden':>5} {'AIC':>12} {'Std resid':>12} {'Coeficientes':>30}")
        print(f"  {'─' * 62}")
        for order, r in d["results_by_order"].items():
            marker = " ← mejor" if order == d["best_order"] else ""
            coef_str = ", ".join(f"φ{i+1}={c:.4f}" for i, c in enumerate(r["coefs"]))
            print(f"  {order:>5} {r['aic']:>12.1f} {r['residual_std']:>12.3f}   "
                  f"{coef_str}{marker}")

        print(f"\n  Modelo seleccionado: AR({d['best_order']})")
        print(f"  Coeficientes: intercept={self.intercept:.4f}, "
              + ", ".join(f"φ{i+1}={c:.4f}" for i, c in enumerate(self.ar_coefs)))

        # Autocorrelación antes/después
        print(f"\n  Autocorrelación de residuos:")
        print(f"  {'Lag':>5} {'Antes':>10} {'Después':>10} {'Reducción':>12}")
        print(f"  {'─' * 40}")
        for lag in [1, 2, 3, 24]:
            before = d["autocorr_before"].get(lag, np.nan)
            after = d["autocorr_after"].get(lag, np.nan)
            reduction = abs(before) - abs(after) if not np.isnan(after) else np.nan
            print(f"  {lag:>5} {before:>10.3f} {after:>10.3f} {reduction:>+10.3f}")

        print(f"\n  Reducción std residual: {d['std_reduction_pct']:.1f}%")
        print("=" * 65)

    # ── Corrección OFFLINE (evaluación / backtest) ──────────────────────────

    def correct_offline(self, y_true: np.ndarray, y_pred: np.ndarray
                        ) -> np.ndarray:
        """
        Corrección secuencial offline — para evaluación y backtest.

        En cada hora t, usa el residuo REAL de t-1 (porque y_true es conocido)
        para corregir la predicción de t. Este es el modo que da la mejora
        máxima y es el escenario realista para intraday corrections donde
        los precios spot se publican hora a hora.

        Args:
            y_true: Valores reales (array 1D temporal)
            y_pred: Predicciones LightGBM sin corregir

        Returns:
            Array de predicciones corregidas (mismo tamaño que y_pred)
        """
        self._check_fitted()
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        n = len(y_pred)
        order = self.best_order

        corrected = y_pred.copy()
        residuals = y_true - y_pred  # Residuos reales

        for t in range(order, n):
            # Predicción AR del residuo en t basada en residuos reales anteriores
            ar_forecast = self.intercept + sum(
                self.ar_coefs[lag] * residuals[t - lag - 1]
                for lag in range(order)
            )
            corrected[t] = y_pred[t] + ar_forecast

        return corrected

    # ── Corrección DAY-AHEAD (producción) ───────────────────────────────────

    def correct_day_ahead(self,
                          preds_future: np.ndarray,
                          last_known_residuals: np.ndarray
                          ) -> np.ndarray:
        """
        Corrección para predicciones day-ahead de producción.

        Dado un vector de predicciones futuras (e.g. 24 horas) y los
        últimos residuos conocidos, propaga la corrección AR hacia adelante.
        La corrección se atenúa naturalmente con la distancia temporal
        (φ₁^h para AR(1)).

        Args:
            preds_future:          Predicciones LightGBM para las próximas N horas
            last_known_residuals:  Últimos `order` residuos conocidos
                                   (ordenados del más antiguo al más reciente)

        Returns:
            Predicciones corregidas para las N horas futuras
        """
        self._check_fitted()
        preds_future = np.asarray(preds_future, dtype=np.float64)
        last_known = np.asarray(last_known_residuals, dtype=np.float64)
        order = self.best_order

        if len(last_known) < order:
            warnings.warn(
                f"Se necesitan {order} residuos conocidos para AR({order}), "
                f"solo se proporcionaron {len(last_known)}. "
                "Rellenando con 0."
            )
            last_known = np.pad(last_known, (order - len(last_known), 0))

        # Tomar los últimos `order` residuos
        recent = list(last_known[-order:])

        corrected = preds_future.copy()
        n = len(preds_future)

        for t in range(n):
            # Predicción AR del residuo
            ar_forecast = self.intercept + sum(
                self.ar_coefs[lag] * recent[-(lag + 1)]
                for lag in range(order)
            )
            corrected[t] = preds_future[t] + ar_forecast

            # Actualizar buffer con el residuo PREDICHO (no real, no lo conocemos)
            recent.append(ar_forecast)

        return corrected

    # ── Corrección ONLINE ROLLING ───────────────────────────────────────────

    def correct_online_rolling(self,
                               preds_24h: np.ndarray,
                               actuals_so_far: np.ndarray,
                               hours_known: int
                               ) -> np.ndarray:
        """
        Corrección híbrida online — actualiza intraday.

        Para las horas ya conocidas, usa residuos reales.
        Para las horas restantes, propaga AR desde el último residuo real.

        Escenario: A las 14:00 del día D+1 ya se conocen los precios
        spot de las horas 00-13. Se corrigen las horas 14-23 usando
        los residuos reales de 11-13 como semilla del AR.

        Args:
            preds_24h:      Predicciones originales (24 horas, 0-23)
            actuals_so_far: Precios reales hasta hours_known (0..hours_known-1)
            hours_known:    Número de horas con precio real conocido

        Returns:
            Predicciones corregidas para las 24 horas
        """
        self._check_fitted()
        preds = np.asarray(preds_24h, dtype=np.float64).copy()
        actuals = np.asarray(actuals_so_far, dtype=np.float64)
        order = self.best_order
        n = len(preds)

        corrected = preds.copy()

        # Fase 1: horas ya conocidas → corrección con residuos reales
        residuals_known = actuals[:hours_known] - preds[:hours_known]

        for t in range(min(order, hours_known), hours_known):
            ar_forecast = self.intercept + sum(
                self.ar_coefs[lag] * residuals_known[t - lag - 1]
                for lag in range(min(order, t))
            )
            corrected[t] = preds[t] + ar_forecast

        # Fase 2: horas futuras → propagación AR
        if hours_known >= order:
            last_resids = list(residuals_known[-(order):])
        else:
            last_resids = list(np.pad(residuals_known, (order - hours_known, 0)))

        for t in range(hours_known, n):
            ar_forecast = self.intercept + sum(
                self.ar_coefs[lag] * last_resids[-(lag + 1)]
                for lag in range(order)
            )
            corrected[t] = preds[t] + ar_forecast
            last_resids.append(ar_forecast)

        return corrected

    # ── Walk-forward con corrección AR ──────────────────────────────────────

    def walk_forward_with_ar(self,
                             X: pd.DataFrame, y: pd.Series,
                             model,
                             wf_start: str = "2024-01-01",
                             verbose: bool = True
                             ) -> pd.DataFrame:
        """
        Walk-forward validation CON corrección AR integrada.

        Para cada mes, entrena LightGBM → calcula residuos en train →
        ajusta AR → corrige predicciones del mes → mide métricas.

        Esto simula el escenario real de producción donde el AR se
        re-calibra periódicamente con datos frescos.

        Args:
            X:        DataFrame de features (completo, con índice temporal)
            y:        Series de target (completo)
            model:    Función que entrena y devuelve modelo LightGBM
                      Signature: model(X_train, y_train) → fitted_model
            wf_start: Inicio de la validación walk-forward
            verbose:  Imprimir resultados por mes

        Returns:
            DataFrame con métricas por mes (con y sin corrección AR)
        """
        wf_start_ts = pd.Timestamp(wf_start, tz="UTC")
        months = pd.date_range(start=wf_start_ts, end=X.index.max(), freq="MS")

        results = []
        for predict_month_start in months:
            predict_month_end = (predict_month_start
                                 + pd.offsets.MonthEnd(1))
            predict_month_end = predict_month_end.replace(
                hour=23, tzinfo=predict_month_start.tzinfo
            )

            train_mask = X.index < predict_month_start
            pred_mask = ((X.index >= predict_month_start) &
                         (X.index <= predict_month_end))

            if train_mask.sum() < 1000 or pred_mask.sum() == 0:
                continue

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_pr, y_pr = X[pred_mask], y[pred_mask]

            # 1. Entrenar LightGBM
            lgbm = model(X_tr, y_tr)

            # 2. Predicciones sin corrección
            preds_raw = lgbm.predict(X_pr)
            preds_train = lgbm.predict(X_tr)

            # 3. Ajustar AR sobre residuos del train
            ar = ARResidualCorrector(max_order=self.max_order)
            ar.fit(y_tr.values, preds_train, verbose=False)

            # 4. Corrección offline (secuencial, residuos reales)
            preds_corrected = ar.correct_offline(y_pr.values, preds_raw)

            # 5. Métricas
            errors_raw = np.abs(y_pr.values - preds_raw)
            errors_cor = np.abs(y_pr.values - preds_corrected)

            mae_raw = float(np.mean(errors_raw))
            mae_cor = float(np.mean(errors_cor))
            rmse_raw = float(np.sqrt(np.mean((y_pr.values - preds_raw) ** 2)))
            rmse_cor = float(np.sqrt(np.mean((y_pr.values - preds_corrected) ** 2)))

            month_str = predict_month_start.strftime("%Y-%m")
            results.append({
                "month": month_str,
                "n_hours": len(X_pr),
                "MAE_raw": round(mae_raw, 3),
                "MAE_corrected": round(mae_cor, 3),
                "MAE_improvement_pct": round((mae_raw - mae_cor) / mae_raw * 100, 1),
                "RMSE_raw": round(rmse_raw, 3),
                "RMSE_corrected": round(rmse_cor, 3),
                "AR_order": ar.best_order,
                "AR_phi1": round(ar.ar_coefs[0], 4),
                "price_mean": round(float(y_pr.mean()), 2),
                "price_std": round(float(y_pr.std()), 2),
            })

            if verbose:
                imp = results[-1]["MAE_improvement_pct"]
                marker = "OK" if imp > 5 else "WARN" if imp > 0 else "ERROR"
                print(f"  {marker} {month_str}: MAE {mae_raw:.2f} → {mae_cor:.2f} "
                      f"({imp:+.1f}%) | AR({ar.best_order}) φ₁={ar.ar_coefs[0]:.3f}")

        df_results = pd.DataFrame(results)
        if verbose and not df_results.empty:
            print(f"\n  Media MAE raw:       {df_results['MAE_raw'].mean():.2f}")
            print(f"  Media MAE corrected: {df_results['MAE_corrected'].mean():.2f}")
            print(f"  Mejora media:        "
                  f"{df_results['MAE_improvement_pct'].mean():.1f}%")

        return df_results

    # ── Corrección de bandas quantile ───────────────────────────────────────

    def correct_quantile_bands(self,
                               bands: Dict[str, np.ndarray],
                               y_true: np.ndarray,
                               preds_point: np.ndarray
                               ) -> Dict[str, np.ndarray]:
        """
        Corrige las bandas de predicción quantile aplicando el mismo
        shift AR que a la predicción puntual.

        La idea: si el AR predice que el residuo en t será +5 EUR,
        entonces P10 y P90 también se desplazan +5 EUR.
        Esto mejora la cobertura de los intervalos de confianza.

        Args:
            bands:       Dict {"P10": array, "P25": ..., "P75": ..., "P90": ...}
            y_true:      Valores reales (para calcular residuos)
            preds_point: Predicciones puntuales del LightGBM

        Returns:
            Dict con bandas corregidas
        """
        self._check_fitted()
        residuals = y_true - preds_point
        n = len(preds_point)
        order = self.best_order

        # Calcular el AR shift para cada hora
        ar_shifts = np.zeros(n)
        for t in range(order, n):
            ar_shifts[t] = self.intercept + sum(
                self.ar_coefs[lag] * residuals[t - lag - 1]
                for lag in range(order)
            )

        # Aplicar shift a todas las bandas
        corrected_bands = {}
        for name, values in bands.items():
            corrected_bands[name] = values + ar_shifts

        return corrected_bands

    # ── Serialización ───────────────────────────────────────────────────────

    def save(self, path: str):
        """Guarda el corrector AR a disco."""
        self._check_fitted()
        state = {
            "max_order": self.max_order,
            "best_order": self.best_order,
            "ar_coefs": self.ar_coefs,
            "intercept": self.intercept,
            "residual_std": self.residual_std,
            "diagnostics": self._fit_diagnostics,
        }
        from src.utils.logger import robust_pickle_dump
        robust_pickle_dump(state, path)

    @classmethod
    def load(cls, path: str) -> 'ARResidualCorrector':
        """Carga un corrector AR desde disco."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(max_order=state["max_order"])
        obj.best_order = state["best_order"]
        obj.ar_coefs = state["ar_coefs"]
        obj.intercept = state["intercept"]
        obj.residual_std = state["residual_std"]
        obj._fit_diagnostics = state["diagnostics"]
        obj.fitted = True
        return obj

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError(
                "Corrector no ajustado. Llama a .fit(y_true, y_pred) primero."
            )

    def get_params(self) -> dict:
        """Devuelve parámetros del modelo para logging."""
        self._check_fitted()
        return {
            "best_order": self.best_order,
            "ar_coefs": self.ar_coefs.tolist(),
            "intercept": self.intercept,
            "residual_std": self.residual_std,
        }

    def __repr__(self):
        if not self.fitted:
            return f"ARResidualCorrector(max_order={self.max_order}, fitted=False)"
        coef_str = ", ".join(f"φ{i+1}={c:.4f}" for i, c in enumerate(self.ar_coefs))
        return (f"ARResidualCorrector(AR({self.best_order}): {coef_str}, "
                f"intercept={self.intercept:.4f})")


# ── Funciones de conveniencia ────────────────────────────────────────────────

def evaluate_ar_correction(y_true: np.ndarray, y_pred: np.ndarray,
                           max_order: int = 3,
                           verbose: bool = True
                           ) -> Tuple[ARResidualCorrector, Dict]:
    """
    Función de alto nivel: ajusta AR, corrige, y reporta métricas.

    Args:
        y_true:    Valores reales
        y_pred:    Predicciones del modelo base
        max_order: Orden máximo del AR
        verbose:   Imprimir resultados

    Returns:
        (corrector, metrics_dict)
    """
    corrector = ARResidualCorrector(max_order=max_order)
    corrector.fit(y_true, y_pred, verbose=verbose)

    # Corrección offline
    y_corrected = corrector.correct_offline(y_true, y_pred)

    # Métricas antes/después
    errors_raw = np.abs(y_true - y_pred)
    errors_cor = np.abs(y_true - y_corrected)

    # Solo medir desde order en adelante (primeras horas sin corrección)
    order = corrector.best_order
    errors_raw_valid = errors_raw[order:]
    errors_cor_valid = errors_cor[order:]
    y_true_valid = y_true[order:]
    y_pred_valid = y_pred[order:]
    y_cor_valid = y_corrected[order:]

    metrics = {
        "MAE_raw": float(np.mean(errors_raw_valid)),
        "MAE_corrected": float(np.mean(errors_cor_valid)),
        "MAE_improvement_abs": float(np.mean(errors_raw_valid) - np.mean(errors_cor_valid)),
        "MAE_improvement_pct": float(
            (np.mean(errors_raw_valid) - np.mean(errors_cor_valid))
            / np.mean(errors_raw_valid) * 100
        ),
        "RMSE_raw": float(np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2))),
        "RMSE_corrected": float(np.sqrt(np.mean((y_true_valid - y_cor_valid) ** 2))),
        "autocorr_raw_lag1": float(pd.Series(y_true - y_pred).autocorr(lag=1)),
        "autocorr_corrected_lag1": float(
            pd.Series(y_true - y_corrected).autocorr(lag=1)
        ),
        "AR_order": corrector.best_order,
        "AR_coefs": corrector.ar_coefs.tolist(),
    }

    if verbose:
        print(f"\n{'─' * 55}")
        print(f"  RESULTADO CORRECCIÓN AR({corrector.best_order})")
        print(f"{'─' * 55}")
        print(f"  MAE sin corrección:  {metrics['MAE_raw']:.3f} EUR/MWh")
        print(f"  MAE con corrección:  {metrics['MAE_corrected']:.3f} EUR/MWh")
        print(f"  Mejora:              {metrics['MAE_improvement_abs']:.3f} EUR/MWh "
              f"({metrics['MAE_improvement_pct']:.1f}%)")
        print(f"  RMSE: {metrics['RMSE_raw']:.3f} → {metrics['RMSE_corrected']:.3f}")
        print(f"  Autocorr lag-1: {metrics['autocorr_raw_lag1']:.3f} → "
              f"{metrics['autocorr_corrected_lag1']:.3f}")
        print(f"{'─' * 55}")

    return corrector, metrics


def simulate_day_ahead_correction(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  corrector: ARResidualCorrector,
                                  verbose: bool = True
                                  ) -> Dict:
    """
    Simula la corrección day-ahead: predice 24h a la vez,
    usando solo el último residuo conocido del día anterior.

    Esto es más realista que offline (no usa residuos intraday)
    pero menos potente. Muestra el beneficio mínimo garantizado.

    Args:
        y_true:    Valores reales (múltiplo de 24h idealmente)
        y_pred:    Predicciones sin corregir
        corrector: Corrector AR ajustado

    Returns:
        Dict con métricas day-ahead
    """
    n = len(y_true)
    order = corrector.best_order
    corrected = y_pred.copy()

    # Procesar día a día (bloques de 24h)
    residuals = y_true - y_pred
    n_days = n // 24

    for d in range(1, n_days):
        start_h = d * 24
        end_h = min(start_h + 24, n)

        # Últimos residuos conocidos: fin del día anterior
        last_resids = residuals[max(0, start_h - order): start_h]
        if len(last_resids) < order:
            last_resids = np.pad(last_resids, (order - len(last_resids), 0))

        # Corregir las 24h del día
        corrected[start_h:end_h] = corrector.correct_day_ahead(
            y_pred[start_h:end_h],
            last_resids
        )

    errors_raw = np.abs(y_true - y_pred)
    errors_cor = np.abs(y_true - corrected)

    # Excluir primer día (sin residuos previos)
    errors_raw = errors_raw[24:]
    errors_cor = errors_cor[24:]

    metrics = {
        "MAE_raw": float(np.mean(errors_raw)),
        "MAE_day_ahead_corrected": float(np.mean(errors_cor)),
        "improvement_pct": float(
            (np.mean(errors_raw) - np.mean(errors_cor))
            / np.mean(errors_raw) * 100
        ),
    }

    if verbose:
        print(f"\n  Day-ahead simulation ({n_days} days):")
        print(f"  MAE raw:       {metrics['MAE_raw']:.3f}")
        print(f"  MAE corrected: {metrics['MAE_day_ahead_corrected']:.3f}")
        print(f"  Improvement:   {metrics['improvement_pct']:.1f}%")

    return metrics


# ── Tests unitarios básicos ──────────────────────────────────────────────────

def _run_tests():
    """Verificación rápida con datos sintéticos."""
    np.random.seed(42)
    n = 2000

    # Generar residuos AR(1) sintéticos con φ₁ = 0.9
    true_phi = 0.9
    innovations = np.random.normal(0, 5, n)
    residuals = np.zeros(n)
    for t in range(1, n):
        residuals[t] = true_phi * residuals[t - 1] + innovations[t]

    # "Predicciones" base y valores reales
    y_pred = np.random.normal(50, 20, n)  # Predicciones LightGBM simuladas
    y_true = y_pred + residuals

    print("=" * 55)
    print("  TEST: Datos sintéticos AR(1) con φ=0.9")
    print("=" * 55)

    corrector, metrics = evaluate_ar_correction(y_true, y_pred, max_order=3)

    # Verificaciones
    # AIC puede preferir AR(2) o AR(3) por ruido — lo importante es que φ₁ ≈ 0.9
    assert corrector.best_order >= 1, f"Orden inválido: {corrector.best_order}"
    assert abs(corrector.ar_coefs[0] - true_phi) < 0.05, \
        f"φ₁ estimado ({corrector.ar_coefs[0]:.3f}) muy lejos de {true_phi}"
    assert metrics["MAE_improvement_pct"] > 20, \
        f"Mejora ({metrics['MAE_improvement_pct']:.1f}%) muy baja para φ=0.9"

    print(f"\n  OK: Test sintético:")
    print(f"     φ₁ estimado: {corrector.ar_coefs[0]:.4f} (real: {true_phi})")
    print(f"     Mejora MAE:  {metrics['MAE_improvement_pct']:.1f}%")

    # Test day-ahead
    da_metrics = simulate_day_ahead_correction(y_true, y_pred, corrector)
    assert da_metrics["improvement_pct"] > 5, "Day-ahead debería mejorar >5%"
    print(f"     Day-ahead:   {da_metrics['improvement_pct']:.1f}% mejora")

    # Test serialización
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    corrector.save(path)
    loaded = ARResidualCorrector.load(path)
    assert np.allclose(loaded.ar_coefs, corrector.ar_coefs)
    os.unlink(path)
    print("     Serialización: OK")

    print("\n  OK Todos los tests pasaron.\n")


if __name__ == "__main__":
    _run_tests()