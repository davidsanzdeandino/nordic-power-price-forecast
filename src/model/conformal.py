"""
conformal.py
============
Conformal Prediction para calibración de intervalos de confianza.

PROBLEMA:
  Los modelos de quantile regression (P10-P90) del LightGBM dan
  cobertura real del 46% cuando deberían dar 80%. Esto ocurre porque:
    1. Los quantile models sobreajustan al train (ven los mismos datos)
    2. En test (Q4 invierno) la volatilidad es mayor que en train
    3. Las colas de la distribución de precios son más pesadas de lo
       que el modelo captura

SOLUCIÓN: Conformal Prediction
  Wrapper post-hoc que ajusta las bandas para que la cobertura
  se cumpla por construcción. No modifica el modelo base — solo
  calibra las bandas usando un set de calibración held-out.

  Garantía teórica: si los datos son intercambiables (exchangeable),
  la cobertura del intervalo conformal es ≥ 1-α con probabilidad
  finita exacta, no asintótica.

DECISIÓN TÉCNICA: ¿Por qué conformal y no simplemente ensanchar las bandas?
  - Ensanchar con un factor fijo (e.g. ×1.5) es arbitrario
  - Conformal se auto-calibra: el factor exacto sale de los datos
  - Conformal adaptativo ajusta el ancho según la dificultad de cada
    predicción (más ancho cuando el modelo está menos seguro)
  - Es model-agnostic: funciona igual con LightGBM, XGBoost o ensemble

ESTRATEGIAS IMPLEMENTADAS:
  1. Split Conformal    — ajuste global único (simple, robusto)
  2. Adaptive Conformal — ancho proporcional a la incertidumbre predicha
  3. Mondrian Conformal — calibración separada por hora del día

Uso:
    from conformal import ConformalCalibrator

    calibrator = ConformalCalibrator(strategy="adaptive")
    calibrator.fit(y_cal, pred_cal, q_lower_cal, q_upper_cal)
    lower_adj, upper_adj = calibrator.calibrate(pred_new, q_lower_new, q_upper_new)

Referencia:
  Romano, Patterson, Candès (2019). "Conformalized Quantile Regression"
  Vovk, Gammerman, Shafer (2005). "Algorithmic Learning in a Random World"
"""

import numpy as np
import pandas as pd
import pickle
from typing import Optional, Tuple, Dict


class ConformalCalibrator:
    """
    Calibrador conformal para intervalos de predicción.

    Toma las predicciones de quantile regression (P10, P90) y las ajusta
    para que la cobertura real sea ≥ target_coverage.

    Strategies:
      "split"    — offset global ±q̂ (simple, robusto)
      "adaptive" — offset ∝ anchura de banda predicha (CQR)
      "mondrian" — offset separado por hora del día
    """

    def __init__(self,
                 strategy: str = "adaptive",
                 target_coverage: float = 0.80):
        if strategy not in ("split", "adaptive", "mondrian"):
            raise ValueError(f"Estrategia desconocida: {strategy}")

        self.strategy = strategy
        self.target_coverage = target_coverage
        self.fitted = False

        self._q_hat = None
        self._q_hat_adaptive = None
        self._q_hat_mondrian = None
        self._fit_diagnostics = {}

    # ── Ajuste ──────────────────────────────────────────────────────────────

    def fit(self,
            y_cal: np.ndarray,
            pred_cal: np.ndarray,
            q_lower_cal: np.ndarray,
            q_upper_cal: np.ndarray,
            timestamps_cal: pd.DatetimeIndex = None,
            verbose: bool = True) -> 'ConformalCalibrator':
        """
        Ajusta el calibrador en el set de calibración.

        El set de calibración DEBE ser datos que el modelo NO ha visto
        durante el entrenamiento. Típicamente el set de validación.

        Args:
            y_cal:           Precios reales (calibración)
            pred_cal:        Predicciones puntuales
            q_lower_cal:     Banda inferior raw (P10)
            q_upper_cal:     Banda superior raw (P90)
            timestamps_cal:  Timestamps (necesario para mondrian)
            verbose:         Imprimir diagnósticos
        """
        y = np.asarray(y_cal)
        pred = np.asarray(pred_cal)
        ql = np.asarray(q_lower_cal)
        qu = np.asarray(q_upper_cal)
        n = len(y)

        if n < 50:
            raise ValueError(f"Calibración muy pequeña ({n} obs). Mínimo: 200.")

        # Corrección de finitud: quantil = ceil((n+1)(1-α))/n
        alpha = 1 - self.target_coverage
        corrected_quantile = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)

        # Cobertura ANTES
        raw_coverage = float(np.mean((y >= ql) & (y <= qu)))
        self._fit_diagnostics["raw_coverage"] = raw_coverage
        self._fit_diagnostics["n_calibration"] = n

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  CONFORMAL CALIBRATION — {self.strategy}")
            print(f"  n={n} | target={self.target_coverage:.0%} | "
                  f"raw coverage={raw_coverage:.1%}")
            print(f"{'=' * 60}")

        # Dispatch
        if self.strategy == "split":
            self._fit_split(y, ql, qu, corrected_quantile, verbose)
        elif self.strategy == "adaptive":
            self._fit_adaptive(y, ql, qu, corrected_quantile, verbose)
        elif self.strategy == "mondrian":
            if timestamps_cal is None:
                raise ValueError("Mondrian requiere timestamps_cal.")
            self._fit_mondrian(y, ql, qu, corrected_quantile,
                               timestamps_cal, verbose)

        self.fitted = True

        # Verificar in-sample
        ql_adj, qu_adj = self.calibrate(pred, ql, qu, timestamps_cal)
        cal_cov = float(np.mean((y >= ql_adj) & (y <= qu_adj)))
        raw_width = float(np.mean(qu - ql))
        cal_width = float(np.mean(qu_adj - ql_adj))

        self._fit_diagnostics.update({
            "calibrated_coverage_insample": cal_cov,
            "raw_band_width": raw_width,
            "calibrated_band_width": cal_width,
            "width_increase_pct": (cal_width - raw_width) / max(raw_width, 1e-6) * 100,
        })

        if verbose:
            print(f"\n  Resultado: {raw_coverage:.1%} → {cal_cov:.1%}")
            print(f"  Ancho banda: {raw_width:.1f} → {cal_width:.1f} EUR/MWh "
                  f"(+{self._fit_diagnostics['width_increase_pct']:.0f}%)")
            print(f"{'=' * 60}\n")

        return self

    def _fit_split(self, y, ql, qu, corrected_quantile, verbose):
        """Split Conformal: offset global q̂ = quantile de los scores."""
        scores = np.maximum(np.maximum(ql - y, y - qu), 0)
        self._q_hat = float(np.quantile(scores, corrected_quantile))
        if verbose:
            print(f"  q̂ = {self._q_hat:.2f} EUR/MWh (offset global ±)")

    def _fit_adaptive(self, y, ql, qu, corrected_quantile, verbose):
        """
        CQR (Romano et al., 2019): score normalizado por anchura.

        score_i = max(ql-y, y-qu, 0) / (qu-ql)
        El offset en producción es q̂_adaptive × anchura_predicha.
        → Bandas más anchas donde el modelo es más incierto.
        """
        widths = np.maximum(qu - ql, 1e-6)
        scores = np.maximum(np.maximum(ql - y, y - qu), 0) / widths
        self._q_hat_adaptive = float(np.quantile(scores, corrected_quantile))

        # Fallback split
        scores_raw = np.maximum(np.maximum(ql - y, y - qu), 0)
        self._q_hat = float(np.quantile(scores_raw, corrected_quantile))

        if verbose:
            w25, w75 = np.percentile(widths, [25, 75])
            print(f"  q̂_adaptive = {self._q_hat_adaptive:.3f} (factor ×anchura)")
            print(f"  Banda estrecha ({w25:.0f} EUR) → ±{self._q_hat_adaptive*w25:.1f}")
            print(f"  Banda ancha ({w75:.0f} EUR) → ±{self._q_hat_adaptive*w75:.1f}")

    def _fit_mondrian(self, y, ql, qu, corrected_quantile,
                      timestamps, verbose):
        """Mondrian: calibración separada por hora del día."""
        hours = pd.DatetimeIndex(timestamps).hour
        scores_all = np.maximum(np.maximum(ql - y, y - qu), 0)
        q_global = float(np.quantile(scores_all, corrected_quantile))
        self._q_hat = q_global

        self._q_hat_mondrian = {}
        MIN_OBS = 30

        if verbose:
            print(f"  {'Hora':>5} {'n':>5} {'q̂':>8}")
            print(f"  {'─' * 20}")

        for h in range(24):
            mask = hours == h
            n_h = mask.sum()
            if n_h >= MIN_OBS:
                q_corr = min(np.ceil((n_h+1)*corrected_quantile)/n_h, 1.0)
                self._q_hat_mondrian[h] = float(
                    np.quantile(scores_all[mask], q_corr)
                )
            else:
                self._q_hat_mondrian[h] = q_global

            if verbose and h % 4 == 0:
                print(f"  {h:>3}h  {n_h:>4}  {self._q_hat_mondrian[h]:>7.2f}")

    # ── Calibración (producción) ────────────────────────────────────────────

    def calibrate(self,
                  pred: np.ndarray,
                  q_lower: np.ndarray,
                  q_upper: np.ndarray,
                  timestamps: pd.DatetimeIndex = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica calibración conformal a nuevas bandas.

        Returns:
            (q_lower_adj, q_upper_adj): Bandas calibradas
        """
        self._check_fitted()
        ql = np.asarray(q_lower, dtype=float)
        qu = np.asarray(q_upper, dtype=float)

        if self.strategy == "split":
            return ql - self._q_hat, qu + self._q_hat

        elif self.strategy == "adaptive":
            widths = np.maximum(qu - ql, 1e-6)
            adj = self._q_hat_adaptive * widths
            return ql - adj, qu + adj

        elif self.strategy == "mondrian":
            if timestamps is None:
                return ql - self._q_hat, qu + self._q_hat
            hours = pd.DatetimeIndex(timestamps).hour
            offsets = np.array([
                self._q_hat_mondrian.get(h, self._q_hat) for h in hours
            ])
            return ql - offsets, qu + offsets

    def calibrate_all_bands(self,
                            pred: np.ndarray,
                            bands: Dict[str, np.ndarray],
                            timestamps: pd.DatetimeIndex = None
                            ) -> Dict[str, np.ndarray]:
        """
        Calibra todas las bandas {P10, P25, P50, P75, P90}.

        P10-P90 recibe el ajuste completo (80% target).
        P25-P75 recibe un ajuste reducido proporcional (50% target).
        P50 pasa sin cambios.
        """
        self._check_fitted()
        result = {}

        # P10-P90: ajuste completo
        if "P10" in bands and "P90" in bands:
            p10_adj, p90_adj = self.calibrate(
                pred, bands["P10"], bands["P90"], timestamps
            )
            result["P10"] = p10_adj
            result["P90"] = p90_adj

        # P25-P75: ajuste reducido
        # Ratio quantiles normales: Φ⁻¹(0.75)/Φ⁻¹(0.90) ≈ 0.526
        INNER_RATIO = 0.526

        if "P25" in bands and "P75" in bands:
            if self.strategy == "split":
                off = self._q_hat * INNER_RATIO
                result["P25"] = bands["P25"] - off
                result["P75"] = bands["P75"] + off

            elif self.strategy == "adaptive":
                ql_outer = bands.get("P10", bands["P25"])
                qu_outer = bands.get("P90", bands["P75"])
                widths = np.maximum(qu_outer - ql_outer, 1e-6)
                adj = self._q_hat_adaptive * INNER_RATIO * widths
                result["P25"] = bands["P25"] - adj
                result["P75"] = bands["P75"] + adj

            elif self.strategy == "mondrian":
                if timestamps is not None:
                    hours = pd.DatetimeIndex(timestamps).hour
                    offsets = np.array([
                        self._q_hat_mondrian.get(h, self._q_hat) * INNER_RATIO
                        for h in hours
                    ])
                else:
                    offsets = self._q_hat * INNER_RATIO
                result["P25"] = bands["P25"] - offsets
                result["P75"] = bands["P75"] + offsets

        if "P50" in bands:
            result["P50"] = bands["P50"]

        return result

    # ── Serialización ───────────────────────────────────────────────────────

    def save(self, path: str):
        self._check_fitted()
        state = {
            "strategy": self.strategy,
            "target_coverage": self.target_coverage,
            "q_hat": self._q_hat,
            "q_hat_adaptive": self._q_hat_adaptive,
            "q_hat_mondrian": self._q_hat_mondrian,
            "diagnostics": self._fit_diagnostics,
        }
        from src.utils.logger import robust_pickle_dump
        robust_pickle_dump(state, path)

    @classmethod
    def load(cls, path: str) -> 'ConformalCalibrator':
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(strategy=state["strategy"],
                  target_coverage=state["target_coverage"])
        obj._q_hat = state["q_hat"]
        obj._q_hat_adaptive = state["q_hat_adaptive"]
        obj._q_hat_mondrian = state["q_hat_mondrian"]
        obj._fit_diagnostics = state["diagnostics"]
        obj.fitted = True
        return obj

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError("Calibrador no ajustado. Llama a .fit().")

    def get_diagnostics(self) -> dict:
        return self._fit_diagnostics.copy()

    def __repr__(self):
        if not self.fitted:
            return f"ConformalCalibrator({self.strategy}, fitted=False)"
        d = self._fit_diagnostics
        return (f"ConformalCalibrator({self.strategy}, "
                f"cov={d.get('calibrated_coverage_insample', 0):.0%}, "
                f"Δwidth=+{d.get('width_increase_pct', 0):.0f}%)")


# ── Función de conveniencia ──────────────────────────────────────────────────

def calibrate_quantile_models(
        y_cal, pred_cal, quantile_preds_cal,
        strategy="adaptive", target_coverage=0.80,
        timestamps_cal=None, verbose=True
) -> ConformalCalibrator:
    """
    Ajusta un calibrador conformal. Función de conveniencia para train.py.
    """
    cal = ConformalCalibrator(strategy=strategy, target_coverage=target_coverage)
    cal.fit(y_cal, pred_cal, quantile_preds_cal["P10"],
            quantile_preds_cal["P90"], timestamps_cal, verbose)
    return cal


# ── Reordenación monótona de cuantiles ────────────────────────────────────────

QUANTILE_ORDER = ("P10", "P25", "P50", "P75", "P90")


def sort_quantile_bands(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Reordenación monótona (Chernozhukov, Fernández-Val y Galichon, 2010): en
    cada hora, los valores de los cuantiles presentes se ordenan de menor a
    mayor y se reparten en el orden P10 ≤ P25 ≤ P50 ≤ P75 ≤ P90.

    Los modelos de cuantiles se ajustan por separado y el conformal ensancha
    cada par por su lado, así que en algunas horas P75 puede quedar por encima
    de P90 (o P10 por encima de P25). Ordenar no cambia el conjunto de valores
    de cada hora y nunca sube la media de la pérdida pinball de los cuantiles
    presentes (desigualdad de reordenación). La previsión puntual no
    pasa por aquí. Las claves que no son cuantiles se devuelven tal cual.
    """
    keys = [q for q in QUANTILE_ORDER if q in bands]
    out = dict(bands)
    if len(keys) < 2:
        return out
    stacked = np.column_stack([np.asarray(bands[q], dtype=float) for q in keys])
    stacked = np.sort(stacked, axis=1)
    for i, q in enumerate(keys):
        out[q] = stacked[:, i]
    return out


def count_crossed_hours(bands: Dict[str, np.ndarray]) -> int:
    """Horas en las que algún cuantil queda por encima del siguiente."""
    keys = [q for q in QUANTILE_ORDER if q in bands]
    if len(keys) < 2:
        return 0
    stacked = np.column_stack([np.asarray(bands[q], dtype=float) for q in keys])
    return int(np.any(np.diff(stacked, axis=1) < 0, axis=1).sum())


# ── Adaptive Conformal Inference (ACI, Gibbs & Candès 2021) ───────────────────

def adaptive_conformal_coverage(y_true, lower, upper,
                                target_coverage: float = 0.80,
                                gamma: float = 0.02,
                                day_labels=None,
                                return_path: bool = False) -> dict:
    """
    Adaptive Conformal Inference (ACI) — cobertura ONLINE robusta a distribution
    shift (Gibbs & Candès, 2021, https://arxiv.org/abs/2106.00170).

    Problema: el conformal estático infra-cubre en periodos volátiles (p.ej. el
    invierno nórdico). ACI ajusta online un único parámetro: ensancha la banda
    cuando ha habido fallos recientes y la estrecha cuando sobra-cubre, de modo
    que la cobertura a largo plazo converge al objetivo SEA CUAL SEA el proceso.

    Aquí se implementa con un multiplicador de anchura k sobre la banda base:
        intervalo_t = centro_t ± k · semianchura_base_t
        err_t = 1 si y_t cae fuera del intervalo_t

    Actualización:
      - Sin day_labels (horaria): k_{t+1} = max(0, k_t + gamma · (err_t − α)).
        Usa el error de la hora anterior, que en day-ahead no se conoce.
      - Con day_labels (día de entrega local de cada hora): k es constante
        dentro de cada día y, al cerrarse el día D,
            k ← max(0, k + gamma · Σ_{h∈D} (err_h − α)),
        así que la banda de D solo usa errores hasta las 23:00 de D-1, que es
        lo que se conoce en la subasta day-ahead.
    En los dos casos k empieza en 1 (la banda base) y α = 1 − cobertura.

    Args:
        y_true, lower, upper: arrays alineados temporalmente (orden cronológico).
        target_coverage:      cobertura objetivo (0.80 → α=0.20).
        gamma:                tasa de aprendizaje (0.01–0.05 típico).
        day_labels:           etiqueta del día de entrega de cada hora (p. ej.
                              index.tz_convert("Europe/Copenhagen").normalize()).
        return_path:          añade 'k_path' (k usado en cada hora) y
                              'covered_path' al resultado.

    Returns:
        dict con cobertura estática vs ACI, anchura media y trayectoria de k.
    """
    y = np.asarray(y_true, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    alpha = 1.0 - target_coverage

    n = len(y)
    k = 1.0
    covered_static = (y >= lo) & (y <= hi)
    covered_aci = np.zeros(n, dtype=bool)
    width_aci = np.zeros(n)
    k_traj = np.zeros(n)

    if day_labels is None:
        for t in range(n):
            lo_t = center[t] - k * half[t]
            hi_t = center[t] + k * half[t]
            covered_aci[t] = (y[t] >= lo_t) & (y[t] <= hi_t)
            width_aci[t] = hi_t - lo_t
            k_traj[t] = k
            err = 0.0 if covered_aci[t] else 1.0
            k = max(0.0, k + gamma * (err - alpha))
        update = "hourly"
    else:
        labels = np.asarray(day_labels)
        if len(labels) != n:
            raise ValueError("day_labels must have one label per observation")
        # Bloques de horas consecutivas con la misma etiqueta (días en orden).
        change = np.r_[True, labels[1:] != labels[:-1]] if n else np.zeros(0, bool)
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            lo_d = center[s:e] - k * half[s:e]
            hi_d = center[s:e] + k * half[s:e]
            cov_d = (y[s:e] >= lo_d) & (y[s:e] <= hi_d)
            covered_aci[s:e] = cov_d
            width_aci[s:e] = hi_d - lo_d
            k_traj[s:e] = k
            k = max(0.0, k + gamma * float(np.sum((~cov_d).astype(float) - alpha)))
        update = "daily"

    out = {
        "coverage_static": float(np.mean(covered_static)),
        "coverage_aci":    float(np.mean(covered_aci)),
        "target":          target_coverage,
        "width_static_mean": float(np.mean(hi - lo)),
        "width_aci_mean":    float(np.mean(width_aci)),
        "k_final":         float(k),
        "k_mean":          float(np.mean(k_traj)),
        "gamma":           gamma,
        "update":          update,
    }
    if return_path:
        out["k_path"] = k_traj
        out["covered_path"] = covered_aci
    return out



# ── Tests ────────────────────────────────────────────────────────────────────

def _run_tests():
    np.random.seed(42)
    print("\n" + "=" * 60)
    print("  TEST: Conformal Prediction")
    print("=" * 60)

    n_cal, n_test = 2000, 1000
    ts_cal = pd.date_range("2024-07-01", periods=n_cal, freq="h", tz="UTC")
    ts_test = pd.date_range("2024-10-01", periods=n_test, freq="h", tz="UTC")

    def gen(n, ts):
        h = ts.hour
        base = 55 + 15 * np.sin(2*np.pi*h/24)
        vol = np.where((h >= 16) & (h <= 20), 25, 12)
        return base + np.random.standard_t(df=5, size=n) * vol

    y_cal, y_test = gen(n_cal, ts_cal), gen(n_test, ts_test)
    pred_cal = y_cal + np.random.normal(2, 8, n_cal)
    pred_test = y_test + np.random.normal(2, 8, n_test)

    # Bandas raw estrechas (simulando 46% cobertura)
    q10_c, q90_c = pred_cal - 8, pred_cal + 8
    q25_c, q75_c = pred_cal - 4, pred_cal + 4
    q10_t, q90_t = pred_test - 8, pred_test + 8
    q25_t, q75_t = pred_test - 4, pred_test + 4

    raw_80 = np.mean((y_test >= q10_t) & (y_test <= q90_t))
    print(f"\n  Raw coverage P10-P90: {raw_80:.1%}")

    # ── Split ──
    print("\n  ── Split Conformal ──")
    c1 = ConformalCalibrator("split", 0.80)
    c1.fit(y_cal, pred_cal, q10_c, q90_c, verbose=True)
    l, u = c1.calibrate(pred_test, q10_t, q90_t)
    cov1 = np.mean((y_test >= l) & (y_test <= u))
    assert cov1 >= 0.78, f"Split {cov1:.1%} < 78%"
    print(f"  OK Split: {cov1:.1%}")

    # ── Adaptive ──
    print("\n  ── Adaptive Conformal ──")
    c2 = ConformalCalibrator("adaptive", 0.80)
    c2.fit(y_cal, pred_cal, q10_c, q90_c, verbose=True)
    l, u = c2.calibrate(pred_test, q10_t, q90_t)
    cov2 = np.mean((y_test >= l) & (y_test <= u))
    assert cov2 >= 0.78, f"Adaptive {cov2:.1%} < 78%"
    print(f"  OK Adaptive: {cov2:.1%}")

    # ── Mondrian ──
    print("\n  ── Mondrian Conformal ──")
    c3 = ConformalCalibrator("mondrian", 0.80)
    c3.fit(y_cal, pred_cal, q10_c, q90_c, timestamps_cal=ts_cal, verbose=True)
    l, u = c3.calibrate(pred_test, q10_t, q90_t, ts_test)
    cov3 = np.mean((y_test >= l) & (y_test <= u))
    assert cov3 >= 0.75, f"Mondrian {cov3:.1%} < 75%"
    print(f"  OK Mondrian: {cov3:.1%}")

    # ── All bands ──
    print("\n  ── All bands calibration ──")
    bands_raw = {"P10": q10_t, "P25": q25_t, "P50": pred_test,
                 "P75": q75_t, "P90": q90_t}
    bands_adj = c2.calibrate_all_bands(pred_test, bands_raw)
    c80 = np.mean((y_test >= bands_adj["P10"]) & (y_test <= bands_adj["P90"]))
    c50 = np.mean((y_test >= bands_adj["P25"]) & (y_test <= bands_adj["P75"]))
    print(f"  P10-P90: {raw_80:.1%} → {c80:.1%}")
    raw_50 = np.mean((y_test >= q25_t) & (y_test <= q75_t))
    print(f"  P25-P75: {raw_50:.1%} → {c50:.1%}")

    # ── Serialización ──
    print("\n  ── Serialización ──")
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    c2.save(path)
    c2_loaded = ConformalCalibrator.load(path)
    l_orig, u_orig = c2.calibrate(pred_test, q10_t, q90_t)
    l_load, u_load = c2_loaded.calibrate(pred_test, q10_t, q90_t)
    assert np.allclose(l_orig, l_load) and np.allclose(u_orig, u_load)
    os.unlink(path)
    print("  OK: Serialización")

    print(f"\n  OK Todos los tests pasaron.\n")


if __name__ == "__main__":
    _run_tests()