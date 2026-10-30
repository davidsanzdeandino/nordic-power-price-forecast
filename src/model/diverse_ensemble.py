"""
diverse_ensemble.py
===================
Ensemble robusto por DIVERSIDAD DE OBJETIVO.

Promedia tres modelos LightGBM entrenados con objetivos distintos sobre las
MISMAS features:
  - L2 (regression)       → media condicional
  - quantile α=0.5        → mediana condicional (robusta a spikes)
  - huber                 → regresión robusta (compromiso L1/L2)

MOTIVACIÓN (validado empíricamente, no por fe):
  El ensemble previo combinaba LightGBM + XGBoost + Ridge, pero LightGBM y
  XGBoost son casi redundantes (correlación de predicciones ≈ 0.99), así que
  aportaban poca diversidad. La diversidad por OBJETIVO sí descorrelaciona los
  errores. En validación walk-forward sobre 2024 (12 meses, DK1 y DK2):
      DirAcc: +2.0 pp de media, MEJOR EN 12/12 MESES
      MAE:    +1.5–1.6 %, mejor en 8–10/12 meses
  frente a un único LightGBM L2. Fundamento en la literatura EPF (Lago et al.
  2021; Marcjasz & Weron): el promedio simple de modelos es repetidamente la
  técnica de combinación más robusta.

Las transformaciones de varianza (asinh) — útiles para modelos lineales/redes —
se DESCARTARON tras comprobar que NO mejoran un modelo de árboles (ver
scripts/experiment_vst.py y docs/decisions.md DT-14).
"""

import pickle
import numpy as np

from src.model.config import LGB_COMMON


class DiverseObjectiveEnsemble:
    """Promedio de 3 LightGBM con objetivos diversos (L2, mediana, Huber)."""

    # (nombre, objetivo LightGBM, kwargs extra del objetivo)
    OBJECTIVES = [
        ("l2",     "regression", {}),
        ("median", "quantile",   {"alpha": 0.5}),
        ("huber",  "huber",      {"alpha": 5.0}),
    ]

    def __init__(self, zone: str = "DK1", base_params: dict | None = None):
        self.zone = zone
        self.base_params = base_params or {}
        self.models: dict = {}
        self.fitted = False

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, verbose: bool = False):
        """Entrena los 3 modelos. Si se dan X_val/y_val, usa early stopping.

        X_val debe ser disjunto de X_tr (en train.py: train hasta sep-2025 y
        early stopping en el trimestre de validación oct-dic 2025).
        """
        import lightgbm as lgb
        for name, objective, extra in self.OBJECTIVES:
            params = dict(self.base_params)
            params.update(objective=objective, metric="mae", **LGB_COMMON)
            params.update(extra)               # alpha del objetivo (sobreescribe)
            model = lgb.LGBMRegressor(**params)
            if X_val is not None and y_val is not None:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                model.fit(X_tr, y_tr)
            self.models[name] = model
            if verbose:
                print(f"    Ensemble diverso · {name:6s} entrenado "
                      f"(n_estimators efectivos: {model.best_iteration_ or model.n_estimators})")
        self.fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Media aritmética simple de los 3 objetivos (la más robusta)."""
        if not self.fitted:
            raise RuntimeError("Ensemble no entrenado.")
        preds = [m.predict(X) for m in self.models.values()]
        return np.mean(preds, axis=0)

    def predict_members(self, X) -> dict:
        """Devuelve las predicciones de cada miembro (diagnóstico/diversidad)."""
        return {name: m.predict(X) for name, m in self.models.items()}

    def save(self, path: str):
        from src.utils.logger import robust_pickle_dump
        robust_pickle_dump({"zone": self.zone, "models": self.models,
                            "objectives": [o[0] for o in self.OBJECTIVES]}, path)

    @classmethod
    def load(cls, path: str) -> "DiverseObjectiveEnsemble":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(zone=state["zone"])
        obj.models = state["models"]
        obj.fitted = True
        return obj

    def __repr__(self):
        return (f"DiverseObjectiveEnsemble(zone={self.zone}, "
                f"members={list(self.models.keys())}, fitted={self.fitted})")
