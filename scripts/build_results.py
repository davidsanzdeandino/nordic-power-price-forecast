"""
build_results.py — Monta los JSON de resultados publicados a partir de los del código
=====================================================================================
Solo lee ficheros que escribe el propio código (nada se copia de los logs):

  results/evaluation_test2026_<zona>.json   src/model/evaluate.py   (pasos 21-23)
  results/training_<zona>.json              src/model/train.py      (pasos 18-20)
  data/metadata/served_metrics_<zona>.json  scripts/gen_served_metrics.py (27-29)
  results/baselines_test2026.json           src/model/baselines.py  (pasos 15-17)
  results/fi_regimes_test2026.json          scripts/evaluate_fi_regimes.py (30)
  results/leakage_sensitivity_test2026.json scripts/leakage_sensitivity.py
  data/master/dataset_features.parquet      src/model/features.py   (paso 14),
                                            solo los precios, para el nivel
                                            medio de train, validación y test

y escribe:

  results/evaluation_test2026.json      resumen del test por zona
  results/model_selection_val2025.json  elección, AR y calibración en validación

Comprueba además que el MAE de served_metrics coincide con la fila «Served»
de la evaluación (misma ruta de código).

Uso:  python scripts/build_results.py
"""

import sys
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
META = ROOT / "data" / "metadata"
FEATURES_PATH = ROOT / "data" / "master" / "dataset_features.parquet"
ZONES = ["DK1", "DK2", "FI"]


def _load(path: Path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def _pct(a, b):
    return round((a - b) / b * 100, 1)


def _battery_gap(adv: dict):
    """Beneficio anual con la previsión servida menos el de la persistencia."""
    srv = (adv.get("battery") or {}).get("profit_forecast_annual")
    per = (adv.get("battery_persistence_forecast") or {}).get("profit_forecast_annual")
    return None if srv is None or per is None else round(srv - per, 1)


def build_zone_eval(zone: str, baselines: dict) -> dict:
    ev = _load(RES / f"evaluation_test2026_{zone}.json")
    tr = _load(RES / f"training_{zone}.json")
    sm = _load(META / f"served_metrics_{zone}.json")
    if ev is None or sm is None:
        raise FileNotFoundError(f"{zone}: faltan evaluation_test2026_{zone}.json o "
                                f"served_metrics_{zone}.json (pasos 21-29)")
    base = baselines["zones"][zone]["baselines"]
    served_row = ev["metrics"]["Served"]
    mae = sm["MAE_test"]
    check = abs(served_row["MAE"] - mae)
    if check > 0.0015:
        raise ValueError(f"{zone}: served_metrics MAE {mae} != evaluate Served {served_row['MAE']}")

    adv = ev.get("advanced", {})
    out = {
        "served_model": {
            "source": f"data/metadata/served_metrics_{zone}.json (scripts/gen_served_metrics.py)",
            "label": sm["production_model"],
            "served_model": sm["served_model"],
            "use_ar": sm["use_ar"],
            "n_hours": sm["n_test"],
            "MAE": sm["MAE_test"],
            "RMSE": sm["RMSE_test"],
            "DirAcc_pct": sm["DirAcc_test"],
            "R2": sm["R2_test"],
            "mean_forecast_minus_actual": sm.get("mean_forecast_minus_actual"),
            "MAE_pct_of_mean_price": sm.get("MAE_pct_of_mean_price"),
            "check_equals_evaluate_served_row": True,
        },
        "served_model_vs_baselines": {
            "source": "served MAE above and results/baselines_test2026.json",
            "MAE_persistence_24h": base["persistence_24h"]["MAE"],
            "MAE_lear": base["lear"]["MAE"],
            "rMAE_vs_persistence_24h": round(mae / base["persistence_24h"]["MAE"], 3),
            "rMAE_vs_lear": round(mae / base["lear"]["MAE"], 3),
            "MAE_change_vs_persistence_24h_pct": _pct(mae, base["persistence_24h"]["MAE"]),
            "MAE_change_vs_lear_pct": _pct(mae, base["lear"]["MAE"]),
        },
        "comparators_raw": {
            "source": f"results/evaluation_test2026_{zone}.json (src/model/evaluate.py)",
            "note": "raw models without the AR term; LightGBM is the 2,000-tree model "
                    "fitted on train plus validation, LightGBM_train the one fitted on "
                    "train only (the selection candidate)",
            **{k: v for k, v in ev["metrics"].items()
               if k in ("LightGBM", "LightGBM_train", "Ensemble")},
        },
        "mean_forecast_minus_actual": ev.get("mean_forecast_minus_actual"),
        "diebold_mariano_test": adv.get("diebold_mariano"),
        "interval_coverage_P10_P90": {
            "split_conformal": ev.get("conformal"),
            "aci_daily": adv.get("aci"),
        },
        "probabilistic_scores": {
            **(adv.get("probabilistic") or {}),
            "bands": "served (AR-shifted, conformal)" if sm["use_ar"] else "served (conformal)",
        },
        "battery_backtest": {
            "setup": "1 MW / 2 MWh, round-trip efficiency 0.90, daily linear program "
                     "(src/model/battery_backtest.py, method lp), settled at actual "
                     "prices, annualised from the test window",
            "served_forecast": adv.get("battery"),
            "persistence_forecast": adv.get("battery_persistence_forecast"),
            "served_minus_persistence_annual": _battery_gap(adv),
        },
        "walk_forward_summary": (tr or {}).get("walk_forward"),
        "test_window_prices": ev.get("test_prices"),
    }
    return out


def price_levels_by_split(features_path: Path = FEATURES_PATH) -> dict:
    """Precio medio, desviación y horas de train, validación y test por zona.

    Sale de la tabla de features del paso 14 (columnas price_<zona>), con los
    cortes de src/model/config.py. Es la fuente de la frase del README sobre el
    nivel de precios del test frente al de entrenamiento.
    """
    import pandas as pd
    from src.model.config import TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END

    if not features_path.exists():
        raise FileNotFoundError(f"falta {features_path.name} (paso 14)")
    cols = [f"price_{z}" for z in ZONES]
    df = pd.read_parquet(features_path, columns=cols)
    idx = pd.to_datetime(df.index, utc=True)

    def ts(s):
        return pd.Timestamp(s, tz="UTC")

    splits = {
        "train": (idx <= ts(TRAIN_END)),
        "validation": (idx >= ts(VAL_START)) & (idx <= ts(VAL_END)),
        "test": (idx >= ts(TEST_START)) & (idx <= ts(TEST_END)),
    }
    out = {}
    for z, col in zip(ZONES, cols):
        out[z] = {}
        for name, mask in splits.items():
            s = df.loc[mask, col].dropna()
            out[z][name] = {"mean": float(s.mean()), "std": float(s.std(ddof=1)),
                            "n_hours": int(len(s))}
    return out


def build_selection(zone: str) -> dict:
    tr = _load(RES / f"training_{zone}.json")
    if tr is None:
        raise FileNotFoundError(f"{zone}: falta results/training_{zone}.json (pasos 18-20)")
    return {
        "source": f"results/training_{zone}.json (src/model/train.py)",
        "hyperparameters": tr["hyperparameters"]["source"],
        "served": tr["served"],
        "selection": tr["selection_val"],
        "ar": tr["ar_val"],
        "quantiles": tr.get("quantiles_val"),
        "conformal": tr.get("conformal_val"),
        "reference_default_lightgbm": tr.get("reference_default_lightgbm"),
    }


def main():
    from src.model.config import TEST_START, TEST_END, VAL_START, VAL_END
    from src.model.results_io import write_json

    baselines = _load(RES / "baselines_test2026.json")
    zones_eval = {z: build_zone_eval(z, baselines) for z in ZONES}
    levels = price_levels_by_split()
    for z in ZONES:
        zones_eval[z]["price_level_by_split"] = {
            "source": "price_<zone> of data/master/dataset_features.parquet (step 14), "
                      "split dates of src/model/config.py",
            **levels[z],
        }
    evaluation = {
        "description": "Out-of-sample evaluation on the test window (January to May 2026, "
                       "hourly, UTC). The served model, the AR term and the interval "
                       "calibration were fixed on the October-December 2025 validation "
                       "quarter; the test window is only scored.",
        "generated_by": "scripts/build_results.py",
        "generated_on": datetime.now().strftime("%Y-%m-%d"),
        "sources": [
            "results/evaluation_test2026_<zone>.json (src/model/evaluate.py)",
            "data/metadata/served_metrics_<zone>.json (scripts/gen_served_metrics.py)",
            "results/training_<zone>.json (src/model/train.py)",
            "results/baselines_test2026.json",
        ],
        "test_period": {"start_utc": TEST_START, "end_utc": TEST_END},
        "units": {
            "MAE, RMSE, pinball, CRPS, prices": "EUR/MWh",
            "battery profit": "EUR per year, annualised from the test window",
            "DirAcc_pct": "% of hours where the sign of the hour-on-hour change is right",
        },
        "zones": zones_eval,
    }
    for name in ("fi_regimes_test2026.json", "leakage_sensitivity_test2026.json"):
        extra = _load(RES / name)
        if extra is not None:
            evaluation.setdefault("related_files", []).append(f"results/{name}")
    write_json(RES / "evaluation_test2026.json", evaluation)

    selection = {
        "description": "Model choice, AR decision and interval calibration on the validation "
                       f"quarter ({VAL_START} to {VAL_END}), with rules written before scoring.",
        "generated_by": "scripts/build_results.py",
        "generated_on": datetime.now().strftime("%Y-%m-%d"),
        "zones": {z: build_selection(z) for z in ZONES},
    }
    write_json(RES / "model_selection_val2025.json", selection)
    for z in ZONES:
        s = zones_eval[z]["served_model"]
        print(f"{z}: {s['label']} | MAE {s['MAE']} | R2 {s['R2']}")
    print("OK -> results/evaluation_test2026.json, results/model_selection_val2025.json")


if __name__ == "__main__":
    main()
