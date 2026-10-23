"""
experiment_fcr_forecast.py — ¿Se pueden predecir los precios FCR nórdicos?
==========================================================================
GAP DETECTADO EN LA LITERATURA (revisión de jun-2026): no
existe ningún paper revisado que pronostique precios FCR-N/FCR-D nórdicos con
métricas publicadas (solo estudios alemanes de pay-as-bid). Este experimento
mide honestamente si un LightGBM bate al forecast naive (lag-24h) que usa el
modo causal del Battery Lab — si lo hace, el uplift es directamente monetizable
en la puja de reservas (en Alemania, 1 €/MW de mejora de MAE ≈ 483-3.631
€/MW/año de ingreso; arXiv:2503.17214).

INFORMACIÓN DISPONIBLE en el gate FCR (18:30 EET D-1) — sin leakage:
  - Precio DA de la zona para el día D (publicado ~13:00 D-1)  → contemporáneo OK
  - Previsiones day-ahead de eólica/carga del TSO para D       → contemporáneo OK
  - Precios FCR hasta el día D-1 incluido                      → lag ≥24h
  - Calendario.
Liquidación: el mercado horario FCR es marginal y sus precios se publican
~22:45 D-1 — el target es el precio del día D, conocido solo después del gate.

Split temporal idéntico al proyecto: train 2023→sep-2025, test ene–may-2026.
Baselines: naive lag-24h (el del modo causal) y lag-168h.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata

logger = get_logger("experiment_fcr")

TARGETS = {
    "FI_fcrn":      "fingrid_FI_fcrn_price",
    "FI_fcrd_up":   "fingrid_FI_fcrd_up_price",
    "FI_fcrd_down": "fingrid_FI_fcrd_down_price",
    "DK2_fcrn":     "energinet_DK2_fcrn_price",
    "DK2_fcrd_up":  "energinet_DK2_fcrd_up_price",
}

# Features disponibles en el gate (por zona)
ZONE_FEATS = {
    "FI":  ["price_FI", "fingrid_FI_wind_fc_mw", "fingrid_FI_load_fc_mw",
            "forecast_wind_FI", "load_fc_FI", "wx_FI_temp_avg"],
    "DK2": ["price_DK2", "forecast_wind_DK2", "load_fc_DK2"],
}

TRAIN_END = "2025-09-30 23:00+00:00"
VAL_END   = "2025-12-31 23:00+00:00"   # val: Q4-2025 (early stopping)
TEST_END  = "2026-05-31 23:00+00:00"


def build_xy(master: pd.DataFrame, target_col: str, zone: str):
    df = pd.DataFrame(index=master.index)
    y = master[target_col]
    # Lags del propio precio FCR (≥24h: lo conocido en el gate)
    for lag in (24, 48, 72, 168):
        df[f"y_lag{lag}"] = y.shift(lag)
    df["y_roll7d"] = y.shift(24).rolling(168).mean()
    df["y_roll7d_max"] = y.shift(24).rolling(168).max()
    # Contemporáneos legítimos (conocidos antes del gate FCR)
    for c in ZONE_FEATS[zone]:
        if c in master.columns:
            df[c] = master[c]
    # El spread DA del día: precio de la hora vs media del día (forma de la curva)
    zprice = ZONE_FEATS[zone][0]
    daily_mean = master[zprice].groupby(master.index.normalize()).transform("mean")
    df["da_vs_daymean"] = master[zprice] - daily_mean
    # Calendario
    df["hour"] = master.index.hour
    df["dow"] = master.index.dayofweek
    df["month"] = master.index.month
    return df, y


def _fit_predict(tr, va, te, target_name):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(tr.drop(columns=target_name), tr[target_name],
              eval_set=[(va.drop(columns=target_name), va[target_name])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model, model.predict(te.drop(columns=target_name))


def run_target(master, name, target_col):
    """
    Tres variantes medidas contra el mismo test (la cultura DT-14 del proyecto:
    medir antes de comprometer):
      A) nivel:      target = precio (formulación obvia — falla con cambio de régimen)
      B) desviación: target = precio − lag24; pred = lag24 + dev (anclado al régimen)
      C) B + ventana corta (entrena solo desde jul-2024, descarta el régimen 2023)
    """
    zone = name.split("_")[0]
    X, y = build_xy(master, target_col, zone)
    data = pd.concat([X, y.rename("target")], axis=1).dropna()
    data["dev"] = data["target"] - data["y_lag24"]
    tr = data.loc[:TRAIN_END]
    va = data.loc[TRAIN_END:VAL_END].iloc[1:]
    te = data.loc[VAL_END:TEST_END].iloc[1:]
    if len(te) < 500 or len(tr) < 5000:
        logger.warning(f"  {name}: datos insuficientes (tr={len(tr)}, te={len(te)})")
        return None

    yt = te["target"].values
    lag24 = te["y_lag24"].values
    mae_n24 = np.abs(lag24 - yt).mean()
    mae_n168 = np.abs(te["y_lag168"].values - yt).mean()

    # A) nivel
    _, pa = _fit_predict(tr.drop(columns="dev"), va.drop(columns="dev"),
                         te.drop(columns="dev"), "target")
    mae_a = np.abs(np.clip(pa, 0, None) - yt).mean()
    # B) desviación vs naive-24h
    cols_b = [c for c in data.columns if c != "target"]
    mb, pb = _fit_predict(tr[cols_b], va[cols_b], te[cols_b], "dev")
    mae_b = np.abs(np.clip(lag24 + pb, 0, None) - yt).mean()
    # C) desviación + ventana corta (régimen reciente)
    tr_c = tr.loc["2024-07-01":]
    _, pc = _fit_predict(tr_c[cols_b], va[cols_b], te[cols_b], "dev")
    mae_c = np.abs(np.clip(lag24 + pc, 0, None) - yt).mean()

    best_name, best_mae = min((("A_nivel", mae_a), ("B_desviacion", mae_b),
                               ("C_desv_corta", mae_c)), key=lambda t: t[1])
    imp = mb.feature_importances_
    top = sorted(zip(cols_b, imp), key=lambda t: -t[1])[:4]
    res = {
        "target": name, "n_test": len(te),
        "test_mean_eur_mw": round(float(yt.mean()), 2),
        "mae_naive24": round(float(mae_n24), 3),
        "mae_naive168": round(float(mae_n168), 3),
        "mae_A_nivel": round(float(mae_a), 3),
        "mae_B_desviacion": round(float(mae_b), 3),
        "mae_C_desv_ventana_corta": round(float(mae_c), 3),
        "best": best_name,
        "best_improvement_vs_naive24_pct": round(float((1 - best_mae / mae_n24) * 100), 1),
        "top_features_B": [t[0] for t in top],
    }
    logger.info(
        f"  {name:14s} naive24 {mae_n24:5.2f} | A:{mae_a:6.2f} B:{mae_b:5.2f} "
        f"C:{mae_c:5.2f} → mejor {best_name} ({res['best_improvement_vs_naive24_pct']:+.1f}%) "
        f"| media test {yt.mean():.1f}")
    return res


def main():
    master = pd.read_parquet(ROOT / "data" / "master" / "dataset_master.parquet")
    logger.info("=== Experimento: forecasting de precios FCR (gap de literatura) ===")
    logger.info(f"Split: train→{TRAIN_END[:10]} | val Q4-2025 | test ene–may 2026\n")
    results = []
    for name, col in TARGETS.items():
        if col not in master.columns:
            logger.warning(f"  {name}: columna {col} ausente")
            continue
        r = run_target(master, name, col)
        if r:
            results.append(r)

    ok = [r for r in results if r["best_improvement_vs_naive24_pct"] > 0]
    logger.info(f"\nVeredicto: la mejor variante bate al naive-24h en {len(ok)}/{len(results)} targets")
    save_run_metadata(script_name="experiment_fcr",
                      params={"train_end": TRAIN_END, "test_end": TEST_END},
                      stats={r["target"]: r["best_improvement_vs_naive24_pct"] for r in results})
    out = ROOT / "data" / "metadata" / "experiment_fcr_results.json"
    import json
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Resultados → {out}")


if __name__ == "__main__":
    main()
