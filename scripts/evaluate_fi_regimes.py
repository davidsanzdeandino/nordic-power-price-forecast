"""
evaluate_fi_regimes.py — Evaluación del modelo FI por régimen
=============================================================
Finlandia es el mercado más volátil de Europa: el MAE se desglosa por los
regímenes que importan y siempre frente al baseline LEAR.

Regímenes:
  - Overall (todo el test out-of-sample, ene–may 2026)
  - Olkiluoto/nuclear OUT  (cuartil superior de indisponibilidad FI)
  - Horas de precio NEGATIVO
  - Horas FRÍAS (temperatura media Helsinki/Vaasa < 0 °C)

Usa la previsión servida (predict.served_predictions, la misma ruta que
evaluate.py y gen_served_metrics.py) y escribe results/fi_regimes_test2026.json.
No reescribe served_metrics_FI.json (eso es scripts/gen_served_metrics.py).

Uso:  python scripts/evaluate_fi_regimes.py
"""

import sys
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mae(a, f):
    m = a.notna() & f.notna()
    return float((f[m] - a[m]).abs().mean()) if m.any() else np.nan


def main():
    from src.model.config import TEST_START, TEST_END
    from src.model.predict import served_predictions, PricePredictor
    from src.model.baselines import LEARBaseline
    from src.model.results_io import write_json

    zone = "FI"
    master = pd.read_parquet(ROOT / "data" / "master" / "dataset_master.parquet")
    master.index = master.index.tz_convert("UTC")

    predictor = PricePredictor(zone=zone)
    srv = served_predictions(zone, TEST_START, TEST_END, predictor=predictor)
    if srv.empty:
        print("No hay predicciones FI — entrena el modelo primero.")
        return
    g = pd.DataFrame({"a": srv["price_actual"], "f": srv["price_pred"]})

    # LEAR (mismo baseline que results/baselines_test2026.json)
    feats = predictor._feature_data
    series = feats[f"price_{zone}"].dropna()
    lear = LEARBaseline(alpha=0.1, zone=zone)
    lear.fit(series, TEST_START, df_extra=feats)
    g["l"] = lear.predict(series, TEST_START, df_extra=feats).reindex(g.index)

    def col(name):
        return master[name].reindex(g.index) if name in master.columns else pd.Series(np.nan, index=g.index)

    unavail = col("unavail_FI_total_mw")
    temp_cols = [c for c in ["wx_HelsinkiFI_temperature_2m", "wx_VaasaFI_temperature_2m"]
                 if c in master.columns]
    temp = sum(col(c) for c in temp_cols) / len(temp_cols) if temp_cols else None

    nuc_thr = float(np.nanpercentile(unavail.dropna(), 75)) if unavail.notna().any() else np.inf
    regimes = {
        "Overall":                                 g.index == g.index,
        "Nuclear/OL3 out (top quartile unavail)":  (unavail > nuc_thr).to_numpy(),
        "Negative-price hours (actual < 0)":       (g["a"] < 0).to_numpy(),
        "Cold hours (temp < 0 °C)":                (temp < 0).to_numpy() if temp is not None
                                                   else np.zeros(len(g), dtype=bool),
    }

    has_lear = g["l"].notna().any()
    print(f"\n=== FI evaluation by regime · test {g.index.min():%Y-%m-%d} → "
          f"{g.index.max():%Y-%m-%d} ({len(g):,} h) ===")
    header = f"{'Regime':42s} {'n':>6s} {'MAE model':>10s}"
    if has_lear:
        header += f" {'MAE LEAR':>10s} {'model-LEAR':>11s}"
    print(header)
    print("-" * len(header))
    rows = {}
    for name, mask in regimes.items():
        sub = g[mask]
        if len(sub) == 0:
            continue
        row = {"n": int(len(sub)), "MAE_served": mae(sub["a"], sub["f"])}
        line = f"{name:42s} {len(sub):6d} {row['MAE_served']:10.2f}"
        if has_lear:
            row["MAE_lear"] = mae(sub["a"], sub["l"])
            row["MAE_served_minus_lear"] = row["MAE_served"] - row["MAE_lear"]
            line += f" {row['MAE_lear']:10.2f} {row['MAE_served_minus_lear']:+11.2f}"
        rows[name] = row
        print(line)

    out = {
        "zone": zone,
        "generated_by": "scripts/evaluate_fi_regimes.py",
        "generated_on": datetime.now().strftime("%Y-%m-%d"),
        "method": "served FI forecast (predict.served_predictions) and LEAR baseline over the "
                  "test window, MAE by regime; descriptive, not used for any choice",
        "test_window_utc": [TEST_START, TEST_END],
        "served_label": predictor.summary()["served_label"],
        "nuclear_threshold_mw": nuc_thr,
        "nuclear_threshold_note": "75th percentile of unavail_FI_total_mw over the test hours",
        "regimes": rows,
    }
    path = write_json(ROOT / "results" / "fi_regimes_test2026.json", out)
    print(f"\nOK: regímenes guardados en results/{path.name}")


if __name__ == "__main__":
    main()
