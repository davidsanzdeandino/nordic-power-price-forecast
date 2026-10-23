"""
evaluate_fi_regimes.py — Evaluación del modelo FI por régimen
=============================================================
Finlandia es el mercado más volátil de Europa: hay que reportar el
MAE NO solo agregado, sino desglosado por los regímenes que de verdad importan,
y siempre contra el baseline estándar LEAR (no la persistencia ingenua).

Regímenes:
  - Overall (todo el test out-of-sample, ene–may 2026)
  - Olkiluoto/nuclear OUT  (unavail_FI alto)  → el sistema se tensiona
  - Horas de precio NEGATIVO (exceso eólico/nuclear) → FI es la zona #1 en Europa
  - Horas FRÍAS (demanda de calefacción)

Uso:  python scripts/evaluate_fi_regimes.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mae(a, f):
    m = a.notna() & f.notna()
    return float((f[m] - a[m]).abs().mean()) if m.any() else np.nan


def main():
    from src.data_loader import load_test_predictions, load_lear_forecast
    master = pd.read_parquet(ROOT / "data" / "master" / "dataset_master.parquet")
    master.index = master.index.tz_convert("UTC")

    df = load_test_predictions("FI")
    if df is None or df.empty:
        print("No hay predicciones FI — entrena el modelo primero.")
        return
    idx = pd.to_datetime(df["timestamp_cet"], utc=True)
    g = pd.DataFrame({"a": df["price_actual"].astype(float).values,
                      "f": df["price_pred"].astype(float).values}, index=idx).dropna()

    # LEAR alineado
    lear = load_lear_forecast("FI")
    if lear is not None and not lear.empty:
        lear.index = pd.to_datetime(lear.index, utc=True)
        g["l"] = lear.reindex(g.index)

    # Indicadores de régimen desde el master (alineados al índice de test)
    def col(name):
        return master[name].reindex(g.index) if name in master.columns else pd.Series(np.nan, index=g.index)

    unavail = col("unavail_FI_total_mw")
    temp = None
    for c in ["wx_HelsinkiFI_temperature_2m", "wx_VaasaFI_temperature_2m"]:
        if c in master.columns:
            temp = col(c) if temp is None else temp + col(c)
    if temp is not None:
        temp = temp / sum(c in master.columns for c in
                          ["wx_HelsinkiFI_temperature_2m", "wx_VaasaFI_temperature_2m"])

    nuc_thr = float(np.nanpercentile(unavail.dropna(), 75)) if unavail.notna().any() else np.inf
    regimes = {
        "Overall":            g.index == g.index,
        "Nuclear/OL3 out (top quartile unavail)": unavail > nuc_thr,
        "Negative-price hours (actual < 0)":       g["a"] < 0,
        "Cold hours (temp < 0 °C)":                (temp < 0) if temp is not None else (g.index != g.index),
    }

    has_lear = "l" in g.columns and g["l"].notna().any()
    print(f"\n=== FI evaluation by regime · test {g.index.min():%Y-%m-%d} → "
          f"{g.index.max():%Y-%m-%d} ({len(g):,} h) ===")
    header = f"{'Regime':42s} {'n':>6s} {'MAE model':>10s}"
    if has_lear:
        header += f" {'MAE LEAR':>10s} {'model-LEAR':>11s}"
    print(header)
    print("-" * len(header))
    for name, mask in regimes.items():
        sub = g[mask]
        if len(sub) == 0:
            continue
        line = f"{name:42s} {len(sub):6d} {mae(sub['a'], sub['f']):10.2f}"
        if has_lear:
            ml = mae(sub['a'], sub['l'])
            line += f" {ml:10.2f} {mae(sub['a'], sub['f']) - ml:+11.2f}"
        print(line)
    print()
    print("Nota: model−LEAR negativo = el modelo bate al baseline LEAR. FI es el "
          "mercado más volátil de Europa, así que un MAE ~21–24 €/MWh es realista y "
          "esperable, no un número inflado.")

    # ── Persistir métricas SERVIDAS (las que de verdad produce predict.py) ─────
    import json
    a, f = g["a"], g["f"]
    # Direction accuracy vs hora previa (mismo criterio que el resto del proyecto)
    da_mask = a.diff().notna() & f.diff().notna()
    diracc = float((np.sign(f.diff()[da_mask]) == np.sign(a.diff()[da_mask])).mean() * 100)
    served = {
        "MAE_test": round(mae(a, f), 3),
        "DirAcc_test": round(diracc, 2),
        "RMSE_test": round(float(np.sqrt(((f - a) ** 2).mean())), 3),
        "n_test": int(len(g)),
        "production_model": "Diverse ensemble (L2+median+Huber) + AR",
    }
    if has_lear:
        served["MAE_lear"] = round(mae(a, g["l"]), 3)
        served["MAE_vs_lear"] = round(mae(a, f) - mae(a, g["l"]), 3)
    out_path = ROOT / "data" / "metadata" / "served_metrics_FI.json"
    json.dump(served, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"\nOK: métricas servidas guardadas en {out_path.name}: "
          f"MAE {served['MAE_test']} | DirAcc {served['DirAcc_test']}% | "
          f"vs LEAR {served.get('MAE_vs_lear')}")


if __name__ == "__main__":
    main()
