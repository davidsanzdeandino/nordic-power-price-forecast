"""
gen_served_metrics.py — Métricas del modelo REALMENTE servido por predict.py, por zona
=======================================================================================
Mide la salida de predict.py (el modelo de producción elegido por DM test: LightGBM
para DK, ensemble para FI) sobre el test out-of-sample y la persiste en
data/metadata/served_metrics_<zone>.json, para que el dashboard y el audit trail
reflejen lo que de verdad se sirve (no un sub-modelo interno).

Uso:  python scripts/gen_served_metrics.py DK1 DK2
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mae(a, f):
    m = a.notna() & f.notna()
    return float((f[m] - a[m]).abs().mean()) if m.any() else np.nan


def served_metrics(zone: str) -> dict:
    from src.data_loader import load_test_predictions, load_lear_forecast
    df = load_test_predictions(zone)
    if df is None or df.empty:
        return {}
    ts = "timestamp_cet" if "timestamp_cet" in df.columns else "timestamp_utc"
    idx = pd.to_datetime(df[ts], utc=True)
    g = pd.DataFrame({"a": df["price_actual"].astype(float).values,
                      "f": df["price_pred"].astype(float).values}, index=idx).dropna()
    a, f = g["a"], g["f"]
    da = a.diff().notna() & f.diff().notna()
    diracc = float((np.sign(f.diff()[da]) == np.sign(a.diff()[da])).mean() * 100)
    ss_res = float(((a - f) ** 2).sum()); ss_tot = float(((a - a.mean()) ** 2).sum())
    out = {
        "MAE_test": round(mae(a, f), 3),
        "RMSE_test": round(float(np.sqrt(((f - a) ** 2).mean())), 3),
        "DirAcc_test": round(diracc, 2),
        "R2_test": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
        "n_test": int(len(g)),
        "production_model": ("Diverse ensemble (L2+median+Huber) + AR"
                             if zone in {"FI"} else "LightGBM + AR"),
    }
    lear = load_lear_forecast(zone)
    if lear is not None and not lear.empty:
        lear.index = pd.to_datetime(lear.index, utc=True)
        l = lear.reindex(g.index)
        out["MAE_lear"] = round(mae(a, l), 3)
        out["MAE_vs_lear"] = round(mae(a, f) - mae(a, l), 3)
    return out


def main():
    zones = sys.argv[1:] or ["DK1", "DK2"]
    for z in zones:
        m = served_metrics(z)
        if not m:
            print(f"{z}: sin predicciones — omitido")
            continue
        path = ROOT / "data" / "metadata" / f"served_metrics_{z}.json"
        json.dump(m, open(path, "w", encoding="utf-8"), indent=2)
        print(f"{z}: MAE {m['MAE_test']} | DirAcc {m['DirAcc_test']}% | "
              f"R2 {m.get('R2_test')} | vs LEAR {m.get('MAE_vs_lear')} -> {path.name}")


if __name__ == "__main__":
    main()
