"""
gen_served_metrics.py — Métricas del modelo servido por predict.py, por zona
=============================================================================
Mide la salida de predict.served_predictions (predict_with_bands: modelo
elegido en validación, AR day-ahead por día de entrega y bandas calibradas)
sobre el test ene-may 2026 y la guarda en data/metadata/served_metrics_<zona>.json.
Es la misma ruta de código que la fila «Served» de src/model/evaluate.py, así
que las dos cifras coinciden.

Los MAE de persistencia y LEAR salen de results/baselines_test2026.json.

Uso:  python scripts/gen_served_metrics.py DK1 DK2 FI
"""
import sys
import json
import pickle
from datetime import datetime
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINES_PATH = ROOT / "results" / "baselines_test2026.json"


def served_metrics(zone: str) -> dict:
    from src.model.config import TEST_START, TEST_END
    from src.model import predict as predict_mod

    with open(predict_mod.MODELS_DIR / f"model_meta_{zone}.pkl", "rb") as f:
        meta = pickle.load(f)
    served_predictions = predict_mod.served_predictions
    served_model_label = predict_mod.served_model_label
    g = served_predictions(zone, TEST_START, TEST_END)
    if g.empty:
        return {}
    a = g["price_actual"].astype(float)
    f = g["price_pred"].astype(float)
    err = f - a
    da = a.diff().notna() & f.diff().notna()
    diracc = float((np.sign(f.diff()[da]) == np.sign(a.diff()[da])).mean() * 100)
    ss_res = float((err ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    mae = float(err.abs().mean())
    served = meta.get("served_model", "lightgbm")
    use_ar = bool(meta.get("use_ar", False))
    out = {
        "generated_by": "scripts/gen_served_metrics.py",
        "generated_on": datetime.now().strftime("%Y-%m-%d"),
        "method": "predict.served_predictions over the test window (same code path as the "
                  "Served row of src/model/evaluate.py)",
        "n_test": int(len(g)),
        "test_window_utc": [TEST_START, TEST_END],
        "MAE_test": round(mae, 3),
        "RMSE_test": round(float(np.sqrt((err ** 2).mean())), 3),
        "DirAcc_test": round(diracc, 2),
        "R2_test": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
        "mean_forecast_minus_actual": round(float(err.mean()), 3),
        "mean_price_test": round(float(a.mean()), 3),
        "MAE_pct_of_mean_price": round(mae / float(a.mean()) * 100, 1) if a.mean() else None,
        "served_model": served,
        "use_ar": use_ar,
        "production_model": served_model_label(served, use_ar),
    }
    if BASELINES_PATH.exists():
        base = json.load(open(BASELINES_PATH, encoding="utf-8"))["zones"].get(zone, {})
        b = base.get("baselines", {})
        if "persistence_24h" in b:
            m_p = float(b["persistence_24h"]["MAE"])
            out["MAE_persistence_24h"] = m_p
            out["rMAE_vs_persistence_24h"] = round(mae / m_p, 3)
            out["MAE_change_vs_persistence_24h_pct"] = round((mae - m_p) / m_p * 100, 1)
        if "lear" in b:
            m_l = float(b["lear"]["MAE"])
            out["MAE_lear"] = m_l
            out["MAE_vs_lear"] = round(mae - m_l, 3)
            out["MAE_change_vs_lear_pct"] = round((mae - m_l) / m_l * 100, 1)
    return out


def main():
    zones = sys.argv[1:] or ["DK1", "DK2", "FI"]
    for z in zones:
        m = served_metrics(z)
        if not m:
            print(f"{z}: sin predicciones — omitido")
            continue
        path = ROOT / "data" / "metadata" / f"served_metrics_{z}.json"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(m, fh, indent=2)
            fh.write("\n")
        print(f"{z}: {m['production_model']} | MAE {m['MAE_test']} | "
              f"DirAcc {m['DirAcc_test']}% | R2 {m.get('R2_test')} | "
              f"vs LEAR {m.get('MAE_change_vs_lear_pct')}% -> {path.name}")


if __name__ == "__main__":
    main()
