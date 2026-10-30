"""Comprobación manual (headless) de la lógica de datos del Battery Lab (sin la UI de Streamlit).

No es un test de pytest: necesita data/master y los modelos que genera
run_pipeline.py, e imprime los resultados para revisarlos a mano.

Usa la previsión servida de FI en el test (predict.served_predictions, la misma
ruta que predict_with_bands y que el dashboard) y las dos baterías del bloque
test_window_2026 de results/battery_lab_fi.json.

Uso:  python checks/test_battery_lab_data.py            # solo imprime
      python checks/test_battery_lab_data.py --write    # y actualiza el bloque
                                                        # test_window_2026 del JSON
"""
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.battery_specs import BATTERY_SPECS  # noqa: E402
from src.model.config import TEST_START, TEST_END  # noqa: E402
from src.model.predict import served_predictions, served_model_label  # noqa: E402
from src.model.revenue_stacking import stacking_backtest  # noqa: E402

OUT = ROOT / "results" / "battery_lab_fi.json"
SPECS = ["generic_fi_2h", "sungrow_powertitan2_2h"]
MODES = ("oracle", "causal", "da_only")

master = pd.read_parquet(ROOT / "data" / "master" / "dataset_master.parquet")

preds = served_predictions("FI", TEST_START, TEST_END)   # indexado por timestamp_utc
with open(ROOT / "data" / "models" / "model_meta_FI.pkl", "rb") as fh:
    meta = pickle.load(fh)
print(f"Predicciones test FI: {len(preds):,} h | {preds.index.min()} -> {preds.index.max()}")

df = pd.DataFrame({
    "price_actual": preds["price_actual"],
    "price_pred": preds["price_pred"],
    "fcrn_price": master["fingrid_FI_fcrn_price"].reindex(preds.index),
    "fcrd_up_price": master["fingrid_FI_fcrd_up_price"].reindex(preds.index),
    "fcrd_dn_price": master["fingrid_FI_fcrd_down_price"].reindex(preds.index),
}, index=preds.index)
print(f"Cobertura reservas en test: {df['fcrn_price'].notna().mean():.0%}")

batteries = {}
for key in SPECS:
    spec = BATTERY_SPECS[key]
    print(f"\n=== {spec.name} | {spec.power_mw} MW / {spec.energy_mwh} MWh | RTE {spec.rte:.3f}")
    res = {}
    raw_gross = {}
    for mode in MODES:
        r = stacking_backtest(df, spec, mode=mode)
        pm = r["per_mw"]
        res[mode] = {
            "rev_da": round(pm["rev_da_annual"] / 1000, 1),
            "rev_fcrn": round(pm["rev_fcrn_annual"] / 1000, 1),
            "rev_fcrd": round(pm["rev_fcrd_annual"] / 1000, 1),
            "gross": round(pm["gross_annual"] / 1000, 1),
            "net": round(pm["net_annual"] / 1000, 1),
            "cycles_per_day": round(float(r["cycles_per_day"]), 2),
        }
        raw_gross[mode] = pm["gross_annual"]
        print(f"[{mode}] " + " | ".join(f"{k} {v}" for k, v in res[mode].items()))
    res["causal_share_of_oracle_pct"] = round(raw_gross["causal"] / raw_gross["oracle"] * 100, 1)
    print(f"  causal / oracle: {res['causal_share_of_oracle_pct']}%")
    batteries[key] = {"power_mw": spec.power_mw, "energy_mwh": spec.energy_mwh,
                      "round_trip_efficiency": spec.rte, **res}

if "--write" in sys.argv[1:]:
    data = json.load(open(OUT, encoding="utf-8"))
    tw = data["test_window_2026"]
    tw["source"] = ("checks/test_battery_lab_data.py --write (stacking_backtest on the served "
                    "FI test forecast from predict.served_predictions), both battery specs below")
    tw["generated_on"] = datetime.now().strftime("%Y-%m-%d")
    tw["forecast"] = served_model_label(meta.get("served_model", "lightgbm"),
                                        bool(meta.get("use_ar", False)))
    tw["period"] = {"start_utc": str(preds.index.min()), "end_utc": str(preds.index.max()),
                    "hours": int(len(preds))}
    for key, vals in batteries.items():
        old = tw["batteries"].get(key, {})
        tw["batteries"][key] = {**vals, "description": old.get("description", "")}
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"\nOK -> results/{OUT.name} (test_window_2026)")
