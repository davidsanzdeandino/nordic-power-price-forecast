"""Comprobación manual (headless) de la lógica de datos del Battery Lab (sin la UI de Streamlit).

No es un test de pytest: necesita data/master y las predicciones de test que
genera run_pipeline.py, e imprime los resultados para revisarlos a mano.

Uso:  python checks/test_battery_lab_data.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_test_predictions
from src.model.battery_specs import BATTERY_SPECS
from src.model.revenue_stacking import stacking_backtest

spec = BATTERY_SPECS["sungrow_powertitan2_2h"]   # el producto típico FI 2025-26
master = pd.read_parquet(ROOT / "data" / "master" / "dataset_master.parquet")

preds = load_test_predictions("FI")
ts = "timestamp_utc" if "timestamp_utc" in preds.columns else "timestamp_cet"
preds = preds.set_index(pd.to_datetime(preds[ts], utc=True))
print(f"Predicciones test FI: {len(preds):,} h | {preds.index.min()} -> {preds.index.max()}")

df = pd.DataFrame({
    "price_actual": preds["price_actual"],
    "price_pred": preds["price_pred"],
    "fcrn_price": master["fingrid_FI_fcrn_price"].reindex(preds.index),
    "fcrd_up_price": master["fingrid_FI_fcrd_up_price"].reindex(preds.index),
    "fcrd_dn_price": master["fingrid_FI_fcrd_down_price"].reindex(preds.index),
}, index=preds.index)
print(f"Cobertura reservas en test: {df['fcrn_price'].notna().mean():.0%}")

for mode in ("oracle", "causal", "da_only"):
    r = stacking_backtest(df, spec, mode=mode)
    print(f"\n[{mode}] (anualizado, por MW)")
    for k in ("rev_da_annual", "rev_fcrn_annual", "rev_fcrd_annual", "gross_annual", "net_annual"):
        print(f"  {k:18s} {r['per_mw'][k]/1000:9.1f} k EUR/MW")
    print(f"  ciclos/dia DA: {r['cycles_per_day']:.2f}")
