"""Comprobación manual del stack DK2 + validación cruzada aFRR FI (Fingrid vs Energinet).

No es un test de pytest: necesita data/master (generado por run_pipeline.py) e
imprime los resultados para revisarlos a mano.

Uso:  python checks/test_dk2_stack.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.battery_specs import BATTERY_SPECS
from src.model.revenue_stacking import yearly_stack_summary

master = pd.read_parquet(ROOT / "data" / "master" / "dataset_master.parquet")

# Validación cruzada: aFRR up FI segun Fingrid (52) vs Energinet (AfrrReservesNordic)
both = master[["fingrid_FI_afrr_up_price", "energinet_FI_afrr_up_price"]].dropna()
corr = both.corr().iloc[0, 1]
diff = (both.iloc[:, 0] - both.iloc[:, 1]).abs().median()
print(f"Validacion cruzada aFRR-up FI (Fingrid 52 vs EDS AfrrReservesNordic):")
print(f"  n={len(both):,} h | corr={corr:.3f} | mediana |dif|={diff:.2f} EUR/MW")

# Medias anuales FCR DK2
res_dk = [c for c in master.columns if c.startswith("energinet_DK2_fcr")]
print("\nMedias anuales FCR DK2 (EUR/MW,h):")
print(master[res_dk].groupby(master.index.year).mean().round(1).to_string())

# Stack anual DK2 (oracle)
spec = BATTERY_SPECS["generic_fi_2h"]
yearly = yearly_stack_summary(
    master, spec, price_col="price_DK2",
    fcrn_col="energinet_DK2_fcrn_price",
    fcrd_up_col="energinet_DK2_fcrd_up_price",
    fcrd_dn_col="energinet_DK2_fcrd_down_price", mode="oracle")
print("\nStack anual DK2 oracle (kEUR/MW):")
print((yearly[["rev_da_per_mw", "rev_fcrn_per_mw", "rev_fcrd_per_mw",
               "gross_per_mw", "net_per_mw"]] / 1000).round(1).to_string())
