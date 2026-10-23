"""Validación del módulo de revenue stacking con datos reales FI del master.

Comprueba: (1) el LP es factible y rápido, (2) el resumen anual 2023-2026
reproduce la narrativa real del mercado (desplome FCR, giro a arbitraje),
(3) el modo causal devuelve menos que el oracle (sin lookahead bias),
(4) los números por MW son del orden de los benchmarks públicos
(Clean Horizon ~400k €/MW/año modelado jul-2025; EBI 70-130k €/MW/año).

Comprobación manual, no un test de pytest: necesita data/master (generado por
run_pipeline.py) e imprime los resultados; solo (3) se verifica con un assert.

Uso:  python checks/test_revenue_stacking.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.battery_specs import BATTERY_SPECS
from src.model.revenue_stacking import stacking_backtest, yearly_stack_summary

master = pd.read_parquet(ROOT / "data" / "master" / "dataset_master.parquet")
spec = BATTERY_SPECS["generic_fi_2h"]

print(f"Spec: {spec.name} | {spec.power_mw} MW / {spec.energy_mwh} MWh | RTE {spec.rte:.0%}")
print(f"Coste ciclo: {spec.cycle_cost_eur_mwh():.1f} EUR/MWh | "
      f"fijos/anio: {spec.fixed_costs_eur_year()['total_eur']:,.0f} EUR")
print()

# ── 1) Resumen anual del POTENCIAL de mercado (oracle) ──────────────────────
t0 = time.time()
yearly = yearly_stack_summary(master, spec, mode="oracle")
print(f"Resumen anual oracle ({time.time()-t0:.1f}s):")
print((yearly / 1000).round(1).to_string())  # k EUR/MW
print()

# ── 2) Mismo resumen SOLO arbitraje (da_only) para ver el delta del stack ───
t0 = time.time()
yearly_da = yearly_stack_summary(master, spec, mode="da_only")
print(f"Solo arbitraje DA ({time.time()-t0:.1f}s):")
print((yearly_da[["rev_da_per_mw", "gross_per_mw", "net_per_mw"]] / 1000).round(1).to_string())
print()

# ── 3) Modo causal en 2025 (reservas decididas con forecast naive lag-24h) ──
y25 = master[master.index.year == 2025]
df25 = pd.DataFrame({
    "price_actual": y25["price_FI"],
    "price_pred": y25["price_FI"],          # aislamos el efecto reserva-causal
    "fcrn_price": y25["fingrid_FI_fcrn_price"],
    "fcrd_up_price": y25["fingrid_FI_fcrd_up_price"],
    "fcrd_dn_price": y25["fingrid_FI_fcrd_down_price"],
}, index=y25.index).dropna()
t0 = time.time()
oracle25 = stacking_backtest(df25, spec, mode="oracle")
causal25 = stacking_backtest(df25, spec, mode="causal")
print(f"2025 oracle vs causal ({time.time()-t0:.1f}s):")
for k in ("rev_da_annual", "rev_fcrn_annual", "rev_fcrd_annual", "gross_annual", "net_annual"):
    print(f"  {k:18s}  oracle {oracle25['per_mw'][k]/1000:8.1f} k/MW   "
          f"causal {causal25['per_mw'][k]/1000:8.1f} k/MW")
assert causal25["gross_annual"] <= oracle25["gross_annual"] * 1.001, \
    "El modo causal no puede superar al oracle"
print()
print("OK: causal <= oracle (sin lookahead bias)")
