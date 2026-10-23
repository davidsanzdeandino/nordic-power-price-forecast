"""
battery_specs.py — Librería de baterías comerciales grid-scale (specs verificadas)
==================================================================================
Especificaciones de productos BESS comerciales reales para parametrizar el
backtest multi-mercado. Cada número lleva su fuente; los campos que NO publica
el fabricante se marcan explícitamente en `estimated_fields` y se rellenan con
rangos de literatura/industria — la honestidad sobre qué es dato y qué es
estimación es parte del diseño.

Fuentes principales (verificadas 10-11 jun 2026, ver docs/research/raw/bess-specs.md):
  - Tesla Megapack 2 XL datasheet REV 1.6 (abr 2023).
  - Wärtsilä Quantum2 spec sheet / Quantum3 solution sheet (wartsila.com).
  - Fluence Gridstack Pro 5000 (batterydesign.net, espejo del spec BR-042-02-EN).
  - Sungrow PowerTitan 2.0 ST5015UX datasheet V5 (2024); RTE 89,5% de nota de prensa.
  - CATL TENER (catl.com; el claim "cero degradación 5 años" se modela como
    sobredimensionamiento oculto, según la lectura de la industria).
  - CAPEX: BNEF ESS Cost Survey 2025 (Europa ~177 $/kWh) + datapoints reales
    finlandeses (Exilion/Merus Kuortti 30 MW/66 MWh ≈ 258 €/kWh turnkey;
    Neve Rovaniemi 30/80 ≈ 163 €/kWh).
  - Degradación: Xu et al. (coste de ciclo rainflow lineal a tramos),
    arXiv:1904.07771 (coste marginal 7→13 $/MWh, LCOD ~17 $/MWh),
    Preger et al. 2020 (LFP 2.500-9.000 EFC hasta 80%).
  - Fees FI: Fingrid capacity fee almacenamiento 87,5 €/MW/mes sobre la SUMA
    de potencia de carga + descarga (desde 1 ago 2025). Impuesto eléctrico:
    el almacenamiento está exento de doble imposición en FI desde 2019.

Uso:
    from src.model.battery_specs import BATTERY_SPECS, BatterySpec
    spec = BATTERY_SPECS["megapack_2xl_2h"]
    spec.cycle_cost_eur_mwh()   # coste de degradación por MWh descargado
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BatterySpec:
    key: str
    name: str
    vendor: str
    # Eléctrico (AC, en punto de conexión, 25 °C salvo nota)
    energy_mwh: float          # energía utilizable
    power_mw: float            # potencia continua
    rte: float                 # eficiencia round-trip AC (0-1)
    # Vida útil
    cycle_life_efc: int        # ciclos equivalentes completos hasta eol_capacity_pct
    eol_capacity_pct: float    # capacidad remanente que define el fin de vida (0-1)
    calendar_life_years: int
    # Económico (turnkey, Nordics 2025-26)
    capex_eur_per_kwh: float
    cell_share_of_capex: float  # fracción del CAPEX que es batería/celdas (para coste de ciclo)
    # Operación
    aux_kw_idle: float         # consumo auxiliar en espera (HVAC/control), media anual nórdica
    aux_kw_cycling: float      # consumo auxiliar adicional ciclando
    availability: float        # disponibilidad anual (0-1)
    warranty_cycles_per_day: float  # límite de ciclado típico de garantía
    # Trazabilidad
    sources: tuple = ()
    estimated_fields: tuple = ()   # campos SIN dato público del fabricante (estimados)
    notes: str = ""

    @property
    def duration_h(self) -> float:
        return self.energy_mwh / self.power_mw

    @property
    def one_way_eff(self) -> float:
        """Eficiencia por sentido (carga = descarga = sqrt(RTE))."""
        return self.rte ** 0.5

    def cycle_cost_eur_mwh(self) -> float:
        """
        Coste marginal de degradación por MWh descargado de celdas:
        amortización de la parte 'batería' del CAPEX sobre la energía total
        descargable en vida (EFC × energía). Convención estándar de la
        literatura de dispatch (Xu et al.; arXiv:1904.07771 da 7-13 $/MWh).
        """
        capex_cells_eur = self.capex_eur_per_kwh * 1000.0 * self.energy_mwh * self.cell_share_of_capex
        lifetime_discharge_mwh = self.cycle_life_efc * self.energy_mwh
        return capex_cells_eur / lifetime_discharge_mwh

    def total_capex_eur(self) -> float:
        return self.capex_eur_per_kwh * 1000.0 * self.energy_mwh

    def fixed_costs_eur_year(self, country: str = "FI") -> dict:
        """
        Costes fijos anuales: O&M (convención NREL ATB: 2,5% CAPEX incl.
        augmentation), seguro (0,75% del valor, centro del rango 0,3-1,2%),
        y fee de red. FI: Fingrid 87,5 €/MW/mes sobre carga+descarga (2×P).
        DK: la batería tarifica hoy como consumidor+productor (sin modelo
        específico BESS publicado); usamos la tarifa de consumo de red de
        transmisión de Energinet sobre la energía cargada — se aplica en el
        backtest por MWh, aquí solo el término fijo (0).
        """
        capex = self.total_capex_eur()
        om = 0.025 * capex
        insurance = 0.0075 * capex
        # FI: 87,5 €/MW/mes × 12 meses × (P_carga + P_descarga = 2×P)
        grid_fixed = 87.5 * 12 * (2 * self.power_mw) if country == "FI" else 0.0
        return {"om_eur": om, "insurance_eur": insurance, "grid_fixed_eur": grid_fixed,
                "total_eur": om + insurance + grid_fixed}


_XU = "Xu et al., 'Factoring the Cycle Aging Cost of Batteries Participating in Electricity Markets'"
_BNEF = "BNEF ESS Cost Survey 2025: Europe 177 $/kWh average turnkey; 2h global 124 $/kWh"
_PREGER = "Preger et al. 2020 (J. Electrochem. Soc. 167:120532): LFP 2,500-9,000 EFC to 80%"


BATTERY_SPECS: dict[str, BatterySpec] = {
    # ── Tesla Megapack 2 XL — el estándar global de proyecto utility ─────────
    "megapack_2xl_2h": BatterySpec(
        key="megapack_2xl_2h", name="Tesla Megapack 2 XL (2h)", vendor="Tesla",
        energy_mwh=3.854, power_mw=1.927, rte=0.917,
        cycle_life_efc=8000, eol_capacity_pct=0.70, calendar_life_years=20,
        capex_eur_per_kwh=240.0, cell_share_of_capex=0.55,
        aux_kw_idle=8.0, aux_kw_cycling=12.0, availability=0.97,
        warranty_cycles_per_day=1.0,
        sources=(
            "Datasheet REV 1.6 (21 Apr 2023): 1,927 kW / 3,854 kWh, RTE 91.7% full cycle incl. PCS and thermal at 25 °C; -30/+50 °C; warranty 15 years, extendable to 20",
            "Online configurator price Jul 2024: 266 $/kWh excluding installation: https://pv-magazine-usa.com/2024/07/03/tesla-battery-deployment-up-157-megapack-pricing-down-44/",
            _BNEF, _PREGER,
        ),
        estimated_fields=("cycle_life_efc", "aux_kw_idle", "aux_kw_cycling", "availability",
                          "capex_eur_per_kwh", "cell_share_of_capex"),
        notes="RTE and power/energy are from the datasheet. Tesla does not publish a cycle "
              "rating: 8,000 EFC is the centre of the LFP range in Preger et al. and of current "
              "314 Ah cell ratings.",
    ),
    "megapack_2xl_4h": BatterySpec(
        key="megapack_2xl_4h", name="Tesla Megapack 2 XL (4h)", vendor="Tesla",
        energy_mwh=3.916, power_mw=0.979, rte=0.937,
        cycle_life_efc=8000, eol_capacity_pct=0.70, calendar_life_years=20,
        capex_eur_per_kwh=210.0, cell_share_of_capex=0.60,
        aux_kw_idle=8.0, aux_kw_cycling=10.0, availability=0.97,
        warranty_cycles_per_day=1.0,
        sources=(
            "Datasheet REV 1.6: 979 kW / 3,916 kWh, RTE 93.7% (4h, 25 °C)",
            _BNEF, _PREGER,
        ),
        estimated_fields=("cycle_life_efc", "aux_kw_idle", "aux_kw_cycling", "availability",
                          "capex_eur_per_kwh", "cell_share_of_capex"),
        notes="4h: BNEF puts 4h systems ~11% cheaper per kWh than 2h.",
    ),
    # ── Wärtsilä Quantum2 ─────────────────────────────────────────────────────
    "wartsila_quantum2_2h": BatterySpec(
        key="wartsila_quantum2_2h", name="Wärtsilä Quantum2 (2h, AC estimated)", vendor="Wärtsilä",
        energy_mwh=4.073, power_mw=2.036, rte=0.88,
        cycle_life_efc=8000, eol_capacity_pct=0.70, calendar_life_years=20,
        capex_eur_per_kwh=250.0, cell_share_of_capex=0.55,
        aux_kw_idle=10.0, aux_kw_cycling=30.0, availability=0.97,
        warranty_cycles_per_day=1.0,
        sources=(
            "Quantum2 spec sheet: DC block 4,073 kWh, nominal C-rate 0.1-0.5C, peak auxiliaries 30 kW, -30/+50 °C, 20' HC container",
            "Quantum3 (2026): AC block 5,016 kWh with integrated string inverters; GEMS Digital Energy Platform",
            _BNEF, _PREGER,
        ),
        estimated_fields=("power_mw", "rte", "cycle_life_efc", "aux_kw_idle", "availability",
                          "capex_eur_per_kwh", "cell_share_of_capex"),
        notes="Wärtsilä publishes the DC block; AC power (0.5C) and the 88% RTE are estimates "
              "(external PCS ~98.5% per direction plus transformer).",
    ),
    # ── Fluence Gridstack Pro 5000 ────────────────────────────────────────────
    "fluence_gridstack_pro_4h": BatterySpec(
        key="fluence_gridstack_pro_4h", name="Fluence Gridstack Pro 5000 (4h)", vendor="Fluence",
        energy_mwh=5.644, power_mw=1.411, rte=0.87,
        cycle_life_efc=10000, eol_capacity_pct=0.70, calendar_life_years=20,
        capex_eur_per_kwh=210.0, cell_share_of_capex=0.60,
        aux_kw_idle=8.0, aux_kw_cycling=17.0, availability=0.987,
        warranty_cycles_per_day=1.0,
        sources=(
            "Specs (mirror of BR-042-02-EN): 5,644 kWh / 1,411 kW continuous (0.25C), RTE 87% AC, AESC 530 Ah LFP cells ≥10,000 cycles to 70% SOH, auxiliaries 17 kW when cycling at 4h",
            "Fleet availability 98.7% (MW-weighted), validated by DNV (9 Jun 2026): https://www.globenewswire.com/news-release/2026/06/09/3308929/0/en/DNV-Validates-Fluence-99-Fleet-Performance-Availability-for-Large-Energy-Storage-Projects.html",
            _BNEF,
        ),
        estimated_fields=("aux_kw_idle", "capex_eur_per_kwh", "cell_share_of_capex"),
        notes="The only product here with a published cell cycle rating (10,000 to 70% SOH) "
              "and publicly audited fleet availability.",
    ),
    # ── Sungrow PowerTitan 2.0 — el más vendido en Finlandia 2025-26 ─────────
    "sungrow_powertitan2_2h": BatterySpec(
        key="sungrow_powertitan2_2h", name="Sungrow PowerTitan 2.0 (2h)", vendor="Sungrow",
        energy_mwh=5.015, power_mw=2.520, rte=0.895,
        cycle_life_efc=8000, eol_capacity_pct=0.70, calendar_life_years=20,
        capex_eur_per_kwh=200.0, cell_share_of_capex=0.55,
        aux_kw_idle=8.0, aux_kw_cycling=15.0, availability=0.97,
        warranty_cycles_per_day=1.0,
        sources=(
            "Datasheet ST5015UX V5: 5,015 kWh, AC 2,520 kVA (2h) at 690 V, transformerless, LFP 314 Ah, -30/+50 °C",
            "System RTE 89.5% (launch press release, not in the datasheet)",
            "Used in FI: RPC Kalanti 50 MW / 100 MWh (PowerTitan 2.0): https://www.energy-storage.news/rpc-sungrow-and-epc-suvic-to-build-50mw-100mwh-finland-bess/",
            _BNEF, _PREGER,
        ),
        estimated_fields=("rte", "cycle_life_efc", "aux_kw_idle", "aux_kw_cycling",
                          "availability", "capex_eur_per_kwh", "cell_share_of_capex"),
        notes="Reference product in the Finnish market 2025-26 (Kalanti).",
    ),
    # ── CATL TENER — el claim 'cero degradación' tratado con escepticismo ────
    "catl_tener_4h": BatterySpec(
        key="catl_tener_4h", name="CATL TENER (4h)", vendor="CATL",
        energy_mwh=6.25, power_mw=1.563, rte=0.88,
        cycle_life_efc=12000, eol_capacity_pct=0.70, calendar_life_years=20,
        capex_eur_per_kwh=190.0, cell_share_of_capex=0.55,
        aux_kw_idle=9.0, aux_kw_cycling=15.0, availability=0.97,
        warranty_cycles_per_day=1.5,
        sources=(
            "CATL press release: 6.25 MWh per 20' container, 'zero degradation in the first 5 years', ≥15,000 cycles (manufacturer figures; no independent test published): https://www.catl.com/en/news/6232.html",
            "Industry commentary on how a zero-degradation figure can be reached (e.g. extra installed capacity): https://www.energy-storage.news/industry-reacts-catl-zero-degradation-tener-battery-storage/",
            _BNEF,
        ),
        estimated_fields=("power_mw", "rte", "cycle_life_efc", "aux_kw_idle", "aux_kw_cycling",
                          "availability", "capex_eur_per_kwh", "cell_share_of_capex"),
        notes="Power rating not published (0.25C assumed). The manufacturer's 15,000 cycles are "
              "cut to 12,000 effective EFC as a conservative assumption, since no independent "
              "test is published.",
    ),
    # ── Genérico LFP 2h — para sensibilidad con CAPEX de tender finlandés ─────
    "generic_fi_2h": BatterySpec(
        key="generic_fi_2h", name="Generic LFP 2h (FI 2025-26 tender)", vendor="—",
        energy_mwh=2.0, power_mw=1.0, rte=0.88,
        cycle_life_efc=7000, eol_capacity_pct=0.70, calendar_life_years=18,
        capex_eur_per_kwh=220.0, cell_share_of_capex=0.55,
        aux_kw_idle=4.0, aux_kw_cycling=6.0, availability=0.97,
        warranty_cycles_per_day=1.0,
        sources=(
            "CAPEX between two published Finnish contract values (scope differs): Neve Rovaniemi 30 MW / 80 MWh for about €13m (~163 €/kWh) and Exilion Kuortti 30 MW / 66 MWh for about €17m (~258 €/kWh), Merus Power financial statements bulletin 2025: https://sijoittajat.meruspower.fi/wp-content/uploads/2026/02/Financial-statements-bulletin-2025.pdf",
            _PREGER, _XU,
        ),
        estimated_fields=("rte", "cycle_life_efc", "aux_kw_idle", "aux_kw_cycling", "availability"),
        notes="Parametric 1 MW / 2 MWh battery, comparable with the Value Calculator case "
              "and linearly scalable.",
    ),
}


def list_specs() -> list[dict]:
    """Resumen tabular para el dashboard."""
    rows = []
    for s in BATTERY_SPECS.values():
        rows.append({
            "key": s.key, "Product": s.name, "Manufacturer": s.vendor,
            "Energy (MWh)": s.energy_mwh, "Power (MW)": s.power_mw,
            "Duration (h)": round(s.duration_h, 2), "RTE (%)": round(s.rte * 100, 1),
            "Cycles (EFC)": s.cycle_life_efc,
            "CAPEX (€/kWh)": s.capex_eur_per_kwh,
            "Cycle cost (€/MWh)": round(s.cycle_cost_eur_mwh(), 1),
            "Estimated fields": len(s.estimated_fields),
        })
    return rows
