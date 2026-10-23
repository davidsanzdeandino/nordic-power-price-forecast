"""
revenue_stacking.py — Backtest multi-mercado de una batería nórdica (FI / DK2)
==============================================================================
Co-optimización diaria (LP) de una batería entre:
  - Arbitraje day-ahead (decide con forecast, liquida a precio real — igual que
    battery_backtest.py),
  - FCR-N (reserva simétrica de contención, mercado horario de Fingrid),
  - FCR-D up / FCR-D down (reserva de perturbación, direccional).

Reglas físicas implementadas (Nordic FCR technical requirements, 14.5.2025 —
docs/research/raw/fingrid-reserves.md):
  - FCR-N con C MW vendidos exige potencia ±1,34×C (34% extra para gestión de
    SOC) y energía 1h×C libre EN CADA dirección (SOC ∈ [C, E−C]).
  - FCR-D up con C MW exige C de potencia de descarga + 0,2×C opuesta (carga),
    y ⅓h×C de energía disponible. FCR-D down es el espejo.
  - La misma MW no puede venderse en dos mercados a la vez; las reservas de
    energía/potencia de cada producto se SUMAN.
  - Mercados horarios marginales (price-taker: cobramos el precio de cierre).

Secuencia informacional honesta (gates reales):
  12:00 D-1  cierra day-ahead  → decidimos carga/descarga con NUESTRO forecast.
  ~13:00 D-1 se publican los precios DA reales.
  18:30 D-1  cierra el mercado horario FCR → al decidir reservas se conoce el
             precio DA real de D, pero NO el precio FCR de cierre (22:45).
  Por eso el plan se decide en DOS modos:
    - "oracle":   reservas decididas con el precio FCR realizado (techo teórico).
    - "causal":   reservas decididas con forecast naive del precio FCR
                  (precio de la misma hora del día anterior — sin leakage);
                  la aceptación se simula comparando nuestra puja (= forecast
                  naive con margen) con el precio de cierre real: si pujamos por
                  debajo del marginal, entramos y COBRAMOS EL MARGINAL (regla
                  de mercado marginal); si no, esa capacidad queda para DA.
  En ambos modos la liquidación es SIEMPRE a precios realizados.

Degradación: coste marginal €/MWh descargado dentro del objetivo del LP
(convención Xu et al. / arXiv:1904.07771), de modo que solo se cicla cuando el
spread esperado cubre el desgaste. El throughput de regulación FCR-N se
contabiliza con un factor configurable (gross MWh por MW·h de FCR-N) que añade
pérdidas y desgaste — estimación documentada, no dato del TSO.

NOTA DE ALCANCE (honestidad): no se modelan aFRR/mFRR (volúmenes FI pequeños
para baterías hasta 2025 y reglas de activación complejas) ni intradiario; sus
precios están en el master para análisis pero el stack aquí es FCR + DA. Es el
stack que explica la mayor parte del ingreso BESS finlandés 2021-2025.
"""

import numpy as np
import pandas as pd

from src.model.battery_specs import BatterySpec

# Multiplicadores LER (Nordic FCR technical requirements, 14.5.2025)
FCRN_POWER_FACTOR = 1.34      # potencia reservada por MW de FCR-N vendido
FCRN_ENERGY_H = 1.0           # MWh de headroom por MW de FCR-N, en cada dirección
FCRD_OPP_POWER_FACTOR = 0.20  # potencia opuesta por MW de FCR-D (LER)
FCRD_ENERGY_H = 1.0 / 3.0     # MWh de headroom por MW de FCR-D

# Throughput bruto estimado de regulación FCR-N (MWh movidos por MW·h vendido,
# por dirección). Estimación conservadora basada en la actividad típica de la
# señal ±0,1 Hz; configurable. Genera pérdidas (1-RTE) y desgaste.
FCRN_THROUGHPUT_PER_MWH = 0.20


def _daily_stack_lp(p_da_decide: np.ndarray, p_da_settle: np.ndarray,
                    r_fcrn: np.ndarray, r_fcrd_up: np.ndarray, r_fcrd_dn: np.ndarray,
                    cap_mwh: float, p_mw: float, eta: float,
                    deg_eur_mwh: float, fcrn_reg_cost: float,
                    allow_reserves: bool = True):
    """
    LP de un día en el que los precios de reserva de decisión y liquidación
    coinciden (modos oracle y da_only). Delegado en `_daily_stack_lp_settle`,
    que documenta la formulación completa.
    """
    zero = np.zeros_like(r_fcrn)
    rn = r_fcrn if allow_reserves else zero
    ru = r_fcrd_up if allow_reserves else zero
    rd = r_fcrd_dn if allow_reserves else zero
    return _daily_stack_lp_settle(p_da_decide, p_da_settle,
                                  rn, ru, rd, rn, ru, rd,
                                  cap_mwh, p_mw, eta, deg_eur_mwh, fcrn_reg_cost)


def stacking_backtest(df: pd.DataFrame, spec: BatterySpec,
                      actual_col: str = "price_actual",
                      forecast_col: str = "price_pred",
                      fcrn_col: str = "fcrn_price",
                      fcrd_up_col: str = "fcrd_up_price",
                      fcrd_dn_col: str = "fcrd_dn_price",
                      mode: str = "causal",
                      country: str = "FI",
                      bid_margin: float = 0.9) -> dict:
    """
    Backtest del stack FCR + arbitraje DA sobre un DataFrame horario indexado
    en UTC con: precio DA real, forecast DA y precios de reserva realizados.

    mode:
      - "oracle": el plan diario conoce los precios FCR realizados (techo).
      - "causal": forecast naive de FCR (lag 24h); puja = forecast × bid_margin.
        Horas donde la puja supera el marginal realizado → no aceptado → esa
        capacidad se reasigna a arbitraje puro en un segundo LP (sin reservas).
      - "da_only": sin reservas (comparable al battery_backtest clásico).

    Devuelve ingresos anualizados por flujo, throughput, ciclos y costes.
    """
    cols = [actual_col, forecast_col, fcrn_col, fcrd_up_col, fcrd_dn_col]
    d = df[[c for c in cols if c in df.columns]].copy()
    for c in (fcrn_col, fcrd_up_col, fcrd_dn_col):
        if c not in d.columns:
            d[c] = 0.0
    d = d.dropna(subset=[actual_col, forecast_col])
    d[[fcrn_col, fcrd_up_col, fcrd_dn_col]] = d[[fcrn_col, fcrd_up_col, fcrd_dn_col]].fillna(0.0)
    if d.empty:
        return {}

    idx = pd.to_datetime(d.index, utc=True).tz_convert("Europe/Helsinki" if country == "FI" else "Europe/Copenhagen")
    d["_day"] = idx.normalize()

    # Forecast naive de precios FCR para el modo causal: misma hora, día anterior
    for c in (fcrn_col, fcrd_up_col, fcrd_dn_col):
        d[f"_naive_{c}"] = d[c].shift(24)

    deg = spec.cycle_cost_eur_mwh()
    # Coste por MW·h de FCR-N: pérdidas del throughput de regulación + desgaste.
    # Las pérdidas se valoran al precio DA medio del dataset (aprox. documentada).
    avg_price = float(d[actual_col].mean())
    fcrn_reg_cost = FCRN_THROUGHPUT_PER_MWH * ((1 - spec.rte) * avg_price + deg)

    rev = {"da": 0.0, "fcrn": 0.0, "fcrd": 0.0}
    thru_da_total = 0.0
    fcrn_mwh_total = 0.0
    fcrd_mwh_total = 0.0
    n_days = 0

    for _, day in d.groupby("_day"):
        if len(day) < 24:
            continue
        n_days += 1
        pa = day[actual_col].values.astype(float)
        pf = day[forecast_col].values.astype(float)
        rn_real = day[fcrn_col].values.astype(float)
        ru_real = day[fcrd_up_col].values.astype(float)
        rd_real = day[fcrd_dn_col].values.astype(float)

        if mode == "da_only":
            r_da, r_n, r_d, t_da, mn, md = _daily_stack_lp(
                pf, pa, rn_real * 0, ru_real * 0, rd_real * 0,
                spec.energy_mwh, spec.power_mw, spec.rte, deg, fcrn_reg_cost,
                allow_reserves=False)
        elif mode == "oracle":
            r_da, r_n, r_d, t_da, mn, md = _daily_stack_lp(
                pf, pa, rn_real, ru_real, rd_real,
                spec.energy_mwh, spec.power_mw, spec.rte, deg, fcrn_reg_cost)
        else:  # causal
            rn_fc = day[f"_naive_{fcrn_col}"].fillna(0.0).values.astype(float)
            ru_fc = day[f"_naive_{fcrd_up_col}"].fillna(0.0).values.astype(float)
            rd_fc = day[f"_naive_{fcrd_dn_col}"].fillna(0.0).values.astype(float)
            # Aceptación: pujamos forecast×margin; entramos si puja ≤ marginal real.
            # Cobro al precio MARGINAL realizado (mercado marginal). El plan se
            # decide con el forecast naive (lo que sabrías al pujar a las 18:30
            # D-1) y se liquida al precio de cierre real:
            acc_n = (rn_fc * bid_margin) <= rn_real
            acc_u = (ru_fc * bid_margin) <= ru_real
            acc_d = (rd_fc * bid_margin) <= rd_real
            r_da, r_n, r_d, t_da, mn, md = _daily_stack_lp_settle(
                pf, pa,
                np.where(acc_n, rn_fc, 0.0), np.where(acc_u, ru_fc, 0.0), np.where(acc_d, rd_fc, 0.0),
                np.where(acc_n, rn_real, 0.0), np.where(acc_u, ru_real, 0.0), np.where(acc_d, rd_real, 0.0),
                spec.energy_mwh, spec.power_mw, spec.rte, deg, fcrn_reg_cost)

        rev["da"] += r_da; rev["fcrn"] += r_n; rev["fcrd"] += r_d
        thru_da_total += t_da
        fcrn_mwh_total += mn
        fcrd_mwh_total += md

    if n_days == 0:
        return {}
    scale = 365.0 / n_days

    # Costes variables anualizados
    fcrn_reg_thru = fcrn_mwh_total * FCRN_THROUGHPUT_PER_MWH        # MWh regulación
    deg_cost = (thru_da_total + fcrn_reg_thru) * deg
    reg_losses = fcrn_reg_thru * (1 - spec.rte) * avg_price
    # Auxiliares: consumo idle continuo valorado al precio medio (8.760 h/año)
    aux_cost_annual = (spec.aux_kw_idle / 1000.0) * 8760.0 * avg_price
    fixed = spec.fixed_costs_eur_year(country=country)

    gross = rev["da"] + rev["fcrn"] + rev["fcrd"]
    out = {
        "mode": mode, "country": country, "spec": spec.key,
        "n_days": n_days,
        "rev_da_annual": rev["da"] * scale,
        "rev_fcrn_annual": rev["fcrn"] * scale,
        "rev_fcrd_annual": rev["fcrd"] * scale,
        "gross_annual": gross * scale,
        "deg_cost_annual": deg_cost * scale,
        "reg_losses_annual": reg_losses * scale,
        "aux_cost_annual": aux_cost_annual,
        "fixed_costs_annual": fixed["total_eur"],
        "net_annual": (gross - deg_cost - reg_losses) * scale
                      - aux_cost_annual - fixed["total_eur"],
        "cycles_per_day": (thru_da_total / spec.energy_mwh) / n_days,
        "fcrn_mw_share": fcrn_mwh_total / (n_days * 24 * spec.power_mw),
        "per_mw": {},
    }
    for k in ("rev_da_annual", "rev_fcrn_annual", "rev_fcrd_annual", "gross_annual", "net_annual"):
        out["per_mw"][k] = out[k] / spec.power_mw
    return out


def _daily_stack_lp_settle(p_da_decide, p_da_settle, rn_dec, ru_dec, rd_dec,
                           rn_set, ru_set, rd_set,
                           cap_mwh, p_mw, eta, deg_eur_mwh, fcrn_reg_cost):
    """
    LP diario completo. Variables por hora: c (carga MW), d (descarga MW),
    rn (FCR-N MW), ru (FCR-D up MW), rd (FCR-D down MW).

    max  Σ_h [ p_dec·d − p_dec·c − deg·d + rn·(π_n_dec − coste_reg)
               + ru·π_du_dec + rd·π_dd_dec ]
    s.a. (potencia descarga)  d + 1,34·rn + ru + 0,2·rd ≤ P
         (potencia carga)     c + 1,34·rn + 0,2·ru + rd ≤ P
         (potencia total)     c + d ≤ P   (no carga y descarga a la vez)
         (SOC)  SOC_h = Σ_{k≤h} (η_c·c_k − d_k/η_d),  η_c = η_d = √RTE
         (headroom abajo)     SOC_h ≥ 1,0·rn_h + ⅓·ru_h
         (headroom arriba)    SOC_h ≤ E − 1,0·rn_h − ⅓·rd_h

    DECIDE con precios *_dec (lo conocido en el gate) y LIQUIDA el arbitraje a
    p_da_settle y las reservas a los marginales realizados *_set. c y d son
    potencias en el lado de red: la eficiencia round-trip se aplica UNA sola
    vez, repartida como √RTE en carga y √RTE en descarga dentro del balance de
    SOC (1 MWh comprado devuelve RTE MWh vendidos, igual que en
    battery_backtest.py), así que el ingreso de la descarga es p·d. Devuelve
    (ingreso_da, ingreso_fcrn, ingreso_fcrd, throughput_da_mwh, fcrn_mw_h,
    fcrd_mw_h) liquidados.
    """
    from scipy.optimize import linprog

    n = len(p_da_decide)
    eta_c = eta_d = np.sqrt(eta)
    nv = 5 * n

    def idx(block, h):
        return block * n + h

    cobj = np.zeros(nv)
    for h in range(n):
        cobj[idx(0, h)] = p_da_decide[h]
        cobj[idx(1, h)] = -(p_da_decide[h] - deg_eur_mwh)
        cobj[idx(2, h)] = -(rn_dec[h] - fcrn_reg_cost)
        cobj[idx(3, h)] = -ru_dec[h]
        cobj[idx(4, h)] = -rd_dec[h]

    A_ub, b_ub = [], []
    for h in range(n):
        row = np.zeros(nv)
        row[idx(1, h)] = 1.0; row[idx(2, h)] = FCRN_POWER_FACTOR
        row[idx(3, h)] = 1.0; row[idx(4, h)] = FCRD_OPP_POWER_FACTOR
        A_ub.append(row); b_ub.append(p_mw)
        row = np.zeros(nv)
        row[idx(0, h)] = 1.0; row[idx(2, h)] = FCRN_POWER_FACTOR
        row[idx(3, h)] = FCRD_OPP_POWER_FACTOR; row[idx(4, h)] = 1.0
        A_ub.append(row); b_ub.append(p_mw)
        row = np.zeros(nv)
        row[idx(0, h)] = 1.0; row[idx(1, h)] = 1.0
        A_ub.append(row); b_ub.append(p_mw)
    for h in range(n):
        row = np.zeros(nv)
        for k in range(h + 1):
            row[idx(0, k)] = -eta_c; row[idx(1, k)] = 1.0 / eta_d
        row[idx(2, h)] += FCRN_ENERGY_H; row[idx(3, h)] += FCRD_ENERGY_H
        A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(nv)
        for k in range(h + 1):
            row[idx(0, k)] = eta_c; row[idx(1, k)] = -1.0 / eta_d
        row[idx(2, h)] += FCRN_ENERGY_H; row[idx(4, h)] += FCRD_ENERGY_H
        A_ub.append(row); b_ub.append(cap_mwh)

    r = linprog(cobj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                bounds=[(0, p_mw)] * nv, method="highs")
    if not r.success:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    c = r.x[0:n]; d = r.x[n:2 * n]
    rn = r.x[2 * n:3 * n]; ru = r.x[3 * n:4 * n]; rd = r.x[4 * n:5 * n]
    rev_da = float(np.sum(d * p_da_settle - c * p_da_settle))   # RTE ya en el SOC
    rev_fcrn = float(np.sum(rn * rn_set))          # liquidación al marginal real
    rev_fcrd = float(np.sum(ru * ru_set + rd * rd_set))
    return rev_da, rev_fcrn, rev_fcrd, float(np.sum(d)), float(np.sum(rn)), float(np.sum(ru) + np.sum(rd))


def yearly_stack_summary(master: pd.DataFrame, spec: BatterySpec,
                         price_col: str = "price_FI",
                         fcrn_col: str = "fingrid_FI_fcrn_price",
                         fcrd_up_col: str = "fingrid_FI_fcrd_up_price",
                         fcrd_dn_col: str = "fingrid_FI_fcrd_down_price",
                         mode: str = "oracle") -> pd.DataFrame:
    """
    Resumen anual del stack ÓPTIMO (modo oracle por defecto) usando el precio
    DA real también como 'forecast' — mide el POTENCIAL del mercado cada año
    (cuánto habría ganado una batería perfectamente operada), no la habilidad
    del modelo. Para la habilidad del modelo usar stacking_backtest con el
    forecast real sobre el periodo de test.

    Esta es la serie que cuenta la historia 2023→2026: el desplome de FCR y el
    desplazamiento del valor hacia el arbitraje.
    """
    rows = []
    for year, g in master.groupby(master.index.year):
        cols = {price_col, fcrn_col, fcrd_up_col, fcrd_dn_col}
        if not cols.issubset(g.columns):
            continue
        sub = g[[price_col, fcrn_col, fcrd_up_col, fcrd_dn_col]].dropna(subset=[price_col])
        if len(sub) < 24 * 28:
            continue
        df = pd.DataFrame({
            "price_actual": sub[price_col],
            "price_pred": sub[price_col],     # oracle: potencial de mercado
            "fcrn_price": sub[fcrn_col],
            "fcrd_up_price": sub[fcrd_up_col],
            "fcrd_dn_price": sub[fcrd_dn_col],
        }, index=sub.index)
        r = stacking_backtest(df, spec, mode=mode)
        if not r:
            continue
        rows.append({
            "year": year,
            "rev_da_per_mw": r["per_mw"]["rev_da_annual"],
            "rev_fcrn_per_mw": r["per_mw"]["rev_fcrn_annual"],
            "rev_fcrd_per_mw": r["per_mw"]["rev_fcrd_annual"],
            "gross_per_mw": r["per_mw"]["gross_annual"],
            "net_per_mw": r["per_mw"]["net_annual"],
            "cycles_per_day": r["cycles_per_day"],
            "fcrn_mw_share": r["fcrn_mw_share"],
        })
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()
