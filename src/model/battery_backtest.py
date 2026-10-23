"""
battery_backtest.py
===================
Backtest económico de arbitraje con batería (BESS) en el mercado day-ahead.

Traduce la calidad del forecast en VALOR ECONÓMICO (€) — el benchmark de
aplicación estándar en la literatura EPF para evaluar forecasts (no basta el MAE).

Estrategia (arbitraje diario, 1 ciclo/día):
  Cada día, con las 24 horas del mercado, una batería de capacidad C (MWh) y
  potencia P (MW) carga en las horas baratas y descarga en las caras. Las horas
  se ELIGEN con el forecast (lo que sabrías al pujar) pero se LIQUIDAN con el
  precio REAL (lo que realmente cobras). Esto mide el valor real del modelo.

Se comparan 3 estrategias:
  - perfect : decide con el precio REAL  → techo teórico (oráculo)
  - forecast: decide con NUESTRO forecast → valor del modelo
  - naive   : horario fijo (carga madrugada, descarga tarde) → suelo

Métrica clave (literatura): % del beneficio del oráculo capturado por el forecast.

NOTA (honestidad): se asume que la batería es tomadora de precios (no mueve el
mercado) y un único ciclo/día. Ignora la dependencia intertemporal fina, como
es habitual en este benchmark. No es un optimizador de trading de producción.
"""

import numpy as np
import pandas as pd


def _daily_arbitrage_profit(prices_decide: np.ndarray, prices_settle: np.ndarray,
                            n_hours: int, power_mw: float, efficiency: float) -> float:
    """
    Beneficio de un día (24 valores). Carga en las `n_hours` más baratas según
    `prices_decide`, descarga en las `n_hours` más caras, y liquida con
    `prices_settle`. Solo opera si el spread esperado es positivo.
    """
    n = len(prices_decide)
    if n < 2 * n_hours:
        n_hours = n // 2
    if n_hours == 0:
        return 0.0
    order = np.argsort(prices_decide)
    charge_h   = order[:n_hours]              # más baratas (según forecast)
    discharge_h = order[-n_hours:]            # más caras (según forecast)
    # Solo arbitrar si, según el forecast, descargar sale a cuenta tras eficiencia
    exp_buy  = prices_decide[charge_h].mean()
    exp_sell = prices_decide[discharge_h].mean()
    if efficiency * exp_sell <= exp_buy:
        return 0.0
    # Liquidación a precio REAL
    cost    = power_mw * prices_settle[charge_h].sum()
    revenue = efficiency * power_mw * prices_settle[discharge_h].sum()
    return float(revenue - cost)


def _soc_power_constraints(n: int, capacity_mwh: float, power_mw: float):
    """
    Restricciones A_ub·x ≤ b_ub del LP diario, con x = [c_0..c_{n-1}, d_0..d_{n-1}]:
      - SOC_h = Σ_{k≤h}(c_k − d_k) ∈ [0, cap]   → 2n desigualdades
      - potencia total por hora: c_h + d_h ≤ P  → n desigualdades
    Sin la última, con precios negativos el LP cargaría y descargaría a la vez
    en la misma hora y apuntaría como beneficio las pérdidas (1 − η)·|p|·P.
    Los límites 0 ≤ c_h, d_h ≤ P van en `bounds`.
    """
    L = np.tril(np.ones((n, n)))
    eye = np.eye(n)
    A_ub = np.vstack([np.hstack([L, -L]),      # SOC ≤ cap
                      np.hstack([-L, L]),      # −SOC ≤ 0
                      np.hstack([eye, eye])])  # c_h + d_h ≤ P
    b_ub = np.concatenate([np.full(n, capacity_mwh), np.zeros(n), np.full(n, power_mw)])
    return A_ub, b_ub


def _daily_lp_dispatch(prices_decide: np.ndarray, prices_settle: np.ndarray,
                       capacity_mwh: float, power_mw: float, efficiency: float) -> float:
    """
    Dispatch ÓPTIMO de un día por programación lineal (LP), decidido sobre
    `prices_decide` (forecast) y liquidado sobre `prices_settle` (real).

    Permite ciclos parciales/múltiples (más general que el greedy de 1 ciclo).
    Variables: carga c_h, descarga d_h (MW) por hora. SOC acumulado ∈ [0, cap]
    y potencia total por hora c_h + d_h ≤ P.
    Maximiza el beneficio esperado; se liquida al precio real.
    """
    from scipy.optimize import linprog
    n = len(prices_decide)
    if n < 2:
        return 0.0
    p = np.asarray(prices_decide, float)
    # min  Σ p·c − η·p·d   (= −beneficio esperado)
    cobj = np.concatenate([p, -efficiency * p])
    A_ub, b_ub = _soc_power_constraints(n, capacity_mwh, power_mw)
    bounds = [(0, power_mw)] * (2 * n)
    r = linprog(cobj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not r.success:
        return 0.0
    c, d = r.x[:n], r.x[n:]
    ps = np.asarray(prices_settle, float)
    return float(np.sum(efficiency * d * ps - c * ps))


def _daily_lp_dispatch_detailed(prices_decide: np.ndarray, prices_settle: np.ndarray,
                                capacity_mwh: float, power_mw: float,
                                efficiency: float,
                                deg_per_mwh: float = 0.0) -> tuple[float, float]:
    """
    Igual que `_daily_lp_dispatch` pero devuelve (beneficio_liquidado, energía
    descargada de las celdas en MWh) para poder contabilizar ciclos/degradación.

    `deg_per_mwh` penaliza cada MWh descargado en la decisión: un operador real
    solo cicla cuando el spread esperado cubre el desgaste de la batería. Con
    deg=0 se recupera el dispatch puramente arbitrador (sobre-ciclador).
    """
    from scipy.optimize import linprog
    n = len(prices_decide)
    if n < 2:
        return 0.0, 0.0
    p = np.asarray(prices_decide, float)
    # min  Σ p·c − (η·p − deg)·d   → solo descarga si η·p_venta − deg supera p_compra
    cobj = np.concatenate([p, -(efficiency * p - deg_per_mwh)])
    A_ub, b_ub = _soc_power_constraints(n, capacity_mwh, power_mw)
    bounds = [(0, power_mw)] * (2 * n)
    r = linprog(cobj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not r.success:
        return 0.0, 0.0
    c, d = r.x[:n], r.x[n:]
    ps = np.asarray(prices_settle, float)
    profit = float(np.sum(efficiency * d * ps - c * ps))
    discharge_mwh = float(np.sum(d))          # energía sacada de las celdas (≈ ciclos)
    return profit, discharge_mwh


def battery_economics_backtest(df: pd.DataFrame,
                               actual_col: str = "price_actual",
                               forecast_col: str = "price_pred",
                               capacity_mwh: float = 2.0,
                               power_mw: float = 1.0,
                               efficiency: float = 0.88,
                               deg_per_mwh: float = 0.0,
                               naive_charge=(0, 1, 2, 3),
                               naive_discharge=(17, 18, 19, 20)) -> dict:
    """
    Backtest de arbitraje con batería que además devuelve la **energía ciclada**
    (para degradación) de las 3 estrategias (oracle / forecast / naive), usando
    dispatch LP óptimo **consciente de la degradación** (`deg_per_mwh`): el
    despacho solo cicla cuando el spread esperado cubre el desgaste, como haría
    un operador real. El beneficio bruto se liquida al precio real; la economía
    final (capex, payback) se aplica aguas arriba.

    Returns dict por estrategia: beneficio bruto anualizado de arbitraje
    (liquidado a precio real) y throughput (MWh descargados) anualizado, más
    metadatos (días, ciclos/día).
    """
    d = df[[actual_col, forecast_col]].dropna().copy()
    if d.empty:
        return {}
    idx = pd.to_datetime(d.index, utc=True).tz_convert("Europe/Copenhagen")
    d["_hour"] = idx.hour
    d["_day"]  = idx.normalize()

    gross = {"perfect": 0.0, "forecast": 0.0, "naive": 0.0}
    thru  = {"perfect": 0.0, "forecast": 0.0, "naive": 0.0}
    n_hours_naive = max(1, int(round(capacity_mwh / power_mw)))
    n_days = 0

    for _, day in d.groupby("_day"):
        if len(day) < 24:
            continue
        actual   = day[actual_col].values
        forecast = day[forecast_col].values
        hours    = day["_hour"].values
        n_days += 1

        p_perf, t_perf = _daily_lp_dispatch_detailed(actual, actual, capacity_mwh, power_mw, efficiency, deg_per_mwh)
        p_fc,   t_fc   = _daily_lp_dispatch_detailed(forecast, actual, capacity_mwh, power_mw, efficiency, deg_per_mwh)
        gross["perfect"]  += p_perf; thru["perfect"]  += t_perf
        gross["forecast"] += p_fc;   thru["forecast"] += t_fc

        # Naive: horas fijas (madrugada/tarde), liquidado a precio real
        ch  = np.array([i for i, h in enumerate(hours) if h in naive_charge])[:n_hours_naive]
        dis = np.array([i for i, h in enumerate(hours) if h in naive_discharge])[:n_hours_naive]
        if len(ch) and len(dis):
            gross["naive"] += float(efficiency * power_mw * actual[dis].sum()
                                    - power_mw * actual[ch].sum())
            thru["naive"]  += float(power_mw * len(dis))

    if n_days == 0:
        return {}
    scale = 365.0 / n_days
    out = {"n_days": n_days, "capacity_mwh": capacity_mwh, "power_mw": power_mw,
           "efficiency": efficiency}
    for k in ("perfect", "forecast", "naive"):
        out[f"gross_{k}_annual"]   = gross[k] * scale
        out[f"thru_{k}_annual"]    = thru[k] * scale
    out["cycles_per_day_forecast"] = (thru["forecast"] / capacity_mwh) / n_days
    return out


def battery_arbitrage_backtest(df: pd.DataFrame,
                               actual_col: str = "price_actual",
                               forecast_col: str = "price_pred",
                               capacity_mwh: float = 2.0,
                               power_mw: float = 1.0,
                               efficiency: float = 0.90,
                               naive_charge=(0, 1, 2, 3),
                               naive_discharge=(17, 18, 19, 20),
                               method: str = "greedy") -> dict:
    """
    Ejecuta el backtest sobre un DataFrame horario con precio real y forecast.

    Args:
        df:            índice temporal horario; columnas actual_col y forecast_col.
        capacity_mwh:  energía de la batería (MWh).
        power_mw:      potencia (MW). n_hours de carga = capacity/power.
        efficiency:    eficiencia round-trip (0-1).
        naive_*:       horas (local) fijas de la estrategia naive.

    Returns:
        dict con beneficios totales y anualizados de las 3 estrategias y el
        % del oráculo capturado por el forecast.
    """
    d = df[[actual_col, forecast_col]].dropna().copy()
    if d.empty:
        return {}
    # Hora local para la estrategia naive
    idx = pd.to_datetime(d.index, utc=True).tz_convert("Europe/Copenhagen")
    d["_hour"] = idx.hour
    d["_day"]  = idx.normalize()

    n_hours = max(1, int(round(capacity_mwh / power_mw)))
    prof = {"perfect": 0.0, "forecast": 0.0, "naive": 0.0}
    n_days = 0

    for _, day in d.groupby("_day"):
        if len(day) < 24:
            continue
        actual   = day[actual_col].values
        forecast = day[forecast_col].values
        hours    = day["_hour"].values
        n_days += 1
        if method == "lp":
            prof["perfect"]  += _daily_lp_dispatch(actual, actual, capacity_mwh, power_mw, efficiency)
            prof["forecast"] += _daily_lp_dispatch(forecast, actual, capacity_mwh, power_mw, efficiency)
        else:
            # Oráculo: decide con el real
            prof["perfect"]  += _daily_arbitrage_profit(actual, actual, n_hours, power_mw, efficiency)
            # Forecast: decide con el forecast, liquida con el real
            prof["forecast"] += _daily_arbitrage_profit(forecast, actual, n_hours, power_mw, efficiency)
        # Naive: horas fijas, pero RESPETANDO la capacidad (n_hours), no más.
        ch = np.array([i for i, h in enumerate(hours) if h in naive_charge])[:n_hours]
        dis = np.array([i for i, h in enumerate(hours) if h in naive_discharge])[:n_hours]
        if len(ch) and len(dis):
            cost = power_mw * actual[ch].sum()
            rev  = efficiency * power_mw * actual[dis].sum()
            prof["naive"] += float(rev - cost)

    if n_days == 0:
        return {}
    scale = 365.0 / n_days
    capture = (prof["forecast"] / prof["perfect"] * 100) if prof["perfect"] > 0 else np.nan
    capture_naive = (prof["naive"] / prof["perfect"] * 100) if prof["perfect"] > 0 else np.nan
    return {
        "n_days": n_days, "method": method,
        "capacity_mwh": capacity_mwh, "power_mw": power_mw, "efficiency": efficiency,
        "profit_perfect_total":  round(prof["perfect"], 1),
        "profit_forecast_total": round(prof["forecast"], 1),
        "profit_naive_total":    round(prof["naive"], 1),
        "profit_perfect_annual":  round(prof["perfect"] * scale, 0),
        "profit_forecast_annual": round(prof["forecast"] * scale, 0),
        "profit_naive_annual":    round(prof["naive"] * scale, 0),
        "pct_of_oracle_forecast": round(capture, 1),
        "pct_of_oracle_naive":    round(capture_naive, 1),
        "forecast_vs_naive_uplift_annual": round((prof["forecast"] - prof["naive"]) * scale, 0),
    }
