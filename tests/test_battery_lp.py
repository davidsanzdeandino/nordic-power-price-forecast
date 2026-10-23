"""(b) Battery dispatch LPs: physical limits and the oracle upper bound.

The LP helpers only return settled profit, so the tests wrap
``scipy.optimize.linprog`` to capture the dispatch the solver actually chose and
rebuild the state of charge from it.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.optimize

from src.model import battery_backtest as bb
from src.model import revenue_stacking as rs

TOL = 1e-6


@pytest.fixture
def captured_lp(monkeypatch):
    """Record every linprog result while still returning it to the caller."""
    results = []
    real = scipy.optimize.linprog

    def spy(*args, **kwargs):
        r = real(*args, **kwargs)
        results.append(r)
        return r

    monkeypatch.setattr(scipy.optimize, "linprog", spy)
    return results


def _day_prices(rng, n=24):
    h = np.arange(n)
    return 60 + 35 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 12, n)


@pytest.mark.parametrize("capacity, power", [(2.0, 1.0), (1.0, 1.0), (4.0, 1.0), (1.5, 2.0)])
def test_lp_respects_soc_and_power(captured_lp, rng, capacity, power):
    for _ in range(25):
        prices = _day_prices(rng)
        forecast = prices + rng.normal(0, 10, len(prices))
        captured_lp.clear()
        bb._daily_lp_dispatch(forecast, prices, capacity, power, 0.88)
        bb._daily_lp_dispatch_detailed(forecast, prices, capacity, power, 0.88, deg_per_mwh=5.0)
        assert len(captured_lp) == 2
        for r in captured_lp:
            assert r.success
            n = len(prices)
            c, d = r.x[:n], r.x[n:]
            assert c.min() >= -TOL and c.max() <= power + TOL
            assert d.min() >= -TOL and d.max() <= power + TOL
            assert np.all(c + d <= power + TOL), "charge plus discharge above the power rating"
            soc = np.cumsum(c - d)
            assert soc.min() >= -TOL, "state of charge below zero"
            assert soc.max() <= capacity + TOL, "state of charge above capacity"


NEG_DAY = np.array([-40.0] * 6 + [20.0] * 12 + [150.0] * 6)


def test_lp_handles_negative_prices(captured_lp):
    """With negative prices the LP charges, and still never exceeds the limits."""
    profit = bb._daily_lp_dispatch(NEG_DAY, NEG_DAY, 2.0, 1.0, 0.9)
    r = captured_lp[-1]
    c, d = r.x[:24], r.x[24:]
    soc = np.cumsum(c - d)
    assert soc.max() <= 2.0 + TOL and soc.min() >= -TOL
    assert c[:6].sum() >= 2.0 - TOL          # fills up while being paid to consume
    assert d[18:].sum() >= 2.0 - TOL         # empties in the expensive block
    # At least the physical plan: charge 2 MWh at -40, sell 2 * 0.9 at 150.
    assert profit >= 2 * 40 + 0.9 * 2 * 150 - 1e-6


def test_lp_never_charges_and_discharges_in_the_same_hour(captured_lp):
    """Without c_h + d_h <= P the solver charged and discharged at once at -40 EUR/MWh."""
    bb._daily_lp_dispatch(NEG_DAY, NEG_DAY, 2.0, 1.0, 0.9)
    bb._daily_lp_dispatch_detailed(NEG_DAY, NEG_DAY, 2.0, 1.0, 0.9)
    for r in captured_lp:
        c, d = r.x[:24], r.x[24:]
        assert np.all(c + d <= 1.0 + TOL)
        assert np.minimum(c, d).max() <= TOL


def test_forecast_profit_never_beats_oracle_per_day(rng):
    for _ in range(60):
        prices = _day_prices(rng)
        forecast = prices + rng.normal(0, 20, len(prices))
        oracle = bb._daily_lp_dispatch(prices, prices, 2.0, 1.0, 0.88)
        fc = bb._daily_lp_dispatch(forecast, prices, 2.0, 1.0, 0.88)
        assert fc <= oracle + 1e-6


def test_degradation_aware_oracle_dominates_on_net_value(rng):
    """With a wear cost the oracle maximises profit minus wear, so compare on that."""
    deg = 8.0
    for _ in range(40):
        prices = _day_prices(rng)
        forecast = prices + rng.normal(0, 20, len(prices))
        p_or, t_or = bb._daily_lp_dispatch_detailed(prices, prices, 2.0, 1.0, 0.88, deg)
        p_fc, t_fc = bb._daily_lp_dispatch_detailed(forecast, prices, 2.0, 1.0, 0.88, deg)
        assert p_fc - deg * t_fc <= p_or - deg * t_or + 1e-6


def _hourly_frame(rng, days=30, noise=15.0):
    idx = pd.date_range("2026-03-02", periods=24 * days, freq="h", tz="Europe/Copenhagen")
    idx = idx.tz_convert("UTC")
    h = np.asarray(idx.tz_convert("Europe/Copenhagen").hour)
    actual = 60 + 35 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 12, len(idx))
    return pd.DataFrame({"price_actual": actual,
                         "price_pred": actual + rng.normal(0, noise, len(idx))}, index=idx)


@pytest.mark.parametrize("method", ["lp", "greedy"])
def test_backtest_capture_is_at_most_100pct(rng, method):
    df = _hourly_frame(rng)
    res = bb.battery_arbitrage_backtest(df, method=method)
    assert res["n_days"] >= 28
    assert res["profit_forecast_total"] <= res["profit_perfect_total"] + 1e-6
    assert res["pct_of_oracle_forecast"] <= 100.0 + 1e-9


def test_perfect_forecast_captures_the_whole_oracle(rng):
    df = _hourly_frame(rng, noise=0.0)
    res = bb.battery_arbitrage_backtest(df, method="lp")
    assert res["pct_of_oracle_forecast"] == pytest.approx(100.0, abs=0.05)


def test_economics_backtest_oracle_bound(rng):
    df = _hourly_frame(rng)
    res = bb.battery_economics_backtest(df, deg_per_mwh=0.0)
    assert res["gross_forecast_annual"] <= res["gross_perfect_annual"] + 1e-6
    assert res["cycles_per_day_forecast"] >= 0


def test_stacking_lp_respects_soc_power_and_headroom(captured_lp, rng):
    n = 24
    cap, p_mw, eta = 2.0, 1.0, 0.88
    eta_c = eta_d = np.sqrt(eta)
    for _ in range(15):
        da = _day_prices(rng)
        fcrn = rng.uniform(0, 40, n)
        fcrd_up = rng.uniform(0, 25, n)
        fcrd_dn = rng.uniform(0, 25, n)
        captured_lp.clear()
        rs._daily_stack_lp(da, da, fcrn, fcrd_up, fcrd_dn, cap, p_mw, eta,
                           deg_eur_mwh=2.0, fcrn_reg_cost=1.0)
        r = captured_lp[-1]
        assert r.success
        c, d, rn, ru, rd = (r.x[k * n:(k + 1) * n] for k in range(5))
        soc = np.cumsum(eta_c * c - d / eta_d)
        assert soc.min() >= -TOL and soc.max() <= cap + TOL
        # Reserved power plus energy trades stay within the inverter rating.
        assert np.all(d + rs.FCRN_POWER_FACTOR * rn + ru + rs.FCRD_OPP_POWER_FACTOR * rd <= p_mw + TOL)
        assert np.all(c + rs.FCRN_POWER_FACTOR * rn + rs.FCRD_OPP_POWER_FACTOR * ru + rd <= p_mw + TOL)
        assert np.all(c + d <= p_mw + TOL)
        # Energy headroom required by the reserves.
        assert np.all(soc >= rs.FCRN_ENERGY_H * rn + rs.FCRD_ENERGY_H * ru - TOL)
        assert np.all(soc <= cap - rs.FCRN_ENERGY_H * rn - rs.FCRD_ENERGY_H * rd + TOL)


def test_simple_lp_round_trip_efficiency_is_applied_once():
    """Buy 1 MWh at 0, sell at 100 with RTE 0.81: the arbitrage is worth 81 EUR."""
    p = np.array([0.0, 100.0])
    assert bb._daily_lp_dispatch(p, p, 10.0, 1.0, 0.81) == pytest.approx(81.0)


def test_stacking_lp_round_trip_efficiency_is_applied_once():
    """sqrt(RTE) on charge and on discharge in the SOC balance, and nowhere else (not RTE**2)."""
    p = np.array([0.0, 100.0])
    z = np.zeros(2)
    rev_da = rs._daily_stack_lp(p, p, z, z, z, 10.0, 1.0, 0.81, 0.0, 0.0, allow_reserves=False)[0]
    assert rev_da == pytest.approx(81.0)


def test_stacking_da_only_forecast_below_oracle(rng):
    n = 24
    zero = np.zeros(n)
    for _ in range(30):
        da = _day_prices(rng)
        fc = da + rng.normal(0, 20, n)
        oracle = rs._daily_stack_lp(da, da, zero, zero, zero, 2.0, 1.0, 0.88, 0.0, 0.0,
                                    allow_reserves=False)[0]
        forecast = rs._daily_stack_lp(fc, da, zero, zero, zero, 2.0, 1.0, 0.88, 0.0, 0.0,
                                      allow_reserves=False)[0]
        assert forecast <= oracle + 1e-6
