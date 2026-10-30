"""(d) The day-ahead AR term only uses residuals of earlier delivery days.

The auction for delivery day D (a local day in Europe/Copenhagen) closes at
12:00 on D-1, when every price of D-1 is already known (it cleared in the
auction of D-2) but none of D. ``day_ahead_offsets`` must therefore build the
offsets of D from residuals up to 23:00 local time on D-1 only, step the AR
path by hours since local midnight (23- and 25-hour days at the clock change),
and use the right step for a partial first day.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.model.ar_correction import (
    ARResidualCorrector, day_ahead_offsets, simulate_day_ahead_correction,
)
from src.model.config import DELIVERY_TZ

ONE_H = pd.Timedelta(hours=1)


def _ar1_series(rng, n, phi=0.85, sigma=5.0, mu=2.0):
    e = rng.normal(0, sigma, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = mu * (1 - phi) + phi * r[t - 1] + e[t]
    return r


@pytest.fixture
def corrector(rng):
    r = _ar1_series(rng, 3000)
    return ARResidualCorrector(max_order=3).fit(r, np.zeros_like(r), verbose=False)


def _residuals(rng, start, end):
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    return pd.Series(_ar1_series(rng, len(idx)), index=idx)


def _local_days(idx):
    return idx.tz_convert(DELIVERY_TZ).normalize()


def test_residuals_from_delivery_day_onwards_do_not_change_its_offsets(corrector, rng):
    resid = _residuals(rng, "2025-11-01", "2025-11-20 23:00")
    target = pd.date_range("2025-11-05", "2025-11-15 23:00", freq="h", tz="UTC")
    ref = day_ahead_offsets(target, resid, corrector)
    days = _local_days(target)
    for day in days.unique()[1:-1]:
        cutoff = day.tz_convert("UTC")
        perturbed = resid.copy()
        perturbed[perturbed.index >= cutoff] += 1000.0
        out = day_ahead_offsets(target, perturbed, corrector)
        in_day = np.asarray(days == day)
        before = np.asarray(target < cutoff)
        np.testing.assert_array_equal(out[in_day], ref[in_day])
        np.testing.assert_array_equal(out[before], ref[before])
        # ...while the next day does see the change (the test can detect a dependency).
        assert not np.allclose(out[np.asarray(days > day)], ref[np.asarray(days > day)])


def test_last_residual_used_is_23h_local_of_previous_day(corrector, rng):
    resid = _residuals(rng, "2026-01-01", "2026-01-10 23:00")
    target = pd.date_range("2026-01-04 23:00", "2026-01-05 22:00", freq="h", tz="UTC")  # 5 Jan local
    ref = day_ahead_offsets(target, resid, corrector)
    last = pd.Timestamp("2026-01-04 23:00", tz=DELIVERY_TZ).tz_convert("UTC")
    p = resid.copy()
    p[last] += 50.0
    assert not np.allclose(day_ahead_offsets(target, p, corrector), ref)
    # Equals the AR path seeded with the residuals up to 23:00 local of D-1.
    order = corrector.best_order
    seed = resid[resid.index <= last].iloc[-order:].to_numpy()
    path = corrector.correct_day_ahead(np.zeros(24), seed)
    np.testing.assert_allclose(ref, path)


@pytest.mark.parametrize("day, n_hours", [("2026-03-29", 23), ("2025-10-26", 25)])
def test_clock_change_days_use_hours_since_local_midnight(corrector, rng, day, n_hours):
    resid = _residuals(rng, "2025-10-01", "2026-04-30 23:00")
    start = pd.Timestamp(day, tz=DELIVERY_TZ)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(DELIVERY_TZ)   # next local midnight
    target = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC") - ONE_H,
                           freq="h", tz="UTC")
    assert len(target) == n_hours
    off = day_ahead_offsets(target, resid, corrector)
    order = corrector.best_order
    seed = resid[resid.index < start.tz_convert("UTC")].iloc[-order:].to_numpy()
    path = corrector.correct_day_ahead(np.zeros(n_hours), seed)
    np.testing.assert_allclose(off, path)


def test_partial_first_day_uses_the_right_step(corrector, rng):
    """The test window starts at 00:00 UTC on 1 January, which is 01:00 local."""
    resid = _residuals(rng, "2025-12-01", "2026-01-03 23:00")
    target = pd.date_range("2026-01-01 00:00", "2026-01-01 22:00", freq="h", tz="UTC")
    off = day_ahead_offsets(target, resid, corrector)
    cutoff = pd.Timestamp("2026-01-01", tz=DELIVERY_TZ).tz_convert("UTC")
    seed = resid[resid.index < cutoff].iloc[-corrector.best_order:].to_numpy()
    path = corrector.correct_day_ahead(np.zeros(24), seed)
    np.testing.assert_allclose(off, path[1:24])     # first requested hour is step 1


def test_missing_previous_day_residual_gives_zero_offset(corrector, rng):
    resid = _residuals(rng, "2026-02-01", "2026-02-10 23:00")
    target = pd.date_range("2026-02-04 23:00", "2026-02-06 22:00", freq="h", tz="UTC")  # 5-6 Feb local
    gap = pd.Timestamp("2026-02-04 23:00", tz=DELIVERY_TZ).tz_convert("UTC")
    with pytest.warns(UserWarning, match="delivery day"):
        off = day_ahead_offsets(target, resid.drop(gap), corrector)
    days = _local_days(target)
    first = np.asarray(days == days[0])
    assert np.all(off[first] == 0.0)
    assert np.any(off[~first] != 0.0)


def test_simulation_with_index_matches_offsets_and_skips_first_local_day(corrector, rng):
    idx = pd.date_range("2026-01-01", "2026-01-20 23:00", freq="h", tz="UTC")
    y_pred = rng.normal(60, 10, len(idx))
    y_true = y_pred + _ar1_series(rng, len(idx))
    res = simulate_day_ahead_correction(y_true, y_pred, corrector, index=idx, verbose=False)
    off = day_ahead_offsets(idx, pd.Series(y_true - y_pred, index=idx), corrector, warn=False)
    np.testing.assert_allclose(res["corrected"], y_pred + off)
    days = _local_days(idx)
    np.testing.assert_array_equal(res["scored"], np.asarray(days != days[0]))
    e = (y_true - res["corrected"])[res["scored"]]
    assert res["RMSE_day_ahead_corrected"] == pytest.approx(np.sqrt(np.mean(e ** 2)))


def test_simulation_without_index_is_deprecated(corrector, rng):
    y = rng.normal(0, 1, 24 * 5)
    with pytest.warns(DeprecationWarning):
        simulate_day_ahead_correction(y, np.zeros_like(y), corrector, verbose=False)


def test_ar_decision_on_validation_is_out_of_sample(rng):
    """train.decide_ar_on_val: each scored month uses an AR fitted on earlier months only."""
    from src.model.train import decide_ar_on_val

    idx = pd.date_range("2025-10-01", "2025-12-31 23:00", freq="h", tz="UTC")
    pred = rng.normal(60, 10, len(idx))
    y = pd.Series(pred + _ar1_series(rng, len(idx), phi=0.95, mu=8.0), index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no day may lack its residuals
        res = decide_ar_on_val(y, pred)
    assert [f["score_month"] for f in res["folds"]] == ["2025-11", "2025-12"]
    local_month = idx.tz_convert(DELIVERY_TZ).month
    assert res["folds"][0]["n_fit_hours"] == int((local_month == 10).sum())
    assert res["folds"][1]["n_fit_hours"] == int(((local_month == 10) | (local_month == 11)).sum())
    assert res["use_ar"] is True                 # persistent bias: the AR term helps
    assert res["MAE_ar_scored"] < res["MAE_raw_scored"]
    assert res["corrector"].fitted
