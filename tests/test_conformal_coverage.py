"""(c) Conformal calibration reaches the target coverage on synthetic series.

Two mechanisms in src/model/conformal.py are checked:
  - ConformalCalibrator(strategy="adaptive"): CQR with width-normalised scores,
    fitted on a calibration block and applied to a later block of the same process.
  - adaptive_conformal_coverage: online ACI (Gibbs & Candes, 2021), which must hold
    the long-run coverage even when the volatility changes half-way through.
"""

import numpy as np
import pandas as pd
import pytest

from src.model.conformal import ConformalCalibrator, adaptive_conformal_coverage


def _heteroscedastic(rng, n, start):
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    h = np.asarray(ts.hour)
    scale = np.where((h >= 16) & (h <= 20), 25.0, 10.0)
    y = 55 + 15 * np.sin(2 * np.pi * h / 24) + rng.standard_t(df=5, size=n) * scale
    pred = 55 + 15 * np.sin(2 * np.pi * h / 24)
    # Quantile bands that know the shape of the uncertainty but are far too narrow.
    half = 0.35 * scale
    return ts, y, pred, pred - half, pred + half


@pytest.mark.parametrize("target", [0.80, 0.90])
def test_adaptive_calibrator_hits_target_out_of_sample(rng, target):
    _, y_cal, p_cal, lo_cal, hi_cal = _heteroscedastic(rng, 3000, "2025-01-01")
    _, y_te, p_te, lo_te, hi_te = _heteroscedastic(rng, 6000, "2025-06-01")

    raw = np.mean((y_te >= lo_te) & (y_te <= hi_te))
    assert raw < target - 0.25, "the raw bands must under-cover for the test to mean anything"

    cal = ConformalCalibrator(strategy="adaptive", target_coverage=target)
    cal.fit(y_cal, p_cal, lo_cal, hi_cal, verbose=False)
    lo, hi = cal.calibrate(p_te, lo_te, hi_te)
    cov = np.mean((y_te >= lo) & (y_te <= hi))
    assert abs(cov - target) < 0.025, f"coverage {cov:.3f} vs target {target}"
    # Adaptive bands stay wider where the base band is wider.
    width = hi - lo
    base = hi_te - lo_te
    assert np.corrcoef(width, base)[0, 1] > 0.99


def test_adaptive_calibrator_in_sample_diagnostics(rng):
    _, y, p, lo, hi = _heteroscedastic(rng, 2000, "2025-01-01")
    cal = ConformalCalibrator(strategy="adaptive", target_coverage=0.80).fit(y, p, lo, hi, verbose=False)
    d = cal.get_diagnostics()
    assert d["calibrated_coverage_insample"] >= 0.80
    assert d["calibrated_band_width"] > d["raw_band_width"]


def test_calibrator_rejects_tiny_calibration_set():
    y = np.zeros(10)
    with pytest.raises(ValueError):
        ConformalCalibrator("adaptive").fit(y, y, y - 1, y + 1, verbose=False)


def test_aci_tracks_target_under_volatility_shift(rng):
    n = 8000
    scale = np.where(np.arange(n) < n // 2, 10.0, 30.0)   # volatility triples
    y = rng.normal(0, 1, n) * scale
    # Static band calibrated for the calm half only (about 80% there).
    lo, hi = np.full(n, -12.8), np.full(n, 12.8)

    gamma = 0.02
    res = adaptive_conformal_coverage(y, lo, hi, target_coverage=0.80, gamma=gamma)
    assert res["coverage_static"] < 0.65                   # static band breaks
    assert abs(res["coverage_aci"] - 0.80) < 0.02          # ACI holds the long-run rate
    assert res["k_final"] > 1.5                            # and did widen the band
    # ACI identity (k starts at 1 and never hits the 0 floor here):
    # k_final - 1 = gamma * sum(err - alpha)  =>  gap = (k_final - 1) / (gamma * n)
    assert res["coverage_aci"] == pytest.approx(0.80 - (res["k_final"] - 1.0) / (gamma * n), abs=1e-9)


def test_aci_gap_shrinks_with_series_length(rng):
    gaps = []
    for n in (2000, 8000, 32000):
        y = rng.normal(0, 1, n) * np.where(np.arange(n) < n // 2, 10.0, 30.0)
        res = adaptive_conformal_coverage(y, np.full(n, -12.8), np.full(n, 12.8), 0.80, 0.02)
        gaps.append(abs(res["coverage_aci"] - 0.80))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 0.005


def test_daily_aci_keeps_k_constant_within_each_delivery_day(rng):
    """Daily ACI: the band of day D only uses errors up to 23:00 of D-1."""
    idx = pd.date_range("2026-01-01", "2026-04-30 23:00", freq="h", tz="UTC")
    n = len(idx)
    days = idx.tz_convert("Europe/Copenhagen").normalize()
    scale = np.where(np.arange(n) < n // 2, 10.0, 30.0)
    y = rng.normal(0, 1, n) * scale
    lo, hi = np.full(n, -12.8), np.full(n, 12.8)

    res = adaptive_conformal_coverage(y, lo, hi, 0.80, gamma=0.03,
                                      day_labels=days, return_path=True)
    k = res["k_path"]
    for day in days.unique():
        in_day = np.asarray(days == day)
        assert np.ptp(k[in_day]) == 0.0
    assert k[0] == 1.0
    assert res["update"] == "daily"
    # Moving the prices of one day changes k only from the next day onwards.
    day = days.unique()[40]
    in_day = np.asarray(days == day)
    y2 = y.copy()
    y2[in_day] += 500.0
    k2 = adaptive_conformal_coverage(y2, lo, hi, 0.80, gamma=0.03,
                                     day_labels=days, return_path=True)["k_path"]
    upto = np.asarray(days <= day)
    np.testing.assert_array_equal(k2[upto], k[upto])
    assert not np.allclose(k2[~upto], k[~upto])
    # Still tracks the target under the volatility shift.
    assert abs(res["coverage_aci"] - 0.80) < 0.03
    assert res["coverage_static"] < 0.65


def test_daily_aci_update_is_the_sum_of_the_day(rng):
    idx = pd.date_range("2026-02-01", periods=72, freq="h", tz="UTC")
    days = idx.tz_convert("Europe/Copenhagen").normalize()
    y = rng.normal(0, 20, len(idx))
    lo, hi = np.full(len(idx), -10.0), np.full(len(idx), 10.0)
    res = adaptive_conformal_coverage(y, lo, hi, 0.80, gamma=0.03,
                                      day_labels=days, return_path=True)
    k, cov = res["k_path"], res["covered_path"]
    first = np.asarray(days == days[0])
    second_start = int(np.argmax(~first))
    expected = max(0.0, 1.0 + 0.03 * float(np.sum((~cov[first]).astype(float) - 0.2)))
    assert k[second_start] == pytest.approx(expected)
