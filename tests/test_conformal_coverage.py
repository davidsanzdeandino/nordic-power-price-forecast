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
