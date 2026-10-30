"""(e) Diebold-Mariano with HAC variance, and the model-choice rule built on it.

Hourly loss differentials are strongly autocorrelated, so the test must use a
Newey-West (HAC) variance over h = 24 lags plus the Harvey-Leybourne-Newbold
correction. With h = 1 and no HLN correction it reduces to the iid version.
"""

import numpy as np
import pytest
from scipy import stats

from src.model import dm
from src.model.evaluate import diebold_mariano as evaluate_dm
from src.model.train import choose_served_model


def _ar1(rng, n, phi):
    x = np.zeros(n)
    e = rng.normal(0, 1, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_h1_without_hln_is_the_iid_statistic(rng):
    e_a = rng.normal(0, 10, 2000)
    e_b = rng.normal(0, 11, 2000)
    stat, p = dm.diebold_mariano(e_a, e_b, loss="mae", h=1, hln=False)
    d = np.abs(e_a) - np.abs(e_b)
    iid = d.mean() / np.sqrt(np.var(d) / len(d))
    assert stat == pytest.approx(iid, rel=1e-12)
    assert p == pytest.approx(2 * (1 - stats.t.cdf(abs(iid), df=len(d) - 1)), rel=1e-9)


def test_sign_convention_negative_means_first_model_better(rng):
    y_err_small = rng.normal(0, 5, 3000)
    y_err_big = rng.normal(0, 8, 3000)
    stat, p = dm.diebold_mariano(y_err_small, y_err_big)
    assert stat < 0 and p < 0.01
    stat2, _ = dm.diebold_mariano(y_err_big, y_err_small)
    assert stat2 == pytest.approx(-stat)


def test_hac_variance_matches_bartlett_formula(rng):
    d = _ar1(rng, 500, 0.7)
    h = 24
    dbar = d.mean()
    expected = np.mean((d - dbar) ** 2)
    for k in range(1, h):
        expected += 2 * (1 - k / h) * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    assert dm.hac_variance(d, h) == pytest.approx(expected, rel=1e-12)


def test_hac_shrinks_the_statistic_under_autocorrelation(rng):
    """Positively autocorrelated differentials: the iid test overstates significance."""
    n = 3600
    base = rng.normal(0, 10, n)
    d_noise = 0.4 + 3.0 * _ar1(rng, n, 0.9)           # persistent loss differential
    e_b = base
    e_a = np.sign(base) * (np.abs(base) + d_noise)    # |e_a| - |e_b| = d_noise
    iid, _ = dm.diebold_mariano(e_a, e_b, h=1, hln=False)
    hac, p_hac = dm.diebold_mariano(e_a, e_b, h=24)
    assert abs(hac) < 0.6 * abs(iid)
    assert p_hac > dm.diebold_mariano(e_a, e_b, h=1, hln=False)[1]


def test_evaluate_reexports_the_shared_test(rng):
    a, b = rng.normal(0, 3, 500), rng.normal(0, 4, 500)
    assert evaluate_dm(a, b) == dm.diebold_mariano(a, b, loss="mae", h=24)


def test_nan_pairs_are_dropped_and_short_series_return_nan(rng):
    a, b = rng.normal(0, 3, 400), rng.normal(0, 4, 400)
    a_nan = a.copy()
    a_nan[:10] = np.nan
    assert dm.diebold_mariano(a_nan, b) == dm.diebold_mariano(a[10:], b[10:])
    s, p = dm.diebold_mariano(a[:20], b[:20])
    assert np.isnan(s) and np.isnan(p)


# ── Model choice on the validation quarter (train.choose_served_model) ──────

def test_choice_lower_mae_ensemble_wins(rng):
    y = rng.normal(60, 20, 2000)
    res = choose_served_model(y, y + rng.normal(0, 5, 2000), y + rng.normal(0, 8, 2000))
    assert res["served_model"] == "ensemble"


def test_choice_lightgbm_needs_significance(rng):
    y = rng.normal(60, 20, 2200)
    noise = rng.normal(0, 6, 2200)
    # LightGBM clearly better: served.
    res = choose_served_model(y, y + noise * 1.6, y + noise)
    assert res["served_model"] == "lightgbm" and res["DM_p_value"] < 0.05
    # LightGBM barely better (not significant): the ensemble is kept.
    other = rng.normal(0, 6, 2200)
    other *= 1.002 * np.abs(noise).mean() / np.abs(other).mean()
    tiny = choose_served_model(y, y + other, y + noise)
    assert tiny["MAE_val_lightgbm_train"] < tiny["MAE_val_ensemble"]
    assert tiny["DM_p_value"] >= 0.05
    assert tiny["served_model"] == "ensemble"
