"""(c2) Quantile bands are sorted per hour after the conformal step.

The P10, P25, P50, P75 and P90 models are fitted separately, so on some hours
one quantile can lie above the next one. src/model/conformal.sort_quantile_bands
sorts the values of each hour (monotone rearrangement). It must keep the set of
values of each hour, remove every crossing, leave hours that were already in
order untouched and never raise the mean pinball loss over the quantiles.
"""

import numpy as np

from src.model.conformal import count_crossed_hours, sort_quantile_bands

LEVELS = {"P10": 0.10, "P25": 0.25, "P50": 0.50, "P75": 0.75, "P90": 0.90}


def _crossed_bands(rng, n=2000):
    centre = 60 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24)
    spread = {"P10": -15.0, "P25": -6.0, "P50": 0.0, "P75": 6.0, "P90": 15.0}
    # Independent noise per quantile, large enough to make neighbours cross.
    bands = {q: centre + s + rng.normal(0, 5, n) for q, s in spread.items()}
    y = centre + rng.standard_t(df=5, size=n) * 10
    return y, bands


def _mean_pinball(y, bands):
    losses = []
    for q, tau in LEVELS.items():
        e = y - bands[q]
        losses.append(np.maximum(tau * e, (tau - 1) * e))
    return float(np.mean(losses))


def test_sorting_removes_crossings_and_keeps_values(rng):
    y, bands = _crossed_bands(rng)
    assert count_crossed_hours(bands) > 100, "the synthetic bands must cross"

    out = sort_quantile_bands(bands)
    assert count_crossed_hours(out) == 0
    stacked = np.column_stack([out[q] for q in LEVELS])
    assert np.all(np.diff(stacked, axis=1) >= 0)
    raw = np.column_stack([bands[q] for q in LEVELS])
    np.testing.assert_array_equal(np.sort(raw, axis=1), stacked)


def test_ordered_hours_are_untouched_and_pinball_does_not_rise(rng):
    y, bands = _crossed_bands(rng)
    out = sort_quantile_bands(bands)
    raw = np.column_stack([bands[q] for q in LEVELS])
    ok = np.all(np.diff(raw, axis=1) >= 0, axis=1)
    assert ok.any()
    for q in LEVELS:
        np.testing.assert_array_equal(out[q][ok], bands[q][ok])
    assert _mean_pinball(y, out) <= _mean_pinball(y, bands) + 1e-12


def test_sorting_leaves_other_keys_alone():
    bands = {"P10": np.array([5.0, 1.0]), "P90": np.array([4.0, 9.0]),
             "price_pred": np.array([3.0, 3.0])}
    out = sort_quantile_bands(bands)
    np.testing.assert_array_equal(out["P10"], [4.0, 1.0])
    np.testing.assert_array_equal(out["P90"], [5.0, 9.0])
    np.testing.assert_array_equal(out["price_pred"], [3.0, 3.0])
    # The input dictionary is not modified in place.
    np.testing.assert_array_equal(bands["P10"], [5.0, 1.0])
