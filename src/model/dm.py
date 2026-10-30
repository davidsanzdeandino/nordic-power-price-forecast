"""
dm.py
=====
Diebold-Mariano test of equal predictive accuracy, shared by train.py (model
choice on the validation quarter), evaluate.py (descriptive tests on the test
window) and the Transparency page.

Hourly forecast errors are strongly autocorrelated, so the variance of the
loss differential uses a HAC (Newey-West) estimator with Bartlett weights over
h - 1 lags, followed by the Harvey-Leybourne-Newbold small-sample correction
and a two-sided p-value from Student's t with n - 1 degrees of freedom
(Diebold and Mariano, 1995; Harvey, Leybourne and Newbold, 1997; the standard
in electricity price forecasting, Lago et al., 2021).

Sign convention: DM < 0 means that forecast A (the first argument) is more
accurate than forecast B.
"""

import numpy as np


def loss_differential(e_a, e_b, loss: str = "mae") -> np.ndarray:
    """d_t = L(e_a,t) - L(e_b,t) with NaN pairs removed."""
    a = np.asarray(e_a, dtype=float)
    b = np.asarray(e_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("e_a and e_b must have the same length")
    if loss == "mse":
        d = a ** 2 - b ** 2
    elif loss == "mae":
        d = np.abs(a) - np.abs(b)
    else:
        raise ValueError(f"unknown loss: {loss}")
    return d[~np.isnan(d)]


def hac_variance(d, h: int) -> float:
    """Newey-West long-run variance of d with Bartlett weights 1 - k/h, k < h."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    dbar = d.mean()
    var = float(np.mean((d - dbar) ** 2))
    for k in range(1, min(h, n)):
        cov = float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
        var += 2.0 * (1.0 - k / h) * cov
    return var


def diebold_mariano(e_a, e_b, loss: str = "mae", h: int = 24,
                    hln: bool = True, min_n: int = 30) -> tuple:
    """
    Diebold-Mariano statistic and two-sided p-value.

    Args:
        e_a, e_b: forecast errors (actual - forecast or the reverse, the same
                  way for both) of models A and B, aligned in time.
        loss:     'mae' (absolute loss) or 'mse' (squared loss).
        h:        forecast horizon in hours; the HAC window. h=1 gives the
                  iid variance.
        hln:      apply the Harvey-Leybourne-Newbold correction.
        min_n:    below this many valid pairs the test returns (nan, nan).

    Returns:
        (DM, p_value). DM < 0 means A is more accurate than B.
    """
    from scipy import stats

    d = loss_differential(e_a, e_b, loss)
    n = len(d)
    if n < max(min_n, 2):
        return np.nan, np.nan
    var = hac_variance(d, h)
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan
    dm = float(d.mean() / np.sqrt(var / n))
    if hln:
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        if factor <= 0:
            return np.nan, np.nan
        dm *= float(np.sqrt(factor))
    p = float(2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1)))
    return dm, p
