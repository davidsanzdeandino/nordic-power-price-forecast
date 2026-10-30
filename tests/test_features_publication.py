"""Inputs whose value is published late, or whose normal and fill values could
see the test window.

1. The seasonal normal of the reservoir anomaly comes from training rows only;
   weeks without training data take the training mean of the zone.
2. A weekly reservoir value is used only for delivery days whose auction comes
   after its publication deadline (third working day after its week, which is
   the Friday of the next week at the latest with two weekday holidays).
3. The load forecast of add_physical_features is filled with the training
   median.
4. Fingrid's day-ahead forecasts cover the Finnish day. The last CET hour of a
   delivery day (00:00-01:00 Finnish time of the next day) is published after
   the auction, so the features of that hour take the previous hour instead.

Every test runs on synthetic frames and calls the feature functions directly.
"""

import numpy as np
import pandas as pd
import pytest

from src.model import features as feat_mod
from src.model.config import DELIVERY_TZ, TRAIN_END

TRAIN_END_TS = pd.Timestamp(TRAIN_END, tz="UTC")


# ── 1-2. Reservoirs ──────────────────────────────────────────────────────────

def _reservoir_frame(rng, start="2025-01-01", end="2026-05-31 23:00") -> pd.DataFrame:
    """Hourly frame with weekly reservoir levels stamped at Monday 00:00 CET.

    As in the master: ENTSO-E stamps each weekly value at the start of its week
    (Sunday 23:00 UTC all year) and integration.py forward-fills it.
    """
    idx = pd.date_range(start, end, freq="h", tz="UTC", name="timestamp_utc")
    week = (idx.tz_localize(None) + pd.Timedelta(hours=1)).to_period("W-SUN")
    df = pd.DataFrame(index=idx)
    for z in ["NO_2", "SE_3"]:
        levels = pd.Series(rng.uniform(1e6, 5e6, week.nunique()), index=week.unique())
        df[f"reservoir_{z}_stored_mwh"] = levels.reindex(week).to_numpy()
    return df


def _week_of(idx: pd.DatetimeIndex) -> pd.PeriodIndex:
    """Reservoir week (Monday 00:00 CET to Sunday 24:00 CET) of each UTC hour."""
    return (idx.tz_localize(None) + pd.Timedelta(hours=1)).to_period("W-SUN")


def test_reservoir_normal_uses_training_rows_only(quiet_logger):
    df = _reservoir_frame(np.random.default_rng(3))
    ref = feat_mod.add_hydro_features(df.copy(), quiet_logger)

    perturbed = df.copy()
    after = perturbed.index > TRAIN_END_TS
    for col in [c for c in perturbed.columns if c.startswith("reservoir_")]:
        perturbed.loc[after, col] *= 10.0
    out = feat_mod.add_hydro_features(perturbed, quiet_logger)

    before = ref.index <= TRAIN_END_TS
    for col in ("feat_reservoir_NO_2_anomaly", "feat_reservoir_SE_3_anomaly",
                "feat_reservoir_nordic_anomaly"):
        assert ref.loc[before, col].notna().any()
        pd.testing.assert_series_equal(ref.loc[before, col], out.loc[before, col])


def test_reservoir_normal_is_weekly_train_mean_with_fallback(quiet_logger):
    df = _reservoir_frame(np.random.default_rng(4))
    out = feat_mod.add_hydro_features(df.copy(), quiet_logger)
    lagged = df["reservoir_NO_2_stored_mwh"].shift(feat_mod.RESERVOIR_LAG)
    train = df.index <= TRAIN_END_TS
    iso = pd.Series(df.index.isocalendar().week.astype(int).to_numpy(), index=df.index)

    # A week with training data: normal = mean of the lagged level over the
    # training hours of that ISO week.
    t = pd.Timestamp("2025-08-20 12:00", tz="UTC")
    wk = iso[t]
    normal = lagged[train & (iso == wk).to_numpy()].mean()
    assert out.loc[t, "feat_reservoir_NO_2_anomaly"] == pytest.approx(lagged[t] - normal)

    # A week with no training hour (the frame starts in January 2025 and
    # training ends in September): training mean.
    t = pd.Timestamp("2025-12-10 12:00", tz="UTC")
    assert not (train & (iso == iso[t]).to_numpy()).any()
    assert out.loc[t, "feat_reservoir_NO_2_anomaly"] == pytest.approx(
        lagged[t] - lagged[train].mean())


def _publication_deadline(week: pd.Period) -> pd.Timestamp:
    """End of the Friday after the week (third working day with two holidays)."""
    end_of_friday_next = week.end_time.normalize() + pd.Timedelta(days=6)   # Saturday 00:00
    return end_of_friday_next.tz_localize(DELIVERY_TZ)


@pytest.mark.parametrize("day", pd.date_range("2025-10-20", "2025-11-09", freq="D"))
def test_reservoir_features_only_use_published_weeks(day, quiet_logger):
    df = _reservoir_frame(np.random.default_rng(5), start="2025-08-01", end="2025-11-30 23:00")
    ref = feat_mod.add_hydro_features(df.copy(), quiet_logger)

    delivery = pd.Timestamp(day).tz_localize(DELIVERY_TZ)
    auction = delivery - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
    weeks = _week_of(df.index)
    unpublished = np.array([_publication_deadline(w) > auction for w in weeks])
    perturbed = df.copy()
    for col in [c for c in perturbed.columns if c.startswith("reservoir_")]:
        perturbed.loc[unpublished, col] += 7e6
    out = feat_mod.add_hydro_features(perturbed, quiet_logger)

    day_end = (delivery + pd.Timedelta(days=1)).tz_convert("UTC")
    upto = ref.index < day_end
    cols = [c for c in ref.columns if c.startswith("feat_reservoir_")]
    a = ref.loc[upto, cols].to_numpy(dtype=float)
    b = out.loc[upto, cols].to_numpy(dtype=float)
    same = np.isclose(a, b, equal_nan=True).all(axis=0)
    assert [c for c, ok in zip(cols, same) if not ok] == []


def test_reservoir_lag_is_the_shortest_safe_one(quiet_logger):
    """The first delivery day that uses week W is the Sunday after W (13 days)."""
    assert feat_mod.RESERVOIR_LAG == 13 * 24
    df = _reservoir_frame(np.random.default_rng(6), start="2025-08-01", end="2025-11-30 23:00")
    out = feat_mod.add_hydro_features(df.copy(), quiet_logger)
    col = f"feat_reservoir_NO_2_{feat_mod.RESERVOIR_LAG_TAG}"
    week = pd.Period("2025-11-03", freq="W-SUN")          # Monday 3 to Sunday 9 November
    level_w = df.loc[_week_of(df.index) == week, "reservoir_NO_2_stored_mwh"].iloc[0]
    for day, uses_w in [("2025-11-15", False), ("2025-11-16", True), ("2025-11-22", True),
                        ("2025-11-23", False)]:
        d = pd.Timestamp(day).tz_localize(DELIVERY_TZ)
        hours = (out.index >= d.tz_convert("UTC")) & \
                (out.index < (d + pd.Timedelta(days=1)).tz_convert("UTC"))
        assert (out.loc[hours, col] == level_w).all() == uses_w, day


# ── 3. Load forecast fill value ──────────────────────────────────────────────

def test_load_fc_fill_value_is_the_training_median(quiet_logger):
    idx = pd.date_range("2025-08-01", "2025-12-31 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(8)
    df = pd.DataFrame(index=idx)
    df["forecast_DK1_Wind Onshore"] = rng.uniform(0, 2500, len(idx))
    df["forecast_DK1_Solar"] = 0.0
    df["load_fc_DK1_Forecasted Load"] = rng.uniform(3000, 4000, len(idx))
    after = idx > TRAIN_END_TS
    df.loc[after, "load_fc_DK1_Forecasted Load"] += 1e5     # val/test far away
    hole = idx[after][50]
    df.loc[hole, "load_fc_DK1_Forecasted Load"] = np.nan
    train_median = df.loc[~after, "load_fc_DK1_Forecasted Load"].median()

    out = feat_mod.add_physical_features(df.copy(), quiet_logger)
    expected = train_median - df.loc[hole, "forecast_DK1_Wind Onshore"]
    assert out.loc[hole, "feat_DK1_residual_load"] == pytest.approx(expected)


# ── 4. Fingrid's last CET hour ───────────────────────────────────────────────

def _fingrid_frame(rng) -> pd.DataFrame:
    idx = pd.date_range("2025-10-15", "2025-11-10 23:00", freq="h", tz="UTC",
                        name="timestamp_utc")
    n = len(idx)
    df = pd.DataFrame(index=idx)
    df["fingrid_FI_load_fc_mw"] = 9000 + rng.normal(0, 300, n)
    df["fingrid_FI_wind_fc_mw"] = rng.uniform(0, 6000, n)
    df["fingrid_FI_solar_fc_mw"] = rng.uniform(0, 500, n)
    df["fingrid_FI_ol3_fc_mw"] = rng.uniform(1300, 1600, n)
    df["load_fc_FI_Forecasted Load"] = 9000 + rng.normal(0, 300, n)
    return df


def _finnish_day(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert("Europe/Helsinki").tz_localize(None).normalize()


def _cet_day(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert(DELIVERY_TZ).tz_localize(None).normalize()


@pytest.mark.parametrize("day", ["2025-10-24", "2025-10-25", "2025-10-26",
                                 "2025-10-27", "2025-11-05"])
def test_fingrid_features_ignore_forecasts_published_after_the_auction(day, quiet_logger):
    """Values of Finnish day F are published at 12:00 Finnish time on F-1, which
    is after the auction of CET day D whenever F > D. They must not move any
    Fingrid feature of day D or earlier (26 October 2025 is the 25-hour day)."""
    df = _fingrid_frame(np.random.default_rng(11))
    ref = feat_mod.add_fingrid_features(df.copy(), quiet_logger)

    d = pd.Timestamp(day)
    late = np.asarray(_finnish_day(df.index) > d)
    perturbed = df.copy()
    for col in feat_mod.FINGRID_FC_COLS:
        perturbed.loc[late, col] += 5e4
    out = feat_mod.add_fingrid_features(perturbed, quiet_logger)

    upto = np.asarray(_cet_day(df.index) <= d)
    cols = [c for c in ref.columns if c.startswith("feat_FI_")]
    assert len(cols) >= 7
    a = ref.loc[upto, cols].to_numpy(dtype=float)
    b = out.loc[upto, cols].to_numpy(dtype=float)
    same = np.isclose(a, b, equal_nan=True).all(axis=0)
    assert [c for c, ok in zip(cols, same) if not ok] == []


def test_fingrid_last_cet_hour_takes_the_previous_hour(quiet_logger):
    df = _fingrid_frame(np.random.default_rng(12))
    out = feat_mod.add_fingrid_features(df.copy(), quiet_logger)
    last = np.asarray(_finnish_day(df.index) > _cet_day(df.index))
    # Exactly the 23:00-24:00 CET hour of each day, also on the 25-hour day.
    assert (last == (df.index.tz_convert(DELIVERY_TZ).hour == 23)).all()

    for raw, feat in [("fingrid_FI_load_fc_mw", "feat_FI_fingrid_load_fc"),
                      ("fingrid_FI_wind_fc_mw", "feat_FI_fingrid_wind_fc"),
                      ("fingrid_FI_ol3_fc_mw", "feat_FI_fingrid_ol3_fc")]:
        expected = df[raw].copy()
        prev = df[raw].shift(1)
        expected[last] = prev[last]
        pd.testing.assert_series_equal(out[feat], expected, check_names=False)
    # The 24-hour lag keeps the published value of the same hour of D-1.
    resid = (df["fingrid_FI_load_fc_mw"] - df["fingrid_FI_wind_fc_mw"]
             - df["fingrid_FI_solar_fc_mw"])
    pd.testing.assert_series_equal(out["feat_FI_fingrid_resid_load_lag24h"],
                                   resid.shift(24), check_names=False)
