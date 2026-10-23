"""(e) integration.safe_join turns 15-minute series into hourly means before joining."""

import numpy as np
import pandas as pd
import pytest

from src.transform.integration import safe_join


def _skeleton(start="2025-09-30 20:00", periods=8):
    idx = pd.date_range(start, periods=periods, freq="h", tz="UTC", name="timestamp_utc")
    return pd.DataFrame(index=idx)


def test_quarter_hour_series_is_averaged_to_hourly(quiet_logger):
    master = _skeleton()
    q_idx = pd.date_range("2025-09-30 20:00", periods=8 * 4, freq="15min", tz="UTC")
    values = np.arange(len(q_idx), dtype=float)          # 0, 1, 2, ... per quarter-hour
    df = pd.DataFrame({"price_DK1": values}, index=q_idx)

    out = safe_join(master, df, "prices", quiet_logger)

    assert list(out.index) == list(master.index)
    expected = values.reshape(-1, 4).mean(axis=1)        # 1.5, 5.5, 9.5, ...
    np.testing.assert_allclose(out["price_DK1"].to_numpy(), expected)


def test_mixed_resolution_switch_keeps_hourly_part(quiet_logger):
    """ENTSO-E moved to 15-minute MTU: hourly history followed by quarter-hours."""
    master = _skeleton("2025-09-30 20:00", 8)
    hourly = pd.Series([10.0, 20.0, 30.0, 40.0],
                       index=pd.date_range("2025-09-30 20:00", periods=4, freq="h", tz="UTC"))
    quarter = pd.Series([1.0, 2.0, 3.0, 6.0] * 4,
                        index=pd.date_range("2025-10-01 00:00", periods=16, freq="15min", tz="UTC"))
    df = pd.concat([hourly, quarter]).to_frame("forecast_DK1_Solar")

    out = safe_join(master, df, "forecast_DK1", quiet_logger)

    np.testing.assert_allclose(out["forecast_DK1_Solar"].to_numpy(),
                               [10.0, 20.0, 30.0, 40.0, 3.0, 3.0, 3.0, 3.0])


def test_partial_hour_and_gaps(quiet_logger):
    master = _skeleton("2025-10-01 00:00", 3)
    idx = pd.DatetimeIndex(["2025-10-01 00:00", "2025-10-01 00:15", "2025-10-01 00:30",
                            "2025-10-01 00:45", "2025-10-01 01:00", "2025-10-01 01:30"], tz="UTC")
    df = pd.DataFrame({"x": [4.0, 8.0, np.nan, 12.0, 1.0, 3.0]}, index=idx)

    out = safe_join(master, df, "x", quiet_logger)

    assert out.loc["2025-10-01 00:00", "x"].item() == pytest.approx(8.0)   # NaN ignored
    assert out.loc["2025-10-01 01:00", "x"].item() == pytest.approx(2.0)   # two quarters present
    assert np.isnan(out.loc["2025-10-01 02:00", "x"].item())               # missing hour stays NaN


def test_hourly_series_is_left_untouched(quiet_logger):
    master = _skeleton()
    df = pd.DataFrame({"load_DK1_Actual Load": np.arange(8, dtype=float) * 100},
                      index=master.index)
    out = safe_join(master, df, "load_DK1", quiet_logger)
    np.testing.assert_allclose(out["load_DK1_Actual Load"].to_numpy(), np.arange(8) * 100.0)


def test_naive_index_is_read_as_utc_and_left_join_keeps_skeleton(quiet_logger):
    master = _skeleton("2025-10-01 00:00", 3)
    naive = pd.date_range("2025-09-30 23:00", periods=5 * 4, freq="15min")   # tz-naive, starts early
    df = pd.DataFrame({"v": np.ones(len(naive))}, index=naive)
    out = safe_join(master, df, "v", quiet_logger)
    assert out.index.equals(master.index)
    assert str(out.index.tz) == "UTC"
    np.testing.assert_allclose(out["v"].to_numpy(), [1.0, 1.0, 1.0])


def test_overlapping_columns_are_prefixed(quiet_logger):
    master = _skeleton()
    master["price_DK1"] = 1.0
    df = pd.DataFrame({"price_DK1": 2.0}, index=master.index)
    out = safe_join(master, df, "entsoe", quiet_logger)
    assert (out["price_DK1"] == 1.0).all()
    assert (out["entsoe_price_DK1"] == 2.0).all()


def test_empty_frame_is_skipped(quiet_logger):
    master = _skeleton()
    out = safe_join(master, pd.DataFrame(), "empty", quiet_logger)
    assert out is master
